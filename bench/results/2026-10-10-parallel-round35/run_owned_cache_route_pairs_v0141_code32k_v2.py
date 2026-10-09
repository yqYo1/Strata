"""SOURCE ONLY / UNEXECUTED: owned cache-route-pairs four-fresh32K diagnostic.

Root-only sequential execution with flushed Level Zero diagnostics and exact
reference heads/live state/LP/MTP. Logged times are excluded from speed claims.
"""
from pathlib import Path
import datetime
import array
import struct
import shutil
import hashlib
import json
import math
import os
import re
import selectors
import subprocess
import sys
import time
import tty
import types
import fcntl
import argparse


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_fresh(result):
    fields = result['protocol'][-1].split()
    assert len(fields) >= 8 and fields[0] == 'DONE'
    generated, prompt = int(fields[1]), int(fields[2])
    prompt_ms, decode_ms = float(fields[3]), float(fields[4])
    resume = [int(x.split()[1]) for x in result['protocol']
              if x.startswith(('RESUME ', 'REUSED '))]
    progress = [int(x.split()[1]) for x in result['protocol'] if x.startswith('PP ')]
    return {
        'full_32768_input': prompt == 32768,
        'fresh_resume_and_reused': bool(resume) and all(x == 0 for x in resume)
            and any(x.startswith('RESUME ') for x in result['protocol'])
            and any(x.startswith('REUSED ') for x in result['protocol']),
        'complete_prefill_prefix': bool(progress) and max(progress) == 32767,
        'visible_output_complete': 0 < generated <= 64
            and len(result['ids']) == generated and len(result['logprobs']) == generated,
        'normal_finish': fields[5] in ['length', 'stop'],
        'finite_logprobs': all(math.isfinite(float(item.rsplit(':', 1)[-1]))
            for value in result['logprobs'] for item in value.split()[1:]),
        'finite_positive_times': all(math.isfinite(x) and x > 0 for x in [prompt_ms, decode_ms]),
        'valid_mtp_counts': 0 <= int(fields[6]) <= int(fields[7]),
    }


def same_output(actual, expected):
    return {key + '_equal': actual[key] == expected[key]
            for key in ['ids', 'logprobs', 'mtp_counts', 'finish_reason']}


def normalize_reference(request):
    done=[value.split() for value in request['protocol'] if value.startswith('DONE ')]
    assert len(done)==1 and len(done[0])>=8, 'reference needs one complete DONE'
    fields=done[0]
    assert int(fields[1])==len(request['ids'])==len(request['logprobs'])==64
    assert int(fields[2])==32768 and fields[5] in ['length','stop']
    result=dict(request)
    result['finish_reason']=fields[5]
    counts=list(map(int,fields[6:8]))
    assert result.get('mtp_counts',counts)==counts
    result['mtp_counts']=counts
    return result


HISTOGRAM_PREFIX = 'strata native dispatch exposure: decode request delta; '
HISTOGRAM_LEGEND = ('; eligible bits=IQ512:1,IQ256:2,KQ256:4,IQ4NL:8,Q2:16; '
    'variant=scalar:0,gather:1,vnni:2,both:3,worker-unresolved:-1; '
    'cells=phase/type/nt/selected/eligible/variant:experts,tokens,output_rows')
HISTOGRAM_END = '; exposure only, not time/traffic/worker imbalance'
HISTOGRAM_KEYS = ('cpu2','cpu512','hybrid','gu512','gu256','gu_kq','down_kq','iq4nl',
    'gu_min','iq3s_min','down_min','gather_setting','iq256_variant_capabilities')


def parse_histogram(text):
    """Fail closed for frozen H's exact default nonhybrid Zen3 schema.

    Selection checks mirror reviewed source only for observed labels; this is
    admission of counters, not a mathematical oracle or new resolver execution.
    """
    assert text.startswith(HISTOGRAM_PREFIX) and text.endswith(HISTOGRAM_END)
    header, cells_text=text[len(HISTOGRAM_PREFIX):-len(HISTOGRAM_END)].split(HISTOGRAM_LEGEND)
    header_items=header.split()
    assert len(header_items)==len(HISTOGRAM_KEYS)
    settings={}
    for key,item in zip(HISTOGRAM_KEYS,header_items):
        match=re.fullmatch(re.escape(key)+r'=(-?\d+)',item);assert match,'invalid header: '+item
        settings[key]=int(match[1])
    expected=dict(zip(HISTOGRAM_KEYS,(1,0,0,0,1,0,0,1,2,2,2,-1,1)))
    assert settings==expected, 'unexpected CPU/default dispatch capabilities or settings'
    words=cells_text.split();assert 0<len(words)<=48,'empty/oversized histogram'
    cells=[];seen=set();totals={phase:{'experts':0,'tokens':0,'output_rows':0} for phase in ('GU','Down')}
    for word in words:
        match=re.fullmatch(r'(GU|Down)/(\d+)/(\d+)/(ggml|kq256|iq512|iq256|iq4nl256|q2-avx2|q2-avx512)/(\d+)/(-?\d+):(\d+),(\d+),(\d+)',word)
        assert match,'malformed/unknown histogram tuple: '+word
        phase,type_text,nt_text,selected,eligible_text,variant_text,experts_text,tokens_text,rows_text=match.groups()
        typ,nt,eligible,variant,experts,tokens,rows=map(int,(type_text,nt_text,eligible_text,variant_text,experts_text,tokens_text,rows_text))
        assert typ in ({18,21,22,23} if phase=='GU' else {20,42}) and 1<=nt<=2, 'configured spec4/split native NT ceiling exceeded'
        key=(phase,typ,nt);assert key not in seen,'duplicate histogram cell';seen.add(key)
        assert 0<experts<2**64 and 0<tokens<2**64 and 0<rows<2**64
        assert tokens==experts*nt and rows==experts*(640 if phase=='GU' else 2560)
        wanted_eligible=2 if phase=='GU' else (8 if typ==20 else 16)
        wanted_selected=('iq256' if nt>=2 else 'ggml') if phase=='GU' else (
            ('iq4nl256' if nt>=2 else 'ggml') if typ==20 else 'q2-avx2')
        assert eligible==wanted_eligible and selected==wanted_selected and variant==0
        row={'phase':phase,'type':typ,'nt':nt,'selected':selected,'eligible':eligible,'variant':variant,
             'experts':experts,'tokens':tokens,'output_rows':rows}
        cells.append(row)
        for name in totals[phase]:totals[phase][name]+=row[name]
    assert all(0<value<2**64 for phase in totals.values() for value in phase.values())
    assert totals['GU']['experts']==totals['Down']['experts']>0
    assert totals['GU']['tokens']==totals['Down']['tokens']>0
    # NT survives the two phase boundaries; also reconcile counts per group width.
    for nt in range(1,9):
        assert sum(c['experts'] for c in cells if c['phase']=='GU' and c['nt']==nt)==sum(
            c['experts'] for c in cells if c['phase']=='Down' and c['nt']==nt)
    return {'passed':True,'raw':text,'settings':settings,'cells':cells,'totals':totals,
            'source_bound_max_nt':2, 'meaning':'request delta logical caller expert jobs, not actual task fragments/time/traffic/worker imbalance'}


base = Path(__file__).parent
observer = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0, str(observer / 'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument('mode', choices=['tasks6'])
cli.add_argument('phase', choices=['diagnostic'])
cli.add_argument('repetition', type=int, choices=[1])
cli.add_argument('--self-sha256', required=True)
cli.add_argument('--build-sha256', required=True)
cli.add_argument('--flags-sha256', required=True)
cli.add_argument('--cpu-preflight', action='store_true')
# Future commit/build/binary identities are deliberately unset until supplied by root.
cli.add_argument('--engine-commit', required=True)
cli.add_argument('--binary-sha256', required=True)
cli.add_argument('--build-receipt', type=Path, required=True)
cli.add_argument('--flags-receipt', type=Path, required=True)
cli.add_argument('--parser-sha256', required=True)
options = cli.parse_args()
for pin in (options.self_sha256, options.build_sha256, options.flags_sha256, options.binary_sha256, options.parser_sha256):
    assert re.fullmatch(r'[0-9a-f]{64}', pin), 'exact caller-supplied SHA256 required'
assert digest(__file__) == options.self_sha256, 'controller source differs from caller pin'
assert digest(base/'cache_route_pairs_parser_v1.py')==options.parser_sha256
from cache_route_pairs_parser_v1 import parse_request as parse_pairs, static_profile, JOB_LIMIT, REQUEST_LIMIT
assert re.fullmatch(r'[0-9a-f]{40}',options.engine_commit), 'exact future reviewed commit required' 
if sys.flags.optimize != 0:
    raise RuntimeError('admission assertions must remain enabled; do not use python -O')
mode, phase, repetition = options.mode, options.phase, options.repetition
review_head=options.engine_commit
engine_commit='96bd5bb4e054f6ddcf677fd96e161b499a013fd7'
root=observer.parent/'diag-sycl-cache-route-pairs-v0141-20261010'
def git(*args):
    return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
assert git('rev-parse','HEAD')==review_head and not git('status','--porcelain')
subprocess.run(['git','merge-base','--is-ancestor',engine_commit,review_head],cwd=root,check=True)
allowed_pair_changes={'sycl/include/strata/core/cache_route_pairs.hpp','sycl/include/strata/core/expert_source.hpp',
    'sycl/src/core/expert_source.cpp','sycl/src/program/generate.cpp','sycl/tools/cache-route-pairs-test.cpp'}
assert all(p in allowed_pair_changes or p.startswith(('bench/','docs/'))
           for p in git('diff','--name-only',engine_commit,review_head).splitlines())
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def terminal_owned(data, *, require_gate=True):
    assert not data['active'] and data['completed'] and data['healthy'] and data['exit_code']==0
    assert not data['exit_signal'] and not data['new_fault_messages'] and not any(data['cleanup'].values())
    assert data['boot_id']==boot
    if require_gate: assert data['math_gate_passed']
    for key in ['inferior','debugger']:
        old=data[key];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
sequence_path=base/'upstream-v0141-tuned-matched32k-comparison-sequence-v1/record.json'
assert digest(sequence_path)=='606e8b8befda5e3720f4197bc4a88894087dcc0ded9b3732fcc7c1300fafe22c'
sequence=json.loads(sequence_path.read_text());assert not sequence['active'] and sequence['passed']
for step in sequence['steps']:
    p=Path(step['receipt']);assert digest(p)==step['receipt_sha256']
    terminal_owned(json.loads(p.read_text()))
    old=step['controller_identity'];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
new_full_path=base/'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json'
assert digest(new_full_path)=='168d7896cbe3a60dfc604a08b6fe310f16d823a17d13d8a013260bfc7facb0a4'
new_full=json.loads(new_full_path.read_text());terminal_owned(new_full)
assert new_full['physical256k_sequence_completed'] and new_full['capacity_sequence_completed']
qualified=new_full
subset_path=base/'upstream-v0141-tuned-fourfresh-numerical-subset-20261009-v1.json'
assert digest(subset_path)=='95ae2a0a7e61f2240f6138f21fdcb35c1a70ec54300b22a7b99fd6b92873b4a7'
subset=json.loads(subset_path.read_text());assert subset['passed'] and not subset['active']
assert subset['binary_sha256']==new_full['binary_sha256']=='86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8'
reference_reads=[normalize_reference(r) for r in subset['requests']]
assert len(reference_reads)==4 and all(r['math_gate_passed'] and all(r['comparison'].values()) for r in reference_reads)
for req in reference_reads:
    assert Path(req['first_head']['file']).is_file() and Path(req['prefill_state']['file']).is_file()
    assert len(req['prefill_state']['parts'])==66
    assert digest(Path(req['first_head']['file']))==req['first_head']['sha256']
    assert req['first_head']['floats']==248320 and req['first_head']['finite']
diagnostic={'requests':reference_reads,'binary_sha256':new_full['binary_sha256']}
build_path=options.build_receipt.resolve()
assert digest(build_path)==options.build_sha256
build=json.loads(build_path.read_text());assert not build['active'] and build['passed'] and build['compiled_engine'] and not build['gpu_tested'] and not build['adopted']
assert build['source_head']==review_head
assert build['complete'] and not build['cleanup'] and not build['survivors']
for owner in build['owners'].values():
    now=process_identity(owner['pid'])
    assert not now or now['start_ticks']!=owner['start_ticks']
binary=Path(build['binary'])
assert binary.resolve().is_relative_to(root.resolve()) and binary.name=='strata'
assert build['binary_sha256']==options.binary_sha256
assert re.fullmatch(r'[0-9a-f]{64}',build['binary_sha256']) and digest(binary)==build['binary_sha256']
for relative, expected_sha in build['source_sha256'].items():
    assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
    assert re.fullmatch(r'[0-9a-f]{64}',expected_sha) and digest(root/relative)==expected_sha
# Clean frozen HEAD pins H's resolver, pool counters and call sites independently
# of receipt formatting; retain every required source pin in the new run receipt.
h_source_files=['src/kernels/cpu/native_expert.cpp','src/kernels/cpu/pool.cpp',
    'include/strata/kernels/cpu/native_expert.hpp','include/strata/kernels/cpu/pool.hpp',
    'src/kernels/cpu/expert_layout.cpp','sycl/src/kernels/cpu/iq_avx2.cpp','sycl/CMakeLists.txt',
    'sycl/src/program/generate.cpp','sycl/src/prefill/prefill.cpp','sycl/include/dpct/device.hpp',
    'sycl/include/strata/core/cache_route_pairs.hpp','sycl/include/strata/core/expert_source.hpp',
    'sycl/src/core/expert_source.cpp','sycl/src/core/expert_cache.cpp']
h_source_pins={relative:digest(root/relative) for relative in h_source_files}
for relative, expected_sha in h_source_pins.items():
    blob=subprocess.check_output(['git','show',review_head+':'+relative],cwd=root)
    assert hashlib.sha256(blob).hexdigest()==expected_sha
    if relative not in allowed_pair_changes:
        baseline_blob=subprocess.check_output(['git','show',engine_commit+':'+relative],cwd=root)
        assert hashlib.sha256(baseline_blob).hexdigest()==expected_sha, 'unreviewed production change' 
pair_source_pins={
    'sycl/include/strata/core/cache_route_pairs.hpp':'c06b174f8059a3fec6638bfdde59820126f00c1db809a30ce89a6087a888ad42',
    'sycl/src/program/generate.cpp':'eb591e263398abd9b178ab00c61e4f6af53a53bad37f837b18cd323af8dbab05',
    'sycl/include/strata/core/expert_source.hpp':'f94d0f2ed4d14f5aa4ad402fcda041b2a46b2690976d08ff1a0b740555043247',
    'sycl/src/core/expert_source.cpp':'41b98031b865338f0429b195ef4aa10ba94cbaf2c0f2224ab1c3a4de7c37f4aa',
}
for relative,sha in pair_source_pins.items():
    assert h_source_pins[relative]==build['source_sha256'][relative]==sha
pair_cpu_path=base/'cache-route-pairs-cpu-validation-v2/record.json'
assert digest(pair_cpu_path)=='2803c78cf7621b8f5ebcda50d67f1ba769a28e07d32568508115e91168e35994'
pair_cpu=json.loads(pair_cpu_path.read_text())
assert pair_cpu['passed'] and pair_cpu['complete'] and not pair_cpu['active'] and not pair_cpu['gpu_executed'] and not pair_cpu['model_opened']
for relative,sha in pair_cpu['pins'].items(): assert digest(root/relative)==sha
profile_path=root/'data/expert-profile.bin'
profile_sha='8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf'
assert digest(profile_path)==profile_sha and profile_path.stat().st_size==196632
resident_pairs,expected_residency_fnv=static_profile(profile_path.read_bytes())
pack=Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s')
pack_pins={
    'native_experts.txt':'d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d',
    'index.txt':'b2a014878c3b61d6a6f915e194055a5e68ede34f9465dd1cb13249d902f008ac',
    'conversions.json':'51df3cd6ffd0d38c2da8e2d3a95a604b4a96c20b06fc639a9bf477282a42b86a',
}
for name,sha in pack_pins.items(): assert digest(pack/name)==sha
conversions=json.loads((pack/'conversions.json').read_text())
model_dir=Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S')
model_identity=[]
for shard in conversions['source_shards']:
    path=model_dir/shard['name'];info=path.stat()
    assert path.is_file() and info.st_size==shard['size']
    model_identity.append({'path':str(path),'bytes':info.st_size,'dev':info.st_dev,'ino':info.st_ino,
                           'mtime_ns':info.st_mtime_ns,'ctime_ns':info.st_ctime_ns})
# Compact provenance and qualified baseline path binding; no full weight hashing.
assert not (pack/'experts.bin').exists()
assert new_full['argv'][new_full['argv'].index('--pack')+1]==str(pack)
assert new_full['argv'][new_full['argv'].index('--native')+1]==model_identity[0]['path']
candidate_source=root/'sycl/src/program/generate.cpp'
candidate_header=root/'sycl/include/dpct/device.hpp'
assert digest(candidate_source)==build['source_sha256']['sycl/src/program/generate.cpp']
assert digest(root/'sycl/src/prefill/prefill.cpp')==build['source_sha256']['sycl/src/prefill/prefill.cpp']=='f3c9bb38b46cdc111e9e663cc606d655f60f07c05873aff5499d51895e886c77'
assert digest(candidate_header)==build['source_sha256']['sycl/include/dpct/device.hpp']=='adbcbd0ac45995348e37ac3f5ca42d9f3d4b0555031aeb1781a1e2166a74ed52'
host_cpu_path=base/'pool-tasks-v0141-source-review-v1.json'
assert digest(host_cpu_path)=='a2c1fc95912ff0f58ce940c981d8a6bab6dbc8b072b45dcc2b670e5721b13794'
host_cpu=json.loads(host_cpu_path.read_text());assert host_cpu['passed'] and not host_cpu['active'] and not host_cpu['adopted']
flags_path=options.flags_receipt.resolve()
assert digest(flags_path)==options.flags_sha256
flags=json.loads(flags_path.read_text());assert flags['passed'] and not flags['active'] and flags['build_receipt_sha256']==digest(build_path)
assert flags['candidate_binary_sha256']==build['binary_sha256']
assert flags['old_compile_count']==flags['candidate_compile_count']==115 and flags['candidate_unique_source_count']==114 and not flags['missing_sources'] and not flags['different_flags']
quiet_path=base/'host-prefill-accounting-v0141-quiet32k-comparison-sequence-v1/record.json'
assert digest(quiet_path)=='d57fba6b0826ad8402505571b8cdc92eee4e0fa5bbd1c1a9b779531bd2f4bc83'
quiet=json.loads(quiet_path.read_text());assert not quiet['active'] and quiet['passed'] and len(quiet['steps'])==6
for step in quiet['steps']:
    p=Path(step['receipt']);assert digest(p)==step['receipt_sha256']
    terminal_owned(json.loads(p.read_text()))
    identity=step['controller_identity'];actual=process_identity(identity['pid'])
    assert not actual or actual['start_ticks']!=identity['start_ticks']
assert digest('/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0.12.0')=='bfdc0f26bf88bd0bccd66fc25302a4b22559f8fe30e499be18529be3ed610e60'
state_comparator=base/'compare_live_prefill_state_v01402_v2.py'
assert digest(state_comparator)=='9a4f55a4760d316045c1ec5179a946de9a9024f7e42fe09032012ceae2475a72'
from compare_live_prefill_state_v01402_v2 import compare_states
assert shutil.disk_usage(base).free>96*1024**3
out=base/'owned-cache-route-pairs-v0141-code32k-tasks6-diagnostic-r1'
assert not out.exists()
lock=(base/'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
commits={mode:review_head}

capture_cpu_path = base / 'owned-native-counter-v01402-cpu-v1/record.json'
capture_cpu = json.loads(capture_cpu_path.read_text())
assert digest(capture_cpu_path) == 'c93b119422938718e230d9e7c02377b7f0e6d48dc2095cb4add442f2e1025e86'
assert capture_cpu['passed'] and not capture_cpu['active']
assert digest(base / 'capture_owned_native_counter_v01402_v1.py') == capture_cpu['capture_controller_sha256']
assert digest(observer / 'sycl/tools/owned_gdb.py') == capture_cpu['owned_helper_sha256']

repeat_path=base/'native-copy-decode-repeat-v0141-comparison-sequence-v1/record.json'
assert digest(repeat_path)=='a9091c40a01af7e725b297d1cb87a01c0348372f6579a34e4db051a9a8981114'
repeat=json.loads(repeat_path.read_text());assert not repeat['active'] and repeat['passed'] and len(repeat['steps'])==18
for step in repeat['steps']:
    receipt=Path(step['receipt']);assert digest(receipt)==step['receipt_sha256'];terminal_owned(json.loads(receipt.read_text()))

previous_T_build_path=base/'decode-pool-phase-timing-v0141-private-build-v2/record.json'
assert digest(previous_T_build_path)=='3632b1514544ef2dd89046c112b823c75a27d0215aca2fce58c4b8df9e43d7f8'
previous_T_build=json.loads(previous_T_build_path.read_text())
assert previous_T_build['passed'] and previous_T_build['compiled_engine'] and not previous_T_build['active']
for prior in previous_T_build['prior_owned_gpu_admission']:
    p=Path(prior['path']);assert digest(p)==prior['sha256']
    terminal_owned(json.loads(p.read_text()),require_gate=not prior['held_prior_candidate_failure_preserved'])
assert sum(x['held_prior_candidate_failure_preserved'] for x in previous_T_build['prior_owned_gpu_admission'])==1

# Next test cannot overlap the currently owned affinity comparison.
affinity_sequence_path=base/'stager-affinity-v0141-decode-repeat-comparison-sequence-v1/record.json'
affinity_sequence=json.loads(affinity_sequence_path.read_text())
assert not affinity_sequence['active'] and affinity_sequence['passed'] and len(affinity_sequence['steps'])==18
assert affinity_sequence['sequence_controller_sha256']=='0600cfad0060fb99c80a306dc6ac85df29d435321d4750a00f9dc77913347cc3'
for step in affinity_sequence['steps']:
    p=Path(step['receipt']);assert digest(p)==step['receipt_sha256'];terminal_owned(json.loads(p.read_text()))
    old=step['controller_identity'];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
for name,expected_sha in [
    ('pool-tasks-cli-validation-v1/record.json','25939ce0bc704d654acb6d8b18ced231e12591797785311e9792787d0f89d1a5'),
    ('pool-tasks-existing-cpu-parity-v2/record.json','1366202e4035b026193c9ea5dfe3b273e8ba5220760f686873ade386c507587a')]:
    p=base/name;assert digest(p)==expected_sha;x=json.loads(p.read_text())
    assert x['passed'] and not x['active'] and not x['gpu_executed'] and not x['model_opened'] and not x['adopted']

# Latest C is a separate failed prefill candidate. Preserve rejection and require
# normal closed ownership; no C timing/adoption or weakened numerical validation.
C_path=base/'owned-layer-major-streamed-kv-v0141-full256k-streamed-lease-diagnostic-r1/record.json'
assert digest(C_path)=='a3413ac4c3f20ae593d8507a7db0ac405cd7fab2de286547460e6ba82b0d9404'
C=json.loads(C_path.read_text());terminal_owned(C,require_gate=False)
assert C['math_gate_passed'] is False and C['full_lifecycle_passed'] is False
assert C['binary_sha256']=='66d5bee60f4bbdb6f710b395ae11a1e6e14a61e04db417bcd6092d4f7956377e'
T_proof=base/'decode-pool-phase-timing-v0141-source-review-v1.json'
assert digest(T_proof)=='d58482d3fa6d8dbe55d3f672a2142862e07c32b465bcc7a9ffcc1bd2efb9325f'
T=json.loads(T_proof.read_text());assert T['passed'] and T['removing_diagnostic_reproduces_entire_original_generate_source_byte_identical']
T_cli=base/'decode-pool-phase-timing-cli-validation-v1.json'
assert digest(T_cli)=='99c95ae009337bec49b3b2c79f5959c88b182a9bdc66f297e32cc76bafd61103'
T_cli_receipt=json.loads(T_cli.read_text());assert T_cli_receipt['passed'] and len(T_cli_receipt['cases'])==19 and not T_cli_receipt['gpu_executed'] and not T_cli_receipt['model_opened']
D_path=base/'owned-pool-tasks-v0141-code32k-tasks6-diagnostic-r1/record.json'
assert digest(D_path)=='116604dfa1206464b668c9e391d6a66f09350fe35abec7071402a1621641323f'
D=json.loads(D_path.read_text());terminal_owned(D);assert len(D['requests'])==4 and all(q['math_gate_passed'] for q in D['requests'])

latest_T_path=base/'owned-decode-pool-phase-timing-v0141-code32k-tasks6-diagnostic-r1/record.json'
assert digest(latest_T_path)=='e770766de708a5ab8842b4aa3ea80a6d1453765059e8b7816e36aed09331148c'
latest_T=json.loads(latest_T_path.read_text());terminal_owned(latest_T)
assert len(latest_T['requests'])==4 and all(q['math_gate_passed'] for q in latest_T['requests'])
latest_D_path=base/'owned-repeat-capture-v0141-two-full-diagnostic-r3/record.json'
assert digest(latest_D_path)=='cc0e49f0dde0642549a5b6ff7b9c2a8d03d7a32d60501dbae69bfeac306e48e3'
latest_D=json.loads(latest_D_path.read_text())
assert latest_D['active'] is False and latest_D['healthy'] is False and latest_D['completed'] is False
assert latest_D['exit_code']==0 and not latest_D['exit_signal'] and not latest_D['new_fault_messages']
assert not any(latest_D['cleanup'].values()) and not latest_D.get('survivors') and latest_D['boot_id']==boot
assert any(q.get('math_gate_passed') is False for q in latest_D['requests']), 'preserve D rejection'
for role in ('inferior','debugger'):
    old=latest_D[role];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
closed_cursor_steps=[x for x in latest_D['steps'] if x['label']=='kernel-after']
assert len(closed_cursor_steps)==1 and closed_cursor_steps[0]['exit_code']==0
assert not closed_cursor_steps[0]['timed_out'] and not closed_cursor_steps[0]['still_alive']
matrix_path=base/'native-wrapper-matched-cpu-matrix-v1/record.json'
assert digest(matrix_path)=='60616fc79ce250e574383e83943cbcaa9564cb1cdab6e81e6a737eefb030dfdc'
matrix=json.loads(matrix_path.read_text())
assert matrix['passed'] and matrix['complete'] and not matrix['active']
assert not matrix['gpu_executed'] and not matrix['model_opened'] and not matrix['cleanup'] and not matrix['survivors']
assert len(matrix['profiles'])==len(matrix['pairs'])==9 and len(matrix['steps'])==18
assert all(pair['T_H_bitwise_equal'] and pair['first_different_byte'] is None for pair in matrix['pairs'])
assert all(step['exit_code']==0 and step['local_passed'] and len(step['stdout_findings']['case_results'])==48 for step in matrix['steps'])
for cpu_step in matrix['steps']:
    old=cpu_step['identity'];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
# Raw reference state files remain required inputs, not a hash-only substitute.
for reference in reference_reads:
    state=reference['prefill_state'];state_file=Path(state['file'])
    assert state_file.stat().st_size==state['bytes']
    with state_file.open('rb') as stream:
        for index,part in enumerate(state['parts']):
            header=stream.read(8);assert len(header)==8 and struct.unpack('=Q',header)[0]==part['bytes']
            assert part['index']==index and stream.tell()==part['offset']
            left=part['bytes'];part_digest=hashlib.sha256()
            while left:
                chunk=stream.read(min(left,1048576));assert chunk
                part_digest.update(chunk);left-=len(chunk)
            assert part_digest.hexdigest()==part['sha256']
        assert not stream.read(1)
a_source=base/'coding-review-32k-tokens.txt'
b_source=base/'full-context-copy-off/coding-context-256k-tokens.txt'
assert digest(a_source)=='137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449'
assert digest(b_source)=='cc29e4427bcacb21c9f7df2d1f1a7d41897fce5bd76a8ca43d8c3c34d914dd2a'
# The intended profile is unchanged default Zen3 dispatch, not a flag sweep.
dispatch_controls=['STRATA_IQ_MT_MIN','STRATA_NO_IQ256','STRATA_NO_IQ512','STRATA_NO_IQ4NL',
    'STRATA_IQ256_GATHER','STRATA_IQ3S_MT1','STRATA_NO_AVXVNNI','STRATA_KQ256',
    'STRATA_FORCE_ISA','STRATA_FORCE_AVX2','STRATA_CPU_YMM','STRATA_IQ_GATHER',
    'STRATA_Q2_BITPLANE','STRATA_NO_Q8K_AVX2','STRATA_IQ_PREFETCH']
assert not any(key in os.environ for key in dispatch_controls), 'default dispatch controls must be unset'
cpuinfo=Path('/proc/cpuinfo').read_text()
assert 'AMD Ryzen 5 5600X' in cpuinfo and 'AuthenticAMD' in cpuinfo
cpu_flags=next(line.split(':',1)[1].split() for line in cpuinfo.splitlines() if line.startswith('flags'))
assert {'avx2','fma','f16c'}<=set(cpu_flags) and not any(flag.startswith('avx512') for flag in cpu_flags)
assert 'avx_vnni' not in cpu_flags and 'avxvnni' not in cpu_flags

if options.cpu_preflight:
    print(json.dumps({'cpu_preflight_passed':True,'gpu_executed':False,'binary_sha256':digest(binary),'reference_binary_sha256':diagnostic['binary_sha256'],'reference_reads':len(reference_reads),'review_head':review_head,'build_receipt_sha256':options.build_sha256,'flags_receipt_sha256':options.flags_sha256,'controller_sha256':options.self_sha256,'runtime_qualification_claimed':False,'parser_sha256':options.parser_sha256,'profile_sha256':profile_sha,'model_weight_payload_hashed':False,'prior_D_closure_only':True}))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();sys.exit(0)

from capture_owned_native_counter_v01402_v1 import capture as capture_native_counter

out.mkdir(mode=0o700)
probes = out / 'probes'
probes.mkdir()
source = observer / 'sycl/tools/recover-xe.sh'
m = types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(source), 'exec'), m.__dict__)
r = m.Runner(probes)
os.chdir(root)
record = {
    'scope':'Cache-route-pairs plus native dispatch histogram diagnostic on existing --pool-tasks6. Four fresh32768 A/B/A/B in ONE process,64 visible outputs, first logits, all66 live state parts and MTP/logprob references against independent qualified integrated869. Histogram+existing phase output enabled; original H production settings retained. Logical caller-job exposure only, not time/traffic/worker imbalance. Flushed UR/ZE/ZEL diagnostics; no speed/full-capacity/adoption qualification.',
    'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':boot,
    'active':True,'completed':False,'healthy':False,'math_gate_passed':False,
    'controller_source_status_at_submission':'SOURCE ONLY / UNEXECUTED until root runs',
    'pair_gate_passed':False,'pair_parser_sha256':options.parser_sha256,'pair_CPU_receipt_sha256':digest(pair_cpu_path),
    'cwd':str(root),'profile':{'path':str(profile_path),'sha256':profile_sha,'resident_pairs':128,'expected_host_res_fnv1a64':expected_residency_fnv},
    'pack_metadata_sha256':pack_pins,'model_identity':model_identity,'weight_payload_hashed':False,
    'D_original_completed':False,'D_original_healthy':False,'D_original_full_math_passed':False,
    'pair_request_byte_limit':REQUEST_LIMIT,'pair_job_byte_limit':JOB_LIMIT,
    'histogram_gate_passed':False,'histogram_counter_semantics':'logical caller expert jobs; not fragments,time,DRAM or worker tails',
    'controller_sha256':options.self_sha256,'matched_CPU_wrapper_receipt_sha256':digest(matrix_path),
    'latest_owned_gpu_fault_cursor_receipt_sha256':digest(latest_D_path),'H_source_pins':h_source_pins,
    'requests':[],'snapshots':[],'steps':[],'mode':mode,'phase':phase,'repetition':repetition,
    'deadline_seconds':1800,'protocol_timeout_seconds':450,'log_limit_bytes':64*1024**2,
    'commit':review_head,'compiled_engine_base_commit':engine_commit,'root':str(root),
    'binary_sha256':digest(binary),'build_receipt_sha256':digest(build_path),
    'candidate_source_sha256':digest(candidate_source),'candidate_header_sha256':digest(candidate_header),'source_preparation_receipt_sha256':digest(host_cpu_path),'uniform_flags_receipt_sha256':digest(flags_path),'completed_host_quiet_sequence_sha256':digest(quiet_path),
    'reference_binary_sha256':diagnostic['binary_sha256'],
    'reference_four_fresh_receipt_sha256':digest(subset_path),
    'baseline_full_capacity_receipt_sha256':digest(new_full_path),
    'baseline_quiet_sequence_receipt_sha256':digest(sequence_path),
    'performance_eligible':False,'full_lifecycle_passed':False,'adopted':False,'C_original_math_gate_passed':False,'C_failure_receipt_sha256':digest(C_path),'T_source_review_sha256':digest(T_proof),'T_cli_receipt_sha256':digest(T_cli),'previous_D_fourfresh_math_receipt_sha256':digest(D_path),
    'source_status_before':'',
    'source_sha256':{str(p):digest(p) for p in [Path(__file__),source,observer/'sycl/tools/owned_gdb.py',base/'capture_owned_native_counter_v01402_v1.py',state_comparator,candidate_source,candidate_header,flags_path]},
}
g = None
master = slave = None
cursor = None
pending = bytearray()
current = None
raw = (out / 'protocol.stdout.raw').open('wb')
events = (out / 'events.jsonl').open('w', buffering=1)
started = time.monotonic()
next_update = started


def save():
    record['elapsed_seconds'] = time.monotonic() - started
    record['steps'] = r.calls
    record['active_request'] = current
    if g:
        record.update(inferior=g.inferior, debugger=g.debugger_identity)
    temporary = out / 'record.json.tmp'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(out / 'record.json')


def event(kind, **fields):
    events.write(json.dumps({'kind': kind, 'elapsed_seconds': time.monotonic() - started, **fields}) + '\n')


def captures(name, required):
    result = {}
    head = out / 'first-head.bin'
    if head.exists():
        kept = out / f'{name}.head.bin'
        assert not kept.exists()
        head.rename(kept)
        data = kept.read_bytes()
        values = array.array('f')
        values.frombytes(data)
        assert len(values) == 248320 and all(map(math.isfinite, values))
        result['first_head'] = {'file': str(kept), 'sha256': hashlib.sha256(data).hexdigest(), 'floats': len(values), 'finite': True}
    state = out / 'prefill-state.bin'
    if state.exists():
        kept = out / f'{name}.state.bin'
        assert not kept.exists()
        state.rename(kept)
        parts = []
        with kept.open('rb') as stream:
            while (header := stream.read(8)):
                assert len(header) == 8
                size = struct.unpack('=Q', header)[0]
                offset = stream.tell()
                left = size
                digest = hashlib.sha256()
                while left:
                    data = stream.read(min(left, 1048576))
                    assert data
                    digest.update(data)
                    left -= len(data)
                parts.append({'index': len(parts), 'offset': offset, 'bytes': size, 'sha256': digest.hexdigest()})
        assert len(parts) == 66 and parts[0]['bytes'] == 8
        result['prefill_state'] = {'file': str(kept), 'bytes': kept.stat().st_size, 'parts': parts}
    if required:
        assert 'first_head' in result and 'prefill_state' in result, 'missing fresh full-read capture'
    return result

def poll():
    global next_update
    g.poll(.01)
    if time.monotonic() - started >= record['deadline_seconds']:
        raise TimeoutError('bounded model job deadline')
    if g.stops and g.stops[-1] != 'resumed' and g.exit_code is None and g.exit_signal is None:
        raise RuntimeError('inferior stopped: ' + g.stops[-1])
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.name.endswith(('.bin','.tmp'))) >= record['log_limit_bytes']:
        raise RuntimeError('total diagnostic text budget exceeded')
    if time.monotonic() >= next_update:
        save()
        next_update = time.monotonic() + 5


def line(seconds=None):
    end = time.monotonic() + (seconds or record['protocol_timeout_seconds'])
    with selectors.DefaultSelector() as ready:
        ready.register(master, selectors.EVENT_READ)
        while time.monotonic() < end:
            if b'\n' in pending:
                value, _, tail = pending.partition(b'\n')
                if len(value) + 1 > 65536:
                    raise ValueError('engine protocol line exceeds 65536-byte bound')
                pending[:] = tail
                return value.decode().strip()
            if len(pending) >= 65536:
                raise ValueError('unterminated engine protocol line exceeds bound')
            poll()
            if not ready.select(.02):
                continue
            try:
                data = os.read(master, 65536)
            except BlockingIOError:
                continue
            if not data:
                raise RuntimeError('protocol EOF')
            raw.write(data)
            raw.flush()
            pending.extend(data)
            event('stdout', bytes=len(data), raw_offset=raw.tell())
    raise TimeoutError('engine protocol deadline')


def send(data):
    end = time.monotonic() + 20
    view = memoryview(data)
    with selectors.DefaultSelector() as ready:
        ready.register(master, selectors.EVENT_WRITE)
        while view:
            poll()
            if time.monotonic() >= end:
                raise TimeoutError('protocol input deadline')
            if not ready.select(.02):
                continue
            try:
                count = os.write(master, view[:8192])
            except BlockingIOError:
                continue
            view = view[count:]


save()
try:
    cpu_path = base / 'owned-main-thread-cpu/record.json'
    cpu = json.loads(cpu_path.read_text())
    assert cpu['passed'] and len(cpu['cases']) == 4
    for case in cpu['cases']:
        if case['name'].startswith('current-'):
            assert case['helper_sha256'] == digest(observer / 'sycl/tools/owned_gdb.py')
        for role in ['inferior', 'debugger']:
            old = case[role]
            now = process_identity(old['pid'])
            assert not now or now['start_ticks'] != old['start_ticks'] or now['state'] == 'Z'
    record['cpu_debugger_guard_sha256'] = digest(cpu_path)
    health_path = latest_D_path
    health = json.loads(health_path.read_text())
    assert not health['healthy'] and not health['active'] and health['boot_id'] == boot
    assert health['exit_code']==0 and not health['exit_signal'] and not health['new_fault_messages'] and not any(health['cleanup'].values())
    old_cursor = next(x['argv'][x['argv'].index('--after-cursor') + 1]
                      for x in health['steps'] if x['label'] == 'kernel-after')
    gap = r.run('kernel-gap', ['/usr/bin/journalctl', '-k', '--after-cursor', old_cursor, '--no-pager', '-o', 'json'], seconds=5)
    rows = [json.loads(s) for s in gap.splitlines() if s.startswith('{')]
    faults = [x['MESSAGE'] for x in rows if ('0000:05:00.0' in x.get('MESSAGE', '') or re.search(r'\bxe\b', x.get('MESSAGE', ''))) and m.FAULT.search(x.get('MESSAGE', ''))]
    record['preflight_fault_messages'] = faults
    record['previous_closed_failed_D_kernel_after_cursor']=old_cursor
    assert not faults
    assert r.run('embedding-state', ['/usr/bin/systemctl', '--user', 'show', 'llama-server-qwen3embed.service', '-p', 'ActiveState'], seconds=5).strip() == 'ActiveState=inactive'
    os.environ.update(NEOReadDebugKeys='1', EnableDirectSubmission='0')
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'), r, Path('/home/yayoi/.local/bin/strata-xe-health'))
    env = m.health_environment()
    env['LD_LIBRARY_PATH'] += ':/opt/intel/oneapi/mkl/2026.1/lib'
    for key in ['LD_PRELOAD', 'ZET_ENABLE_METRICS']:
        env.pop(key, None)
    env = {k: v for k, v in env.items() if not k.startswith(('UNITRACE_', 'XPTI_'))}
    env.update(NEOReadDebugKeys='1', EnableDirectSubmission='0', STRATA_PREFILL_FIRST='0', STRATA_PREFILL_RING='8')
    env.update(STRATA_PREFILL_COMPACT='2', STRATA_PREFILL_ATTN_LAYOUT='1', STRATA_PREFILL_LAYER_MAJOR='0', STRATA_KV_STAGE_OWN='1', STRATA_KV_PREFETCH='0')
    env.pop('STRATA_PREFILL_COPY_ENGINE', None)
    env.pop('STRATA_PREFILL_COPY_PHASE_RELEASE', None)
    env.pop('STRATA_STAGER_CPU_LIST',None)
    assert not any(key in env for key in dispatch_controls), 'inherited health environment changed dispatch'
    env.update(STRATA_CACHE_ROUTE_PAIRS='1',STRATA_NATIVE_DISPATCH_HISTOGRAM='1',STRATA_DECODE_TIMING='1',STRATA_DUMP_FIRST_LOGITS=str(out/'first-head.bin'),STRATA_PREFILL_DUMP_STATE=str(out/'prefill-state.bin'))
    if phase == 'diagnostic':
        env.update(ZEL_ENABLE_LOADER_LOGGING='1',ZEL_LOADER_LOG_CONSOLE='1',ZEL_LOADER_LOGGING_LEVEL='warn',ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',ZE_ENABLE_VALIDATION_LAYER='1',ZE_ENABLE_PARAMETER_VALIDATION='1',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',STRATA_TRACE='1')
        for key in ['UR_ENABLE_LAYERS','UR_LOG_TRACING']:env.pop(key,None)
    if phase == 'clean':
        assert not any(k.startswith(('UR_LOG_', 'ZE_ENABLE_', 'ZEL_')) or k in ['UR_ENABLE_LAYERS', 'STRATA_TRACE'] for k in env)
    assert not any(k in env for k in ['STRATA_PREFILL_SYNC', 'STRATA_PREFILL_TIMING', 'STRATA_TRANSFER_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING', 'STRATA_VERIFY_NO_HOST', 'STRATA_VERIFY_DEVICE_PLAN', 'STRATA_PREFILL_HOST_TIMING'])
    record['environment'] = {k: v for k, v in env.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')) or k in ['LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission']}
    args = list(json.loads((base / 'model-lease-copy-off/normal-mtp.json').read_text())['args'])
    for key, value in [('--pcie-frac', '0'), ('--max-context', '262144'), ('--prefill', '8192'), ('--expert-cache', '128'), ('--conversation-cache-mib', '0')]:
        args[args.index(key) + 1] = value
    args += ['--kv', 'int8', '--kv-resident', '32768', '--prompt-cache', '1', '--prompt-cache-every', '262139', '--prompt-cache-root', '0', '--turn-token', '-1']
    assert args == new_full['argv'][1:] == qualified['argv'][1:]
    args += ['--pool-tasks','6']
    argv = [str(binary)] + args
    record['argv'] = argv
    record['configuration_note'] = 'Qualified baseline arguments plus --pool-tasks6, with the same requested environment except diagnostic output destinations. No affinity/native-copy/release setting. Fresh A/B/A/B has no reused prompt tokens. API logs are never timing evidence.'
    record['prior_affinity_sequence_receipt_sha256']=digest(affinity_sequence_path)
    record['pool_tasks_cli_receipt_sha256']=digest(base/'pool-tasks-cli-validation-v1/record.json')
    record['pool_tasks_cpu_parity_receipt_sha256']=digest(base/'pool-tasks-existing-cpu-parity-v2/record.json')
    a_source = base / 'coding-review-32k-tokens.txt'
    b_source = base / 'full-context-copy-off/coding-context-256k-tokens.txt'
    assert digest(a_source) == '137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449'
    assert digest(b_source) == 'cc29e4427bcacb21c9f7df2d1f1a7d41897fce5bd76a8ca43d8c3c34d914dd2a'
    fixtures = {
        'A': list(map(int, a_source.read_text().split()))[:32768],
        'B': list(map(int, b_source.read_text().split()))[:32759] + [248046, 198, 248045, 74455, 198, 248068, 198, 248069, 271],
    }
    record['prompt_fixtures'] = {}
    for key, tokens in fixtures.items():
        assert len(tokens) == 32768
        path = out / ('input-' + key + '-tokens.txt')
        path.write_text(' '.join(map(str, tokens)) + '\n')
        record['prompt_fixtures'][key] = {'file': str(path), 'sha256': digest(path), 'tokens': len(tokens)}
    cursor = m.journal_cursor(r, 'kernel-before')
    save()
    master, slave = os.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    g = OwnedGdb(argv, out / 'debugger', env, inferior_tty_fd=slave)
    g.command('-gdb-set may-call-functions off')
    g.command('-gdb-set debug-file-directory /usr/lib/debug:/home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/runtime-symbols/extracted/usr/lib/debug')
    g.run()
    record['startup'] = []
    while True:
        value = line()
        record['startup'].append(value)
        if value.startswith('ERR'):
            raise RuntimeError(value)
        if value.startswith('READY '):
            break
    os.close(slave)
    slave = None
    identity = process_identity(g.inferior['pid'])
    assert identity and identity['start_ticks'] == g.inferior['start_ticks']
    actual_exe = Path('/proc', str(identity['pid']), 'exe')
    assert actual_exe.resolve() == binary.resolve() and digest(actual_exe) == record['binary_sha256']
    record['actual_executable_identity'] = {'inferior': identity, 'sha256': record['binary_sha256'], 'boot_id': boot}
    actual = dict(item.decode().split('=', 1) for item in Path('/proc', str(identity['pid']), 'environ').read_bytes().split(b'\0') if b'=' in item)
    actual_relevant = {k: v for k, v in actual.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')) or k in ['LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission']}
    assert actual_relevant == record['environment']
    assert 'STRATA_STAGER_CPU_LIST' not in actual_relevant
    actual_argv=[v.decode() for v in Path('/proc',str(identity['pid']),'cmdline').read_bytes().split(b'\0') if v]
    assert actual_argv==argv
    record['actual_target_argv']=actual_argv
    actual_cwd = Path('/proc', str(identity['pid']), 'cwd').resolve()
    assert actual_cwd == root.resolve()
    record['actual_target_cwd'] = str(actual_cwd)
    assert 'STRATA_PREFILL_COPY_ENGINE' not in actual_relevant and 'STRATA_PREFILL_COPY_PHASE_RELEASE' not in actual_relevant
    assert not any(k.startswith(('UNITRACE_', 'XPTI_')) or k in ['LD_PRELOAD', 'ZET_ENABLE_METRICS'] for k in actual)
    record['actual_target_environment'] = actual_relevant
    save()
    seen = {}
    math_ok = True
    for read_index, key in enumerate(['A', 'B', 'A', 'B']):
        current = f'{key}-read{read_index}'
        result = {'stderr_begin_offset':(out/'debugger/inferior.stderr').stat().st_size, 'name': current, 'fixture': key, 'read_index': read_index, 'first_process_read': read_index == 0, 'ids': [], 'logprobs': [], 'protocol': []}
        record['requests'].append(result)
        save()
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        request_start = time.monotonic()
        send(('GEN 64 ckpt=1 logprobs=5 ' + ','.join(map(str, fixtures[key])) + '\n').encode())
        event('request', name=current)
        while True:
            value = line()
            result['protocol'].append(value)
            if value.startswith('ERR'):
                raise RuntimeError(value)
            if value.startswith('T '):
                result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):
                result['logprobs'].append(value)
            if value.startswith('DONE '):
                break
        result['stderr_end_offset']=(out/'debugger/inferior.stderr').stat().st_size
        fields = value.split()
        result['finish_reason'] = fields[5]
        result['mtp_counts'] = list(map(int, fields[6:8]))
        result['resume_tokens'] = [int(x.split()[1]) for x in result['protocol'] if x.startswith(('RESUME ', 'REUSED '))]
        result['validation'] = validate_fresh(result)
        result['measurement'] = {'generated_tokens': int(fields[1]), 'prompt_tokens': int(fields[2]), 'prompt_ms': float(fields[3]), 'decode_ms': float(fields[4]), 'wall_seconds': time.monotonic() - request_start, 'purpose': 'quiet timing' if phase == 'clean' else 'logged correctness only, excluded from speed'}
        mm = result['measurement']
        if mm['prompt_ms'] > 0 and mm['decode_ms'] > 0:
            mm['prefill_tok_s'] = 1000 * mm['prompt_tokens'] / mm['prompt_ms']
            mm['decode_tok_s'] = 1000 * mm['generated_tokens'] / mm['decode_ms']
        result.update(captures(current,True))
        expected=reference_reads[read_index]
        state_cmp=compare_states(result['prefill_state'],expected['prefill_state'],32767)
        result['live_prefill_comparison']=state_cmp
        result['head_and_live_state_comparison']={
            'first_head_equal':result['first_head']['sha256']==expected['first_head']['sha256'],
            'all66_live_state_parts_equal':not state_cmp['different_live_parts'],
            'actual8192_chunks':[int(x.split()[1]) for x in result['protocol'] if x.startswith('PP ')]==[8192,16384,24576,32767]}
        checks = list(result['validation'].values())
        checks.extend(result['head_and_live_state_comparison'].values())
        if key in seen:
            result['repeat_comparison'] = same_output(result, seen[key])
            checks.extend(result['repeat_comparison'].values())
        seen[key] = result
        if diagnostic:
            result['qualified_physical_reference_output_comparison'] = same_output(result, diagnostic['requests'][read_index])
            checks.extend(result['qualified_physical_reference_output_comparison'].values())
        result['math_gate_passed'] = all(checks)
        math_ok &= result['math_gate_passed']
        save()
        if not result['math_gate_passed']:
            failed_path=out/'numerical-failure.session.bin'
            assert not failed_path.exists()
            send(('SAVE '+str(failed_path)+'\n').encode())
            responses=[]
            while True:
                response=line();responses.append(response)
                if response.startswith(('SAVED ', 'ERR')): break
            record['numerical_failure_preservation']={'request':current,'protocol':responses,'file':str(failed_path),'exists':failed_path.exists()}
            if failed_path.exists():
                record['numerical_failure_preservation'].update(bytes=failed_path.stat().st_size,sha256=digest(failed_path))
            save()
            break


    current = None
    send(b'QUIT\n')
    event('quit')
    end = time.monotonic() + 30
    with selectors.DefaultSelector() as ready:
        ready.register(master, selectors.EVENT_READ)
        while g.exit_code is None and g.exit_signal is None and time.monotonic() < end:
            poll()
            if ready.select(.01):
                try:
                    data = os.read(master, 65536)
                except (BlockingIOError, OSError):
                    continue
                if data:
                    raw.write(data)
                    raw.flush()
                    event('quit-stdout', bytes=len(data))
    assert g.exit_code == 0 and g.exit_signal is None
    engine_log = out / 'debugger/inferior.stderr'
    assert engine_log.stat().st_size<=record['log_limit_bytes']
    record['project_messages_duplicate_skipped'] = True
    record['pool_tasks_configuration_gate']={'passed':True,'requested_tasks':6,'actual_cli':record['actual_target_argv'][-2:],'source_review_sha256':digest(host_cpu_path),'existing_cpu_parity_sha256':digest(base/'pool-tasks-existing-cpu-parity-v2/record.json'),'note':'Actual compiled CLI supplies the existing ExpertPool fifth argument. No new pool/kernel math or synchronization. Full request numerical gates retained.'}
    pool_phase=[]
    with engine_log.open() as checked:
        for value in checked:
            if value.startswith('strata decode CPU pool phases: '):
                match=re.search(r'phases: (\d+) windows; GU ([0-9.]+), FF-quant ([0-9.]+), Down ([0-9.]+) ms/window .*bytes ([0-9.]+)/window',value)
                assert match,value
                windows=int(match.group(1));gu,quant,down,submitted=map(float,match.groups()[1:])
                assert all(math.isfinite(v) for v in [gu,quant,down,submitted])
                assert windows>0 and gu>0 and quant>=0 and down>0 and submitted>0
                pool_phase.append({'windows':windows,'GU_ms_per_window':gu,'FF_quant_ms_per_window':quant,'Down_ms_per_window':down,'packed_blob_submitted_bytes_per_window':submitted,'raw':value.strip(),'limit':'Host phase elapsed nested with GPU work; not additive serial latency or measuredDRAM traffic.'})
    assert len(pool_phase)==len(record['requests'])==4
    record['decode_CPU_phase_counter_gate']={'passed':True,'requests':pool_phase,'source_proof_sha256':digest(T_proof),'performance_eligible':False}
    histogram_lines=[];phase_lines=[]
    with engine_log.open('rb') as checked:
        while raw_line:=checked.readline(65537):
            assert len(raw_line)<=65536, 'engine diagnostic line exceeds parser bound'
            start_offset=checked.tell()-len(raw_line)
            if raw_line.startswith(b'strata decode CPU pool phases: '):
                assert raw_line.endswith(b'\n')
                phase_lines.append((start_offset,checked.tell(),raw_line.decode('ascii').strip()))
            if raw_line.startswith(b'strata native dispatch exposure: '):
                assert raw_line.endswith(b'\n'), 'incomplete histogram report'
                histogram_lines.append((start_offset,checked.tell(),raw_line.decode('ascii').rstrip('\n')))
    assert len(phase_lines)==4
    assert len(histogram_lines)==len(record['requests'])==4, 'need exactly four complete request reports'
    for index,request in enumerate(record['requests']):
        matching=[text for begin,end,text in histogram_lines if
            request['stderr_begin_offset']<=begin<end<=request['stderr_end_offset']]
        assert len(matching)==1, 'request needs exactly one owned histogram report'
        phases_in_request=[text for begin,end,text in phase_lines if
            request['stderr_begin_offset']<=begin<end<=request['stderr_end_offset']]
        assert phases_in_request==[pool_phase[index]['raw']], 'one phase report must belong to the same request'
        parsed=parse_histogram(matching[0])
        request['native_dispatch_histogram']=parsed
        request['decode_CPU_phase_counters']=pool_phase[index]
    record['native_dispatch_histogram_gate']={'passed':True,'requests':[
        {'name':q['name'],'histogram':q['native_dispatch_histogram']} for q in record['requests']],
        'meaning':'logical caller jobs; counts are not fragments, time, DRAM traffic or worker tails',
        'Zen3_default_resolver_source_sha256':h_source_pins['src/kernels/cpu/native_expert.cpp'],
        'caller_histogram_source_sha256':h_source_pins['src/kernels/cpu/pool.cpp']}
    pair_total_bytes=0
    for index,request in enumerate(record['requests']):
        begin=request['stderr_begin_offset'];end=request['stderr_end_offset']
        assert 0<=begin<end<=engine_log.stat().st_size and end-begin<=record['log_limit_bytes']
        with engine_log.open('rb') as checked:
            checked.seek(begin);interval=checked.read(end-begin)
        done=[line for line in request['protocol'] if line.startswith('DONE ')]
        assert len(done)==1
        parsed=parse_pairs(interval,index+1,done[0],resident_pairs,expected_residency_fnv)
        pair_total_bytes+=parsed['frame_bytes'];assert pair_total_bytes<=JOB_LIMIT
        cpu_callback_pairs=sum(pair['callback_pairs'] for pair in parsed['pairs']
                               if (pair['layer'],pair['expert']) not in resident_pairs)
        native=request['native_dispatch_histogram']['totals']
        assert all(native[phase]['tokens']==parsed['end']['refused'] and
                   native[phase]['experts']==cpu_callback_pairs for phase in ('GU','Down'))
        request['cache_route_native_job_reconciliation']={'passed':True,'CPU_distinct_refused_callback_pairs':cpu_callback_pairs,
            'CPU_refused_token_entries':parsed['end']['refused'],
            'meaning':'fixed static successful native route only: refused callback pairs match logical CPU expert jobs; all callback pairs also include GPU hits'}
        request['cache_route_pairs']=parsed
    # Reject extra frames outside owned request intervals, including startup/QUIT.
    pair_lines_total=0
    with engine_log.open('rb') as checked:
        while raw_line:=checked.readline(65537):
            assert len(raw_line)<=65536
            if b'CACHE_ROUTE_PAIRS_V1' in raw_line:
                assert raw_line.startswith(b'CACHE_ROUTE_PAIRS_V1 '), 'misplaced/noncanonical pair prefix'
                start=checked.tell()-len(raw_line);end=checked.tell()
                assert sum(q['stderr_begin_offset']<=start<end<=q['stderr_end_offset'] for q in record['requests'])==1
                pair_lines_total+=1
    assert pair_lines_total==sum(len(q['cache_route_pairs']['pairs'])+3 for q in record['requests'])
    record['cache_route_pairs_gate']={'passed':True,'requests':[{'name':q['name'],'ordinal':q['cache_route_pairs']['ordinal'],'end':q['cache_route_pairs']['end']} for q in record['requests']],
        'bytes':pair_total_bytes,'semantics':'main target evaluated token entries including rejected speculation; distinct callback pairs are not CPU/native jobs/time/traffic'}
    record['pair_gate_passed']=True
    record['histogram_gate_passed']=True
    for identity in model_identity:
        info=Path(identity['path']).stat()
        assert (info.st_size,info.st_dev,info.st_ino,info.st_mtime_ns,info.st_ctime_ns)==(
            identity['bytes'],identity['dev'],identity['ino'],identity['mtime_ns'],identity['ctime_ns'])
    for name,sha in pack_pins.items(): assert digest(pack/name)==sha
    assert digest(profile_path)==profile_sha
    record['model_and_profile_identity_unchanged']=True
    record['engine_log_bytes'] = engine_log.stat().st_size
    record['engine_log_sha256'] = digest(engine_log)
    record['source_status_after'] = git('status', '--porcelain')
    assert not record['source_status_after'] and git('rev-parse', 'HEAD') == commits[mode]
    record['completed'] = True
    record['math_gate_passed'] = math_ok and len(record['requests']) == 4
except BaseException as error:
    record['error'] = repr(error)
    if g:
        try:
            snap = g.snapshot('failure', resume=False)
            record['snapshots'].append(snap)
            if snap:
                record['native_counter_at_failure'] = capture_native_counter(g, snap, out / 'native-counter-at-failure-v1.json')
        except BaseException as inspect:
            record['snapshot_error'] = repr(inspect)
finally:
    if g:
        record['exit_code'] = g.exit_code
        record['exit_signal'] = g.exit_signal
        record['cleanup'] = g.close()
    for fd in [master, slave]:
        if fd is not None:
            os.close(fd)
    raw.close()
    events.close()
    if cursor:
        text = r.run('kernel-after', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'], seconds=5)
        rows = [json.loads(s) for s in text.splitlines() if s.startswith('{')]
        record['new_fault_messages'] = [x['MESSAGE'] for x in rows if (('0000:05:00.0' in x.get('MESSAGE', '') or re.search(r'\bxe\b', x.get('MESSAGE', ''))) and m.FAULT.search(x.get('MESSAGE', ''))) or ('strata' in x.get('MESSAGE', '') and 'segfault' in x.get('MESSAGE', ''))]
    record['healthy'] = record['completed'] and record['exit_code']==0 and record['exit_signal'] is None and record['histogram_gate_passed'] and record['pair_gate_passed'] and not record.get('new_fault_messages') and not record.get('error') and not any(record.get('cleanup', {}).values())
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
print(json.dumps({k: record.get(k) for k in ['mode', 'phase', 'healthy', 'math_gate_passed', 'histogram_gate_passed', 'pair_gate_passed', 'elapsed_seconds', 'error', 'exit_code', 'exit_signal', 'new_fault_messages', 'cleanup']} | {'measurements': [x.get('measurement') for x in record['requests']]}, indent=2))
if not record['healthy'] or not record['math_gate_passed'] or not record['histogram_gate_passed'] or not record['pair_gate_passed']:
    raise SystemExit(1)
