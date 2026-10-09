"""Owned four-fresh32K layer-major streamed INT8 diagnostic qualification.

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
import ast
import textwrap


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


base = Path(__file__).parent
observer = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0, str(observer / 'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity

mode, phase, repetition = sys.argv[1], sys.argv[2], int(sys.argv[3])
assert mode == 'streamed-lease' and phase == 'diagnostic' and repetition == 1
assert sys.argv[4:] in [[], ['--cpu-preflight']]
review_head='6cdf81078128488e5797afb9e92824b7ebe51124'
engine_commit=review_head
root=observer.parent/'perf-sycl-layer-major-streamed-kv-v0141-20261009'
def git(*args):
    return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
assert git('rev-parse','HEAD')==review_head and not git('status','--porcelain')
subprocess.run(['git','merge-base','--is-ancestor',engine_commit,review_head],cwd=root,check=True)
assert all(p=='sycl/serve/strata-sycl.sh' or p.startswith(('bench/','docs/')) for p in git('diff','--name-only',engine_commit,review_head).splitlines())
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
build_path=base/'layer-major-streamed-kv-v0141-private-build-v1/record.json'
assert digest(build_path)=='c3d9b24cc9bfd30cf121e34d9325bc145fcaeb467d650857dbb16aa9d030670f'
build=json.loads(build_path.read_text());assert not build['active'] and build['passed'] and build['compiled_engine'] and not build['gpu_tested'] and not build['adopted']
assert build['source_head']==review_head
binary=Path(build['binary']);assert digest(binary)==build['binary_sha256']=='66d5bee60f4bbdb6f710b395ae11a1e6e14a61e04db417bcd6092d4f7956377e'
candidate_source=root/'sycl/src/prefill/prefill.cpp'
candidate_header=root/'sycl/include/dpct/device.hpp'
assert digest(candidate_source)==build['source_sha256']['sycl/src/prefill/prefill.cpp']=='92b849df3e19ad77a8a56927cae48cd91d4be0f73f4edc1ff189d2177c0cd8c1'
assert digest(root/'sycl/src/program/generate.cpp')==build['source_sha256']['sycl/src/program/generate.cpp']=='4309a974a0af70d2144a7ba3d05f7780ee31016c3d4898f9fa1a980bce96b552'
assert digest(candidate_header)==build['source_sha256']['sycl/include/dpct/device.hpp']=='adbcbd0ac45995348e37ac3f5ca42d9f3d4b0555031aeb1781a1e2166a74ed52'
host_cpu_path=base/'layer-major-streamed-kv-v0141-root-source-review-v1.json'
assert digest(host_cpu_path)=='217617371c7819315df13b082a53340915b9a8653d2b0df9a712a6024624b264'
host_cpu=json.loads(host_cpu_path.read_text());assert host_cpu['passed'] and not host_cpu['active'] and not host_cpu['adopted']
flags_path=base/'layer-major-streamed-kv-v0141-uniform-build-flags-v1.json'
assert digest(flags_path)=='c92f61aab4cb99fb77dd84cabbdd895c7cc0635dacfbd2ed0c4d1ebce46652a6'
flags=json.loads(flags_path.read_text());assert flags['passed'] and not flags['active'] and flags['build_receipt_sha256']==digest(build_path)
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
assert shutil.disk_usage(base).free>170*1024**3
capture_cpu_path = base / 'owned-native-counter-v01402-cpu-v1/record.json'
capture_cpu = json.loads(capture_cpu_path.read_text())
assert digest(capture_cpu_path) == 'c93b119422938718e230d9e7c02377b7f0e6d48dc2095cb4add442f2e1025e86'
assert capture_cpu['passed'] and not capture_cpu['active']
assert digest(base / 'capture_owned_native_counter_v01402_v1.py') == capture_cpu['capture_controller_sha256']
assert digest(observer / 'sycl/tools/owned_gdb.py') == capture_cpu['owned_helper_sha256']
from capture_owned_native_counter_v01402_v1 import capture as capture_native_counter

repeat_path=base/'native-copy-decode-repeat-v0141-comparison-sequence-v1/record.json'
assert digest(repeat_path)=='a9091c40a01af7e725b297d1cb87a01c0348372f6579a34e4db051a9a8981114'
repeat=json.loads(repeat_path.read_text());assert not repeat['active'] and repeat['passed'] and len(repeat['steps'])==18
for step in repeat['steps']:
    receipt=Path(step['receipt']);assert digest(receipt)==step['receipt_sha256'];terminal_owned(json.loads(receipt.read_text()))

for prior in build['prior_owned_gpu_admission']:
    p=Path(prior['path']);assert digest(p)==prior['sha256']
    terminal_owned(json.loads(p.read_text()),require_gate=not prior['held_prior_candidate_failure_preserved'])
assert sum(x['held_prior_candidate_failure_preserved'] for x in build['prior_owned_gpu_admission'])==1

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

last_pool_path=base/'owned-pool-tasks-v0141-code32k-tasks6-diagnostic-r1/record.json'
assert digest(last_pool_path)=='116604dfa1206464b668c9e391d6a66f09350fe35abec7071402a1621641323f'
terminal_owned(json.loads(last_pool_path.read_text()))
prefix_path=base/'streamed-kv-prefix-layout-host-v1/record.json'
assert digest(prefix_path)=='d83922d9e5ab3092e25b1f2af14d550c54e53870999b70ced29178675dd14a85'
prefix=json.loads(prefix_path.read_text());assert prefix['passed'] and not prefix['active'] and not prefix['gpu_executed']
assert prefix['source_sha256']==digest(candidate_source)
assert prefix['result']==dict(prefix_cases=11,rows_checked=5767168,seed_bytes_checked=657471936,owner_cases=14)


# Admit the exact previous allocation rejection separately. It remains failed;
# terminal_owned() and every normal numerical/fault gate are unchanged.
safety_path=base/'streamed-kv-capacity-failure-safety-review-v1/record.json'
assert digest(safety_path)=='0208be3bfc73efd1ec61f46e844b35036be6735160435c36227f073a11135dfa'
safety=json.loads(safety_path.read_text())
assert safety['active'] is False and safety['passed'] and safety['healthy'] and safety['gpu_executed']
assert safety['boot_id']==boot and not safety['preflight_fault_messages'] and not safety['log_budget_exceeded']
assert safety['binary_sha256']==digest(binary) and safety['source_head']==review_head
assert safety['preserved_failure_status']==dict(completed=False,healthy=False,math_gate_passed=False)
assert safety['owned_failed_processes_absent'] and safety['prior_capacity_failure_exact']
for call in safety['steps']:
    assert call['exit_code']==0 and not call['timed_out'] and not call['still_alive']
    assert not process_identity(call['pid'])
failed_path=base/'owned-layer-major-streamed-kv-v0141-code32k-streamed-diagnostic-r1/record.json'
assert digest(failed_path)==safety['failed_receipt_sha256']=='9f76562cc4ced2a60363e7b01b31cfd6da4bbfde8abc62c45c0abe737027f0a1'
failed=json.loads(failed_path.read_text())
assert failed['active'] is False and failed['completed'] is False and failed['healthy'] is False and failed['math_gate_passed'] is False
assert not failed['new_fault_messages'] and failed['cleanup']==dict(forced=True,inferior_survived=False,gdb_survived=False)
for role in ['inferior','debugger']:
    now=process_identity(failed[role]['pid']);assert not now or now['start_ticks']!=failed[role]['start_ticks']


# The completed C four-read RAM-lease gate qualifies this exact source/binary.
first_path=base/'owned-layer-major-streamed-kv-v0141-code32k-streamed-lease-diagnostic-r1/record.json'
assert digest(first_path)=='f54eb757d9553403b59a3f72eb07b036c5d90068f57466b1c97f629b7003b7b2'
first=json.loads(first_path.read_text());terminal_owned(first)
assert first['binary_sha256']==digest(binary) and first['commit']==review_head
assert first['verified_ram_lease_gate']['passed'] and first['streamed_layer_major_configuration_gate']['passed']
assert len(first['requests'])==4 and all(q['math_gate_passed'] for q in first['requests'])
assert [q['fixture'] for q in first['requests']]==['A','B','A','B']
assert all(q['first_head']['finite'] and q['first_head']['floats']==248320 and len(q['prefill_state']['parts'])==66 for q in first['requests'])
assert first['candidate_source_sha256']==digest(candidate_source) and first['candidate_header_sha256']==digest(candidate_header)
assert first['source_preparation_receipt_sha256']==digest(host_cpu_path) and first['prefix_layout_cpu_receipt_sha256']==digest(prefix_path)
assert first['prior_capacity_failure_safety_review_sha256']==digest(safety_path)
reference_build_path=base/'integrated-upstream-v0.1.41-20261009-v2/build-record.json'
reference_build=json.loads(reference_build_path.read_text())
assert reference_build['passed'] and not reference_build['active'] and reference_build['binary_sha256']==new_full['binary_sha256']
baseline_full_path=new_full_path
baseline_full=new_full
qualified_numerical={'requests':reference_reads}
# Same engine version: compare complete semantic hashes, including config and all
# inactive MTP/KV bytes. The older cross-version comparator remains unchanged.
version_cpu_path=base/'upstream-v0141-full-history-and-version-comparator-host-check-20261009-v1.json'
assert digest(version_cpu_path)=='c083340eae6fc5a62ebb079fc6124a50fcc8ee875f316b40296a55a2024157a0'
version_cpu=json.loads(version_cpu_path.read_text());assert version_cpu['passed'] and not version_cpu['active']
version_comparator_path=base/'compare_saved_session_versions_v0141_v1.py'
assert digest(version_comparator_path)==version_cpu['comparator_sha256']=='0260db007151cde044651208dc531249985d3afb1212b1b19500b883c2eba8e0'
for q in qualified['requests']:
    if 'first_head' in q:
        p=Path(q['first_head']['file']);assert p.is_file() and digest(p)==q['first_head']['sha256']
    if 'prefill_state' in q:
        p=Path(q['prefill_state']['file']);assert p.is_file() and p.stat().st_size==q['prefill_state']['bytes']
assert len(qualified['requests'])==12 and len(qualified['sessions'])==6
for q in qualified['sessions']:
    if 'image' in q:
        assert q['image']['semantic']['config']==643052213586580166
# New debug source qualification is independent of the old C32 pass. Keep all
# original C admissions/pins above as historical evidence, not a debug math pass.
C_root=root; C_build=build; C_build_path=build_path
C_candidate_source=candidate_source
last_C_path=base/'owned-layer-major-streamed-kv-v0141-full256k-streamed-lease-diagnostic-r1/record.json'
assert digest(last_C_path)=='a3413ac4c3f20ae593d8507a7db0ac405cd7fab2de286547460e6ba82b0d9404'
last_C=json.loads(last_C_path.read_text());terminal_owned(last_C,require_gate=False)
assert not last_C['math_gate_passed'] and not last_C['full_lifecycle_passed']
last_T_path=base/'owned-decode-pool-phase-timing-v0141-code32k-tasks6-diagnostic-r1/record.json'
assert digest(last_T_path)=='e770766de708a5ab8842b4aa3ea80a6d1453765059e8b7816e36aed09331148c'
last_T=json.loads(last_T_path.read_text());terminal_owned(last_T)
root=observer.parent/'debug-sycl-layer-major-repeat-v0141-20261009'
review_head=engine_commit='9c2ebde89e5157c81a9c9ae135719452a0244ca3'
assert git('rev-parse','HEAD')==review_head and not git('status','--porcelain')
assert git('merge-base',review_head,C_build['source_head'])==C_build['source_head']
assert all(p.startswith(('bench/','docs/')) or p in ['sycl/src/prefill/prefill.cpp','sycl/src/prefill/repeat_capture.hpp'] for p in git('diff','--name-only',C_build['source_head'],review_head).splitlines())
build_path=base/'repeat-capture-v0141-private-build-v1/record.json'
assert digest(build_path)=='f7d8b0b1efec0aa7ad21fb937d92166db608fe0c8a9253a6998c9a7e24f5b166'
build=json.loads(build_path.read_text())
assert build['passed'] and not build['active'] and build['compiled_engine'] and not build['gpu_tested'] and not build['adopted']
assert build['source_head']==review_head
binary=Path(build['binary']);assert digest(binary)==build['binary_sha256']=='4fa4944923c5e0509c71014d75646476c0eeef965b79f459066908bb2f50b83c'
candidate_source=root/'sycl/src/prefill/prefill.cpp'
candidate_header=root/'sycl/include/dpct/device.hpp'
assert digest(candidate_source)=='acd062d1a4f4ab29230630e066fbc29080f92c270cb1fc3e900be12502a17b56'
assert digest(candidate_header)=='adbcbd0ac45995348e37ac3f5ca42d9f3d4b0555031aeb1781a1e2166a74ed52'
assert digest(root/'sycl/src/prefill/repeat_capture.hpp')=='1db25fde8987a4f778d9cba2a5dfdc811dc4f0e36778b9e6b34a779fdc7bd509'
assert digest(root/'sycl/src/program/generate.cpp')=='4309a974a0af70d2144a7ba3d05f7780ee31016c3d4898f9fa1a980bce96b552'
debug_source_proof=base/'repeat-capture-v0141-source-review-v2.json'
assert digest(debug_source_proof)=='a5ed2740c973830ed67275a89f8f911910abaa6d497a2f3424f5b45a595838b7'
flags_path=base/'repeat-capture-v0141-uniform-build-flags-v1.json'
assert digest(flags_path)=='e02a70304abc0d1ccb3c46c67cbe4e7663ef6d9586ccabbcc180c58c3f9ce7e8'
flags=json.loads(flags_path.read_text());assert flags['passed'] and not flags['active'] and flags['build_receipt_sha256']==digest(build_path)
assert flags['candidate_compile_count']==115 and flags['candidate_unique_source_count']==114 and not flags['different_flags'] and not flags['missing_sources']
mock_path=base/'repeat-capture-cpu-validation-v2/record.json'
assert digest(mock_path)=='09d22091a053ef9b3bb356ae0b760a4298608e84f12f5085f9d169fe60017cfe'
mock=json.loads(mock_path.read_text());assert mock['passed'] and mock['complete'] and not mock['active'] and not mock['gpu_executed'] and not mock['cleanup'] and not mock['survivors']
assert len(mock['cases'])==12 and mock['production_header_sha256']==digest(root/'sycl/src/prefill/repeat_capture.hpp')
assert not any(process_identity(x['pid']) and process_identity(x['pid'])['start_ticks']==x['start_ticks'] for x in mock['owners'].values())
assert all(not process_identity(x['identity']['pid']) or process_identity(x['identity']['pid'])['start_ticks']!=x['identity']['start_ticks'] for x in build['steps'] if 'identity' in x)
out=base/'owned-repeat-capture-v0141-two-full-diagnostic-r1'
assert not out.exists()
lock=(base/'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
if sys.argv[4:]==['--cpu-preflight']:
    print(json.dumps({'cpu_preflight_passed':True,'gpu_executed':False,'binary_sha256':digest(binary),'historical_C_four_read_receipt_sha256':digest(first_path),'full_reference_sha256':digest(new_full_path),'same_version_full_saved_semantic_comparison':True}))
    sys.exit(0)

os.chdir(root)  # Preserve relative expert-profile lookup in the frozen source tree.
out.mkdir(mode=0o700)
probes=out/'probes';probes.mkdir()
source=observer/'sycl/tools/recover-xe.sh'
m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
helper_python=source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
runner_class=next(x for x in ast.parse(helper_python).body if isinstance(x,ast.ClassDef) and x.name=='Runner')
run_node=next(x for x in runner_class.body if isinstance(x,ast.FunctionDef) and x.name=='run')
run_source=textwrap.dedent(ast.get_source_segment(helper_python,run_node))
run_predicate='while child.poll() is None and time.monotonic() < deadline:'
assert run_source.count(run_predicate)==1
run_source=run_source.replace(run_predicate,'while child.poll() is None and time.monotonic() < deadline and sum(p.stat().st_size for p in self.output.iterdir() if p.is_file()) < 64 * 1024**2:')
exec(compile(run_source,str(Path(__file__)),'exec'),m.__dict__)
m.Runner.run=m.run
r=m.Runner(probes)

record={'scope':'Bounded intermediate-row diagnosis based on numerically rejectedC; two fresh262140+4 reaching physicalcell262143 with exact original control/resume/full1/control/full2 history. Before/after GEN byte ledger partitions captures by request. Full1 state/session compared to canonical baseline, repeat to first. No clipped/RESTORE/refusal/later-control lifecycle in this diagnostic. Diagnostic wait can mask original failure; no performance eligibility/adoption/full-lifecycle qualification.128MiB binary and64MiB textcap.',
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'active':True,'completed':False,'healthy':False,'requests':[],'snapshots':[],'steps':[],
        'deadline_seconds':5400,'protocol_timeout_seconds':2400,'capture_limit_bytes':128*1024**2,'capture_request_offsets':[],'log_limit_bytes':64*1024**2}
g=None;master=slave=None;cursor=None;pending=bytearray();current=None
capture_file=out/'repeat-rows.bin'
raw=(out/'protocol.stdout.raw').open('wb')
events=(out/'events.jsonl').open('w',buffering=1)
started=time.monotonic();next_update=started

def save():
    record['elapsed_seconds'] = time.monotonic() - started
    record['steps'] = r.calls
    record['active_request'] = current
    if g:
        record.update(inferior=g.inferior, debugger=g.debugger_identity)
    temporary = out / 'record.json.tmp'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(out / 'record.json')


def event(kind,**fields):
    events.write(json.dumps({'kind':kind,'elapsed_seconds':time.monotonic()-started,**fields})+'\n')

def poll():
    global next_update
    g.poll(.01)
    if time.monotonic() - started >= record['deadline_seconds']:
        raise TimeoutError('bounded model job deadline')
    if g.stops and g.stops[-1] != 'resumed' and g.exit_code is None and g.exit_signal is None:
        raise RuntimeError('inferior stopped: ' + g.stops[-1])
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.name.endswith(('.bin','.tmp'))) >= record['log_limit_bytes']:
        raise RuntimeError('model log size limit')
    if time.monotonic() >= next_update:
        save()
        next_update = time.monotonic() + 5


def line(seconds=None):
    end=time.monotonic()+(seconds or record['protocol_timeout_seconds'])
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_READ)
        while time.monotonic()<end:
            if b'\n' in pending:
                value,_,tail=pending.partition(b'\n');pending[:]=tail
                return value.decode().strip()
            poll()
            if not ready.select(.02):continue
            try:data=os.read(master,65536)
            except BlockingIOError:continue
            except OSError:
                if g.exit_code is not None or g.exit_signal is not None:raise RuntimeError('engine exited before protocol reply')
                raise
            if not data:raise RuntimeError('protocol EOF')
            raw.write(data);raw.flush();pending.extend(data)
            event('stdout',bytes=len(data),raw_offset=raw.tell())
    raise TimeoutError('engine protocol deadline')

def send(data):
    end=time.monotonic()+20;view=memoryview(data)
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_WRITE)
        while view:
            poll()
            if time.monotonic()>=end:raise TimeoutError('protocol input deadline')
            if not ready.select(.02):continue
            try:n=os.write(master,view[:8192])
            except BlockingIOError:continue
            view=view[n:]

try:
    cpu=json.loads((base/'owned-main-thread-cpu/record.json').read_text())
    assert cpu['passed'] and len(cpu['cases'])==4
    current_helper=hashlib.sha256((observer/'sycl/tools/owned_gdb.py').read_bytes()).hexdigest()
    assert all(v['helper_sha256']==current_helper for v in cpu['cases'] if v['name'].startswith('current-'))
    for v in cpu['cases']:
        for key in ['inferior','debugger']:
            now=process_identity(v[key]['pid'])
            assert not now or now['start_ticks']!=v[key]['start_ticks'] or now['state']=='Z'
    record['cpu_debugger_guard_sha256']=hashlib.sha256((base/'owned-main-thread-cpu/record.json').read_bytes()).hexdigest()
    health_path=base/'post-device-profile-abort-v01402-health-v1/record.json'
    assert hashlib.sha256(health_path.read_bytes()).hexdigest()=='27eda22ac64827a042db7ef90c204a644e8dbc1d56ed2c4132aae57768f91cff'
    health=json.loads(health_path.read_text())
    full=json.loads((base/'full-context-eager-verifier-serve/record.json').read_text())
    assert not full['active'] and not full['new_fault_messages']
    assert not full['cleanup']['inferior_survived'] and not full['cleanup']['gdb_survived']
    assert health['started_utc'] > full['finished_utc']
    for key in ['inferior','debugger']:
        old=full[key];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']

    assert health['healthy'] and health['boot_id']==record['boot_id']
    for name in ['full-context-entry-capacity','full-context-entry-csr','full-context-layer-trace-serve','full-context-expert-wait-32-serve','full-context-expert-wait-1-serve','full-context-legacy-l0-serve']:
        old=json.loads((base/name/'record.json').read_text())
        assert not old['active'] and not old['new_fault_messages']
        for key in ['inferior','debugger']:
            now=process_identity(old[key]['pid'])
            assert not now or now['start_ticks']!=old[key]['start_ticks'] or now['state']=='Z'
    for p in base.rglob('stalled-writer.json'):
        old=json.loads(p.read_text());now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks'] or now['state']=='Z'
    old_cursor=next(x['argv'][x['argv'].index('--after-cursor')+1] for x in last_T['steps'] if x['label']=='kernel-after')
    gap=r.run('kernel-gap',['/usr/bin/journalctl','-k','--after-cursor',old_cursor,'--no-pager','-o','json'],seconds=5)
    rows=[json.loads(s) for s in gap.splitlines() if s.startswith('{')]
    faults=[x['MESSAGE'] for x in rows if ('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and m.FAULT.search(x.get('MESSAGE',''))]
    record['preflight_fault_messages']=faults;assert not faults
    assert r.run('embedding-state',['/usr/bin/systemctl','--user','show','llama-server-qwen3embed.service','-p','ActiveState'],seconds=5).strip()=='ActiveState=inactive'
    reference=json.loads((base/'model-lease-copy-off/normal-mtp.json').read_text())
    assert reference['exit_code']==0 and len(reference['requests'])==4
    record['binary_sha256']=build['binary_sha256']
    record['build_receipt_sha256']=hashlib.sha256(build_path.read_bytes()).hexdigest()
    record['updated32k_comparison_receipt_sha256']=digest(sequence_path)
    record['previous_capacity_failure_receipt_sha256']=digest(failed_path)
    record['prior_capacity_failure_safety_review_sha256']=digest(safety_path)
    record['version_identity_comparator_sha256']=hashlib.sha256(version_comparator_path.read_bytes()).hexdigest()
    record['version_identity_cpu_check_sha256']=hashlib.sha256(version_cpu_path.read_bytes()).hexdigest()
    record['same_version_full_reference_receipt_sha256']=digest(new_full_path)
    record['physical256k_sequence_completed']=False
    capture_cpu_path=base/'owned-native-counter-v01402-cpu-v1/record.json'
    assert hashlib.sha256(capture_cpu_path.read_bytes()).hexdigest()=='c93b119422938718e230d9e7c02377b7f0e6d48dc2095cb4add442f2e1025e86'
    capture_cpu=json.loads(capture_cpu_path.read_text());assert capture_cpu['passed'] and not capture_cpu['active']
    capture_path=base/'capture_owned_native_counter_v01402_v1.py'
    assert hashlib.sha256(capture_path.read_bytes()).hexdigest()==capture_cpu['capture_controller_sha256']
    assert hashlib.sha256((observer/'sycl/tools/owned_gdb.py').read_bytes()).hexdigest()==capture_cpu['owned_helper_sha256']
    from capture_owned_native_counter_v01402_v1 import capture as capture_native_counter
    record['native_counter_capture_controller_sha256']=capture_cpu['capture_controller_sha256']
    os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'))
    env = m.health_environment()
    env['LD_LIBRARY_PATH'] += ':/opt/intel/oneapi/mkl/2026.1/lib'
    for key in ['LD_PRELOAD', 'ZET_ENABLE_METRICS']:
        env.pop(key, None)
    env = {k: v for k, v in env.items() if not k.startswith(('UNITRACE_', 'XPTI_'))}
    env.update(NEOReadDebugKeys='1', EnableDirectSubmission='0', STRATA_PREFILL_FIRST='0', STRATA_PREFILL_RING='8')
    env.update(STRATA_PREFILL_COMPACT='2', STRATA_PREFILL_ATTN_LAYOUT='1', STRATA_PREFILL_LAYER_MAJOR='1', STRATA_PREFILL_LAYER_MAJOR_STREAMED_KV='1', STRATA_KV_STAGE_OWN='1', STRATA_KV_PREFETCH='0')
    env.pop('STRATA_PREFILL_COPY_ENGINE', None)
    env.pop('STRATA_PREFILL_COPY_PHASE_RELEASE', None)
    env.pop('STRATA_STAGER_CPU_LIST',None)
    for key in ['STRATA_PREFILL_RELEASE_CACHE','STRATA_PREFILL_CACHE_RELEASE_FRAC','STRATA_PREFILL_CACHE_RESTORE','STRATA_PREFILL_LAYER_MAJOR_R_GPU','STRATA_PREFILL_LAYER_MAJOR_R_INPLACE']:
        env.pop(key,None)
    env.update(STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_RELEASE_FRAC='1',STRATA_PREFILL_CACHE_RESTORE='ram',STRATA_PREFILL_CACHE_VERIFY='1',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1')
    env.update(STRATA_DECODE_TIMING='1',STRATA_DUMP_FIRST_LOGITS=str(out/'first-head.bin'),STRATA_PREFILL_DUMP_STATE=str(out/'prefill-state.bin'),STRATA_PREFILL_REPEAT_CAPTURE=str(capture_file))
    if phase == 'diagnostic':
        env = m.diagnostic_environment(env)
        env.update(ZEL_LOADER_LOGGING_LEVEL='warn',ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        env.pop('UR_LOG_TRACING',None)
        env['UR_ENABLE_LAYERS']=','.join(x for x in env.get('UR_ENABLE_LAYERS','').split(',') if x and x!='UR_LAYER_TRACING')
        if not env['UR_ENABLE_LAYERS']: env.pop('UR_ENABLE_LAYERS')
    if phase == 'clean':
        assert not any(k.startswith(('UR_LOG_', 'ZE_ENABLE_', 'ZEL_')) or k in ['UR_ENABLE_LAYERS', 'STRATA_TRACE'] for k in env)
    assert not any(k in env for k in ['STRATA_PREFILL_SYNC', 'STRATA_PREFILL_TIMING', 'STRATA_TRANSFER_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING', 'STRATA_VERIFY_NO_HOST', 'STRATA_VERIFY_DEVICE_PLAN', 'STRATA_PREFILL_HOST_TIMING'])
    record['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission','EnableImplicitConvertionToCounterBasedEvents','MKL_CBWR']}
    record['candidate_build_receipt_sha256']=hashlib.sha256(build_path.read_bytes()).hexdigest()
    record['compatibility_change']='Debug9c based on rejectedC6cdf; bounded98304..98335 rows0..12 synchronous readbacks, unknown-destination quarantine.115/114 uniform compile and12CPU fake-queue cases passed; no prior debug model math qualification. Extra waits may mask originalC mismatch.'
    record['decode_host_counters']='STRATA_DECODE_TIMING prints existing unconditional host counters once per request. No STRATA_VERIFY_PROFILE, GPU timestamps, added synchronization or GPU profiler. Logged diagnostic times are excluded from speed comparisons.'
    record['first_four_fresh_C_lease_receipt_sha256']=hashlib.sha256(first_path.read_bytes()).hexdigest()
    record['baseline_full_capacity_receipt_sha256']=hashlib.sha256(baseline_full_path.read_bytes()).hexdigest()
    record['uniform_flags_receipt_sha256']=hashlib.sha256(flags_path.read_bytes()).hexdigest()
    record['root']=str(root);record['source_head']=build['source_head']
    record['performance_eligible']=False;record['adopted']=False;record['full_lifecycle_passed']=False
    args=list(reference['args'])
    for key,value in [('--pcie-frac','0'),('--max-context','262144'),('--prefill','8192'),('--expert-cache','128')]:args[args.index(key)+1]=value
    args+=['--kv','int8','--kv-resident','32768','--prompt-cache','1','--prompt-cache-every','262139','--prompt-cache-root','0','--turn-token','-1']
    args[args.index('--conversation-cache-mib')+1]='0'
    argv=[str(binary)]+args;record['argv']=argv
    record['mode']=mode;record['phase']=phase;record['repetition']=repetition
    record['scope']='Bounded intermediate-row diagnosis based on numerically rejectedC; two fresh262140+4 reaching physicalcell262143 with exact original control/resume/full1/control/full2 history. Before/after GEN byte ledger partitions captures by request. Full1 state/session compared to canonical baseline, repeat to first. No clipped/RESTORE/refusal/later-control lifecycle in this diagnostic. Diagnostic wait can mask original failure; no performance eligibility/adoption/full-lifecycle qualification.128MiB binary and64MiB textcap.'
    previous_cnr=json.loads((base/'event-ack-registered-copy-cb-v01402-state-sequence/record.json').read_text())
    assert previous_cnr['passed'] and not previous_cnr['active']
    record['previous_goal_turn']='C full1 pass/full2 reject preserved. Debug realbuild andCPU mocks passed, not model-qualified; baseline869 and priorC32 do not qualify debug.'
    record['original_C_numerical_failure_receipt_sha256']=digest(last_C_path)
    record['original_C_math_gate_passed']=False
    record['last_T_terminal_receipt_sha256']=digest(last_T_path)
    record['capture_cpu_receipt_sha256']=digest(mock_path)
    record['capture_source_review_sha256']=digest(debug_source_proof)
    record['capture_binary_file']=str(capture_file)
    record['diagnostic_sequence_completed']=False
    record['configuration_note']='Debug enabled repeat rows, same baselineargv: context262144/chunk8192/INT8resident32768/cache128/workers5/pcie0/MTP4/PC1/ckpt1/every262139/root0/turn-token-1. C layer-major1/streamedKV/full-cacheRAM/MTPRAM with payload verification and required graph retirement/rebuild. Host residual plane, no GPU residual/native copy factory/phase flags/prefetch/affinity/profiler. Warning diagnostic budget64MiB.'
    record['reference_binary_sha256']=qualified['binary_sha256']
    record['reference_result_sha256']=hashlib.sha256((base/'model-lease-copy-off/normal-mtp.json').read_bytes()).hexdigest()
    record['source_sha256']={str(p):digest(p) for p in [Path(__file__),source,observer/'sycl/tools/owned_gdb.py',candidate_source,candidate_header,flags_path,base/'capture_owned_native_counter_v01402_v1.py']}
    record['prefix_layout_cpu_receipt_sha256']=digest(prefix_path)
    record['candidate_source_sha256']=digest(candidate_source)
    record['candidate_header_sha256']=digest(candidate_header)
    record['source_preparation_receipt_sha256']=digest(host_cpu_path)
    record['source_review_sha256']=hashlib.sha256((base/'main-cache-repeat32k-v01402-source-review/record.json').read_bytes()).hexdigest()
    live_gate=json.loads((base/'live-prefill-state-v01402-host-check-v2/record.json').read_text())
    assert live_gate['passed'] and not live_gate['active']
    record['live_state_comparator_sha256']=hashlib.sha256((base/'compare_live_prefill_state_v01402_v2.py').read_bytes()).hexdigest()
    record['host_lease_check_sha256']=hashlib.sha256((base/'main-cache-lease-v01402-host-check/record.json').read_bytes()).hexdigest()
    reader_check=base/'saved-session-reader-v01402-host-check-v2/record.json'
    cpu_reader=json.loads(reader_check.read_text())
    assert cpu_reader['passed'] and not cpu_reader['active']
    assert cpu_reader['reader_sha256']==hashlib.sha256((base/'read_saved_session_v01402_v2.py').read_bytes()).hexdigest()
    record['session_reader_cpu_check_sha256']=hashlib.sha256(reader_check.read_bytes()).hexdigest()
    earlier=base/'owned-full-kv-access-v01402-full256k-diagnostic-r1/record.json'
    assert hashlib.sha256(earlier.read_bytes()).hexdigest()=='833dcb26548cd298b49f2da746fa48675f04e75d40f47c1409f5a3d3b60caf1d'
    negative=json.loads(earlier.read_text())
    assert negative['healthy'] and negative['completed'] and not negative['active'] and not negative['math_gate_passed']
    assert negative['exit_code']==0 and not negative['new_fault_messages']
    record['previous_config_rejection']={'receipt_sha256':hashlib.sha256(earlier.read_bytes()).hexdigest(),
        'reason':'PC1/default turn-token split the32K final chunk at32761+6, unlike the accepted PC0 geometry32767. Same64 IDs but different intermediate state/head/logprobs and42/69 rather than43/66 MTP counts. Normal exit, no GPU fault. Not a source-change attribution.'}
    from read_saved_session_v01402_v2 import read_session, full_kv_gate
    fixture=base/'coding-review-32k-tokens.txt'
    full_fixture=base/'full-context-copy-off/coding-context-256k-tokens.txt'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449'
    assert hashlib.sha256(full_fixture.read_bytes()).hexdigest()=='cc29e4427bcacb21c9f7df2d1f1a7d41897fce5bd76a8ca43d8c3c34d914dd2a'
    control=list(map(int,fixture.read_text().split()))[:32768]
    full_source=list(map(int,full_fixture.read_text().split()))
    suffix=[248046,198,248045,74455,198,248068,198,248069,271]
    def full_prompt(n):
        assert 32768<=n<=262144
        ids=full_source[:n-len(suffix)]+suffix
        assert len(ids)==n
        return ids
    full=full_prompt(262140)
    common=next(i for i,(a,b) in enumerate(zip(control,full)) if a!=b)
    assert common<32767, 'interposed control must clear the full-prefix checkpoint'
    record['fixtures']={'control_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
                        'full_sha256':hashlib.sha256(full_fixture.read_bytes()).hexdigest(),
                        'common_prefix_tokens':common}
    baseline=next(x for x in qualified['requests'] if x['name']=='control32k-before')
    alternate=full_prompt(32768)
    alternate_baseline=next(x for x in qualified_numerical['requests'] if x['name']=='alternate32k-first')
    assert baseline['mtp_counts']==[41,66] and alternate_baseline['mtp_counts']==[38,75]
    cursor=m.journal_cursor(r,'kernel-before');save()
    master,slave=os.openpty();tty.setraw(slave);os.set_blocking(master,False)
    g=OwnedGdb(argv,out/'debugger',env,inferior_tty_fd=slave)
    g.command('-gdb-set may-call-functions off')
    g.command('-gdb-set debug-file-directory /usr/lib/debug:/home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/runtime-symbols/extracted/usr/lib/debug')
    g.run();record['startup']=[]
    while True:
        value=line();record['startup'].append(value)
        if value.startswith('ERR'):raise RuntimeError(value)
        if value.startswith('READY '):break
    os.close(slave);slave=None
    actual_identity=process_identity(g.inferior['pid'])
    assert actual_identity and actual_identity['start_ticks']==g.inferior['start_ticks']
    assert Path(os.readlink('/proc/'+str(actual_identity['pid'])+'/exe')).resolve()==binary.resolve()
    assert hashlib.sha256(Path('/proc',str(actual_identity['pid']),'exe').read_bytes()).hexdigest()==record['binary_sha256']
    record['actual_executable_identity']={'inferior':actual_identity,'sha256':record['binary_sha256'],'boot_id':record['boot_id']}
    actual=dict(item.decode().split('=',1) for item in Path('/proc',str(actual_identity['pid']),'environ').read_bytes().split(b'\0') if b'=' in item)
    assert not any(k.startswith(('UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS'] for k in actual)
    record['actual_target_environment']={k:v for k,v in actual.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    assert argv[1:]==qualified['argv'][1:]
    save()

    from compare_live_prefill_state_v01402_v2 import compare_states

    stderr=out/'debugger/inferior.stderr'
    record['sessions']=[]
    math_ok=True
    assert int(record['startup'][-1].split()[1])==262144

    def captures(name, required):
        result={}
        head=out/'first-head.bin'
        if head.exists():
            kept=out/f'{name}.head.bin';assert not kept.exists();head.rename(kept)
            data=kept.read_bytes();values=array.array('f');values.frombytes(data)
            assert len(values)==248320 and all(map(math.isfinite,values))
            result['first_head']={'file':str(kept),'sha256':hashlib.sha256(data).hexdigest(),'floats':len(values),'finite':True}
        state=out/'prefill-state.bin'
        if state.exists():
            kept=out/f'{name}.state.bin';assert not kept.exists();state.rename(kept)
            parts=[]
            with kept.open('rb') as stream:
                while header:=stream.read(8):
                    assert len(header)==8
                    size=struct.unpack('=Q',header)[0];offset=stream.tell();left=size;digest=hashlib.sha256()
                    while left:
                        data=stream.read(min(left,1048576));assert data;digest.update(data);left-=len(data)
                    parts.append({'index':len(parts),'offset':offset,'bytes':size,'sha256':digest.hexdigest()})
            assert len(parts)==66 and parts[0]['bytes']==8
            result['prefill_state']={'file':str(kept),'bytes':kept.stat().st_size,'parts':parts}
        if required:
            assert 'first_head' in result and 'prefill_state' in result, 'missing fresh full-read capture'
        return result

    def request(name,tokens,new,expected=None,fresh=False,allow=True,tail=False):
        global current,math_ok
        current=name
        assert len(tokens)>=32768
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        start_offset=stderr.stat().st_size
        command=f'GEN {new} ckpt=1 logprobs=5 '+','.join(map(str,tokens))+'\n'
        (out/f'{name}.input-tokens.txt').write_text(' '.join(map(str,tokens))+'\n')
        before_capture=capture_file.stat().st_size if capture_file.exists() else 0
        assert before_capture<=record['capture_limit_bytes']
        result={'name':name,'capture_before_bytes':before_capture,'input_tokens':len(tokens),'max_new':new,'ids':[],'logprobs':[],'protocol':[],
                'request_sha256':hashlib.sha256(command.encode()).hexdigest(),'allowed':allow}
        record['requests'].append(result);save();send(command.encode());event('request',name=name)
        request_started=time.monotonic()
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith('T '):result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):
                assert all(math.isfinite(float(x.rsplit(':',1)[-1])) for x in value.split()[1:])
                result['logprobs'].append(value)
            if value.startswith(('DONE ','ERR ')):break
        result['diagnostic_wall_seconds']=time.monotonic()-request_started
        after_capture=capture_file.stat().st_size if capture_file.exists() else 0
        assert before_capture<=after_capture<=record['capture_limit_bytes']
        result['capture_after_bytes']=after_capture
        record['capture_request_offsets'].append({'name':name,'begin':before_capture,'end':after_capture})
        if name.startswith('full256k-'): assert after_capture>before_capture
        else: assert after_capture==before_capture
        windows=[]
        with stderr.open('rb') as stream:
            stream.seek(start_offset)
            for item in stream:
                if item.startswith(b'strata trace: window '):
                    match=re.search(rb'window (-?\d+) (-?\d+)',item);assert match
                    windows.append(list(map(int,match.groups())))
        result['verify_windows']=windows
        assert all(pos>=0 and count>0 and pos+count<=262144 for pos,count in windows)
        if not allow:
            assert value.startswith('ERR prompt') and not result['ids'] and not result['logprobs'] and not windows
            assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
            result['math_gate_passed']=True;save();return result
        assert value.startswith('DONE '),value
        fields=value.split()
        assert int(fields[2])==len(tokens) and int(fields[1])==new and len(result['ids'])==new and len(result['logprobs'])==new
        result['mtp_counts']=list(map(int,fields[6:8]))
        result['measurement']={'generated_tokens':int(fields[1]),'prompt_tokens':int(fields[2]),
                               'prompt_ms':float(fields[3]),'decode_ms':float(fields[4]),'purpose':'diagnostic only; excluded from performance'}
        result['resume_tokens']=[int(v.split()[1]) for v in result['protocol'] if v.startswith(('RESUME ','REUSED '))]
        assert result['resume_tokens']
        if fresh:assert all(n==0 for n in result['resume_tokens']), 'a repeated full read was silently reused'
        result.update(captures(name,fresh))
        result['math_gate_passed']=True
        if expected is not None:
            comparison={'ids_equal':result['ids']==expected['ids'],
                        'logprobs_equal':result['logprobs']==expected['logprobs']}
            if fresh:
                comparison['first_head_equal']=result['first_head']['sha256']==expected['first_head']['sha256']
                state_cmp=compare_states(result['prefill_state'],expected['prefill_state'],len(tokens)-1)
                result['live_prefill_comparison']=state_cmp
                comparison['all_live_state_equal']=not state_cmp['different_live_parts']
                comparison['mtp_counts_equal']=result['mtp_counts']==expected.get('mtp_counts',[43,66])
            elif 'first_head' in result and 'first_head' in expected:
                comparison['first_head_equal']=result['first_head']['sha256']==expected['first_head']['sha256']
            comparison['mtp_counts_equal']=result['mtp_counts']==expected['mtp_counts']
            expected_done=next(v.split() for v in expected['protocol'] if v.startswith('DONE '))
            comparison['finish_reason_equal']=fields[5]==expected_done[5]
            comparison['finite_positive_times']=all(math.isfinite(float(v)) and float(v)>0 for v in fields[3:5])
            if not fresh and ('prefill_state' in result or 'prefill_state' in expected):
                assert 'prefill_state' in result and 'prefill_state' in expected
                state_cmp=compare_states(result['prefill_state'],expected['prefill_state'],len(tokens)-1)
                result['live_prefill_comparison']=state_cmp
                comparison['all_live_state_equal']=not state_cmp['different_live_parts']
            result['comparison']=comparison
            result['math_gate_passed']=all(comparison.values())
        if tail:
            assert windows and max(pos+count for pos,count in windows)==262144
            result['last_executed_physical_cell']=262143
            if new==2:
                assert windows[-1][1]==2, 'expected a clipped two-token verification window'
                result['clipped_verify_tail']=2
        math_ok=math_ok and result['math_gate_passed'];save()
        return result

    def session(command,name,path,expected_tokens):
        global current
        current=name
        if command=='SAVE':assert not path.exists()
        result={'name':name,'command':command,'file':str(path),'protocol':[]}
        record['sessions'].append(result);save();send(f'{command} {path}\n'.encode())
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith(('SAVED ','RESTORED ','SERR ')):break
        assert value.startswith('SAVED ' if command=='SAVE' else 'RESTORED '),value
        fields=value.split();assert int(fields[1])==expected_tokens and int(fields[2])==path.stat().st_size
        result['tokens']=int(fields[1]);result['bytes']=int(fields[2]);result['diagnostic_ms']=float(fields[3])
        if command=='SAVE':
            result['image']=read_session(path)
            sem=result['image']['semantic']
            dead=sem['live']['state']['dead']
            with path.open('rb') as checked_file:
                checked_file.seek(dead['offset']);dead_bytes=checked_file.read(dead['bytes'])
                assert len(dead_bytes)==12*512
                for layer in range(12):
                    pooled=sem['kv'][layer]['parts']['pooled']
                    row=expected_tokens//4
                    assert row<sem['kv'][layer]['shape'][5]
                    checked_file.seek(pooled['offset']+row*512)
                    assert checked_file.read(512)==dead_bytes[layer*512:(layer+1)*512], ('noncanonical saved spare',name,layer,row)
            result['all_saved_main_indexer_spares_equal_live_dead']=True
            assert result['image']['semantic']['live']['ids']['count']==expected_tokens
            if name=='save-control32k':
                visible_ids=control+control_first['ids'][:-1]
            elif name in ['save-full256k-first','save-full256k-repeat']:
                visible_ids=full+full_first['ids'][:-1]
            else:visible_ids=None
            if visible_ids is not None:
                exact=hashlib.sha256(array.array('i',visible_ids).tobytes()).hexdigest()
                assert result['image']['semantic']['live']['ids']['sha256']==exact
                result['saved_ids_match_exact_consumed_visible_prefix']=True
        else:result['engine_checksum_and_compatibility_validated']=True
        result['passed']=True;save();return result

    def reject():
        raise ValueError('mathematical gate rejected; no timing/adoption')

    # A mismatch leaves the engine waiting for input, so QUIT can close it normally.
    try:
        control_first=request('control32k-before',control,64,baseline,fresh=True)
        if not math_ok:reject()
        control_file=out/'control32k.session.bin'
        control_saved=session('SAVE','save-control32k',control_file,32831)
        continuation=control+control_first['ids'];assert len(continuation)==32832
        resume_reference=request('resume32k-reference',continuation,64,next(q for q in qualified['requests'] if q['name']=='resume32k-reference'))
        assert max(resume_reference['resume_tokens'])==32831
        # The4-read numerical gate was already completed in a separate owned run.
        # Do not prime additional MTP cells before the all-byte full-state check.
        record['prior_four_fresh32k_heads_state_output_passed']=True
        record['prior_four_fresh32k_numerical_subset_sha256']=digest(subset_path)
        record['prior_C_four_fresh32k_math_receipt_sha256']=digest(first_path)
        record['matched_before_full_request_history']=[x['name'] for x in record['requests']]
        assert record['matched_before_full_request_history']==[x['name'] for x in qualified['requests'][:2]]
        save()
        previous_full=next(x for x in qualified['requests'] if x['name']=='full256k-first')
        full_first=request('full256k-first',full,4,previous_full,fresh=True,tail=True)
        if not math_ok:reject()
        full_file=out/'full256k-first.session.bin'
        full_saved=session('SAVE','save-full256k-first',full_file,262143)
        full_saved['full_capacity_gate']=full_kv_gate(full_saved['image'])
        dd5_full_saved=next(x for x in qualified['sessions'] if x['name']=='save-full256k-first')
        full_saved['all_saved_state_and_kv_bytes_equal_same_version_baseline']=full_saved['image']['semantic_sha256']==dd5_full_saved['image']['semantic_sha256']
        full_saved['version_identity_comparison']={'config':643052213586580166,'tensor_bytes_ignored':0,'same_version_complete_semantic_digest':True}
        math_ok=math_ok and full_saved['all_saved_state_and_kv_bytes_equal_same_version_baseline'];save()
        if not math_ok:reject()
        request('control32k-between-full-reads',control,64,baseline,fresh=True)
        if not math_ok:reject()
        full_repeat=request('full256k-repeat',full,4,full_first,fresh=True,tail=True)
        if not math_ok:reject()
        repeat_saved=session('SAVE','save-full256k-repeat',out/'full256k-repeat.session.bin',262143)
        repeat_saved['full_capacity_gate']=full_kv_gate(repeat_saved['image'])
        repeat_saved['all_saved_state_and_kv_bytes_equal']=repeat_saved['image']['semantic_sha256']==full_saved['image']['semantic_sha256']
        math_ok=math_ok and repeat_saved['all_saved_state_and_kv_bytes_equal'];save()
        if not math_ok:reject()
        record['diagnostic_sequence_completed']=True
    except ValueError as rejected:
        if not math_ok:
            record['mathematical_rejection']=str(rejected)
            failure_file=out/'numerical-failure.session.bin'
            session('SAVE','save-numerical-failure',failure_file,record['requests'][-1]['input_tokens']+len(record['requests'][-1]['ids'])-1)
        else:raise

    current=None;send(b'QUIT\n');event('quit')
    end=time.monotonic()+30
    with selectors.DefaultSelector() as ready:
        ready.register(master,selectors.EVENT_READ)
        while g.exit_code is None and g.exit_signal is None and time.monotonic()<end:
            poll()
            if ready.select(.01):
                try:data=os.read(master,65536)
                except (BlockingIOError,OSError):continue
                if data:raw.write(data);raw.flush();event('quit-stdout',bytes=len(data))
    assert g.exit_code==0 and g.exit_signal is None
    engine_log=out/'debugger/inferior.stderr'
    project=[]
    with engine_log.open(errors='replace') as log:
        for item in log:
            if item.startswith('strata '):project.append(item.rstrip())
    (out/'project-messages.txt').write_text('\n'.join(project)+'\n')
    text=(out/'project-messages.txt').read_text()
    seeds=re.findall(r'^strata trace: streamed layer-major QSA (\d+) seed scheduled_cells=(\d+) bytes=(\d+) \(enqueue order\)$',text,re.M)
    drains=re.findall(r'^strata trace: streamed layer-major QSA (\d+) drained_cells=(\d+) chunks=(\d+) seed_bytes=(\d+) retained_chunks=(\d+)$',text,re.M)
    fresh_reads=[q for q in record['requests'] if q['allowed'] and q.get('resume_tokens') and all(n==0 for n in q['resume_tokens'])]
    count=len(fresh_reads)
    assert len(seeds)==len(drains)==12*count,'all12 QSA owners must enter/drain each fresh request'
    expected_ordinals=list(range(12))*count
    assert [int(x[0]) for x in seeds]==[int(x[0]) for x in drains]==expected_ordinals
    assert all(list(map(int,x[1:]))==[0,0] for x in seeds)
    expected_drains=[]
    for q in fresh_reads:
        end=q['input_tokens']-1;chunks=(end+8191)//8192
        expected_drains.extend([[end,chunks,0,chunks-1]]*12)
    assert [list(map(int,x[1:])) for x in drains]==expected_drains
    if record.get('diagnostic_sequence_completed'):
        assert [q['name'] for q in fresh_reads]==['control32k-before','full256k-first','control32k-between-full-reads','full256k-repeat']
    infos=[x for x in record['startup'] if x.startswith('INFO ')]
    assert len(infos)==1
    info=dict(x.split('=',1) for x in infos[0].split()[1:])
    assert all(info[k]==v for k,v in dict(context='262144',kv='int8',kv_resident='32768',expert_slots='128',expert_cache_mib='325',spec='4',arena_mib='47962',pool_workers='5',engine='0.1.41').items())
    record['streamed_layer_major_configuration_gate']={'passed':True,'source_review_sha256':digest(host_cpu_path),'prefix_cpu_sha256':digest(prefix_path),'actual_info':info,'seeds':seeds,'drains':drains,'note':'Each fresh request uses12INT8 identity owners,8K/tail chunks determined by its physical prefix length,zero seed bytes,and all sequential retained chunks. Full cases drain262139cells; decode must separately reach262143. Nonzero prefix seeding in the layer-major path is not claimed by this sequence.'}

    cache_release=re.findall(r'^strata prefill cache release: (\d+) physical bytes, (\d+) restored bytes, source (\w+), suspend ([0-9.]+) ms, restore ([0-9.]+) ms, graph ([0-9.]+) ms, same_address ([01]), slots restored$',text,re.M)
    cache_verify=re.findall(r'^strata prefill cache verify: (\d+) occupied tail bytes matched before and after restoration; (\d+) experts differ from the initial admission map$',text,re.M)
    mtp_release=re.findall(r'^strata mtp decode release: (\d+) physical bytes, experts and head; K/V retained; total ([0-9.]+) ms, wait ([0-9.]+) ms, unmap ([0-9.]+) ms, verify ([0-9.]+) ms, verified=(\d), graph_drop ([0-9.]+) ms$',text,re.M)
    mtp_restore=re.findall(r'^strata mtp decode restore: (\d+) payload bytes from RAM, same addresses; total ([0-9.]+) ms, map ([0-9.]+) ms, copy ([0-9.]+) ms, verify ([0-9.]+) ms, verified=(\d)$',text,re.M)
    assert len(cache_release)==len(cache_verify)==len(mtp_release)==len(mtp_restore)==count,'each request requires both complete verified release/restore pairs'
    assert all(int(x[0])>0 and int(x[1])>0 and x[2]=='ram' and all(math.isfinite(float(v)) and float(v)>=0 for v in x[3:6]) for x in cache_release)
    assert all(int(x[0])>0 for x in cache_verify)
    assert all(int(x[0])>0 and x[5]=='1' and all(math.isfinite(float(v)) and float(v)>=0 for v in x[1:5]+x[6:]) for x in mtp_release)
    assert all(int(x[0])>0 and x[5]=='1' and all(math.isfinite(float(v)) and float(v)>=0 for v in x[1:5]) for x in mtp_restore)
    memory_points=['before decode cache release','main decode cache released','MTP decode weights released','temporary layer cache allocated','prefill completed with temporary layer cache','temporary buffers released','main decode cache restored, verifier graphs rebuilt','MTP decode weights restored']
    memory=re.findall(r'^strata prefill memory: ([^;]+); free=(\d+) total=(\d+) bytes$',text,re.M)
    assert len(memory)==len(memory_points)*count and [x[0] for x in memory]==memory_points*count,'each lease boundary requires a valid memory query in source order'
    assert all(0<=int(x[1])<=int(x[2]) and int(x[2])>0 for x in memory)
    record['verified_ram_lease_gate']={'passed':True,'cache_release':cache_release,'cache_verify':cache_verify,'mtp_release':mtp_release,'mtp_restore':mtp_restore,'memory_points':memory_points,'memory':memory,'note':'Payloads verified before/after release; current-map RAM sources and verifier graph rebuild required; MTP graphs retired before unmap and recaptured by normal decode. Logged durations are not performance samples.'}

    record['engine_log_bytes']=engine_log.stat().st_size
    record['engine_log_sha256']=hashlib.file_digest(engine_log.open('rb'),'sha256').hexdigest()
    if mode=='streamed-lease':
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==build['source_head']
        record['source_status_after']=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)
        assert not record['source_status_after']
    record['completed']=True
    record['math_gate_passed']=math_ok and bool(record.get('diagnostic_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])
    record['full_lifecycle_passed']=False
    assert 'repeat capture close failed' not in engine_log.read_text(errors='replace')
    # A normal mathematical rejection is terminal evidence, not a GPU-health failure.
except BaseException as error:
    record['error']=repr(error)
    if g:
        try:
            snap=g.snapshot('failure',resume=False);record['snapshots'].append(snap)
            if snap:
                record['native_counter_at_failure']=capture_native_counter(g,snap,out/'native-counter-at-failure-v1.json')
        except BaseException as inspect:record['snapshot_error']=repr(inspect)
finally:
    if g:
        record['exit_code']=g.exit_code;record['exit_signal']=g.exit_signal
        record['cleanup']=g.close()
    for fd in [master,slave]:
        if fd is not None:os.close(fd)
    raw.close();events.close()
    if cursor:
        text=r.run('kernel-after',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'],seconds=5)
        rows=[json.loads(s) for s in text.splitlines() if s.startswith('{')]
        record['new_fault_messages']=[x['MESSAGE'] for x in rows if (('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and m.FAULT.search(x.get('MESSAGE',''))) or ('strata' in x.get('MESSAGE','') and 'segfault' in x.get('MESSAGE',''))]
    record['healthy']=record['completed'] and not record.get('new_fault_messages') and not record.get('error') and not any(record.get('cleanup',{}).get(k) for k in ['inferior_survived','gdb_survived'])
    record['full_lifecycle_passed']=False  # This bounded diagnostic omits the complete lifecycle.
    record['adopted']=False
    record['performance_eligible']=False
    record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps({k:record.get(k) for k in ['mode','phase','healthy','elapsed_seconds','error','exit_code','exit_signal','new_fault_messages','cleanup']} | {'measurements':[x['measurement'] for x in record['requests'] if 'measurement' in x]},indent=2))
if not record['healthy'] or not record.get('math_gate_passed'):raise SystemExit(1)
