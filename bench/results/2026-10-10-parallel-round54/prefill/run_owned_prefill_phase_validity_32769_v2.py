"""Owned four-fresh32K opt-in --pool-tasks6 diagnostic qualification.

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
        'actual_32768_batched_input': prompt == 32769,
        'fresh_resume_and_reused': bool(resume) and all(x == 0 for x in resume)
            and any(x.startswith('RESUME ') for x in result['protocol'])
            and any(x.startswith('REUSED ') for x in result['protocol']),
        'complete_prefill_prefix': progress == [8192,16384,24576,32768],
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
assert mode in ['baseline','pool0'] and phase == 'diagnostic' and repetition == 1
assert sys.argv[4:] in [[], ['--cpu-preflight']]
review_head='7d0105f2a942ac72ca20fef69d4c2e7960653de3'
engine_commit=review_head
root=observer.parent/'perf-sycl-decode-pool-phase-timing-v0141-20261009'
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
build_path=base/'decode-pool-phase-timing-v0141-private-build-v2/record.json'
assert digest(build_path)=='3632b1514544ef2dd89046c112b823c75a27d0215aca2fce58c4b8df9e43d7f8'
build=json.loads(build_path.read_text());assert not build['active'] and build['passed'] and build['compiled_engine'] and not build['gpu_tested'] and not build['adopted']
assert build['source_head']==review_head
binary=Path(build['binary']);assert digest(binary)==build['binary_sha256']=='f303b9b9a75bcb47acbfdce42a032734d299c7143f5f32933916865fd859ea43'
candidate_source=root/'sycl/src/program/generate.cpp'
candidate_header=root/'sycl/include/dpct/device.hpp'
assert digest(candidate_source)==build['source_sha256']['sycl/src/program/generate.cpp']=='8bb78a21e3cbb9f03e40db3699d056225181e9ebb638d7eea04fadb41d852ced'
assert digest(root/'sycl/src/prefill/prefill.cpp')==build['source_sha256']['sycl/src/prefill/prefill.cpp']=='f3c9bb38b46cdc111e9e663cc606d655f60f07c05873aff5499d51895e886c77'
assert digest(candidate_header)==build['source_sha256']['sycl/include/dpct/device.hpp']=='adbcbd0ac45995348e37ac3f5ca42d9f3d4b0555031aeb1781a1e2166a74ed52'
host_cpu_path=base/'pool-tasks-v0141-source-review-v1.json'
assert digest(host_cpu_path)=='a2c1fc95912ff0f58ce940c981d8a6bab6dbc8b072b45dcc2b670e5721b13794'
host_cpu=json.loads(host_cpu_path.read_text());assert host_cpu['passed'] and not host_cpu['active'] and not host_cpu['adopted']
flags_path=base/'decode-pool-phase-timing-v0141-uniform-build-flags-v1.json'
assert digest(flags_path)=='a85a6bd9e93c3d35a5596831ad888e57a657d79713bd28da6dc708d91ed6bc24'
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
assert shutil.disk_usage(base).free>96*1024**3
out=base/'owned-prefill-phase-validity-32769-v2'
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


cpu_preflight_v1_sha256='frozen before metadata-only source identification correction'
baseline_32769_path=base/'owned-decode-pool-phase-32769-v1-baseline/record.json'
baseline_32769=None
if mode=='pool0':
    baseline_32769=json.loads(baseline_32769_path.read_text())
    terminal_owned(baseline_32769)
    assert len(baseline_32769['requests'])==3
    assert baseline_32769['requests'][0]['measurement']['prompt_tokens']==32769
    baseline_checkpoint=baseline_32769['checkpoint']
    assert digest(baseline_checkpoint['file'])==baseline_checkpoint['sha256']
if mode=='baseline':
    measured_build_path=base/'integrated-upstream-v0.1.41-20261009-v2/build-record.json'
    assert digest(measured_build_path)=='20589f1486dedaf825052e95560155acd30e6017b7967e2df476b5dd41989c43'
    measured_build=json.loads(measured_build_path.read_text())
    assert measured_build['passed'] and not measured_build['active']
    binary=Path(measured_build['binary'])
    assert digest(binary)==measured_build['binary_sha256']=='86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8'
else:
    measured_build_path=build_path

assert mode=='pool0'
prefill_validity_build_path=base/'prefill-phase-validity-cpu-build-v1/record.json'
prefill_validity_build=json.loads(prefill_validity_build_path.read_text())
assert prefill_validity_build['passed'] and prefill_validity_build['complete'] and not prefill_validity_build['active']
assert not prefill_validity_build['gpu_tested'] and not prefill_validity_build['adopted']
assert prefill_validity_build['commit']=='185fa78098bee4be01ac81ad4f791673d24246f3'
root=Path(prefill_validity_build['root'])
review_head=prefill_validity_build['commit']
assert git('rev-parse','HEAD')==review_head and not git('status','--porcelain')
prefill_validity_source=root/'sycl/src/prefill/prefill.cpp'
assert digest(prefill_validity_source)==prefill_validity_build['source_sha256']=='1a6598a158e0b11e79c7a9f93ba7e452e3c316f108e38a07ed88c88a04cc8b72'
binary=Path(prefill_validity_build['binary'])
assert digest(binary)==prefill_validity_build['binary_sha256']
measured_build_path=prefill_validity_build_path

if sys.argv[4:]==['--cpu-preflight']:
    print(json.dumps({'cpu_preflight_passed':True,'gpu_executed':False,'binary_sha256':digest(binary),'reference_binary_sha256':diagnostic['binary_sha256'],'reference_reads':len(reference_reads),'review_head':review_head}))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();sys.exit(0)

out.mkdir(mode=0o700)
probes = out / 'probes'
probes.mkdir()
source = observer / 'sycl/tools/recover-xe.sh'
m = types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(source), 'exec'), m.__dict__)
r = m.Runner(probes)
os.chdir(root)
record = {
    'scope':'Current default-auto CPU phase diagnostic. Fresh32769 input gives32768 actual PP positions then64 outputs. Two RESTORE+decode repeats use the identical32832-consumed-position checkpoint; no PP on restores. Baseline869 creates matched references; telemetryT uses --pool-tasks0. Validation warnings/progress, full first head/live state/output parity; no optimization, clean speed or physical256K claim.',
    'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':boot,
    'active':True,'completed':False,'healthy':False,'math_gate_passed':False,
    'requests':[],'snapshots':[],'steps':[],'mode':mode,'phase':phase,'repetition':repetition,
    'deadline_seconds':1800,'protocol_timeout_seconds':450,'log_limit_bytes':64*1024**2,
    'commit':measured_build['commit'] if mode=='baseline' else review_head,'compiled_engine_base_commit':measured_build['commit'] if mode=='baseline' else engine_commit,'root':measured_build['root'] if mode=='baseline' else str(root),'controller_review_root':str(root),
    'binary_sha256':digest(binary),'build_receipt_sha256':digest(measured_build_path),'telemetry_build_receipt_sha256':digest(build_path),
    'candidate_source_sha256':digest(candidate_source),'candidate_header_sha256':digest(candidate_header),'source_preparation_receipt_sha256':digest(host_cpu_path),'uniform_flags_receipt_sha256':digest(flags_path),'completed_host_quiet_sequence_sha256':digest(quiet_path),
    'reference_binary_sha256':diagnostic['binary_sha256'],
    'reference_four_fresh_receipt_sha256':digest(subset_path),
    'baseline_full_capacity_receipt_sha256':digest(new_full_path),
    'baseline_quiet_sequence_receipt_sha256':digest(sequence_path),
    'performance_eligible':False,'full_lifecycle_passed':False,'adopted':False,'C_original_math_gate_passed':False,'C_failure_receipt_sha256':digest(C_path),'T_source_review_sha256':digest(T_proof),'T_cli_receipt_sha256':digest(T_cli),'previous_D_fourfresh_math_receipt_sha256':digest(D_path),
    'source_status_before':'',
    'source_sha256':{str(p):digest(p) for p in [Path(__file__),source,observer/'sycl/tools/owned_gdb.py',base/'capture_owned_native_counter_v01402_v1.py',state_comparator,candidate_source,candidate_header,flags_path]},
}
record['scope']='Opt-in valid prefillphase measurement, actual32768PP and two matchedcheckpoint-restored decodes; original native arithmetic and schedules unchanged, no extra permark waits. Numerical parity required; phase admission parsed separately. Not clean model speed orfull256K/adoption.'
record['source_sha256'][str(prefill_validity_source)]=digest(prefill_validity_source)
record['prefill_validity_build_receipt_sha256']=digest(prefill_validity_build_path)
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
                pending[:] = tail
                return value.decode().strip()
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
    health_path = Path(sequence['steps'][-1]['receipt'])
    health = json.loads(health_path.read_text())
    assert health['healthy'] and not health['active'] and health['boot_id'] == boot
    old_cursor = next(x['argv'][x['argv'].index('--after-cursor') + 1]
                      for x in health['steps'] if x['label'] == 'health-kernel')
    gap = r.run('kernel-gap', ['/usr/bin/journalctl', '-k', '--after-cursor', old_cursor, '--no-pager', '-o', 'json'], seconds=5)
    rows = [json.loads(s) for s in gap.splitlines() if s.startswith('{')]
    faults = [x['MESSAGE'] for x in rows if ('0000:05:00.0' in x.get('MESSAGE', '') or re.search(r'\bxe\b', x.get('MESSAGE', ''))) and m.FAULT.search(x.get('MESSAGE', ''))]
    record['preflight_fault_messages'] = faults
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
    env.update(STRATA_DECODE_TIMING='1',STRATA_DUMP_FIRST_LOGITS=str(out/'first-head.bin'),STRATA_PREFILL_DUMP_STATE=str(out/'prefill-state.bin'))
    assert 'STRATA_PREFILL_SYNC' not in env and 'STRATA_PREFILL_TRANSFER_TIMING' not in env
    env['STRATA_PREFILL_TIMING']='1'
    if phase == 'diagnostic':
        env.update(ZEL_ENABLE_LOADER_LOGGING='1',ZEL_LOADER_LOG_CONSOLE='1',ZEL_LOADER_LOGGING_LEVEL='warn',ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',ZE_ENABLE_VALIDATION_LAYER='1',ZE_ENABLE_PARAMETER_VALIDATION='1',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',STRATA_TRACE='1')
        for key in ['UR_ENABLE_LAYERS','UR_LOG_TRACING']:env.pop(key,None)
    if phase == 'clean':
        assert not any(k.startswith(('UR_LOG_', 'ZE_ENABLE_', 'ZEL_')) or k in ['UR_ENABLE_LAYERS', 'STRATA_TRACE'] for k in env)
    assert not any(k in env for k in ['STRATA_PREFILL_SYNC', 'STRATA_TRANSFER_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING', 'STRATA_VERIFY_NO_HOST', 'STRATA_VERIFY_DEVICE_PLAN', 'STRATA_PREFILL_HOST_TIMING'])
    assert env['STRATA_PREFILL_TIMING']=='1'
    record['environment'] = {k: v for k, v in env.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')) or k in ['LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission']}
    args = list(json.loads((base / 'model-lease-copy-off/normal-mtp.json').read_text())['args'])
    for key, value in [('--pcie-frac', '0'), ('--max-context', '262144'), ('--prefill', '8192'), ('--expert-cache', '128'), ('--conversation-cache-mib', '0')]:
        args[args.index(key) + 1] = value
    args += ['--kv', 'int8', '--kv-resident', '32768', '--prompt-cache', '1', '--prompt-cache-every', '262139', '--prompt-cache-root', '0', '--turn-token', '-1']
    assert args == new_full['argv'][1:] == qualified['argv'][1:]
    if mode=='pool0': args += ['--pool-tasks','0']
    argv = [str(binary)] + args
    record['argv'] = argv
    record['configuration_note'] = 'Qualified baseline arguments; T0 explicit auto tasks equals baseline implicit auto. Fresh32769 then64 outputs, two identical checkpoint restores. Timers are existing host spans, not worker busy time. Detailed API tracing off; exact-queue opt-in GPU phase marker profiling enabled, validity reconciliation mandatory; validation warnings and Strata progress captured. Logged durations are separate diagnostic observations.'
    record['prior_affinity_sequence_receipt_sha256']=digest(affinity_sequence_path)
    record['pool_tasks_cli_receipt_sha256']=digest(base/'pool-tasks-cli-validation-v1/record.json')
    record['pool_tasks_cpu_parity_receipt_sha256']=digest(base/'pool-tasks-existing-cpu-parity-v2/record.json')
    a_source = base / 'coding-review-32k-tokens.txt'
    b_source = base / 'full-context-copy-off/coding-context-256k-tokens.txt'
    assert digest(a_source) == '137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449'
    assert digest(b_source) == 'cc29e4427bcacb21c9f7df2d1f1a7d41897fce5bd76a8ca43d8c3c34d914dd2a'
    fixtures = {
        'A': list(map(int,a_source.read_text().split()))[:32759] + [198] + list(map(int,a_source.read_text().split()))[32759:],
        'B': list(map(int, b_source.read_text().split()))[:32760] + [248046, 198, 248045, 74455, 198, 248068, 198, 248069, 271],
    }
    record['prompt_fixtures'] = {}
    for key, tokens in fixtures.items():
        assert len(tokens) == 32769
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
    assert 'STRATA_PREFILL_COPY_ENGINE' not in actual_relevant and 'STRATA_PREFILL_COPY_PHASE_RELEASE' not in actual_relevant
    assert not any(k.startswith(('UNITRACE_', 'XPTI_')) or k in ['LD_PRELOAD', 'ZET_ENABLE_METRICS'] for k in actual)
    record['actual_target_environment'] = actual_relevant
    save()

    record['sessions']=[]
    record['stderr_request_boundaries']=[]
    math_ok=True
    def session(command,path,expected_tokens=None,expected_bytes=None):
        item={'command':command,'path':str(path),'protocol':[]}
        record['sessions'].append(item);save()
        send((command+' '+str(path)+'\n').encode())
        while True:
            response=line();item['protocol'].append(response)
            if response.startswith(('SAVED ','RESTORED ','SERR ','ERR')):break
        fields=response.split()
        assert fields[0]==('SAVED' if command=='SAVE' else 'RESTORED'),response
        assert len(fields)>=4 and int(fields[1])>0 and 0<int(fields[2])<2*1024**3
        if expected_tokens is not None: assert int(fields[1])==expected_tokens
        if expected_bytes is not None: assert int(fields[2])==expected_bytes
        item.update(tokens=int(fields[1]),bytes=int(fields[2]),elapsed_ms=float(fields[3]),passed=True)
        save();return item
    for read_index in range(3):
        priming=read_index==0
        if not priming:
            checkpoint=record['checkpoint']
            assert digest(checkpoint['file'])==checkpoint['sha256']
            session('RESTORE',Path(checkpoint['file']),32832,checkpoint['bytes'])
        tokens=fixtures['A'] if priming else fixtures['A']+record['requests'][0]['ids']
        assert len(tokens)==(32769 if priming else 32833)
        current='fresh32769-firstdecode' if priming else f'restored-decode{read_index}'
        result={'name':current,'fixture':'A','read_index':read_index,'phase_kind':'fresh-first' if priming else 'restored','first_process_read':priming,'ids':[],'logprobs':[],'protocol':[]}
        record['requests'].append(result);save()
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        log_path=out/'debugger/inferior.stderr'
        stderr_begin=log_path.stat().st_size
        request_start=time.monotonic()
        send(('GEN 64 ckpt=1 logprobs=5 '+','.join(map(str,tokens))+'\n').encode())
        event('request',name=current)
        while True:
            value=line();result['protocol'].append(value)
            if value.startswith('ERR'): raise RuntimeError(value)
            if value.startswith('T '):result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '):result['logprobs'].append(value)
            if value.startswith('DONE '):break
        fields=value.split()
        result['finish_reason']=fields[5];result['mtp_counts']=list(map(int,fields[6:8]))
        result['resume_tokens']=[int(x.split()[1]) for x in result['protocol'] if x.startswith(('RESUME ','REUSED '))]
        if priming:result['validation']=validate_fresh(result)
        else:result['validation']={
            'restored32832':result['resume_tokens']==[32832,32832],
            'request32833':int(fields[2])==32833,
            'no_PP':not any(x.startswith('PP ') for x in result['protocol']),
            'exact64_outputs':int(fields[1])==len(result['ids'])==len(result['logprobs'])==64,
            'normal_finish':fields[5]=='length',
            'finite_logprobs':all(math.isfinite(float(x.rsplit(':',1)[-1])) for v in result['logprobs'] for x in v.split()[1:])}
        result['validation']['exact64_outputs']=int(fields[1])==len(result['ids'])==len(result['logprobs'])==64
        result['measurement']={'generated_tokens':int(fields[1]),'prompt_tokens':int(fields[2]),'actual_batched_positions':32768 if priming else 0,'prompt_ms':float(fields[3]),'decode_ms':float(fields[4]),'wall_seconds':time.monotonic()-request_start,'purpose':'Diagnostic phase observation; excluded from clean throughput'}
        assert math.isfinite(result['measurement']['decode_ms']) and result['measurement']['decode_ms']>0
        result.update(captures(current,priming))
        checks=list(result['validation'].values())
        if mode=='pool0':
            expected=baseline_32769['requests'][read_index]
            result['qualified_reference_output_comparison']=same_output(result,expected)
            checks.extend(result['qualified_reference_output_comparison'].values())
            if 'first_head' in expected:
                assert 'first_head' in result
                result['first_head_equal']=result['first_head']['sha256']==expected['first_head']['sha256']
                checks.append(result['first_head_equal'])
            if 'prefill_state' in expected:
                assert 'prefill_state' in result
                cmp=compare_states(result['prefill_state'],expected['prefill_state'],32768 if priming else 32832)
                result['live_prefill_comparison']=cmp
                checks.append(not cmp['different_live_parts'])
        elif read_index==2:
            result['repeat_comparison']=same_output(result,record['requests'][1])
            checks.extend(result['repeat_comparison'].values())
        result['math_gate_passed']=all(checks);math_ok &= result['math_gate_passed']
        assert math_ok,'numerical/invariant mismatch: stop; no retry'
        # Wait for stderr report write, which precedes DONE in this source.
        stderr_end=log_path.stat().st_size
        with log_path.open('rb') as stream:stream.seek(stderr_begin);segment=stream.read(stderr_end-stderr_begin).decode(errors='strict')
        counter=[x for x in segment.splitlines() if x.startswith('strata decode timing: ')]
        phases=[x for x in segment.splitlines() if x.startswith('strata decode CPU pool phases: ')]
        assert len(counter)==1 and len(phases)==(1 if mode=='pool0' else 0)
        window_match=re.search(r'timing: (\d+) windows',counter[0]);assert window_match
        windows=int(window_match.group(1));assert windows>0
        row={'request':current,'ordinal':read_index,'stderr_begin':stderr_begin,'stderr_end':stderr_end,'decode_timing_line':counter[0],'windows':windows}
        if phases:
            match=re.search(r'phases: (\d+) windows; GU ([0-9.]+), FF-quant ([0-9.]+), Down ([0-9.]+) ms/window .*bytes ([0-9.]+)/window',phases[0]);assert match
            assert int(match.group(1))==windows
            values=list(map(float,match.groups()[1:]))
            assert all(math.isfinite(v) and v>=0 for v in values) and values[-1]>0
            row.update(pool_phase_line=phases[0],GU_ms_per_window=values[0],FF_quant_ms_per_window=values[1],Down_ms_per_window=values[2],logical_blob_bytes_per_window=values[3])
        record['stderr_request_boundaries'].append(row);save()
        if priming:
            if mode=='baseline':
                checkpoint_path=out/'matched32832.session.bin'
                info=session('SAVE',checkpoint_path,32832)
                assert checkpoint_path.stat().st_size==info['bytes']
                record['checkpoint']={'file':str(checkpoint_path),'tokens':info['tokens'],'bytes':info['bytes'],'sha256':digest(checkpoint_path)}
            else:record['checkpoint']=dict(baseline_32769['checkpoint'])
            save()


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
    with engine_log.open(errors='replace') as stream, (out / 'project-messages.txt').open('w') as project:
        for value in stream:
            if value.startswith('strata '):
                project.write(value)
    record['pool_tasks_configuration_gate']={'passed':True,'tasks':'automatic','requested_cli':None if mode=='baseline' else 0,'note':'Existing auto partition, five workers plus caller; no task candidate or kernel optimization.'}
    record['decode_CPU_phase_counter_gate']={'passed':len(record['stderr_request_boundaries'])==3,'requests':record['stderr_request_boundaries'],'source_proof_sha256':digest(T_proof),'performance_eligible':False}
    record['engine_log_bytes'] = engine_log.stat().st_size
    record['engine_log_sha256'] = digest(engine_log)
    record['source_status_after'] = git('status', '--porcelain')
    assert not record['source_status_after'] and git('rev-parse', 'HEAD') == commits[mode]
    record['completed'] = True
    record['math_gate_passed'] = math_ok and len(record['requests']) == 3 and len(record['stderr_request_boundaries'])==3
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
    record['healthy'] = record['completed'] and not record.get('new_fault_messages') and not record.get('error') and not any(record.get('cleanup', {}).values())
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
print(json.dumps({k: record.get(k) for k in ['mode', 'phase', 'healthy', 'math_gate_passed', 'elapsed_seconds', 'error', 'exit_code', 'exit_signal', 'new_fault_messages', 'cleanup']} | {'measurements': [x.get('measurement') for x in record['requests']]}, indent=2))
if not record['healthy'] or not record['math_gate_passed']:
    raise SystemExit(1)
