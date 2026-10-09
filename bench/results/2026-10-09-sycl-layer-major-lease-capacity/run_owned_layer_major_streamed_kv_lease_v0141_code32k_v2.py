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
assert shutil.disk_usage(base).free>96*1024**3
out=base/'owned-layer-major-streamed-kv-v0141-code32k-streamed-lease-diagnostic-r1'
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

if sys.argv[4:]==['--cpu-preflight']:
    print(json.dumps({'cpu_preflight_passed':True,'gpu_executed':False,'binary_sha256':digest(binary),'reference_binary_sha256':diagnostic['binary_sha256'],'reference_reads':len(reference_reads),'review_head':review_head}))
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();sys.exit(0)

out.mkdir(mode=0o700)
probes = out / 'probes'
probes.mkdir()
source = observer / 'sycl/tools/recover-xe.sh'
m = types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(source), 'exec'), m.__dict__)

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

os.chdir(root)
record = {
    'scope':'Layer-major streamed INT8 plus existing full main-cache/MTP RAM leases. Four fresh32768 A/B/A/B requests,64 outputs; exact heads/all66 live state/LP/MTP against qualified integrated869. Physical262144 capacity and8K chunks retained. Full payload verification and graph lifecycle traced with validation/warnings/progress. Logged times excluded; no speed/full-lifecycle/adoption claim.',
    'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':boot,
    'active':True,'completed':False,'healthy':False,'math_gate_passed':False,
    'requests':[],'snapshots':[],'steps':[],'mode':mode,'phase':phase,'repetition':repetition,
    'deadline_seconds':1800,'protocol_timeout_seconds':450,'log_limit_bytes':64*1024**2,
    'commit':review_head,'compiled_engine_base_commit':engine_commit,'root':str(root),
    'binary_sha256':digest(binary),'build_receipt_sha256':digest(build_path),
    'candidate_source_sha256':digest(candidate_source),'candidate_header_sha256':digest(candidate_header),'source_preparation_receipt_sha256':digest(host_cpu_path),'uniform_flags_receipt_sha256':digest(flags_path),'completed_host_quiet_sequence_sha256':digest(quiet_path),
    'reference_binary_sha256':diagnostic['binary_sha256'],
    'reference_four_fresh_receipt_sha256':digest(subset_path),
    'baseline_full_capacity_receipt_sha256':digest(new_full_path),
    'baseline_quiet_sequence_receipt_sha256':digest(sequence_path),
    'performance_eligible':False,'full_lifecycle_passed':False,'adopted':False,
    'source_status_before':'','prior_capacity_failure_safety_review_sha256':digest(safety_path),'diagnostic_log_profile':'ZE parameter validation + warning/error logs + Strata progress; no all-success API history',
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
        raise RuntimeError('model log size limit')
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
    old_cursor = safety['health_kernel_cursor']
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
    env.update(STRATA_PREFILL_COMPACT='2', STRATA_PREFILL_ATTN_LAYOUT='1', STRATA_PREFILL_LAYER_MAJOR='1', STRATA_PREFILL_LAYER_MAJOR_STREAMED_KV='1', STRATA_KV_STAGE_OWN='1', STRATA_KV_PREFETCH='0')
    env.pop('STRATA_PREFILL_COPY_ENGINE', None)
    env.pop('STRATA_PREFILL_COPY_PHASE_RELEASE', None)
    env.pop('STRATA_STAGER_CPU_LIST',None)
    for key in ['STRATA_PREFILL_RELEASE_CACHE','STRATA_PREFILL_CACHE_RELEASE_FRAC','STRATA_PREFILL_CACHE_RESTORE','STRATA_PREFILL_LAYER_MAJOR_R_GPU','STRATA_PREFILL_LAYER_MAJOR_R_INPLACE']:
        env.pop(key,None)
    env.update(STRATA_PREFILL_RELEASE_CACHE='1',STRATA_PREFILL_CACHE_RELEASE_FRAC='1',STRATA_PREFILL_CACHE_RESTORE='ram',STRATA_PREFILL_CACHE_VERIFY='1',STRATA_PREFILL_RELEASE_DRAFT='1',STRATA_PREFILL_DRAFT_VERIFY='1')
    env.update(STRATA_DECODE_TIMING='1',STRATA_DUMP_FIRST_LOGITS=str(out/'first-head.bin'),STRATA_PREFILL_DUMP_STATE=str(out/'prefill-state.bin'))
    if phase == 'diagnostic':
        env = m.diagnostic_environment(env)
        env.update(ZEL_LOADER_LOGGING_LEVEL='warn',ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        env.pop('UR_LOG_TRACING',None)
        env['UR_ENABLE_LAYERS']=','.join(x for x in env.get('UR_ENABLE_LAYERS','').split(',') if x and x!='UR_LAYER_TRACING')
        if not env['UR_ENABLE_LAYERS']: env.pop('UR_ENABLE_LAYERS')
    if phase == 'clean':
        assert not any(k.startswith(('UR_LOG_', 'ZE_ENABLE_', 'ZEL_')) or k in ['UR_ENABLE_LAYERS', 'STRATA_TRACE'] for k in env)
    assert not any(k in env for k in ['STRATA_PREFILL_SYNC', 'STRATA_PREFILL_TIMING', 'STRATA_TRANSFER_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING', 'STRATA_VERIFY_NO_HOST', 'STRATA_VERIFY_DEVICE_PLAN', 'STRATA_PREFILL_HOST_TIMING'])
    record['environment'] = {k: v for k, v in env.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')) or k in ['LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission']}
    args = list(json.loads((base / 'model-lease-copy-off/normal-mtp.json').read_text())['args'])
    for key, value in [('--pcie-frac', '0'), ('--max-context', '262144'), ('--prefill', '8192'), ('--expert-cache', '128'), ('--conversation-cache-mib', '0')]:
        args[args.index(key) + 1] = value
    args += ['--kv', 'int8', '--kv-resident', '32768', '--prompt-cache', '1', '--prompt-cache-every', '262139', '--prompt-cache-root', '0', '--turn-token', '-1']
    assert args == new_full['argv'][1:] == qualified['argv'][1:]
    argv = [str(binary)] + args
    record['argv'] = argv
    record['configuration_note'] = 'Qualified baseline argv identical. Layer-major1/streamedINT8 and existing main full-cache RAM lease/MTP full-expert+head RAM lease, both payload verifications enabled. Host residual plane retained. No extra CLI task argument/import/native copy factory/affinity. Fresh A/B/A/B no reused tokens; logged times excluded.'
    record['prior_affinity_sequence_receipt_sha256']=digest(affinity_sequence_path)
    record['prior_pool_tasks_diagnostic_receipt_sha256']=digest(last_pool_path)
    record['prefix_layout_cpu_receipt_sha256']=digest(prefix_path)
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
    assert 'STRATA_PREFILL_COPY_ENGINE' not in actual_relevant and 'STRATA_PREFILL_COPY_PHASE_RELEASE' not in actual_relevant
    assert not any(k.startswith(('UNITRACE_', 'XPTI_')) or k in ['LD_PRELOAD', 'ZET_ENABLE_METRICS'] for k in actual)
    record['actual_target_environment'] = actual_relevant
    save()
    seen = {}
    math_ok = True
    for read_index, key in enumerate(['A', 'B', 'A', 'B']):
        current = f'{key}-read{read_index}'
        result = {'name': current, 'fixture': key, 'read_index': read_index, 'first_process_read': read_index == 0, 'ids': [], 'logprobs': [], 'protocol': []}
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
    with engine_log.open(errors='replace') as stream, (out / 'project-messages.txt').open('w') as project:
        for value in stream:
            if value.startswith('strata '):
                project.write(value)
    text=(out/'project-messages.txt').read_text()
    seeds=re.findall(r'^strata trace: streamed layer-major QSA (\d+) seed scheduled_cells=(\d+) bytes=(\d+) \(enqueue order\)$',text,re.M)
    drains=re.findall(r'^strata trace: streamed layer-major QSA (\d+) drained_cells=(\d+) chunks=(\d+) seed_bytes=(\d+) retained_chunks=(\d+)$',text,re.M)
    count=len(record['requests'])
    assert len(seeds)==len(drains)==12*count,'all12 QSA owners must enter/drain each fresh request'
    expected_ordinals=list(range(12))*count
    assert [int(x[0]) for x in seeds]==[int(x[0]) for x in drains]==expected_ordinals
    assert all(list(map(int,x[1:]))==[0,0] for x in seeds)
    assert all(list(map(int,x[1:]))==[32767,4,0,3] for x in drains)
    infos=[x for x in record['startup'] if x.startswith('INFO ')]
    assert len(infos)==1
    info=dict(x.split('=',1) for x in infos[0].split()[1:])
    assert all(info[k]==v for k,v in dict(context='262144',kv='int8',kv_resident='32768',expert_slots='128',expert_cache_mib='325',spec='4',arena_mib='47962',pool_workers='5',engine='0.1.41').items())
    record['streamed_layer_major_configuration_gate']={'passed':True,'source_review_sha256':digest(host_cpu_path),'prefix_cpu_sha256':digest(prefix_path),'actual_info':info,'seeds':seeds,'drains':drains,'note':'Each fresh request used12INT8 identity owners, four8K/tail chunks, zero prefix seed bytes, three retained chunks; actual settings and exact numerical gates required.'}
    
    cache_release=re.findall(r'^strata prefill cache release: (\d+) physical bytes, (\d+) restored bytes, source (\w+), suspend ([0-9.]+) ms, restore ([0-9.]+) ms, graph ([0-9.]+) ms, same_address ([01]), slots restored$',text,re.M)
    cache_verify=re.findall(r'^strata prefill cache verify: (\d+) occupied tail bytes matched before and after restoration; (\d+) experts differ from the initial admission map$',text,re.M)
    mtp_release=re.findall(r'^strata mtp decode release: (\d+) physical bytes, experts and head; K/V retained; total ([0-9.]+) ms, wait ([0-9.]+) ms, unmap ([0-9.]+) ms, verify ([0-9.]+) ms, verified=(\d), graph_drop ([0-9.]+) ms$',text,re.M)
    mtp_restore=re.findall(r'^strata mtp decode restore: (\d+) payload bytes from RAM, same addresses; total ([0-9.]+) ms, map ([0-9.]+) ms, copy ([0-9.]+) ms, verify ([0-9.]+) ms, verified=(\d)$',text,re.M)
    assert len(cache_release)==len(cache_verify)==len(mtp_release)==len(mtp_restore)==count==4,'each request requires both complete verified release/restore pairs'
    assert all(int(x[0])>0 and int(x[1])>0 and x[2]=='ram' and all(math.isfinite(float(v)) and float(v)>=0 for v in x[3:6]) for x in cache_release)
    assert all(int(x[0])>0 for x in cache_verify)
    assert all(int(x[0])>0 and x[5]=='1' and all(math.isfinite(float(v)) and float(v)>=0 for v in x[1:5]+x[6:]) for x in mtp_release)
    assert all(int(x[0])>0 and x[5]=='1' and all(math.isfinite(float(v)) and float(v)>=0 for v in x[1:5]) for x in mtp_restore)
    memory_points=['before decode cache release','main decode cache released','MTP decode weights released','temporary layer cache allocated','prefill completed with temporary layer cache','temporary buffers released','main decode cache restored, verifier graphs rebuilt','MTP decode weights restored']
    memory=re.findall(r'^strata prefill memory: ([^;]+); free=(\d+) total=(\d+) bytes$',text,re.M)
    assert len(memory)==len(memory_points)*count and [x[0] for x in memory]==memory_points*count,'each lease boundary requires a valid memory query in source order'
    assert all(0<=int(x[1])<=int(x[2]) and int(x[2])>0 for x in memory)
    record['verified_ram_lease_gate']={'passed':True,'cache_release':cache_release,'cache_verify':cache_verify,'mtp_release':mtp_release,'mtp_restore':mtp_restore,'memory_points':memory_points,'memory':memory,'note':'Payloads verified before/after release; current-map RAM sources and verifier graph rebuild required; MTP graphs retired before unmap and recaptured by normal decode. Logged durations are not performance samples.'}

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
    record['healthy'] = record['completed'] and not record.get('new_fault_messages') and not record.get('error') and not any(record.get('cleanup', {}).values())
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
print(json.dumps({k: record.get(k) for k in ['mode', 'phase', 'healthy', 'math_gate_passed', 'elapsed_seconds', 'error', 'exit_code', 'exit_signal', 'new_fault_messages', 'cleanup']} | {'measurements': [x.get('measurement') for x in record['requests']]}, indent=2))
if not record['healthy'] or not record['math_gate_passed']:
    raise SystemExit(1)
