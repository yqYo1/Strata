"""Owned 32K update comparison: raw upstream, integrated update, qualified old control.

First-use diagnostics are separate from quiet timings. Every read is fresh.
This controller never changes sources, resets the GPU, or adopts a binary.
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
assert mode in ['baseline','nativeoff','nativeon'] and phase in ['clean','diagnostic'] and 1 <= repetition <= 6
assert sys.argv[4:] in [[], ['--cpu-preflight']]
candidate_head='6caa1421f9212750a425fe1729139ffdde6e9f9a'
candidate_root=observer.parent/'perf-sycl-prefill-native-copy-v0141-20261009'
review_head='0ed619e5239bebc87f018cd2f9b8443e37105c72' if mode=='baseline' else candidate_head
engine_commit='1eb89482a4afd20277ae0405780ed4f8eb98eb20' if mode=='baseline' else candidate_head
root=observer.parent/'sync-upstream-v0.1.41-20261009' if mode=='baseline' else candidate_root
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
build_path=base/'native-expert-copy-v0141-private-build-v1/record.json'
assert digest(build_path)=='74b48ad90996c942fd74fa826d22dfecf76145d8dec82a40cfa0d35dc9b65fad'
build=json.loads(build_path.read_text());assert not build['active'] and build['passed'] and build['compiled_engine'] and not build['gpu_tested'] and not build['adopted']
assert build['source_head']==candidate_head
binary=Path(build['binary']);assert digest(binary)==build['binary_sha256']=='2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f'
candidate_source=candidate_root/'sycl/src/prefill/prefill.cpp'
candidate_header=candidate_root/'sycl/include/dpct/device.hpp'
assert digest(candidate_source)==build['source_sha256']['sycl/src/prefill/prefill.cpp']=='6399d65feea884f932a5a1079a91c80228293bfc3be371732b48413977229f84'
assert digest(candidate_header)==build['source_sha256']['sycl/include/dpct/device.hpp']=='8805874ecb1d58d7acea8ca931154a38d0ad3d623f7c5ee8d599fa2ba56a841b'
host_cpu_path=base/'native-expert-copy-v0141-source-preparation-v1.json'
assert digest(host_cpu_path)=='9ee58f5ebfa8f2de86c36b40b7edd0a519a123c18ec1426eb16e8930d5dcf027'
host_cpu=json.loads(host_cpu_path.read_text());assert host_cpu['prepared'] and not host_cpu['active'] and not host_cpu['adopted']
flags_path=base/'native-expert-copy-v0141-uniform-build-flags-v1.json'
assert digest(flags_path)=='54a1a491a7f00874a90af1b7776d0654b8b001e9aea3da99f6e0b1dfe0a83e53'
flags=json.loads(flags_path.read_text());assert flags['passed'] and not flags['active'] and flags['build_receipt_sha256']==digest(build_path)
assert flags['old_compile_count']==flags['candidate_compile_count']==114 and not flags['missing_sources'] and not flags['different_flags']
quiet_path=base/'host-prefill-accounting-v0141-quiet32k-comparison-sequence-v1/record.json'
assert digest(quiet_path)=='d57fba6b0826ad8402505571b8cdc92eee4e0fa5bbd1c1a9b779531bd2f4bc83'
quiet=json.loads(quiet_path.read_text());assert not quiet['active'] and quiet['passed'] and len(quiet['steps'])==6
for step in quiet['steps']:
    p=Path(step['receipt']);assert digest(p)==step['receipt_sha256']
    terminal_owned(json.loads(p.read_text()))
    identity=step['controller_identity'];actual=process_identity(identity['pid'])
    assert not actual or actual['start_ticks']!=identity['start_ticks']
assert digest('/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0.12.0')=='bfdc0f26bf88bd0bccd66fc25302a4b22559f8fe30e499be18529be3ed610e60'
first_path=base/'owned-native-expert-copy-v0141-code32k-diagnostic-r1/record.json'
assert digest(first_path)=='83670ee10b75d702c621e0ce029c5be1a4701ac6684d08b4f22de905704ae908'
first=json.loads(first_path.read_text());terminal_owned(first)
assert first['native_copy_queue_startup_gate_passed'] and len(first['requests'])==4
assert first['native_copy_queue_ordinals'] and all(x==1 for x in first['native_copy_queue_ordinals'])
assert first['binary_sha256']==build['binary_sha256']
if mode=='baseline':
    baseline_path=base/'integrated-upstream-v0.1.41-20261009-v2/build-record.json'
    baseline=json.loads(baseline_path.read_text());assert baseline['passed'] and not baseline['active']
    binary=Path(baseline['binary']);assert digest(binary)==baseline['binary_sha256']==diagnostic['binary_sha256']
    build_path=baseline_path
for previous_path in base.glob('owned-native-expert-copy-v0141-code32k-*-clean-r*/record.json'):
    terminal_owned(json.loads(previous_path.read_text()))
state_comparator=base/'compare_live_prefill_state_v01402_v2.py'
assert digest(state_comparator)=='9a4f55a4760d316045c1ec5179a946de9a9024f7e42fe09032012ceae2475a72'
from compare_live_prefill_state_v01402_v2 import compare_states
full_path=base/'owned-native-expert-copy-v0141-full256k-diagnostic-r1/record.json'
assert digest(full_path)=='7f206dc65c01d05192d54211e6ba1f52eb34dd029f25e9a7c2235b1a2567d533'
native_full=json.loads(full_path.read_text());terminal_owned(native_full)
assert native_full['full_lifecycle_passed'] and native_full['physical256k_sequence_completed']
assert native_full['binary_sha256']=='2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f'
sync_path=base/'owned-native-stream-order-v0141-code32k-ring8-diagnostic-r1/record.json'
assert digest(sync_path)=='ddaf94e2fdce81adc83d0955290d9cd4a3fd774f5bda9be93f7a2ab79badcc9c'
terminal_owned(json.loads(sync_path.read_text()))
saved=next(x for x in native_full['sessions'] if x['name']=='save-control32k')
checkpoint=Path(saved['file'])
assert checkpoint.stat().st_size==saved['bytes']==619343700
assert digest(checkpoint)==saved['image']['sha256']=='96a96b4946c495977198c0b915be243960f74fa2f5f2c3dfd5079eb25f36bc5e'
assert saved['tokens']==32831
resume_reference=next(x for x in native_full['requests'] if x['name']=='resume32k-restored')
resume_reference=dict(resume_reference)
resume_reference['finish_reason']=resume_reference['protocol'][-1].split()[5]
assert resume_reference['mtp_counts']==[46,51] and len(resume_reference['ids'])==len(resume_reference['logprobs'])==64
assert resume_reference['resume_tokens']==[32831,32831]
assert same_output(resume_reference,next(dict(x,finish_reason=x['protocol'][-1].split()[5]) for x in new_full['requests'] if x['name']=='resume32k-restored'))==dict.fromkeys(['ids_equal','logprobs_equal','mtp_counts_equal','finish_reason_equal'],True)
for prior in base.glob('owned-native-copy-decode-repeat-v0141-*/record.json'):
    terminal_owned(json.loads(prior.read_text()))
assert shutil.disk_usage(base).free>64*1024**3
out=base/f'owned-native-copy-decode-repeat-v0141-{mode}-{phase}-r{repetition}'
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
    'scope':'Quiet native expert-copy comparison: qualified baseline869 versus uniformly rebuilt candidate272 with native queue disabled0/enabled1. Four fresh32768 A/B/A/B and64 outputs with qualified numerical references. No API logs, profiler, dumps, timestamp queries or extra phase waits. First/later separated. No adoption/full-capacity claim.',
    'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':boot,
    'active':True,'completed':False,'healthy':False,'math_gate_passed':False,
    'requests':[],'snapshots':[],'steps':[],'mode':mode,'phase':phase,'repetition':repetition,
    'deadline_seconds':1500,'protocol_timeout_seconds':450,'log_limit_bytes':(16*1024**3 if phase=='diagnostic' else 128*1024**2),
    'commit':review_head,'compiled_engine_base_commit':engine_commit,'root':str(root),
    'binary_sha256':digest(binary),'build_receipt_sha256':digest(build_path),
    'candidate_source_sha256':digest(candidate_source) if mode!='baseline' else None,
    'measured_prefill_source_sha256':digest(candidate_source) if mode!='baseline' else digest(root/'sycl/src/prefill/prefill.cpp'),
    'candidate_header_sha256':digest(candidate_header) if mode!='baseline' else None,'source_preparation_receipt_sha256':digest(host_cpu_path),'uniform_flags_receipt_sha256':digest(flags_path),
    'reference_binary_sha256':diagnostic['binary_sha256'],
    'reference_four_fresh_receipt_sha256':digest(subset_path),
    'baseline_full_capacity_receipt_sha256':digest(new_full_path),
    'baseline_quiet_sequence_receipt_sha256':digest(sequence_path),
    'performance_eligible':phase=='clean','full_lifecycle_passed':False,'adopted':False,
    'first_native_copy_diagnostic_receipt_sha256':digest(first_path),
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
    if sum(p.stat().st_size for p in (out / 'debugger').glob('*.stderr')) >= record['log_limit_bytes']:
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
    env.pop('STRATA_PREFILL_HOST_TIMING',None)
    env.pop('STRATA_PREFILL_COPY_ENGINE',None)
    env['STRATA_DECODE_TIMING']='1'
    assert 'STRATA_VERIFY_PROFILE' not in env
    if mode!='baseline':
        env['STRATA_PREFILL_COPY_ENGINE']={'nativeoff':'0','nativeon':'1'}[mode]
    if phase == 'diagnostic':
        env['STRATA_TRACE']='1'
        env = m.diagnostic_environment(env)
    if phase == 'clean':
        assert not any(k.startswith(('UR_LOG_', 'ZE_ENABLE_', 'ZEL_')) or k in ['UR_ENABLE_LAYERS', 'STRATA_TRACE'] for k in env)
    assert not any(k in env for k in ['STRATA_PREFILL_SYNC', 'STRATA_PREFILL_TIMING', 'STRATA_TRANSFER_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING', 'STRATA_VERIFY_NO_HOST', 'STRATA_VERIFY_DEVICE_PLAN'])
    record['environment'] = {k: v for k, v in env.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')) or k in ['LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission']}
    args = list(json.loads((base / 'model-lease-copy-off/normal-mtp.json').read_text())['args'])
    for key, value in [('--pcie-frac', '0'), ('--max-context', '262144'), ('--prefill', '8192'), ('--expert-cache', '128'), ('--conversation-cache-mib', '0')]:
        args[args.index(key) + 1] = value
    args += ['--kv', 'int8', '--kv-resident', '32768', '--prompt-cache', '1', '--prompt-cache-every', '262139', '--prompt-cache-root', '0', '--turn-token', '-1']
    assert args == new_full['argv'][1:] == qualified['argv'][1:]
    argv = [str(binary)] + args
    record['argv'] = argv
    record['configuration_note'] = 'Exactly the same requested argv/environment for all three modes; actual workspace/cache/VRAM admission can differ by implementation and remains in project messages. Different compiler code/source identities are explicit. Per-process read0 is reported separately from reads1..3. Fresh A/B/A/B means no reused prompt tokens. API logs are never timing evidence.'
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
    assert not any(k.startswith(('UNITRACE_', 'XPTI_')) or k in ['LD_PRELOAD', 'ZET_ENABLE_METRICS'] for k in actual)
    record['actual_target_environment'] = actual_relevant
    save()
    assert record['startup'][-1].split()[1]=='262144'
    record['scope']='Phase-specific decode comparison: qualified baseline869, candidate272 flag0, candidate272 flag1. Prime each process with the same fresh32768 A/64-output prefill, retaining its workspace and queue. Before every decode trial RESTORE the same verified32831-token checkpoint, then generate64 from32832 input tokens. One warm trial excluded; twelve measured repeats per quiet process. Six order-balanced independent process blocks. Exact IDs/logprobs/MTP counts required; restoration and prefill timings excluded from decode statistics.'
    record['checkpoint_identity']={'file':str(checkpoint),'sha256':saved['image']['sha256'],'bytes':saved['bytes'],'tokens':saved['tokens'],'source_receipt_sha256':digest(full_path)}
    record['sessions']=[]
    record['configuration_note']='Context262144, resident32768, chunk8192, expert-cache128, LP5, greedy MTP4. All modes share argv/environment except executable and explicit prefill-only queue flag. Priming creates the actual prefill queue/workspace. Each decode begins from the same saved numerical state; no PP chunks or prefix growth. Warm trial reported separately. Quiet host counters add no GPU timestamps/waits. Interleaved independent processes are the units for uncertainty; repeated requests within a process are not independent process samples.'
    continuation=fixtures['A']+reference_reads[0]['ids']
    assert len(continuation)==32832
    math_ok=True

    def restore(trial):
        global current
        current=f'restore-{trial}'
        item={'name':current,'command':'RESTORE','file':str(checkpoint),'protocol':[]}
        record['sessions'].append(item);save()
        send(f'RESTORE {checkpoint}\n'.encode())
        while True:
            value=line();item['protocol'].append(value)
            if value.startswith(('RESTORED ','SERR ')):break
        fields=value.split()
        assert fields[0]=='RESTORED' and int(fields[1])==32831 and int(fields[2])==619343700,value
        item.update(tokens=int(fields[1]),bytes=int(fields[2]),restore_ms=float(fields[3]),passed=True)
        save()

    trials=2 if phase=='diagnostic' else 13
    for read_index in range(1+trials):
        key='A'
        priming=read_index==0
        if not priming:restore(read_index-1)
        tokens=fixtures['A'] if priming else continuation
        current = 'prefill-prime' if priming else f'decode-trial{read_index-1}'
        result = {'name':current,'fixture':key,'read_index':read_index,'phase_kind':'prefill-prime' if priming else 'decode-only','warmup_decode':read_index==1,'timing_eligible':phase=='clean' and read_index>=2,'first_process_read':priming,'ids':[],'logprobs':[],'protocol':[]}
        record['requests'].append(result)
        save()
        assert not (out/'first-head.bin').exists() and not (out/'prefill-state.bin').exists()
        request_start = time.monotonic()
        send(('GEN 64 ckpt=1 logprobs=5 ' + ','.join(map(str, tokens)) + '\n').encode())
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
        if priming:
            result['validation']=validate_fresh(result)
        else:
            result['validation']={
                'fixed_restored_input':int(fields[2])==32832,
                'same_resume_and_reused':result['resume_tokens']==[32831,32831],
                'no_prefill_chunks':not any(x.startswith('PP ') for x in result['protocol']),
                'exact64_visible_outputs':int(fields[1])==len(result['ids'])==len(result['logprobs'])==64,
                'normal_finish':fields[5]=='length',
                'finite_logprobs':all(math.isfinite(float(x.rsplit(':',1)[-1])) for line_value in result['logprobs'] for x in line_value.split()[1:]),
                'finite_positive_times':math.isfinite(float(fields[4])) and float(fields[4])>0,
            }
        result['measurement'] = {'generated_tokens': int(fields[1]), 'prompt_tokens': int(fields[2]), 'prompt_ms': float(fields[3]), 'decode_ms': float(fields[4]), 'wall_seconds': time.monotonic() - request_start, 'purpose': 'quiet timing' if phase == 'clean' else 'logged correctness only, excluded from speed'}
        mm = result['measurement']
        if mm['prompt_ms'] > 0 and mm['decode_ms'] > 0:
            mm['prefill_tok_s'] = 1000 * mm['prompt_tokens'] / mm['prompt_ms']
            mm['decode_tok_s'] = 1000 * mm['generated_tokens'] / mm['decode_ms']
        checks = list(result['validation'].values())
        if priming:
            assert [int(x.split()[1]) for x in result['protocol'] if x.startswith('PP ')]==[8192,16384,24576,32767]
        expected=reference_reads[0] if priming else resume_reference
        result['qualified_reference_output_comparison']=same_output(result,expected)
        checks.extend(result['qualified_reference_output_comparison'].values())
        result['math_gate_passed'] = all(checks)
        math_ok &= result['math_gate_passed']
        save()
        if not math_ok:break

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
    queue_lines=[value for value in (out/'project-messages.txt').read_text().splitlines() if value.startswith('strata prefill copy queue: ')]
    if mode=='nativeon':
        assert queue_lines, 'native copy-only queue creation required'
        prefix='strata prefill copy queue: native copy-only, ordinal '
        suffix=', index0, in-order, profiling0'
        ordinals=[]
        for value in queue_lines:
            assert value.startswith(prefix) and value.endswith(suffix), 'native copy-only queue metadata required'
            ordinal=value[len(prefix):-len(suffix)]
            assert ordinal.isdecimal() and int(ordinal)==1, 'recorded B570 copy-only queue ordinal required'
            ordinals.append(int(ordinal))
        record['native_copy_queue_ordinals']=ordinals
        record['native_copy_queue_startup_gate_passed']=True
        record['native_copy_queue_note']='Factory validates COPY without COMPUTE, immediate/order/native identity and existing context/device, then registers the imported queue. This metadata and four-fresh numerical gate do not establish full-capacity correctness or general stall prevention.'
    else:
        assert not queue_lines, 'native queue must be absent for baseline/off'
        record['native_copy_queue_ordinals']=[]
        record['native_copy_queue_startup_gate_passed']=True
    record['engine_log_bytes'] = engine_log.stat().st_size
    record['engine_log_sha256'] = digest(engine_log)
    record['source_status_after'] = git('status', '--porcelain')
    assert not record['source_status_after'] and git('rev-parse', 'HEAD') == commits[mode]
    record['completed'] = True
    record['math_gate_passed'] = math_ok and len(record['requests'])==1+trials and len(record['sessions'])==trials
    record['phase_specific_decode_sequence_completed']=record['math_gate_passed']
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
