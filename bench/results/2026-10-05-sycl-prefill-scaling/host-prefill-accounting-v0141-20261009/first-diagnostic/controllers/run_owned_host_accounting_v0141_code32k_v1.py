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
assert mode == 'accounting' and phase == 'diagnostic' and repetition == 1
assert sys.argv[4:] in [[], ['--cpu-preflight']]
review_head='16b1b04ffd77101c4b49543507c06fc71fbf2368'
engine_commit='1eb89482a4afd20277ae0405780ed4f8eb98eb20'
root=observer.parent/'sync-upstream-v0.1.41-20261009'
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
build_path=base/'host-prefill-accounting-v0141-private-build-v1/record.json'
assert digest(build_path)=='88fb31211c098c60791c3f6a9c5fbca77bdf4caf094e9366bb4fc2cbf9139283'
build=json.loads(build_path.read_text());assert not build['active'] and build['passed'] and build['compiled_engine'] and not build['gpu_tested'] and not build['adopted']
assert build['source_head_observed']==review_head and build['compiled_engine_source_commit']==engine_commit
assert build['all_other_archive_members_identical'] and build['original_link_inputs_unchanged']
binary=Path(build['binary']);assert digest(binary)==build['binary_sha256']=='494cf4be288595f7abf4d56412fb519155cfccf602a2a060093a1f8fcd988123'
candidate_source=base/'host-prefill-accounting-v0141-source-v1/prefill.cpp'
assert digest(candidate_source)==build['candidate_source_sha256']=='c8f6356707af8aaa521e273cf0733d8e650811f67474ba5c249db255d67f05f1'
host_cpu_path=base/'host-prefill-accounting-v0141-cpu-check-v1/record.json'
assert digest(host_cpu_path)=='568de39ce1dc0df39e753f405d2fa3816e578e8fee462a0ef844f20e1b39830a'
host_cpu=json.loads(host_cpu_path.read_text());assert host_cpu['passed'] and not host_cpu['active']
state_comparator=base/'compare_live_prefill_state_v01402_v2.py'
assert digest(state_comparator)=='9a4f55a4760d316045c1ec5179a946de9a9024f7e42fe09032012ceae2475a72'
from compare_live_prefill_state_v01402_v2 import compare_states
assert shutil.disk_usage(base).free>96*1024**3
out=base/'owned-host-accounting-v0141-code32k-diagnostic-r1'
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
    'scope':'First host-accounting binary qualification: four fresh32768 A/B/A/B with64 visible outputs, first logits, all66 live state parts and MTP/logprob references against qualified integrated869. Flushed UR/ZE/ZEL diagnostics; host counts only, no native event queries/extra phase waits/profiler. Logged times are excluded; no full-capacity or adoption claim.',
    'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':boot,
    'active':True,'completed':False,'healthy':False,'math_gate_passed':False,
    'requests':[],'snapshots':[],'steps':[],'mode':mode,'phase':phase,'repetition':repetition,
    'deadline_seconds':3600,'protocol_timeout_seconds':700,'log_limit_bytes':64*1024**3,
    'commit':review_head,'compiled_engine_base_commit':engine_commit,'root':str(root),
    'binary_sha256':digest(binary),'build_receipt_sha256':digest(build_path),
    'candidate_source_sha256':digest(candidate_source),'host_cpu_receipt_sha256':digest(host_cpu_path),
    'reference_binary_sha256':diagnostic['binary_sha256'],
    'reference_four_fresh_receipt_sha256':digest(subset_path),
    'baseline_full_capacity_receipt_sha256':digest(new_full_path),
    'baseline_quiet_sequence_receipt_sha256':digest(sequence_path),
    'performance_eligible':False,'full_lifecycle_passed':False,'adopted':False,
    'source_status_before':'',
    'source_sha256':{str(p):digest(p) for p in [Path(__file__),source,observer/'sycl/tools/owned_gdb.py',base/'capture_owned_native_counter_v01402_v1.py',state_comparator,candidate_source]},
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
    env.update(STRATA_PREFILL_HOST_TIMING='1',STRATA_DUMP_FIRST_LOGITS=str(out/'first-head.bin'),STRATA_PREFILL_DUMP_STATE=str(out/'prefill-state.bin'))
    if phase == 'diagnostic':
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
    reports=[]
    for line_value in (out/'project-messages.txt').read_text().splitlines():
        if line_value.startswith('strata prefill host: '):
            item=json.loads(line_value.split(': ',1)[1])
            assert item['tokens']==32767 and item['chunks']==4
            assert item['expert_copies']>0 and item['expert_bytes']>0
            assert sum(x['copies'] for x in item['layers'])==item['expert_copies']
            assert sum(x['bytes'] for x in item['layers'])==item['expert_bytes']
            assert [x['layer'] for x in item['layers']]==list(range(len(item['layers'])))
            assert all(math.isfinite(v) and v>=0 for k,v in item.items() if k.endswith('_ms'))
            reports.append(item)
    assert len(reports)==4, 'all4 fresh prefill host reports required'
    record['host_accounting_reports']=reports
    record['host_accounting_gate_passed']=True
    record['host_accounting_note']='CPU submit and group-wait times include logging, queue dependencies and compute; not DMA active time. RAM-worker sums may overlap. Exact expert bytes/copies are useful; logged timing is not throughput evidence.'
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
