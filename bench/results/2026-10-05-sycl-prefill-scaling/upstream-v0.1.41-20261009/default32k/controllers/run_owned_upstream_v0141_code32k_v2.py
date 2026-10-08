"""Owned 32K update comparison: raw upstream, integrated update, qualified old control.

First-use diagnostics are separate from quiet timings. Every read is fresh.
This controller never changes sources, resets the GPU, or adopts a binary.
"""
from pathlib import Path
import datetime
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


base = Path(__file__).parent
observer = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0, str(observer / 'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity

mode, phase, repetition = sys.argv[1], sys.argv[2], int(sys.argv[3])
assert mode in ['pure', 'integrated', 'control']
assert phase in ['diagnostic', 'clean'] and 1 <= repetition <= 5
out = base / f'owned-v0141-code32k-{mode}-{phase}-r{repetition}'
assert not out.exists()
lock = (base / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def terminal_owned(record, *, require_gate=False):
    assert not record['active']
    assert record['healthy'] and record['completed'] and record['exit_code'] == 0
    assert not record['exit_signal'] and not record['new_fault_messages']
    assert not any(record['cleanup'].values())
    if require_gate:
        assert record['math_gate_passed']
    assert record['boot_id'] == boot
    for role in ['inferior', 'debugger']:
        old = record[role]
        now = process_identity(old['pid'])
        assert not now or now['start_ticks'] != old['start_ticks']


phase_path = base / 'owned-dd5-phase-v2-v01402-code32k-diagnostic-r1/record.json'
preceding = json.loads(phase_path.read_text())
terminal_owned(preceding, require_gate=True)
assert preceding['profiling_gate_passed']
qualified_path = base / 'owned-profile-definition-v01402-full256k-diagnostic-r7/record.json'
qualified = json.loads(qualified_path.read_text())
terminal_owned(qualified, require_gate=True)
assert qualified['physical256k_sequence_completed'] and qualified['capacity_sequence_completed']
known_path = base / 'owned-v0141-code32k-pure-diagnostic-r1/record.json'
known = json.loads(known_path.read_text())
assert not known['active'] and known['error'] == "IndexError('list index out of range')"
assert known['completed'] is False and not known['new_fault_messages']
assert len(known['requests']) == 1 and all(validate_fresh(known['requests'][0]).values())
for role in ['inferior', 'debugger']:
    old = known[role]
    now = process_identity(old['pid'])
    assert not now or now['start_ticks'] != old['start_ticks']
assert not known['cleanup']['inferior_survived'] and not known['cleanup']['gdb_survived']
for path in base.glob('owned-v0141-code32k-*/record.json'):
    if path == known_path:
        continue
    terminal_owned(json.loads(path.read_text()))

build_paths = {
    'pure': base / 'pure-upstream-v0.1.41-20261009/build-record.json',
    'integrated': base / 'integrated-upstream-v0.1.41-20261009-v2/build-record.json',
    'control': base / 'dpct-profile-definition-v01402-build-v1/record.json',
}
builds = {key: json.loads(path.read_text()) for key, path in build_paths.items()}
assert all(not build['active'] and build['passed'] for build in builds.values())
roots = {
    'pure': observer.parent / 'bench-upstream-v0.1.41-20261009',
    'integrated': observer.parent / 'sync-upstream-v0.1.41-20261009',
    'control': observer,
}
commits = {
    'pure': 'fb58e0dbc8399662c0e47c76578c6e878b14f6cf',
    'integrated': '1eb89482a4afd20277ae0405780ed4f8eb98eb20',
    'control': '452044ab2546185bebad51c2cb16b9375b6683e3',
}
root = roots[mode]
binary = Path(builds[mode]['binary']) if mode != 'control' else Path(qualified['argv'][0])
assert digest(binary) == (builds[mode]['binary_sha256'] if mode != 'control' else qualified['binary_sha256'])
def git(*args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
assert git('rev-parse', 'HEAD') == commits[mode] and not git('status', '--porcelain')
for key in ['pure', 'integrated']:
    assert builds[key]['commit'] == commits[key]
    assert builds[key]['ggml_commit'] == '3cf03257f219afbe7334045ff7c6a06ac68c627d'
tests_path = base / 'upstream-v0.1.41-server-tests-20261009-v2/record.json'
tests = json.loads(tests_path.read_text())
assert tests['passed'] and not tests['active'] and tests['exit_code'] == 0
diagnostic_rep = 2 if mode == 'pure' else 1
diagnostic_path = base / f'owned-v0141-code32k-{mode}-diagnostic-r{diagnostic_rep}/record.json'
diagnostic = None
if phase == 'clean':
    diagnostic = json.loads(diagnostic_path.read_text())
    terminal_owned(diagnostic, require_gate=True)
    assert len(diagnostic['requests']) == 4 and diagnostic['binary_sha256'] == digest(binary)

capture_cpu_path = base / 'owned-native-counter-v01402-cpu-v1/record.json'
capture_cpu = json.loads(capture_cpu_path.read_text())
assert digest(capture_cpu_path) == 'c93b119422938718e230d9e7c02377b7f0e6d48dc2095cb4add442f2e1025e86'
assert capture_cpu['passed'] and not capture_cpu['active']
assert digest(base / 'capture_owned_native_counter_v01402_v1.py') == capture_cpu['capture_controller_sha256']
assert digest(observer / 'sycl/tools/owned_gdb.py') == capture_cpu['owned_helper_sha256']
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
    'scope': 'Same-day comparison of unmodified upstream v0.1.41 main, integrated update, and previously qualified DD5 control. Four fresh32768 A/B/A/B reads, up to64 output tokens, context262144, chunk8192, int8 KV resident32768, MTP4, cache128 requested, workers5, pcie0, PC0/ckpt0. Initial logged tests excluded from performance. This is not a physical256K or disk-restore qualification.',
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'boot_id': boot, 'active': True, 'completed': False, 'healthy': False,
    'math_gate_passed': False, 'requests': [], 'snapshots': [], 'steps': [],
    'mode': mode, 'phase': phase, 'repetition': repetition,
    'deadline_seconds': 3600 if phase == 'diagnostic' else 1500,
    'protocol_timeout_seconds': 700 if phase == 'diagnostic' else 300,
    'log_limit_bytes': 64 * 1024**3 if phase == 'diagnostic' else 128 * 1024**2,
    'commit': commits[mode], 'root': str(root), 'binary_sha256': digest(binary),
    'source_status_before': '',
    'build_receipt_sha256': digest(build_paths[mode]),
    'rejected_controller_failure_receipt_sha256': digest(known_path),
    'rejected_controller_failure_note': 'Known v1 Python logprob parsing failure after one complete32K generation; not GPU stall or performance evidence. Owned processes absent and a new logged exact-word GPU health check is required below before model launch.',
    'server_tests_receipt_sha256': digest(tests_path),
    'preceding_logged_phase_receipt_sha256': digest(phase_path),
    'previous_qualified_full_receipt_sha256': digest(qualified_path),
    'actual_control_source_binding': 'Qualified private DD5 build and full lifecycle receipts; tracked control worktree HEAD is archive provenance, not a source-identical rebuild.' if mode == 'control' else None,
    'diagnostic_reference_receipt_sha256': digest(diagnostic_path) if diagnostic else None,
    'performance_eligible': phase == 'clean',
    'source_sha256': {str(p): digest(p) for p in [Path(__file__), source, observer / 'sycl/tools/owned_gdb.py', base / 'capture_owned_native_counter_v01402_v1.py']},
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
    health_path = base / 'post-device-profile-abort-v01402-health-v1/record.json'
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
    if phase == 'diagnostic':
        env = m.diagnostic_environment(env)
    if phase == 'clean':
        assert not any(k.startswith(('UR_LOG_', 'ZE_ENABLE_', 'ZEL_')) or k in ['UR_ENABLE_LAYERS', 'STRATA_TRACE'] for k in env)
    assert not any(k in env for k in ['STRATA_PREFILL_SYNC', 'STRATA_PREFILL_TIMING', 'STRATA_TRANSFER_TIMING', 'STRATA_VERIFY_NO_HOST', 'STRATA_VERIFY_DEVICE_PLAN'])
    record['environment'] = {k: v for k, v in env.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')) or k in ['LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission']}
    args = list(json.loads((base / 'model-lease-copy-off/normal-mtp.json').read_text())['args'])
    for key, value in [('--pcie-frac', '0'), ('--max-context', '262144'), ('--prefill', '8192'), ('--expert-cache', '128'), ('--conversation-cache-mib', '0')]:
        args[args.index(key) + 1] = value
    args += ['--kv', 'int8', '--kv-resident', '32768', '--prompt-cache', '0']
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
        request_start = time.monotonic()
        send(('GEN 64 ckpt=0 logprobs=5 ' + ','.join(map(str, fixtures[key])) + '\n').encode())
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
        checks = list(result['validation'].values())
        if key in seen:
            result['repeat_comparison'] = same_output(result, seen[key])
            checks.extend(result['repeat_comparison'].values())
        seen[key] = result
        if diagnostic:
            result['logged_reference_comparison'] = same_output(result, diagnostic['requests'][read_index])
            checks.extend(result['logged_reference_comparison'].values())
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
