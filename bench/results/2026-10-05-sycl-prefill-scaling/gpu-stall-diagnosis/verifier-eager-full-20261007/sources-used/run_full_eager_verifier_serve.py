"""Real 262144-cell normal-MTP serve capacity under an owned debugger.

Includes both full and clipped speculative tails, refusals and a later valid
request. No reset, service operation or throughput claim is made here.
"""
from pathlib import Path
import array
import datetime
import hashlib
import json
import math
import os
import re
import selectors
import sys
import time
import tty
import types

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
os.chdir(root)
sys.path.insert(0, str(root / 'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity

out = base / 'full-context-eager-verifier-serve'
out.mkdir(mode=0o700)
probes = out / 'probes'
probes.mkdir()
helper = root / 'sycl/tools/recover-xe.sh'
module = types.ModuleType('capacity_readonly')
exec(compile(helper.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(helper), 'exec'), module.__dict__)
runner = module.Runner(probes)
record = {'scope': 'Actual normal-MTP 262144-cell capacity and repeated-memory-restoration check using the unchanged 3f3e binary and existing STRATA_VERIFY_EAGER=1, with first full output compared to the lazy candidate using STRATA_PREFILL_SYNC=1 with expert wait 0, validated same-executable short profile, warnings/parameter checks/progress and read-only CSR wait reader; no timing or prevention claim',
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'active': True, 'completed': False, 'healthy': False,
          'requests': [], 'snapshots': [], 'steps': [],
          'deadline_seconds': 10800, 'protocol_timeout_seconds': 5400,
          'write_timeout_seconds': 30, 'shutdown_timeout_seconds': 30,
          'log_limit_bytes': 4 * 1024**3}
g = None
master = slave = None
cursor = None
pending = bytearray()
current = None
started = time.monotonic()
next_update = started
next_metric = started
idle_start = started
previous_signature = None
last_snapshot = 0
raw = (out / 'protocol.stdout.raw').open('wb')
events = (out / 'events.jsonl').open('w', buffering=1)
metrics = (out / 'metrics.jsonl').open('w', buffering=1)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save():
    record['elapsed_seconds'] = time.monotonic() - started
    record['active_request'] = current
    record['steps'] = runner.calls
    if g:
        record.update(inferior=g.inferior, debugger=g.debugger_identity)
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

def event(kind, **fields):
    events.write(json.dumps({'kind': kind, 'elapsed_seconds': time.monotonic() - started, **fields}) + '\n')

def read_metrics():
    proc = Path('/proc') / str(g.inferior['pid'])
    stat = (proc / 'stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    assert int(fields[19]) == g.inferior['start_ticks']
    io = {k: int(v) for k, v in (line.split(':', 1) for line in (proc / 'io').read_text().splitlines())}
    gpu = []
    for path in (proc / 'fdinfo').iterdir():
        try:
            text = path.read_text()
        except FileNotFoundError:
            continue
        if re.search(r'^drm-pdev:\s*0000:05:00.0\s*$', text, re.M):
            gpu.append({'fd': path.name, 'raw': text})
    return {'pid': g.inferior['pid'], 'start_ticks': int(fields[19]), 'state': fields[0],
            'minor_faults': int(fields[7]), 'major_faults': int(fields[9]),
            'user_ticks': int(fields[11]), 'system_ticks': int(fields[12]),
            'io': io, 'gpu': gpu, 'elapsed_seconds': time.monotonic() - started}

def inspect_submission(label):
    snap = g.snapshot(label, resume=False)
    if not snap:
        return None
    start = g.raw.tell()
    g.command('-interpreter-exec console ' + json.dumps('source ' + str(base / 'read-csr-wait.gdb')))
    g.raw.flush()
    dump = out / 'debugger' / (label + '.csr.mi.txt')
    with (out / 'debugger/gdb-mi.stdout').open('rb') as source, dump.open('wb') as dest:
        source.seek(start)
        import shutil
        shutil.copyfileobj(source, dest)
    snap['read_only_csr_dump'] = str(dump)
    decoded = ''
    for line in Path(snap['path']).read_text().splitlines():
        if line.startswith('~'):
            try:
                decoded += json.loads(line[1:])
            except ValueError:
                pass
    main = re.search(r'^Thread 1 \(.*?\n(.*?)(?:\n\n|\Z)', decoded, re.M | re.S)
    counter = re.search(r'NEO::IoctlHelperXe::execBuffer[^\n]*counterValue=(\d+)', main[1]) if main else None
    snap['submission_counter'] = int(counter[1]) if counter else None
    main_registers = decoded.split('strata diagnostic: main thread ', 1)[1] if 'strata diagnostic: main thread ' in decoded else ''
    snap['syscall_eagain'] = bool(re.search(r'^rax\s+.*?\s-11\s*$', main_registers, re.M))
    previous = record['snapshots'][-1] if record['snapshots'] else None
    stalled = bool(previous and previous.get('syscall_eagain') and snap['syscall_eagain']
                   and snap['submission_counter'] is not None
                   and previous.get('submission_counter') == snap['submission_counter'])
    record['snapshots'].append(snap)
    if stalled:
        raise RuntimeError('Repeated EAGAIN at the same submission counter after unchanged progress/I/O')
    if 'signal-name="SIGINT"' in snap['stop']:
        g.command('-exec-continue --all')
        snap['resumed'] = True
        g.stops.append('resumed')
    return snap

def poll():
    global next_update, next_metric, previous_signature, idle_start, last_snapshot
    g.poll(.01)
    if time.monotonic() - started >= record['deadline_seconds']:
        raise TimeoutError('Finite diagnostic deadline; not proof of a workload hang')
    if g.stops and g.stops[-1] != 'resumed' and g.exit_code is None and g.exit_signal is None:
        raise RuntimeError('Inferior stopped: ' + g.stops[-1])
    size = sum(p.stat().st_size for p in (out / 'debugger').glob('*.stderr'))
    if size >= record['log_limit_bytes']:
        raise RuntimeError('Diagnostic log size limit')
    control = out / 'control.json'
    if control.exists():
        command = json.loads(control.read_text())
        control.unlink()
        if command['action'] == 'stop':
            raise RuntimeError(command.get('reason', 'Requested diagnostic stop'))
        assert command['action'] == 'snapshot'
        inspect_submission('requested-' + str(len(record['snapshots'])))
    if g.inferior and g.exit_code is None and g.exit_signal is None and time.monotonic() >= next_metric:
        try:
            sample = read_metrics()
        except (FileNotFoundError, ProcessLookupError):
            identity = process_identity(g.inferior['pid'])
            if identity and identity['start_ticks'] == g.inferior['start_ticks'] and identity['state'] != 'Z':
                raise
            # The final /proc entries can disappear before GDB delivers its
            # normal-exit MI event. Keep polling the same owned debugger.
            g.poll(0)
            next_metric = time.monotonic() + .1
            return
        metrics.write(json.dumps(sample) + '\n')
        record['latest_metrics'] = sample
        signature = (size, raw.tell(), sample['minor_faults'], sample['major_faults'],
                     sample['io']['rchar'], sample['io']['read_bytes'])
        if signature != previous_signature:
            idle_start = time.monotonic()
            previous_signature = signature
        record['unchanged_progress_seconds'] = time.monotonic() - idle_start
        if sample['gpu'] and record['unchanged_progress_seconds'] >= 40 and len(record['snapshots']) < 4 and time.monotonic() - last_snapshot >= 30:
            inspect_submission('unchanged-progress-' + str(len(record['snapshots'])))
            last_snapshot = time.monotonic()
        next_metric = time.monotonic() + 5
    if time.monotonic() >= next_update:
        save()
        next_update = time.monotonic() + 5

def line():
    end = time.monotonic() + record['protocol_timeout_seconds']
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
                raise RuntimeError('Protocol EOF before reply')
            raw.write(data)
            raw.flush()
            pending.extend(data)
            event('stdout', bytes=len(data), raw_offset=raw.tell())
    raise TimeoutError('Engine protocol deadline')

def send(data):
    end = time.monotonic() + record['write_timeout_seconds']
    view = memoryview(data)
    with selectors.DefaultSelector() as ready:
        ready.register(master, selectors.EVENT_WRITE)
        while view:
            poll()
            if time.monotonic() >= end:
                raise TimeoutError('Protocol request-write deadline')
            if not ready.select(.02):
                continue
            try:
                n = os.write(master, view[:8192])
            except BlockingIOError:
                continue
            view = view[n:]

try:
    previous = json.loads((base / 'full-context-verifier-lazy-restore-serve/record.json').read_text())
    health = json.loads((base / 'post-full-verifier-lazy-restore-health/record.json').read_text())
    dequant = json.loads((base / 'dequant-actual-wrapper-diagnostic-v2/record.json').read_text())
    continuation = json.loads((base / 'profiler-continuation.json').read_text())
    assert not continuation['live_gpu_jobs']
    assert not previous['active'] and not previous['new_fault_messages']
    assert not previous['cleanup']['inferior_survived'] and not previous['cleanup']['gdb_survived']
    assert health['healthy'] and health['boot_id'] == previous['boot_id'] == record['boot_id']
    assert health['started_utc'] > previous['finished_utc']
    assert dequant['passed'] and not dequant['active'] and not dequant['new_fault_messages']
    assert dequant['boot_id'] == record['boot_id']
    for owner in [previous['inferior'], previous['debugger']] + [job[key] for job in dequant['runs'] for key in ('inferior', 'debugger')]:
        current_owner = process_identity(owner['pid'])
        assert not current_owner or current_owner['start_ticks'] != owner['start_ticks']
    eager_short = json.loads((base / 'large-kv-short-eager-control/record.json').read_text())
    assert eager_short['healthy'] and eager_short['completed'] and not eager_short['active']
    assert eager_short['environment']['STRATA_VERIFY_EAGER'] == '1'
    assert all(eager_short['requests'][0]['same_configuration_comparison'].values())
    assert not eager_short['new_fault_messages'] and not any(eager_short['cleanup'].values())
    for key in ['inferior', 'debugger']:
        assert not process_identity(eager_short[key]['pid'])
    record['validated_eager_short_record_sha256'] = digest(base / 'large-kv-short-eager-control/record.json')
    cli = json.loads((base / 'full-context-layer-trace-csr/record.json').read_text())
    assert cli['completed'] and cli['healthy'] and not cli['active']
    assert cli['boot_id'] == record['boot_id']
    for key in ['inferior', 'debugger']:
        assert not process_identity(cli[key]['pid'])
    assert not any(cli['cleanup'].values())
    assert not cli['new_fault_messages'] and cli['last_executed_kv_cell'] == 262143
    assert cli['verify_windows'] == [[262141, 1], [262142, 2]]
    short = json.loads((base / 'owned-workspace-reclaim-short/record.json').read_text())
    assert short['completed'] and short['healthy'] and not short['active']
    assert short['boot_id'] == record['boot_id'] and not short['new_fault_messages']
    assert not any(short['cleanup'].values())
    for key in ['inferior', 'debugger']:
        assert not process_identity(short[key]['pid'])
    assert not short['expert_wait_batches'] and short['phase_sync_mark_count']>0
    assert short['environment']['STRATA_PREFILL_SYNC']=='1'
    for name in ['full-context-expert-wait-1-serve','full-context-legacy-l0-serve']:
        old=json.loads((base/name/'record.json').read_text())
        assert not old['active'] and not old['new_fault_messages']
        for key in ['inferior','debugger']:
            now=process_identity(old[key]['pid'])
            assert not now or now['start_ticks']!=old[key]['start_ticks'] or now['state']=='Z'
    assert short['release_pairs'] == {'release': 6, 'restore': 6}
    assert len(short['requests']) == 4 and all(all(r['equality'].values()) for r in short['requests'])
    for name in ['full-context-phase-sync-serve', 'vram-restore-probe-default-single', 'vram-restore-probe-default-split']:
        old = json.loads((base / name / 'record.json').read_text())
        assert not old['active'] and not old['new_fault_messages']
        for key in ['inferior', 'debugger']:
            now = process_identity(old[key]['pid'])
            assert not now or now['start_ticks'] != old[key]['start_ticks'] or now['state'] == 'Z'
    assert digest(base / 'prefill-workspace-reclaim-build/prefill.cpp') == 'bdc0dea41753d345f5b5c329d807d3f9013280e2488e39df1d615a3b84506b92'
    executable = base / 'strata-prefill-workspace-reclaim-candidate'
    assert digest(executable) == short['binary_sha256'] == '3f3ed0526848f6c7273a70da8802cbe00389b67791629c000c26399bc50823f2'
    record['baseline_cli_binary_sha256'] = cli['binary_sha256']
    record['binary_sha256'] = digest(executable)
    reference = json.loads((base / 'full-context-copy-off/reference.json').read_text())
    original_args = reference['runs'][0]['args']
    def value(key):
        return original_args[original_args.index(key) + 1]
    source = list(map(int, (base / 'full-context-copy-off/coding-context-256k-tokens.txt').read_text().split()))
    assert len(source) >= 262144
    record['source_fixture_sha256'] = digest(base / 'full-context-copy-off/coding-context-256k-tokens.txt')
    suffix = [248046, 198, 248045, 74455, 198, 248068, 198, 248069, 271]
    def prompt_ids(n):
        ids = source[:n - len(suffix)] + suffix
        assert len(ids) == n
        return ids
    baseline_env = json.loads((base / 'full-context-copy-off/environment.json').read_text())
    original_tuning = json.loads((base / 'full-context-copy-off/cli-262144-strata-residency-candidate-draft-lease-verified/summary.json').read_text())['env']
    baseline_env.update(original_tuning)
    env = module.diagnostic_environment(baseline_env)
    env['UR_ENABLE_LAYERS'] = ','.join(v for v in env['UR_ENABLE_LAYERS'].split(',') if v != 'UR_LAYER_TRACING')
    for key in ['UR_LOG_TRACING', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO']:
        env[key] = 'level:warning;flush:warning;output:stderr'
    env.update(ZEL_LOADER_LOGGING_LEVEL='warn', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',
               ZEL_LOADER_LOG_PATTERN='[thread-id:%t] [%l] %v', STRATA_TRACE='1')
    assert all(env[k] == v for k, v in original_tuning.items() if k != 'STRATA_TRACE')
    head = out / 'first-head.bin'
    env['STRATA_DUMP_FIRST_LOGITS'] = str(head)
    env['STRATA_PREFILL_EXPERT_WAIT_BATCH'] = '0'
    env['STRATA_PREFILL_SYNC'] = '1'
    env['STRATA_VERIFY_EAGER'] = '1'
    assert eager_short['binary_sha256'] == digest(executable)
    record['environment'] = {k: v for k, v in env.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')) or k in ['LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission']}
    record['source_sha256'] = {str(p): digest(p) for p in [Path(__file__), helper, root / 'sycl/tools/owned_gdb.py', root / 'sycl/src/program/generate.cpp', root / 'sycl/src/prefill/prefill.cpp', root / 'sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp', base / 'prefill-workspace-reclaim-build/prefill.cpp', base / 'read-csr-wait.gdb']}
    argv = [str(executable), '--pack', value('--pack'), '--native', value('--native'),
            '--max-context', '262144', '--kv', 'int8', '--spec', '4', '--spec-min-p', '0',
            '--suffix-draft', '0', '--no-prefill-borrow', '--expert-cache', '128',
            '--expert-profile', value('--expert-profile'), '--expert-cache-per-layer',
            '--pool-workers', '5', '--pcie-frac', '0', '--adapt-swaps', '0', '--ple-io', 'direct',
            '--greedy', '--serve', '--mtp', str(Path.home() / '.local/share/strata-sycl/mtp/rt'),
            '--spec-split', '--prefill', '1024', '--prompt-cache', '0']
    record['argv'] = argv
    record['assistant_suffix'] = suffix
    record['capacity'] = 262144
    save()
    assert runner.run('embedding-state', ['/usr/bin/systemctl', '--user', 'show', 'llama-server-qwen3embed.service', '-p', 'ActiveState'], seconds=5).strip() == 'ActiveState=inactive'
    previous_cursor = next(v['argv'][v['argv'].index('--after-cursor') + 1] for v in short['steps'] if v['label'] == 'kernel-after')
    gap = runner.run('kernel-gap', ['/usr/bin/journalctl', '-k', '--after-cursor', previous_cursor, '--no-pager', '-o', 'json'])
    rows = [json.loads(s) for s in gap.splitlines() if s.startswith('{')]
    record['preflight_fault_messages'] = [r.get('MESSAGE', '') for r in rows if ('0000:05:00.0' in r.get('MESSAGE', '') or re.search(r'\bxe\b', r.get('MESSAGE', ''))) and module.FAULT.search(r.get('MESSAGE', ''))]
    assert not record['preflight_fault_messages']
    cursor = module.journal_cursor(runner, 'kernel-before')
    master, slave = os.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    g = OwnedGdb(argv, out / 'debugger', env, inferior_tty_fd=slave)
    g.command('-gdb-set may-call-functions off')
    g.command('-gdb-set debug-file-directory /usr/lib/debug:/home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/runtime-symbols/extracted/usr/lib/debug')
    g.run()
    record['startup'] = []
    while True:
        text = line()
        record['startup'].append(text)
        if text.startswith('ERR'):
            raise RuntimeError(text)
        if text.startswith('READY '):
            break
    os.close(slave)
    slave = None
    stderr = out / 'debugger/inferior.stderr'
    record['ready_capacity'] = int(record['startup'][-1].split()[1])
    assert record['ready_capacity'] == 262144
    cases = [('fills-context', 262140, 4, True), ('fills-context-tail-2', 262142, 2, True),
             ('no-room', 262144, 1, False), ('one-too-many', 262142, 3, False),
             ('works-after-refusal', 37, 2, True)]
    for name, n, new, allowed in cases:
        current = name
        trace_start = stderr.stat().st_size
        if allowed:
            head.unlink(missing_ok=True)
        request = f'GEN {new} logprobs=5 ' + ','.join(map(str, prompt_ids(n))) + '\n'
        send(request.encode())
        event('request', name=name, request_bytes=len(request))
        result = {'name': name, 'input_tokens': n, 'max_new': new, 'ids': [], 'logprobs': [], 'protocol': [],
                  'request_sha256': hashlib.sha256(request.encode()).hexdigest()}
        request_started = time.monotonic()
        while True:
            text = line()
            result['protocol'].append(text)
            if text.startswith('T '):
                result['ids'].append(int(text.split()[1]))
            if text.startswith('LP '):
                assert all(math.isfinite(float(v.rsplit(':', 1)[-1])) for v in text.split()[1:])
                result['logprobs'].append(text)
            if text.startswith(('DONE ', 'ERR ')):
                break
        result['diagnostic_request_seconds'] = time.monotonic() - request_started
        record['requests'].append(result)
        save()
        with stderr.open('rb') as trace:
            trace.seek(trace_start)
            text = ''.join(line.decode(errors='replace') for line in trace
                if any(marker in line for marker in (b'strata trace: window ', b'strata mtp decode release:', b'strata mtp decode restore:')))
        windows = [list(map(int, v)) for v in re.findall(r'strata trace: window (-?\d+) (-?\d+)', text)]
        result['verify_windows'] = windows
        assert all(pos >= 0 and count > 0 and pos + count <= 262144 for pos, count in windows)
        releases = re.findall(r'strata mtp decode release:.*?verified=(\d+)', text)
        restores = re.findall(r'strata mtp decode restore:.*?verified=(\d+)', text)
        result['draft_releases'] = len(releases)
        result['draft_restores'] = len(restores)
        assert len(releases) == len(restores) and all(v == '1' for v in releases + restores)
        if name == 'fills-context':
            assert releases
        if allowed:
            assert result['protocol'][-1].startswith('DONE ')
            assert len(result['ids']) == new and len(result['logprobs']) == new
            done = result['protocol'][-1].split()
            assert len(done) >= 6 and int(done[1]) == new and int(done[2]) == n and done[5] == 'length'
            assert all(math.isfinite(float(v)) and float(v) >= 0 for v in done[3:5])
            payload = head.read_bytes()
            values = array.array('f')
            values.frombytes(payload)
            assert len(values) == 248320 and all(map(math.isfinite, values))
            result['head'] = {'bytes': len(payload), 'floats': len(values), 'sha256': hashlib.sha256(payload).hexdigest()}
            if name == 'fills-context':
                reference_request = previous['requests'][0]
                result['lazy_first_request_comparison'] = {
                    'ids': result['ids'] == reference_request['ids'],
                    'logprobs': result['logprobs'] == reference_request['logprobs'],
                    'whole_head': result['head']['sha256'] == reference_request['head']['sha256']}
                assert all(result['lazy_first_request_comparison'].values())
            (out / (name + '.head.bin')).write_bytes(payload)
            if name.startswith('fills-context'):
                assert n + len(result['ids']) == 262144
                assert windows and max(pos + count for pos, count in windows) == 262144
                result['last_executed_kv_cell'] = 262143
                if name == 'fills-context-tail-2':
                    assert windows == [[262141, 1], [262142, 2]]
                    result['clipped_verify_tail'] = 2
                    result['cli_head_equal'] = result['head']['sha256'] == cli['head_sha256']
                    result['cli_ids_equal'] = result['ids'] == cli['ids']
        else:
            assert result['protocol'][-1].startswith('ERR prompt') and not result['ids']
            assert not windows and not result['logprobs']
            assert not any(v.startswith('REUSED ') for v in result['protocol'])
        save()
    current = None
    send(b'QUIT\n')
    event('quit')
    end = time.monotonic() + record['shutdown_timeout_seconds']
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
        # Drain remaining buffered PTY bytes after the normal-exit MI event.
        drain_end = time.monotonic() + 2
        while time.monotonic() < drain_end and ready.select(.01):
            try:
                data = os.read(master, 65536)
            except (BlockingIOError, OSError):
                break
            if not data:
                break
            raw.write(data)
            raw.flush()
            event('final-stdout', bytes=len(data))
    assert g.exit_code == 0 and g.exit_signal is None
    record['completed'] = True
except BaseException as error:
    record['error'] = repr(error)
    if g:
        try:
            inspect_submission('failure')
        except BaseException as inspection:
            record['snapshot_error'] = repr(inspection)
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
    metrics.close()
    if cursor:
        try:
            text = runner.run('kernel-after', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
            rows = [json.loads(s) for s in text.splitlines() if s.startswith('{')]
            record['new_fault_messages'] = [r.get('MESSAGE', '') for r in rows if ('0000:05:00.0' in r.get('MESSAGE', '') or re.search(r'\bxe\b', r.get('MESSAGE', ''))) and module.FAULT.search(r.get('MESSAGE', ''))]
        except BaseException as error:
            record['kernel_error'] = repr(error)
    record['active'] = False
    record['healthy'] = bool(record['completed'] and not record.get('error') and not record.get('kernel_error') and not record.get('new_fault_messages') and not any(record.get('cleanup', {}).values()))
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({k: v for k, v in record.items() if k not in ['environment', 'argv', 'steps']}, indent=2))
raise SystemExit(0 if record['healthy'] else 1)
