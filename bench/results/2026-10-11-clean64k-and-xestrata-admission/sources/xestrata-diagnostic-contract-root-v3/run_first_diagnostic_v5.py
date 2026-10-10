"""Private root-owned first XeStrata diagnostic; never a throughput benchmark.

Protocol/owner pattern adapted from run_owned_fixed12k_chunk_major_code32k_v1.py.
No fork source modification, GPU reset, retries, or adoption.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import sys
import time
import tty
import types

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
OBSERVER = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05/sycl/tools')
COMMIT = '39bdadcc9e2b89b1e3c8be7bb2a603b04fa0e197'
HELPER_HASH = '61e1d206bf57bb320051cdd65943d23722bc523a602d80c54b7e7a2ad0d3f0d1'
FIXTURE_HASH = '137fa1157c697295238d2fd487c09f9df60ab9cfee421e8c7e0b743b240af449'
REFERENCE_HASH = '95ae2a0a7e61f2240f6138f21fdcb35c1a70ec54300b22a7b99fd6b92873b4a7'
OWNER_SOURCE = B / 'run_gdn_gate_factor_probe_v2.py'
OWNER_HASH = '7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def explicit_environments():
    """Construct all execution environments from an allowlist, never inheritance."""
    require(os.geteuid() != 0, 'ordinary-account controller required')
    base = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8',
                LC_CTYPE='C.UTF-8', LC_ALL='C.UTF-8')
    for key in ('XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS'):
        value = os.environ.get(key)
        if value is not None:
            require(len(value) < 4096, 'desktop environment bound')
            base[key] = value
    runtime = dict(base, LD_LIBRARY_PATH=':'.join([
        '/opt/intel/oneapi/compiler/2026.1/lib',
        '/opt/intel/oneapi/compiler/2026.1/opt/compiler/lib',
        '/opt/intel/oneapi/umf/1.1/lib', '/usr/lib/x86_64-linux-gnu',
        '/opt/intel/oneapi/mkl/2026.1/lib']),
        SYCL_CACHE_PERSISTENT='0',
        UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',
        UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
        NEOReadDebugKeys='1', EnableDirectSubmission='0')
    return base, runtime


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(result):
    done = [v.split() for v in result['protocol'] if v.startswith('DONE ')]
    require(len(done) == 1 and len(done[0]) >= 16, 'incomplete DONE')
    d = done[0]
    n = int(d[1])
    require(0 < n <= 64 and len(result['ids']) == len(result['logprobs']) == n, 'output counts')
    require(all(0 <= v < 248320 for v in result['ids']), 'output ID range')
    require(int(d[2]) == 32768 and int(d[8]) == 0 and int(d[14]) == 32768, 'fresh complete prompt counts')
    require(d[5] in ('length', 'stop') and 0 <= int(d[6]) <= int(d[7]), 'finish/draft counts')
    require(all(math.isfinite(float(d[i])) and float(d[i]) > 0 for i in (3, 4)), 'finite positive phase times')
    for tag in ('RESUME', 'REUSED'):
        values = [int(v.split()[1]) for v in result['protocol'] if v.startswith(tag + ' ')]
        require(values == [0], 'fresh ' + tag)
    pp = [v.split() for v in result['protocol'] if v.startswith('PP ')]
    require([int(v[1]) for v in pp] == [8192, 16384, 24576, 32767], 'fixed8K complete PP positions')
    require(all(int(v[2]) == 32768 and all(math.isfinite(float(x)) for x in v[3:]) for v in pp), 'PP totals/finite')
    for value in result['logprobs']:
        fields = value.split()
        require(len(fields) == 7 and math.isfinite(float(fields[1])), 'full LP5')
        for field in fields[2:]:
            token, probability = field.split(':')
            require(0 <= int(token) < 248320 and math.isfinite(float(probability)), 'LP ID/finite')
    return {'generated': n, 'prompt_tokens': 32768, 'actual_batched_positions': 32767,
            'prompt_ms': float(d[3]), 'decode_ms': float(d[4]), 'finish_reason': d[5],
            'mtp_counts': list(map(int, d[6:8])), 'fresh_complete_finite': True}


def preflight(receipt_path):
    build = json.loads(receipt_path.read_text())
    require(build.get('passed') is True and build.get('active') is False and build.get('completed') is True, 'closed successful root build receipt')
    require(build.get('source_commit') == COMMIT and build.get('fork_sources_unchanged') is True, 'pristine pinned fork attestation')
    require(build.get('license_mode') == 'contrib-icpx' and build.get('compiler_version') == '2026.1.1', 'build mode/compiler')
    require(build.get('ggml_commit') == '3cf03257f219afbe7334045ff7c6a06ac68c627d', 'GGML identity')
    root, binary = Path(build['root']).resolve(), Path(build['binary']).resolve()
    require(binary.is_relative_to(root) and binary.is_file() and os.access(binary, os.X_OK), 'owned worktree executable')
    require(sha(binary) == build['binary_sha256'], 'built binary identity')
    require(bool(build.get('source_sha256')), 'root source manifest required')
    for name, expected in build['source_sha256'].items():
        p = (root / name).resolve()
        require(p.is_relative_to(root) and sha(p) == expected, 'source manifest ' + name)
    source = root / 'src/program/generate.cpp'
    require(sha(source) == '25854acf5a75ba7cba4e7aa7f27d762c8a961385cd6c45df7c3deeab7b293802', 'frozen CLI source')
    require(sha(OBSERVER / 'owned_gdb.py') == HELPER_HASH, 'qualified owner helper')
    require(sha(OWNER_SOURCE) == OWNER_HASH, 'qualified auxiliary owner source')
    fixture = B / 'coding-review-32k-tokens.txt'
    reference_path = B / 'upstream-v0141-tuned-fourfresh-numerical-subset-20261009-v1.json'
    require(sha(fixture) == FIXTURE_HASH and sha(reference_path) == REFERENCE_HASH, 'frozen fixture/reference')
    tokens = list(map(int, fixture.read_text().split()))
    require(len(tokens) == 32768 and all(0 <= t < 248320 for t in tokens), 'exact fixture without truncation')
    baseline = json.loads(reference_path.read_text())
    require(baseline['passed'] and not baseline['active'], 'closed baseline')
    args = list(baseline['argv'][1:])
    for key, value in {'--prefill': '8192', '--max-context': '262144', '--expert-cache': '128', '--pool-workers': '5', '--spec': '4', '--kv-resident': '32768'}.items():
        require(args.count(key) == 1 and args[args.index(key) + 1] == value, 'fixed parameter ' + key)
    require('--no-prefill-borrow' in args and '--serve' in args, 'owned serving prompt buffers')
    # Preserve the baseline profile FILE, not a different relative fork file.
    index = args.index('--expert-profile') + 1
    profile = Path(args[index])
    if not profile.is_absolute():
        profile = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-events-20261010') / profile
    require(profile.is_file(), 'baseline expert profile unavailable')
    args[index] = str(profile.resolve())
    text = source.read_text()
    require(all(('a == "' + flag + '"') in text for flag in args if flag.startswith('--')), 'fork CLI flag unsupported')
    require('STRATA_DUMP_FIRST_LOGITS' not in text, 'dump support changed: review capture contract')
    return root, binary, [str(binary)] + args, tokens, baseline, profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpu-preflight', action='store_true')
    options = parser.parse_args()
    root, binary, argv, tokens, baseline, profile = preflight(options.build_receipt)
    if options.cpu_preflight:
        print(json.dumps({'passed': True, 'gpu_executed': False, 'argv': argv, 'tokens': len(tokens),
                          'binary_sha256': sha(binary), 'profile_sha256': sha(profile), 'dump_support': False}))
        return 0
    require(options.output is not None and options.output.is_absolute(), 'explicit absolute private output')
    out = options.output
    require(not out.exists() and out.parent.resolve().is_relative_to(B), 'new owned output under B')
    lock = (B / 'owned-v0141-measurement.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(shutil.disk_usage(B).free > 8 * 1024**3, 'free artifact capacity')
    memory = {v.split(':')[0]: int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith(('MemTotal:', 'MemAvailable:'))}
    require(memory['MemAvailable'] > 64*1024**3, 'free host memory admission')
    out.mkdir(mode=0o700)
    sys.path.insert(0, str(OBSERVER))
    from owned_gdb import OwnedGdb, process_identity
    health_source = OBSERVER / 'recover-xe.sh'
    health = types.ModuleType('read_only_health')
    require(sha(health_source) == '2f90005f79d2309747d917cff51dc53d8ee6192d68cb1feca074f84af8b0796c', 'qualified health helper')
    exec(compile(health_source.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(health_source), 'exec'), health.__dict__)
    (out / 'probes').mkdir()
    owner_module = types.ModuleType('qualified_auxiliary_owner')
    # EXACT qualified source prefix; no copied/reimplemented/weakened Owner.
    owner_source = OWNER_SOURCE.read_text().split('\ndef parse_probe(', 1)[0]
    exec(compile(owner_source, str(OWNER_SOURCE), 'exec'), owner_module.__dict__)
    owner_module.W = root
    auxiliary_owner = owner_module.Owner(out / 'probes')
    base_env, runtime_env = explicit_environments()
    # Original health checks/diagnostic transformation, pure environment source.
    health.health_environment = lambda: dict(runtime_env)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    start = time.monotonic()
    record = dict(active=True, completed=False, diagnostic_passed=False, math_equivalence=False,
                  performance_eligible=False, adopted=False, full256k_qualified=False, source_commit=COMMIT,
                  argv=argv, binary_sha256=sha(binary), build_receipt_sha256=sha(options.build_receipt),
                  controller_sha256=sha(__file__), owner_helper_sha256=HELPER_HASH,
                  auxiliary_owner_source_sha256=OWNER_HASH,
                  health_source_sha256=sha(health_source), reference_sha256=REFERENCE_HASH,
                  prompt_fixture_sha256=FIXTURE_HASH, expert_profile_sha256=sha(profile),
                  initial_memory=memory, boot_id=boot,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='one fresh raw-fork diagnostic; output divergence is evidence, not quality equivalence',
                  dump_support=False, head_comparison=None, live_state_comparison=None,
                  limits=dict(wall_seconds=1000, protocol_seconds=700, logs_bytes=64*1024**2,
                              RSS_bytes=104*1024**3, artifacts_bytes=2*1024**3))
    g = None
    gdb_launch_attempted = False
    master = slave = cursor = None
    pending = bytearray()
    next_save = start + 5
    raw = (out / 'protocol.stdout.raw').open('wb')

    def save():
        record['elapsed_seconds'] = time.monotonic() - start
        if g:
            record.update(inferior=g.inferior, debugger=g.debugger_identity)
        temp = out / 'record.json.tmp'
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(out / 'record.json')

    auxiliary_owner.persist = save
    class Runner:
        output = out / 'probes'
        calls = auxiliary_owner.commands

        def run(self, label, arguments, seconds=20, env=None):
            require(auxiliary_owner.active is None, 'unresolved auxiliary ownership; no further launch')
            files = [p for p in out.rglob('*') if p.is_file()]
            require(sum(p.stat().st_size for p in files) < 2*1024**3, 'auxiliary artifact budget')
            require(sum(p.stat().st_size for p in files if p.suffix in ('.stderr', '.raw', '.stdout')) < 64*1024**2, 'auxiliary log budget')
            remaining = 1000 - (time.monotonic() - start)
            require(remaining > 0, 'whole diagnostic wall budget')
            execution_env = dict(base_env if env is None else env)
            entry, stdout, stderr = auxiliary_owner.run(label, arguments, execution_env,
                wall=min(seconds, remaining), text_cap=64 << 20, file_cap=64 << 20)
            require(owner_module.completed(entry) and entry['exit_code'] == 0,
                    'auxiliary did not close normally: ' + label)
            files = [p for p in out.rglob('*') if p.is_file()]
            require(sum(p.stat().st_size for p in files) < 2*1024**3, 'auxiliary artifact budget at exit')
            require(sum(p.stat().st_size for p in files if p.suffix in ('.stderr', '.raw', '.stdout')) < 64*1024**2, 'auxiliary log budget at exit')
            return stdout.read_text(errors='replace')
    runner = Runner()
    record['health_calls'] = auxiliary_owner.commands

    def poll():
        nonlocal next_save
        g.poll(.01)
        require(time.monotonic() - start < 1000, 'wall budget')
        require(not (g.stops and g.stops[-1] != 'resumed' and g.exit_code is None and g.exit_signal is None), 'unexpected inferior stop')
        files = [p for p in out.rglob('*') if p.is_file()]
        require(sum(p.stat().st_size for p in files) < 2*1024**3, 'artifact budget')
        require(sum(p.stat().st_size for p in files if p.suffix in ('.stderr', '.raw', '.stdout')) < 64*1024**2, 'log budget')
        if g.inferior:
            status = Path('/proc', str(g.inferior['pid']), 'status')
            if status.exists():
                rss = [int(v.split()[1])*1024 for v in status.read_text().splitlines() if v.startswith('VmRSS:')]
                require(not rss or rss[0] < 104*1024**3, 'RSS budget')
        if time.monotonic() >= next_save:
            save()
            next_save = time.monotonic() + 5

    def line():
        deadline = time.monotonic() + 700
        with selectors.DefaultSelector() as selector:
            selector.register(master, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                if b'\n' in pending:
                    value, _, tail = pending.partition(b'\n')
                    pending[:] = tail
                    return value.decode().strip()
                poll()
                require(len(pending) < 1024**2, 'protocol line bound')
                if selector.select(.02):
                    try:
                        data = os.read(master, 65536)
                    except BlockingIOError:
                        continue
                    require(bool(data), 'protocol EOF')
                    raw.write(data); raw.flush(); pending.extend(data)
        raise TimeoutError('protocol deadline')

    def send(data):
        deadline = time.monotonic() + 20
        view = memoryview(data)
        while view:
            poll()
            require(time.monotonic() < deadline, 'input deadline')
            try:
                count = os.write(master, view[:8192])
            except BlockingIOError:
                continue
            require(count > 0, 'input closed')
            view = view[count:]

    save()
    try:
        os.chdir(root)
        require(not Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists(), 'existing GPU dump; do not retry/reset')
        require(runner.run('embedding-state', ['/usr/bin/systemctl', '--user', 'show', 'llama-server-qwen3embed.service', '-p', 'ActiveState'], seconds=5).strip() == 'ActiveState=inactive', 'embedding service active')
        cursor = health.journal_cursor(runner, 'kernel-before')
        health.check_health(types.SimpleNamespace(bdf='0000:05:00.0'), runner, Path('/home/yayoi/.local/bin/strata-xe-health'), cursor=cursor)
        env = health.diagnostic_environment(dict(runtime_env))
        for key in list(env):
            if key.startswith(('UNITRACE_', 'XPTI_')) or key in ('LD_PRELOAD', 'ZET_ENABLE_METRICS', 'UR_LOG_TRACING'):
                env.pop(key)
        env['UR_ENABLE_LAYERS'] = ','.join(v for v in env.get('UR_ENABLE_LAYERS', '').split(',') if v and v != 'UR_LAYER_TRACING')
        env.update(ZEL_LOADER_LOGGING_LEVEL='warn', ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',
                   UR_LOG_LOADER='level:warning;flush:warning;output:stderr', UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr',
                   NEOReadDebugKeys='1', EnableDirectSubmission='0', STRATA_PREFILL_RING='8')
        record['environment'] = dict(env)
        master, slave = os.openpty(); tty.setraw(slave); os.set_blocking(master, False)
        gdb_launch_attempted = True
        g = OwnedGdb(argv, out / 'debugger', env, inferior_tty_fd=slave)
        g.command('-gdb-set may-call-functions off'); g.run()
        record['startup'] = []
        while True:
            value = line(); record['startup'].append(value)
            require(len(record['startup']) < 1000 and not value.startswith('ERR'), 'startup failure/bound')
            if value.startswith('READY '):
                require(value.split()[1] == '262144', 'READY context'); break
        os.close(slave); slave = None
        identity = process_identity(g.inferior['pid'])
        require(identity and identity['start_ticks'] == g.inferior['start_ticks'], 'inferior start identity')
        exe = Path('/proc', str(identity['pid']), 'exe')
        require(exe.resolve() == binary and sha(exe) == record['binary_sha256'], 'actual inferior binary')
        actual = dict(v.decode().split('=', 1) for v in Path('/proc', str(identity['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
        require(actual == record['environment'], 'actual full initial environment equals explicit allowlist')
        record['actual_inferior'] = identity; record['actual_target_environment'] = dict(env)
        info = next(v for v in record['startup'] if v.startswith('INFO '))
        settings = dict(v.split('=', 1) for v in info.split()[1:] if '=' in v)
        require(all(settings.get(k) == v for k, v in {'context':'262144', 'kv_resident':'32768', 'expert_slots':'128', 'pool_workers':'5', 'spec':'4'}.items()), 'actual startup parameters')
        record['startup_vram_free_mib'] = int(settings.get('vram_free_mib', '-1'))
        require(record['startup_vram_free_mib'] >= 512, 'startup VRAM reserve below512MiB; no prompt launch')
        result = {'ids': [], 'logprobs': [], 'protocol': []}; record['request'] = result
        send(('GEN 64 ckpt=1 logprobs=5 ' + ','.join(map(str, tokens)) + '\n').encode())
        while True:
            value = line(); result['protocol'].append(value)
            require(len(result['protocol']) < 2000 and not value.startswith('ERR'), 'request failure/bound')
            if value.startswith('T '): result['ids'].append(int(value.split()[1]))
            if value.startswith('LP '): result['logprobs'].append(value)
            if value.startswith('DONE '): break
        result['validation'] = validate(result)
        reference = baseline['requests'][0]
        result['comparison'] = {k + '_equal': result[k] == reference[k] for k in ('ids', 'logprobs')}
        reference_done = next(v.split() for v in reference['protocol'] if v.startswith('DONE '))
        result['comparison'].update(finish_reason_equal=result['validation']['finish_reason'] == reference_done[5],
                                    mtp_counts_equal=result['validation']['mtp_counts'] == list(map(int, reference_done[6:8])))
        record['math_equivalence'] = False  # missing independent full head/live-state gates
        record['output_comparison_equal'] = all(result['comparison'].values())
        save(); send(b'QUIT\n')
        end = time.monotonic() + 30
        while g.exit_code is None and g.exit_signal is None and time.monotonic() < end:
            poll()
            try:
                data = os.read(master, 65536)
                if data: raw.write(data); raw.flush()
            except (BlockingIOError, OSError):
                pass
        require(g.exit_code == 0 and g.exit_signal is None, 'normal owned PID exit')
        record['completed'] = True
    except BaseException as error:
        record['error'] = repr(error)
        if g:
            try: record['failure_snapshot'] = g.snapshot('failure', resume=False)
            except BaseException as inspect: record['snapshot_error'] = repr(inspect)
    finally:
        if g:
            record.update(exit_code=g.exit_code, exit_signal=g.exit_signal)
            try: record['cleanup'] = g.close()
            except BaseException as error: record['cleanup_error'] = repr(error)
        for fd in (master, slave):
            if fd is not None:
                try: os.close(fd)
                except OSError as error: record['fd_cleanup_error'] = repr(error)
        raw.close()
        # No post-GPU auxiliary launch until OwnedGdb closure is proven.
        # Missing/failed closure is unknown ownership, never a successful stop.
        cleanup = record.get('cleanup')
        gdb_closed = (not gdb_launch_attempted or (g is not None and isinstance(cleanup, dict)
            and cleanup.get('inferior_survived') is False and cleanup.get('gdb_survived') is False
            and not record.get('cleanup_error')))
        aux_closed = (auxiliary_owner.active is None and all(
            entry.get('direct_child_reaped') and entry.get('session_empty')
            and entry.get('ownership_status') == 'observed' and not entry.get('errors')
            for entry in auxiliary_owner.commands))
        if cursor and gdb_closed and aux_closed:
            try:
                text = runner.run('kernel-after', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'], seconds=5)
                rows = [json.loads(v) for v in text.splitlines() if v.startswith('{')]
                record['new_fault_messages'] = [v['MESSAGE'] for v in rows if (((('0000:05:00.0' in v.get('MESSAGE', '') or re.search(r'\bxe\b', v.get('MESSAGE', ''))) and health.FAULT.search(v.get('MESSAGE', ''))) or ('strata' in v.get('MESSAGE', '') and 'segfault' in v.get('MESSAGE', ''))))]
            except BaseException as error: record['kernel_gate_error'] = repr(error)
        record['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == boot
        # Reconcile after the audit too: a journal utility can become unresolved.
        aux_closed = (auxiliary_owner.active is None and all(
            entry.get('direct_child_reaped') and entry.get('session_empty')
            and entry.get('ownership_status') == 'observed' and not entry.get('errors')
            for entry in auxiliary_owner.commands))
        record['owned_gdb_closed'] = gdb_closed
        record['auxiliary_ownership_closed'] = aux_closed
        record['active'] = not (gdb_closed and aux_closed)
        record['diagnostic_passed'] = bool(record['completed'] and not record['active'] and cursor and 'new_fault_messages' in record and not any(record.get(k) for k in ('error', 'kernel_gate_error', 'cleanup_error', 'fd_cleanup_error')) and not record['new_fault_messages'] and record['boot_unchanged'] and not any(record.get('cleanup', {'forced': True}).values()) and all(owner_module.completed(entry) and entry.get('exit_code') == 0 for entry in auxiliary_owner.commands))
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record['health_calls'] = runner.calls
        save(); lock.close()
    print(json.dumps({k: record.get(k) for k in ('diagnostic_passed', 'math_equivalence', 'output_comparison_equal', 'error', 'exit_code', 'exit_signal', 'cleanup')}))
    return 0 if record['diagnostic_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
