"""One bounded 262144-cell diagnostic, with an owned debugger and API logs.

This is not a timing run or an automatic recovery/reset. A finite diagnostic
deadline is not proof of a workload hang or a successful capacity check.
"""
from pathlib import Path
import array
import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
import types

root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
base = Path(__file__).parent
sys.path.insert(0, str(root / 'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity

out = base / 'full-context-layer-trace-csr'
out.mkdir(mode=0o700)
probes = out / 'probes'; probes.mkdir()
src = root / 'sycl/tools/recover-xe.sh'
m = types.ModuleType('owned_readonly')
exec(compile(src.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(src), 'exec'), m.__dict__)
r = m.Runner(probes)
record = {'scope': 'Actual 262144-cell CLI correctness run with owned GDB, Level Zero warnings/parameter validation and layer progress; 7200-second deadline and read-only stalled-submission inspection; original tuning; not a throughput run',
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'healthy': False, 'completed': False, 'active': True,
          'diagnostic_deadline_seconds': 7200, 'log_limit_bytes': 512 * 1024**2,
          'snapshots': [], 'steps': []}

def save():
    record['steps'] = r.calls
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

metric_source = out / 'read_metrics.py'
metric_source.write_text('''from pathlib import Path
import json,re,sys
p=Path('/proc')/sys.argv[1]
s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
assert int(f[19])==int(sys.argv[2])
io={k:int(v.strip()) for k,v in (line.split(':',1) for line in (p/'io').read_text().splitlines())}
gpu=[]
for q in (p/'fdinfo').iterdir():
 t=q.read_text()
 if re.search(r'^drm-pdev:\\s*0000:05:00.0\\s*$',t,re.M):
  gpu.append({'fd':q.name,'busy_cycles':{k:int(v) for k,v in re.findall(r'^drm-cycles-(\\w+):\\s*(\\d+)',t,re.M)},'raw':t})
print(json.dumps({'pid':int(sys.argv[1]),'start_ticks':int(f[19]),'state':f[0],'minor_faults':int(f[7]),'major_faults':int(f[9]),'user_ticks':int(f[11]),'system_ticks':int(f[12]),'io':io,'gpu':gpu}))
''')

g = None
cursor = None
try:
    health = json.loads((base / 'owned-layer-trace-short/record.json').read_text())
    assert health['healthy'] and health['completed'] and health['boot_id'] == record['boot_id']
    for key in ['inferior','debugger']:
        assert not process_identity(health[key]['pid'])
    assert not health['cleanup']['inferior_survived'] and not health['cleanup']['gdb_survived']
    owned = json.loads((base/'full-context-entry-capacity/record.json').read_text())
    assert not owned['active'] and not owned['new_fault_messages']
    assert not owned['cleanup']['inferior_survived'] and not owned['cleanup']['gdb_survived']
    for key in ['inferior', 'debugger']:
        assert not Path('/proc',str(owned[key]['pid'])).exists()
    prior = json.loads((base / 'full-context-copy-off/cli-supervisor/record.json').read_text())
    assert prior.get('finished_utc') and prior['active_stage'] is None
    assert not prior.get('new_fault_messages') and not Path('/proc/428740').exists()
    for p in base.rglob('stalled-writer.json'):
        stale = json.loads(p.read_text()); current = process_identity(stale['pid'])
        assert not current or current['start_ticks'] != stale['start_ticks'] or current['state'] == 'Z', str(p)
    exe = base / 'strata-prefill-layer-trace-candidate'
    record['binary_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
    assert record['binary_sha256'] == 'b81a7d6fbc1c6d524cf3b64e196f5866c91ec56c76e460a66b63b28ce3aed461'
    argv = json.loads((base / 'full-context-copy-off/original-cli-argv.json').read_text())
    argv[0] = str(exe)
    original = Path(argv[argv.index('--tokens-file') + 1])
    fixture = out / 'fills-context.tokens.txt'; fixture.write_bytes(original.read_bytes())
    argv[argv.index('--tokens-file') + 1] = str(fixture)
    assert len(fixture.read_text().split()) == 262142
    record['fixture_sha256'] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    head = out / 'first-head.bin'
    baseline_env = json.loads((base / 'full-context-copy-off/environment.json').read_text())
    original_summary = json.loads((base / 'full-context-copy-off/cli-262144-strata-residency-candidate-draft-lease-verified/summary.json').read_text())
    baseline_env.update(original_summary['env'])
    assert baseline_env['STRATA_PREFILL_LAYER_MAJOR'] == '1' and baseline_env['STRATA_PREFILL_FIRST'] == '0'
    env = m.diagnostic_environment(baseline_env)
    # This exact binary already passed the short complete API-trace check.
    # Keep validation, warnings and layer progress for the full-capacity run;
    # verbose entry logs made the earlier diagnostic much more intrusive.
    env['UR_ENABLE_LAYERS'] = ','.join(x for x in env['UR_ENABLE_LAYERS'].split(',') if x != 'UR_LAYER_TRACING')
    for key in ['UR_LOG_TRACING', 'UR_LOG_LOADER', 'UR_LOG_LEVEL_ZERO']:
        env[key] = 'level:warning;flush:warning;output:stderr'
    env['ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT'] = '0'
    env['ZEL_LOADER_LOGGING_LEVEL'] = 'warn'
    env['ZEL_LOADER_LOG_PATTERN'] = '[thread-id:%t] [%l] %v'
    record['diagnostic_profile'] = 'Level Zero warning/error logging with parameter validation and Strata layer/chunk progress; verbose API-entry tracing already checked on this exact binary in real short normal-MTP comparison; parent metrics retain elapsed time; not a throughput run'
    record['capacity_deadline_seconds'] = 7200
    record['diagnostic_only_deadline_seconds'] = 7200
    record['original_tuning_environment_matches'] = all(env[k] == v for k, v in original_summary['env'].items() if k != 'STRATA_TRACE')
    assert record['original_tuning_environment_matches']
    env['STRATA_DUMP_FIRST_LOGITS'] = str(head)
    record['environment'] = {k: v for k, v in env.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')) or k in ('LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission')}
    record['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), base/'read-csr.gdb', src, root/'sycl/tools/owned_gdb.py', root/'sycl/src/program/generate.cpp',root/'sycl/src/prefill/prefill.cpp']}
    record['argv'] = argv
    save()
    assert r.run('embedding-state', ['/usr/bin/systemctl', '--user', 'show', 'llama-server-qwen3embed.service', '-p', 'ActiveState'], seconds=5).strip() == 'ActiveState=inactive'
    short_cursor = next(v['argv'][v['argv'].index('--after-cursor') + 1]
                        for v in health['steps'] if v['label'] == 'kernel-after')
    gap = r.run('kernel-gap', ['/usr/bin/journalctl', '-k', '--after-cursor', short_cursor, '--no-pager', '-o', 'json'])
    gap_rows = [json.loads(s) for s in gap.splitlines() if s.startswith('{')]
    record['preceding_fault_messages'] = [v.get('MESSAGE', '') for v in gap_rows
        if ('xe' in v.get('MESSAGE', '') or '0000:05:00.0' in v.get('MESSAGE', ''))
        and m.FAULT.search(v.get('MESSAGE', ''))]
    assert not record['preceding_fault_messages']
    cursor = m.journal_cursor(r, 'kernel-before')
    g = OwnedGdb(argv, out / 'debugger', env)
    g.command('-gdb-set debug-file-directory /usr/lib/debug:/home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/runtime-symbols/extracted/usr/lib/debug')
    g.command('-gdb-set may-call-functions off')
    g.run()
    started = time.monotonic(); next_metric = started; idle_start = started; stalled = False
    previous_signature = None; last_snapshot = 0; snapshots = 0
    with (out / 'metrics.jsonl').open('w') as metrics:
        while g.exit_code is None and g.exit_signal is None:
            g.poll(.05)
            if g.exit_code is not None or g.exit_signal is not None:
                break
            record['inferior'] = g.inferior
            record['debugger'] = g.debugger_identity
            record['elapsed_seconds'] = time.monotonic() - started
            if g.stops and g.stops[-1] != 'resumed':
                record['snapshots'].append(g.snapshot('first-stop', resume=False))
                record['stop_reason'] = 'inferior signal or unexpected debugger stop'
                break
            total_log = sum(p.stat().st_size for p in (out / 'debugger').glob('*.stderr'))
            if total_log >= record['log_limit_bytes']:
                record['snapshots'].append(g.snapshot('log-limit'))
                record['stop_reason'] = 'diagnostic log size limit'
                break
            if time.monotonic() - started >= record['diagnostic_deadline_seconds']:
                record['snapshots'].append(g.snapshot('observation-deadline'))
                record['stop_reason'] = 'finite diagnostic observation deadline; not proof of hang'
                break
            control = out / 'control.json'
            if control.exists():
                request = json.loads(control.read_text()); control.unlink()
                if request['action'] == 'stop':
                    record['snapshots'].append(g.snapshot('requested-stop', resume=False))
                    record['stop_reason'] = request.get('reason', 'explicit diagnostic stop request')
                    break
                assert request['action'] == 'snapshot'
                record['snapshots'].append(g.snapshot('requested-' + str(len(record['snapshots']))))
                save()
            if g.inferior and time.monotonic() >= next_metric:
                label = 'metrics-' + str(len(r.calls))
                current = json.loads(r.run(label, ['/usr/bin/python3', str(metric_source), str(g.inferior['pid']), str(g.inferior['start_ticks'])], seconds=2))
                current['elapsed_seconds'] = time.monotonic() - started
                metrics.write(json.dumps(current) + '\n'); metrics.flush()
                record['latest_metrics'] = current
                signature = (total_log, current['minor_faults'], current['major_faults'], current['io']['rchar'], current['io']['read_bytes'])
                if signature != previous_signature:
                    idle_start = time.monotonic(); previous_signature = signature
                record['unchanged_progress_seconds'] = time.monotonic() - idle_start
                if current['gpu'] and record['unchanged_progress_seconds'] >= 60 and snapshots < 4 and time.monotonic() - last_snapshot >= 30:
                    # Only pause after observed lack of device/read/fault progress.
                    # Debugger pauses can affect host-dependent device execution.
                    snap = g.snapshot('unchanged-api-and-io-' + str(snapshots), resume=False)
                    if snap:
                        start = g.raw.tell()
                        g.command('-interpreter-exec console ' + json.dumps('source ' + str(base/'read-csr.gdb')))
                        g.raw.flush()
                        dump = out/'debugger'/('csr-' + str(snapshots) + '.mi.txt')
                        with (out/'debugger/gdb-mi.stdout').open('rb') as source, dump.open('wb') as dest:
                            source.seek(start)
                            import shutil
                            shutil.copyfileobj(source, dest)
                        snap['read_only_csr_dump'] = str(dump)
                        decoded = ''
                        for raw_line in Path(snap['path']).read_text().splitlines():
                            if raw_line.startswith('~'):
                                try: decoded += json.loads(raw_line[1:])
                                except ValueError: pass
                        main = re.search(r'^Thread 1 \(.*?\n(.*?)(?:\n\n|\Z)', decoded, re.M | re.S)
                        counter = re.search(r'NEO::IoctlHelperXe::execBuffer[^\n]*counterValue=(\d+)', main[1]) if main else None
                        snap['submission_counter'] = int(counter[1]) if counter else None
                        snap['syscall_eagain'] = bool(re.search(r'^rax\s+.*?\s-11\s*$',decoded,re.M))
                        previous = record['snapshots'][-1] if record['snapshots'] else None
                        stalled = bool(previous and previous.get('syscall_eagain') and snap['syscall_eagain'] and snap['submission_counter'] is not None and previous.get('submission_counter') == snap['submission_counter'])
                        if not stalled and 'signal-name="SIGINT"' in snap['stop']:
                            g.command('-exec-continue --all')
                            snap['resumed'] = True
                            g.stops.append('resumed')
                    record['snapshots'].append(snap)
                    snapshots += 1; last_snapshot = time.monotonic()
                    if snap and stalled:
                        record['stop_reason'] = 'Repeated EAGAIN at the same host submission counter after unchanged warning/progress log and I/O; owned process stopped for investigation'
                        break
                next_metric = time.monotonic() + 5
                save()
    record['exit_code'] = g.exit_code
    record['exit_signal'] = g.exit_signal
    if g.exit_code == 0:
        raw_text = (out / 'debugger/gdb-mi.stdout').read_text(errors='replace')
        decoded = []
        for raw_line in raw_text.splitlines():
            if raw_line.startswith(('~','@')):
                try: decoded.append(json.loads(raw_line[1:]))
                except ValueError: pass
        text = raw_text + '\n' + ''.join(decoded) + '\n' + (out / 'debugger/engine-and-gdb.stderr').read_text(errors='replace')
        match = re.search(r'^output\s*:\s*(.*)$', text, re.M)
        assert match
        record['ids'] = list(map(int, match[1].split())); assert len(record['ids']) == 2
        assert 'prefill 262141 tokens' in text
        values = array.array('f'); values.frombytes(head.read_bytes())
        assert len(values) == 248320 and all(map(math.isfinite, values))
        record['head_sha256'] = hashlib.sha256(head.read_bytes()).hexdigest()
        record['verify_windows'] = [(int(p), int(n)) for p, n in re.findall(r'strata trace: window (-?\d+) (-?\d+)', text)]
        assert record['verify_windows'] == [(262141, 1), (262142, 2)]
        record['prefill_layer_ranges'] = [list(map(int,x)) for x in re.findall(r'strata trace: prompt chunk \d+ of \d+, layers \[(\d+), (\d+)\)', text)]
        assert record['prefill_layer_ranges']
        record.update(completed=True, last_executed_kv_cell=262143)
except BaseException as e:
    record['error'] = repr(e)
finally:
    if g:
        record['cleanup'] = g.close()
    if cursor:
        try:
            text = r.run('kernel-after', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
            rows = [json.loads(s) for s in text.splitlines() if s.startswith('{')]
            (out/'kernel.json').write_text(json.dumps(rows, indent=2)+'\n')
            record['new_fault_messages'] = [v.get('MESSAGE', '') for v in rows if ('xe' in v.get('MESSAGE', '') or '0000:05:00.0' in v.get('MESSAGE', '')) and m.FAULT.search(v.get('MESSAGE', ''))]
        except BaseException as e:
            record['kernel_error'] = repr(e)
    record['active'] = False
    record['healthy'] = bool(record['completed'] and not record.get('error') and not record.get('kernel_error') and not record.get('new_fault_messages') and not any(record.get('cleanup', {}).values()))
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps(record, indent=2))
raise SystemExit(0 if record['healthy'] else 1)
