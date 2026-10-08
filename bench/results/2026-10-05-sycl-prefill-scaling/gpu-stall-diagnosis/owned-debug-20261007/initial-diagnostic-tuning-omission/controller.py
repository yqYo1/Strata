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

out = base / 'full-context-owned-debug'
out.mkdir(mode=0o700)
probes = out / 'probes'; probes.mkdir()
src = root / 'sycl/tools/recover-xe.sh'
m = types.ModuleType('owned_readonly')
exec(compile(src.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(src), 'exec'), m.__dict__)
r = m.Runner(probes)
record = {'scope': 'Actual 262144-cell CLI diagnostic with owned GDB, Level Zero and UR tracing; 600-second observation deadline, not a timing run or completed capacity proof',
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'healthy': False, 'completed': False, 'active': True,
          'diagnostic_deadline_seconds': 600, 'log_limit_bytes': 2 * 1024**3,
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
    health = json.loads((base / 'post-capacity-timeout-health/record.json').read_text())
    assert health['healthy'] and health['boot_id'] == record['boot_id']
    prior = json.loads((base / 'full-context-copy-off/cli-supervisor/record.json').read_text())
    assert prior.get('finished_utc') and prior['active_stage'] is None
    assert not prior.get('new_fault_messages') and not Path('/proc/428740').exists()
    for p in base.rglob('stalled-writer.json'):
        stale = json.loads(p.read_text()); current = process_identity(stale['pid'])
        assert not current or current['start_ticks'] != stale['start_ticks'] or current['state'] == 'Z', str(p)
    exe = base / 'strata-residency-trace-candidate'
    record['binary_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
    assert record['binary_sha256'] == '3a83e2c0c8b916225669275217b9a194ebabbe6101faf0ea7a110913d061e9b9'
    argv = json.loads((base / 'full-context-copy-off/original-cli-argv.json').read_text())
    argv[0] = str(exe)
    original = Path(argv[argv.index('--tokens-file') + 1])
    fixture = out / 'fills-context.tokens.txt'; fixture.write_bytes(original.read_bytes())
    argv[argv.index('--tokens-file') + 1] = str(fixture)
    assert len(fixture.read_text().split()) == 262142
    record['fixture_sha256'] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    head = out / 'first-head.bin'
    env = m.diagnostic_environment(json.loads((base / 'full-context-copy-off/environment.json').read_text()))
    env['STRATA_DUMP_FIRST_LOGITS'] = str(head)
    record['environment'] = {k: v for k, v in env.items() if k.startswith(('STRATA_', 'SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_')) or k in ('LD_LIBRARY_PATH', 'NEOReadDebugKeys', 'EnableDirectSubmission')}
    record['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), src, root/'sycl/tools/owned_gdb.py', root/'sycl/src/program/generate.cpp']}
    record['argv'] = argv
    save()
    assert r.run('embedding-state', ['/usr/bin/systemctl', '--user', 'show', 'llama-server-qwen3embed.service', '-p', 'ActiveState'], seconds=5).strip() == 'ActiveState=inactive'
    cursor = m.journal_cursor(r, 'kernel-before')
    g = OwnedGdb(argv, out / 'debugger', env)
    g.command('-gdb-set debug-file-directory /usr/lib/debug:/home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/runtime-symbols/extracted/usr/lib/debug')
    g.run()
    started = time.monotonic(); next_metric = started; idle_start = started
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
                signature = (current['minor_faults'], current['major_faults'], current['io']['rchar'], current['io']['read_bytes'], json.dumps([v['busy_cycles'] for v in current['gpu']], sort_keys=True))
                if signature != previous_signature:
                    idle_start = time.monotonic(); previous_signature = signature
                record['unchanged_progress_seconds'] = time.monotonic() - idle_start
                if current['gpu'] and record['unchanged_progress_seconds'] >= 60 and snapshots < 2 and time.monotonic() - last_snapshot >= 30:
                    # Only pause after observed lack of device/read/fault progress.
                    # Debugger pauses can affect host-dependent device execution.
                    record['snapshots'].append(g.snapshot('unchanged-progress-' + str(snapshots)))
                    snapshots += 1; last_snapshot = time.monotonic()
                next_metric = time.monotonic() + 5
                save()
    record['exit_code'] = g.exit_code
    record['exit_signal'] = g.exit_signal
    if g.exit_code == 0:
        text = (out / 'debugger/gdb-mi.stdout').read_text(errors='replace') + (out / 'debugger/engine-and-gdb.stderr').read_text(errors='replace')
        match = re.search(r'^output\s*:\s*(.*)$', text, re.M)
        assert match
        record['ids'] = list(map(int, match[1].split())); assert len(record['ids']) == 2
        assert 'prefill 262141 tokens' in text
        values = array.array('f'); values.frombytes(head.read_bytes())
        assert len(values) == 248320 and all(map(math.isfinite, values))
        record['head_sha256'] = hashlib.sha256(head.read_bytes()).hexdigest()
        record['verify_windows'] = [(int(p), int(n)) for p, n in re.findall(r'strata trace: window (-?\d+) (-?\d+)', text)]
        assert record['verify_windows'] == [(262141, 1), (262142, 2)]
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
