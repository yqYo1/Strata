#!/usr/bin/env python3
"""Independently close a failed capped trace; never changes its failed status."""
import fcntl, hashlib, importlib.util, json, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
BOOT = 'cfda0b6a-7c13-4fe2-bdf5-8f9202770840'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')

def pin(path):
    p = Path(path)
    with p.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest)

def main():
    root = B / 'main-pr6-model-trace-stop-review-v1'
    dst = A / 'own-pr6-model-trace-stop-review-v1.json'
    assert not root.exists() and not dst.exists()
    old_path = A / 'own-pr6-model-small-v2.json'
    old = json.loads(old_path.read_text())
    assert not old['passed'] and not old['complete'] and not old['active']
    owner_path = A.parents[2] / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_path)['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('trace_stop_review_owner', owner_path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    checks = []
    for c in old['commands']:
        # The capped command's normal observation is deliberately incomplete.
        # Verify physical closure separately; this cannot qualify that command.
        assert c['ownership_status'] == 'observed' and c['direct_child_reaped'] and c['session_empty'] and not c['survivors'] and not c['errors']
        for ident in c['owners'] + [c['direct_identity']]:
            now = m.identity(ident['pid'])
            assert now is None or now['start_ticks'] != ident['start_ticks']
            checks.append(dict(old=ident, current=now))
    model, = [c for c in old['commands'] if c['label'] == 'model-small-0']
    assert model['exit_code'] == -25 and model['error'] == "RuntimeError('combined log cap')"
    stderr = B / 'main-pr6-model-small-v2/owned/model-small-0.stderr'
    assert stderr.stat().st_size == 64 << 20
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); root.mkdir(mode=0o700)
        rec = dict(schema='strata-pr6-model-trace-stop-review-v1', passed=False, complete=False,
            active=False, commands=[], GPU_executed=False, original_receipt=pin(old_path),
            original_failed_status_unchanged=True, original_native_exit=-25,
            original_error=model['error'], direct_reaped_and_sessions_empty=True,
            original_normal_observation_complete=model['observation_complete'],
            old_PID_start_absence=checks, full_trace=pin(stderr),
            cause='Full successful UR traffic reached the prelaunch64MiB text/file cap. Native SIGXFSZ and Main cleanup; original followup journal hit same aggregate root cap. Separate root closes observation only.',
            measurement_qualified=False, model_quality_qualified=False,
            next='Use fixed preexec parameter-validation/warning logs plus existing STRATA_TRACE/progress, with all timing excluded from performance claims.')
        owner = None
        def persist():
            rec['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(rec, indent=2) + '\n')
        owner = m.Owner(root / 'owned', persist); rec['commands'] = owner.commands
        try:
            env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
            argv = ['/usr/bin/journalctl', '-k', '-b', '--after-cursor', old['journal_before_cursor'], '--no-pager', '-o', 'json', '--show-cursor']
            c, out, err = owner.run('fault-window', argv, env, A.parents[2], wall=30,
                cpu=60, rss_cap=1 << 30, text_cap=16 << 20, file_cap=16 << 20, total_cap=32 << 20)
            assert m.closed(c) and c['exit_code'] == 0
            raw = out.read_text(); rows = [json.loads(x) for x in raw.splitlines() if x.startswith('{')]
            assert all(x.get('_BOOT_ID') == BOOT.replace('-', '') for x in rows)
            rec.update(journal_before_cursor=old['journal_before_cursor'], journal_after_cursor=re.search(r'-- cursor: (\S+)', raw)[1],
                journal_rows=len(rows), journal_output=pin(out), journal_stderr=err.read_text(),
                fault_filter=FAULT.pattern, fresh_faults=[x for x in rows if FAULT.search(x.get('MESSAGE', ''))],
                boot_unchanged=Path('/proc/sys/kernel/random/boot_id').read_text().strip() == BOOT,
                driver_dump_entries=[str(p) for p in Path('/sys/bus/pci/devices/0000:05:00.0').glob('devcoredump*')])
            assert rec['boot_unchanged'] and not rec['fresh_faults'] and not rec['driver_dump_entries']
            rec.update(passed=True, complete=True, physical_ownership_closed=True)
        except BaseException as e:
            import traceback
            rec.update(error=repr(e), traceback=traceback.format_exc())
        finally:
            rec['ownership_closed'] = owner.active is None and all(m.closed(c) for c in owner.commands)
            persist(); dst.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(receipt=str(dst), passed=rec['passed'], physical_ownership_closed=rec.get('physical_ownership_closed'),
            fresh_fault_count=len(rec.get('fresh_faults', [])), original_model_run_still_failed=True)))
        return 0 if rec['passed'] and rec['ownership_closed'] else 1

if __name__ == '__main__': sys.exit(main())
