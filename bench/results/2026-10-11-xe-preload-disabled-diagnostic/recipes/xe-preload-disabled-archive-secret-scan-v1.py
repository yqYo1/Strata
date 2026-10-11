import fcntl, importlib.util, json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-preload-disabled-diagnostic'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner pin')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-preload-disabled-archive-secret-scan-root-v1'
    m.require(not out.exists(), 'immutable scan root')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, cleanup_executed=False, scope='Actual redacted scanner result; manual review remains separate')
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    try:
        g = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
        report = out / 'gitleaks-redacted.json'
        c, so, se = owner.run('gitleaks', [str(g), 'dir', str(R), '--no-banner', '--redact',
                             '--report-format', 'json', '--report-path', str(report)], ENV, W,
                             wall=60, cpu=120, text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c['exit_code'] in [0, 1], 'normal closed scanner')
        findings = json.loads(report.read_bytes())
        r['actual_scanner_exit_code'] = c['exit_code']
        r['finding_count'] = len(findings)
        r['redacted_report_sha256'] = m.sha(report)
        r['public_finding_locations'] = [dict(file=v['File'], line=v['StartLine'], rule=v['RuleID']) for v in findings]
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ['passed','error','actual_scanner_exit_code','finding_count','public_finding_locations']}))
    raise SystemExit(0 if r['passed'] else 1)
