import base64, fcntl, hashlib, importlib.util, json, os
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-native-dequant-prefill-batch'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
G = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner identity')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-native-dequant-prefill-boundary-publish-root-v2'
    m.require(not out.exists(), 'immutable publication')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), model_executed=False,
             gpu_executed=False, reset_executed=False, cleanup_executed=False)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, args, expected=(0,)):
        c, so, se = owner.run(label, args, ENV, W, wall=60, cpu=120,
                             text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c['exit_code'] in expected, 'normal closed ' + label)
        return so.read_text().strip(), c
    try:
        manifest = json.loads((R / 'archive-manifest.json').read_text())
        for entry in manifest['byte_exact_copies']:
            m.require(m.sha(Path(entry['source'])) == entry['sha256'] == m.sha(W / entry['archive']), 'exact archived evidence')
        for entry in manifest['lossless_encoded_copies']:
            encoded_path = W / entry['archive']
            encoded = json.loads(encoded_path.read_text())
            decoded = base64.b64decode(encoded['data'], validate=True)
            m.require(m.sha(encoded_path) == entry['archive_sha256'] and hashlib.sha256(decoded).hexdigest() == entry['decoded_sha256'] and len(decoded) == entry['decoded_bytes'] and decoded == Path(entry['source']).read_bytes(), 'exact decoded patch')
        registry = json.loads((R / 'report-registry-v130.json').read_text())
        m.require(registry['research_completed'] == 527 and len(registry['new_reports']) == 19, 'correct incremental report count')
        for report in registry['new_reports']:
            m.require(report['main_full_report_review'] and m.sha(W / report['archive']) == report['sha256'], 'reviewed report')
        for p in (R / 'receipts').glob('*.json'):
            d = json.loads(p.read_text())
            m.require(not d['active'] and all(c['direct_child_reaped'] and c['session_empty'] and not c['survivors'] for c in d['commands']), 'closed original owners')
        for p in R.rglob('*.json'):
            json.loads(p.read_text())
        r['verified_byte_exact_copies'] = len(manifest['byte_exact_copies'])
        r['archive_manifest_sha256'] = m.sha(R / 'archive-manifest.json')
        r['retention_plan_sha256'] = m.sha(R / 'retention-v1.json')
        r['registry_sha256'] = m.sha(R / 'report-registry-v130.json')
        text, _ = run('source-status', ['/usr/bin/git', 'status', '--porcelain', '--untracked-files=normal'])
        m.require(text and all(line[3:].startswith('bench/results/2026-10-11-xe-native-dequant-prefill-batch/') for line in text.splitlines()), 'only reviewed archive changed')
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        report = out / 'gitleaks-redacted.json'
        _, scan = run('gitleaks', [str(G), 'dir', str(R), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(report)], expected=(0, 1))
        findings = json.loads(report.read_text())
        exact = {('reviews/xe-native-dequant-prefill-batch-applied-main-review-v1.json', 15, 'generic-api-key'),
                 ('research-xe-joint-matrix-packed-a-lowering-next-v1.txt', 17, 'generic-api-key')}
        observed = {(str(Path(v['File']).relative_to(R)), v['StartLine'], v['RuleID']) for v in findings}
        m.require(scan['exit_code'] == 1 and len(findings) == 2 and observed == exact, 'only two manually verified public identifiers')
        first = json.loads((R / 'reviews/xe-native-dequant-prefill-batch-applied-main-review-v1.json').read_text())
        m.require(first['api_byte_receipt_sha256'] == m.sha(B / 'xe-native-dequant-group-byte-gpu-root-v4-free-full/record.json'), 'measurement receipt identifier')
        line = (R / 'research-xe-joint-matrix-packed-a-lowering-next-v1.txt').read_text().splitlines()[16]
        m.require('b72640fff05098f1762cb3bf9c427a5f8d7db31c' in line, 'known public source commit identifier')
        r['secret_scan_review'] = dict(actual_exit_code=1, false_positives=2, identity_checks_passed=True, suppressions_added=False)
        run('stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W))])
        run('staged-diff-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'docs(sycl): record native prefill batch qualification and startup JIT failure'])
        r['committed_revision'], _ = run('committed-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push-own-fork', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        text, _ = run('final-status', ['/usr/bin/git', 'status', '--porcelain'])
        m.require(not text, 'clean published report')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ['passed', 'error', 'committed_revision', 'verified_byte_exact_copies']}))
    raise SystemExit(0 if r['passed'] else 1)
