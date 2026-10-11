import base64, fcntl, hashlib, importlib.util, json, os
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
R = W / 'bench/results/2026-10-11-xe-preload-disabled-diagnostic'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner pin')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-preload-disabled-post-retention-publish-root-v1'
    m.require(not out.exists(), 'immutable publication')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, cleanup_executed=False)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, argv, expected=(0,)):
        c, so, se = owner.run(label, argv, ENV, W, wall=60, cpu=120,
                             text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(c) and c['exit_code'] in expected, 'normal closed ' + label)
        return so.read_text().rstrip('\n'), c
    try:
        manifest = json.loads((R / 'post-retention-archive-manifest.json').read_bytes())
        historical = json.loads((R / 'archive-manifest.json').read_bytes())
        for source in [Path(__file__), B / 'xe-preload-disabled-post-retention-archive-v1.py']:
            target = R / 'recipes' / source.name
            m.require(not target.exists(), 'new post-retention recipe')
            target.write_bytes(source.read_bytes())
            manifest['byte_exact_copies'].append(dict(source=str(source), archive=str(target.relative_to(W)),
                                                      bytes=target.stat().st_size, sha256=m.sha(target)))
        (R / 'post-retention-archive-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        for entry in manifest['byte_exact_copies'] + historical['byte_exact_copies']:
            m.require(m.sha(Path(entry['source'])) == entry['sha256'] == m.sha(W / entry['archive']), 'exact archive input')
        for entry in historical['lossless_encoded_copies']:
            target = W / entry['archive']
            encoded = json.loads(target.read_bytes())
            decoded = base64.b64decode(encoded['data'], validate=True)
            m.require(m.sha(target) == entry['archive_sha256'] and len(decoded) == entry['decoded_bytes'] and
                      hashlib.sha256(decoded).hexdigest() == entry['decoded_sha256'] and
                      decoded == Path(entry['source']).read_bytes(), 'lossless source/patch identity')
        for p in R.rglob('*.json'):
            json.loads(p.read_bytes())
        registry = json.loads((R / 'report-registry-v132.json').read_bytes())
        m.require(registry['research_completed'] == 547 and len(registry['new_reports']) == 2, 'correct report count')
        for entry in registry['new_reports']:
            archived = W / entry['archive']
            if entry.get('archive_encoding') == 'base64':
                data = base64.b64decode(json.loads(archived.read_bytes())['data'], validate=True)
                m.require(m.sha(archived) == entry['archive_sha256'] and hashlib.sha256(data).hexdigest() == entry['sha256'], 'lossless original report identity')
            else:
                m.require(m.sha(archived) == entry['sha256'], 'exact original report identity')
            m.require(entry['main_full_report_review'], 'reviewed report')
        plan = json.loads((R / 'retention-executed-v1.json').read_bytes())
        m.require(plan['status'] == 'EXECUTED' and len(plan['files']) == 402 and all(v['unlink_status'] == 'REMOVED' and not Path(v['path']).exists() for v in plan['files']), 'actual completed exact retention')
        m.require(plan['original_plan_sha256'] == m.sha(R / 'retention-v1.json'), 'immutable original plan preserved')
        r['verified_byte_exact_copies'] = len(manifest['byte_exact_copies']) + len(historical['byte_exact_copies'])
        r['lossless_encoded_copies'] = len(historical['lossless_encoded_copies'])
        r['post_retention_manifest_sha256'] = m.sha(R / 'post-retention-archive-manifest.json')
        r['historical_manifest_sha256'] = m.sha(R / 'archive-manifest.json')
        r['retention_executed_sha256'] = m.sha(R / 'retention-executed-v1.json')
        text, _ = run('identity-origin', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(all(line.startswith('file:/home/yayoi/.config/git/config') for line in text.splitlines()), 'global identity only')
        text, _ = run('identity-name', ['/usr/bin/git', 'config', '--global', 'user.name'])
        m.require(text == 'yqYoi', 'expected global author')
        text, _ = run('identity-email', ['/usr/bin/git', 'config', '--global', 'user.email'])
        m.require(text == '52419113+yqYo1@users.noreply.github.com', 'expected global email')
        text, _ = run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin'])
        m.require(text == 'git@github.com:yqYo1/Strata.git', 'own fork SSH remote')
        text, _ = run('source-status', ['/usr/bin/git', 'status', '--porcelain', '--untracked-files=normal'])
        m.require(text and all(line[3:].startswith(str(R.relative_to(W)) + '/') for line in text.splitlines()), 'only archive changes')
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        g = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks')
        report = out / 'gitleaks-redacted.json'
        _, scan = run('gitleaks-final-archive', [str(g), 'dir', str(R), '--no-banner', '--redact',
                          '--report-format', 'json', '--report-path', str(report)])
        m.require(scan['exit_code'] == 0 and json.loads(report.read_bytes()) == [], 'actual scanner zero findings')
        r['secret_scan'] = dict(actual_exit_code=0, findings=0, redacted_report_sha256=m.sha(report), suppressions_added=False)
        run('stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W))])
        run('staged-diff-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'docs(sycl): record closed-log retirement and timing review'])
        r['committed_revision'], _ = run('committed-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push-own-fork', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        text, _ = run('final-status', ['/usr/bin/git', 'status', '--porcelain'])
        m.require(not text, 'clean published evidence')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ['passed','error','committed_revision','verified_byte_exact_copies','secret_scan']}))
    raise SystemExit(0 if r['passed'] else 1)
