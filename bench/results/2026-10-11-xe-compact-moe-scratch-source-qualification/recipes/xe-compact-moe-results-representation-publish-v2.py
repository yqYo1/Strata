"""Main: preserve a literal source patch losslessly and publish reviewed evidence."""
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
S = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
R = W / 'bench/results/2026-10-11-xe-compact-moe-scratch-source-qualification'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
assert hashlib.sha256(O.read_bytes()).hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
spec = importlib.util.spec_from_file_location('representation_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-results-representation-publish-root-v2'
    m.require(not out.exists(), 'new immutable publication')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             recipe_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False,
             model_executed=False, original_results_modified=False)

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands

    def run(label, argv, cwd=W):
        c, so, _ = owner.run(label, argv, ENV, cwd, wall=60, cpu=120, text_cap=8 << 20,
                             total_cap=32 << 20, rss_cap=2 << 30, file_cap=64 << 20)
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] == 0, 'closed ' + label)
        return so.read_text().strip()

    save()
    try:
        prior_path = B / 'xe-compact-moe-results-publish-root-v1/record.json'
        prior = json.loads(prior_path.read_text())
        m.require(prior['passed'] is False and prior['complete'] is False and prior['active'] is False
                  and prior['error'] == "RuntimeError('closed staged-check')"
                  and all(m.closed(c) and c.get('session_anchored') is True for c in prior['commands']),
                  'original failed publication fully closed')
        m.require(run('initial-head', ['/usr/bin/git', 'rev-parse', 'HEAD']) ==
                  '3dc65b994ce09d75b9a96d077a265d3d846cb389', 'unchanged report base')
        status = run('initial-status', ['/usr/bin/git', 'status', '--porcelain'])
        allowed = {str((R / p).relative_to(W)) for p in []}
        for line in status.splitlines():
            path = line[3:]
            m.require(path.startswith(str(R.relative_to(W)) + '/') or
                      path == 'bench/results/2026-10-11-upstream-parameter-audit/TODO.md', 'only expected staged boundary')
        identity = run('identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(identity and all(v.startswith('file:/home/yayoi/.config/git/config') for v in identity.splitlines()), 'global identity')
        m.require(run('origin', ['/usr/bin/git', 'remote', 'get-url', '--push', 'origin']) ==
                  'git@github.com:yqYo1/Strata.git', 'own fork only')
        m.require(run('source-head', ['/usr/bin/git', 'rev-parse', 'HEAD'], S) ==
                  '39e395a103af36323262cdc0dbaea27cba81f3e4' and
                  not run('source-status', ['/usr/bin/git', 'status', '--porcelain'], S), 'qualified source unchanged')
        manifest_path = R / 'source-and-binaries.json'
        manifest = json.loads(manifest_path.read_text())
        for item in manifest['artifacts']:
            m.require(m.sha(Path(item['source'])) == item['sha256'] and
                      m.sha(W / item['archive']) == item['sha256'], 'actual byte-identical preexisting archive')
        patch = R / 'source.patch'
        data = patch.read_bytes()
        source = B / 'xe-compact-moe-results-publish-root-v1/commands/published-source-diff.stdout'
        m.require(data == source.read_bytes(), 'original exact source patch')
        target = R / 'source.patch.lossless.json'
        m.require(not target.exists(), 'new lossless representation')
        packed = dict(encoding='base64', original_path=str(source), decoded_bytes=len(data),
                      decoded_sha256=m.sha(source), data_base64=base64.b64encode(data).decode('ascii'))
        target.write_text(json.dumps(packed, indent=2) + '\n')
        decoded = base64.b64decode(json.loads(target.read_text())['data_base64'], validate=True)
        m.require(decoded == data and hashlib.sha256(decoded).hexdigest() == packed['decoded_sha256'], 'lossless verification')
        for item in manifest['artifacts']:
            if item['archive'] == str(patch.relative_to(W)):
                item.update(archive=str(target.relative_to(W)), archive_encoding='base64-json',
                            archive_sha256=m.sha(target), decoded_bytes=len(data), decoded_sha256=m.sha(source))
        manifest['publication_representation_correction'] = dict(original_publication_passed=False,
            original_receipt_sha256=m.sha(prior_path), reason='Literal git patch has a context blank line containing one space; staged whitespace check rejected the archive, not implementation source.',
            original_source_capture_retained=True, source_or_results_changed=False)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        failed_copy = R / 'receipts/xe-compact-moe-results-publish-root-v1.json'
        m.require(not failed_copy.exists(), 'new failed-publication receipt')
        failed_copy.write_bytes(prior_path.read_bytes())
        m.require(m.sha(failed_copy) == m.sha(prior_path), 'preserved original FAILED')
        recipe_copy = R / 'recipes' / Path(__file__).name
        recipe_copy.write_bytes(Path(__file__).read_bytes())
        m.require(m.sha(recipe_copy) == m.sha(__file__), 'exact correction recipe')
        patch.unlink()  # Only uncommitted duplicate representation; original capture retained.
        readme = R / 'README.md'
        readme.write_text(readme.read_text() + '\n[Source diff](source.patch.lossless.json) preserves the exact original patch as base64 JSON.\nThe original publication failed only on the patch archive context-line whitespace and stays FAILED.\nNo source, command result, or research report was rewritten to correct that representation.\n')
        scan = out / 'gitleaks-redacted.json'
        run('secret-scan', [str(S.parent / 'fix-xestrata-e32-host-handoff-20261011/.lint/bin/gitleaks'),
            'dir', str(R), '--no-banner', '--redact', '--report-format', 'json', '--report-path', str(scan)])
        m.require(json.loads(scan.read_text()) == [], 'no credentials archived')
        run('stage', ['/usr/bin/git', 'add', '--', str(R.relative_to(W)),
                      'bench/results/2026-10-11-upstream-parameter-audit/TODO.md'])
        run('staged-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'docs(sycl): record compact MoE qualification and upstream tuning audit'])
        r['committed_revision'] = run('head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('push-own-fork', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean published boundary')
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed','error','committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
