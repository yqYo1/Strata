import fcntl
import importlib.util
import json
import os
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-prefill-route-census-source-publish-root-v1'
    m.require(not out.exists(), 'immutable output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], gpu_executed=False,
             model_executed=False, started_utc=m.utc(), controller_sha256=m.sha(__file__),
             base_revision='9c0b291d85adecb6c1dc489c3e9965b82226b799', source_tree=str(W),
             scope='Publish optional host diagnostics after both modes release/host sanitizer and actual baseline/current mandatory lints; no performance adoption')
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()
    def run(label, argv, wall=60):
        e, so, se = owner.run(label, argv, ENV, W, wall=wall, cpu=int(wall)*2,
                             text_cap=2 << 20, total_cap=8 << 20, rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0, 'closed ' + label)
        return so.read_text().strip()
    try:
        m.require(run('head', ['/usr/bin/git', 'rev-parse', 'HEAD']) == r['base_revision'], 'qualified base')
        m.require(run('changed', ['/usr/bin/git', 'diff', '--name-only', 'HEAD']) == 'src/prefill/prefill.cpp', 'only optional prefill diagnostics')
        m.require(not run('untracked', ['/usr/bin/git', 'ls-files', '--others', '--exclude-standard']), 'no unrelated untracked files')
        review = B / 'xe-prefill-route-census-applied-main-review-v1.json'
        v = json.loads(review.read_text())
        m.require(v['main_reviewed'] and v['applied'], 'combined Main review')
        r['main_review_sha256'] = m.sha(review)
        r['qualifications'] = {}
        pins = None
        for mode in ('free', 'contrib-llvm'):
            for kind, version in [('full-build', 15), ('asan-full-build', 12)]:
                p = B / f'xe-e32-{mode}-{kind}-root-v{version}/record.json'
                a = json.loads(p.read_text())
                m.require(a['passed'] and a['complete'] and not a['active'] and all(m.closed(c) for c in a['commands']), 'closed qualification')
                m.require(all(m.sha(W/n) == h for n,h in a['source_pins'].items()), 'qualified source unchanged')
                if pins is None:
                    pins = a['source_pins']
                m.require(pins == a['source_pins'], 'same all-mode source')
                m.require(m.sha(Path(a['binary']['path'])) == a['binary']['sha256'], 'qualified binary unchanged')
                r['qualifications'][mode+'-'+kind] = dict(path=str(p), sha256=m.sha(p))
        lp = B / 'xe-e32-lints-baseline-compare-root-v10/record.json'
        a = json.loads(lp.read_text())
        m.require(a['passed'] and a['complete'] and not a['active'] and a['source_stable'] and
                  all(m.closed(c) for c in a['commands']), 'actual closed lints')
        m.require(a['baseline_revision'] == r['base_revision'] and all(not c['new_findings'] for c in a['comparisons'].values()), 'no new lint finding')
        m.require(all(m.sha(W/n) == h for n,h in a['source_pins'].items()), 'actual lint source')
        r['qualifications']['lint'] = dict(path=str(lp), sha256=m.sha(lp))
        r['source_pins'] = pins
        r['source_patch_sha256'] = m.sha(B / 'xe-e32-free-full-build-root-v15/commands/source-patch.stdout')
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        run('stage', ['/usr/bin/git', 'add', '--', 'src/prefill/prefill.cpp'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'feat(sycl): count prefill routes and expert kernel branches'])
        r['committed_revision'] = run('committed-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        m.require(not run('committed-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean committed source')
        m.require(all(m.sha(W/n) == h for n,h in pins.items()), 'exact built source after commit')
        run('publish', ['/usr/bin/git', 'push', 'git@github.com:yqYo1/Strata.git', 'HEAD:refs/heads/fix/xestrata-e32-host-handoff-20261011'])
        r['build_provenance'] = 'Qualified9c base plus exact reviewed route census patch is byte-identical to committed source; no build-to-run source changes.'
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    p = B / 'xe-prefill-route-census-source-commit-v1.json'
    m.require(not p.exists(), 'immutable source publication proof')
    p.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps({k: r.get(k) for k in ('passed', 'error', 'committed_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
