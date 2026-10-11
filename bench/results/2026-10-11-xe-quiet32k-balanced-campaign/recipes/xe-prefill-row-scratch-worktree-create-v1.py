"""Main: create a source-only candidate worktree from the qualified source."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
P = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
W = P.parent / 'perf-xe-prefill-row-bounded-scratch-20261011'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
with O.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-prefill-row-scratch-worktree-create-root-v1'
    m.require(not out.exists() and not W.exists(), 'new worktree and receipt')
    out.mkdir(mode=0o700)
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             recipe_sha256=m.sha(__file__), owner_sha256=m.sha(O), worktree=str(W),
             branch='perf/xe-prefill-row-bounded-scratch-20261011', gpu_executed=False,
             model_executed=False, source_changes_applied=False)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    def run(label, argv, cwd=P):
        c, stdout, stderr = owner.run(label, argv, ENV, cwd, wall=30, cpu=60,
                                      text_cap=2 << 20, total_cap=8 << 20, rss_cap=1 << 30)
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] == 0, 'closed ' + label)
        return stdout.read_text().strip()
    save()
    try:
        r['base_revision'] = run('parent-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        m.require(r['base_revision'] == '0740b43488d812117f021ccbe4bd0e15d223f955'
                  and not run('parent-tracked-status', ['/usr/bin/git', 'status', '--porcelain', '--untracked-files=no']), 'qualified clean parent')
        run('create-worktree', ['/usr/bin/git', 'worktree', 'add', '-b', r['branch'], str(W), r['base_revision']])
        r['registered_worktrees'] = run('worktree-list', ['/usr/bin/git', 'worktree', 'list', '--porcelain'])
        m.require('worktree '+str(W) in r['registered_worktrees'], 'registered worktree path')
        m.require(run('candidate-head', ['/usr/bin/git', 'rev-parse', 'HEAD'], W) == r['base_revision']
                  and not run('candidate-status', ['/usr/bin/git', 'status', '--porcelain'], W), 'unchanged clean new candidate')
        identity = run('candidate-identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'], W)
        m.require(identity and all(x.startswith('file:/home/yayoi/.config/git/config') for x in identity.splitlines()), 'global identity only')
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ('passed', 'error', 'worktree', 'base_revision')}))
    raise SystemExit(0 if r['passed'] else 1)
