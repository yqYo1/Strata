"""Main: publish six opt-in source files after actual both-mode host qualifications."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/perf-xe-prefill-row-bounded-scratch-20261011')
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('publish_owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner identity')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-compact-moe-source-publish-root-v1'
    m.require(not out.exists(), 'immutable source publication')
    out.mkdir()
    review_path = B / 'xe-compact-moe-applied-main-review-v2.json'
    review = json.loads(review_path.read_text())
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             recipe_sha256=m.sha(__file__), owner_sha256=m.sha(O), source_tree=str(W),
             historical_build_base=review['base_revision'], source_pins=review['source_pins'],
             main_review_sha256=m.sha(review_path), gpu_executed=False, model_executed=False,
             adopted=False, performance_eligible=False, full262144_runtime_qualified=False)
    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')
    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    def run(label, args):
        c, so, se = owner.run(label, args, ENV, W, wall=60, cpu=120, text_cap=8 << 20,
                              total_cap=32 << 20, rss_cap=2 << 30, file_cap=64 << 20)
        m.require(m.closed(c) and c.get('session_anchored') is True and c['exit_code'] == 0, 'closed ' + label)
        return so.read_text().strip()
    save()
    try:
        m.require(review['main_reviewed'] and all(m.sha(W / p) == h for p, h in r['source_pins'].items()), 'reviewed source stable')
        m.require(run('head-before', ['/usr/bin/git', 'rev-parse', 'HEAD']) == review['base_revision'], 'exact prepublication base')
        files = sorted(review['changed_source_pins'])
        changed = run('changed', ['/usr/bin/git', 'diff', '--name-only', 'HEAD']).splitlines()
        changed += run('new', ['/usr/bin/git', 'ls-files', '--others', '--exclude-standard']).splitlines()
        m.require(set(changed) == set(files), 'only six reviewed source files')
        run('index-empty', ['/usr/bin/git', 'diff', '--cached', '--quiet'])
        identity = run('identity', ['/usr/bin/git', 'config', '--show-origin', '--get-regexp', r'^user\.(name|email)$'])
        m.require(identity and all(v.startswith('file:/home/yayoi/.config/git/config') for v in identity.splitlines()), 'global identity only')
        r['qualification_receipts'] = {}
        for phase in ['host', 'free', 'contrib-llvm', 'free-asan', 'contrib-llvm-asan']:
            p = B / ('xe-compact-moe-' + phase + '-qualification-root-v2/record.json')
            v = json.loads(p.read_text())
            m.require(v['passed'] and v['complete'] and not v['active'] and v['phase'] == phase,
                      'actual completed qualification ' + phase)
            m.require(all(m.closed(c) and c.get('session_anchored') is True for c in v['commands']), 'all anchored owners closed ' + phase)
            m.require(v['source_pins'] == r['source_pins'] and v['source_stable'] is True, 'same exact sources ' + phase)
            if phase != 'host':
                m.require(m.sha(Path(v['binary']['path'])) == v['binary']['sha256'], 'qualified binary ' + phase)
                m.require('100% tests passed, 0 tests failed out of 14' in v['cpu_test_output'] and
                          'expert_multi_test (Skipped)' in v['cpu_test_output'], 'actual selected 13 CPU PASS and AVX512 skip ' + phase)
            r['qualification_receipts'][phase] = dict(path=str(p), sha256=m.sha(p))
        lint_path = B / 'xe-compact-moe-lint-qualification-root-v1/record.json'
        lint = json.loads(lint_path.read_text())
        m.require(lint['passed'] and lint['complete'] and not lint['active'] and lint['source_pins'] == r['source_pins'], 'actual mandatory both-mode lint proof')
        m.require(all(not v['new_findings'] for v in lint['comparisons'].values()) and
                  set(lint['comparisons']) == {'free', 'contrib-llvm'}, 'no new findings in both modes')
        r['lint_receipt'] = dict(path=str(lint_path), sha256=m.sha(lint_path))
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        run('stage', ['/usr/bin/git', 'add', '--', *files])
        m.require(set(run('staged-files', ['/usr/bin/git', 'diff', '--cached', '--name-only']).splitlines()) == set(files), 'exact six-file commit')
        run('staged-check', ['/usr/bin/git', 'diff', '--cached', '--check'])
        run('commit', ['/usr/bin/git', 'commit', '-m', 'perf(prefill): add opt-in row-bounded MoE scratch', '-m',
            'Keep logical dequant groups and route-sized FP32 Down output. Reuse T-row FP16 gathered inputs and SwiGLU output and FP32 GU through checked contiguous subgroups on the same ordered queue. Persistent device bounds avoid asynchronous host descriptor reuse.\n\nDefault OFF. Reject MMQ and whole-route NaN diagnostic combinations. Built free and contrib-llvm, with host release and address/undefined sanitizer fixtures plus selected CPU CTest. GPU correctness, physical VRAM, performance, RESTORE and full 262144 occupancy remain unverified; no default or adoption change.'])
        r['published_head'] = run('head-after', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        r['branch'] = run('branch', ['/usr/bin/git', 'branch', '--show-current'])
        m.require(r['branch'] == 'perf/xe-prefill-row-bounded-scratch-20261011', 'assigned worktree branch')
        run('push-own-fork', ['/usr/bin/git', 'push', 'git@github.com:yqYo1/Strata.git', 'HEAD:refs/heads/' + r['branch']])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean published source')
        m.require(all(m.sha(W / p) == h for p, h in r['source_pins'].items()), 'published bytes equal built source')
        bridge = dict(main_reviewed=True, historical_build_base=r['historical_build_base'],
                      published_head=r['published_head'], source_tree=str(W), production_source_pins=r['source_pins'],
                      clean_tracked_working_tree=True, source_bytes_identical_to_all_qualified_builds=True,
                      qualification_receipts=r['qualification_receipts'], lint_receipt=r['lint_receipt'],
                      gpu_executed=False, model_executed=False, performance_eligible=False, adopted=False,
                      limitations=['Historical build base plus reviewed uncommitted six-file change, not a rebuilt postcommit banner',
                                   'GPU numerical, allocation, lifecycle and 262144 occupancy pending'])
        p = B / 'xe-compact-moe-source-publication-main-review-v1.json'
        m.require(not p.exists(), 'new publication bridge')
        p.write_text(json.dumps(bridge, indent=2) + '\n')
        r['publication_bridge'] = dict(path=str(p), sha256=m.sha(p))
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ['passed', 'error', 'published_head']}))
    raise SystemExit(0 if r['passed'] else 1)
