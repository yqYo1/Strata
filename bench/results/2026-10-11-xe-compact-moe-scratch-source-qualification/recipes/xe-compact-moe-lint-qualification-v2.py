"""Main: actual mandatory lints against clean parent, same flags and no suppression."""
import collections
import difflib
import fcntl
import importlib.util
import json
import re
import shlex
import time
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
P = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
W = P.parent / 'perf-xe-prefill-row-bounded-scratch-20261011'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('lint_owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner identity')
LLVM = P / '.tools/intel-llvm-v711/install'
LINT = P / '.lint'
ENV = dict(PATH=str(LINT / 'venv/bin') + ':' + str(LINT / 'bin') + ':' + str(LINT / 'cppcheck/usr/bin') + ':/usr/bin:/bin',
           HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
           LD_LIBRARY_PATH=str(LINT / 'cppcheck/usr/lib/x86_64-linux-gnu') + ':' + str(LLVM / 'lib') +
           ':/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib')

def strong(c):
    return m.closed(c) and c.get('session_anchored') is True

def main():
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        review_path = B / 'xe-compact-moe-applied-main-review-v2.json'
        review = json.loads(review_path.read_text())
        m.require(review['main_reviewed'] and all(m.sha(W / f) == h for f, h in review['source_pins'].items()), 'exact reviewed source')
        files = sorted(review['changed_source_pins'])
        baseline_files = [f for f in files if (P / f).exists()]
        out = B / 'xe-compact-moe-lint-qualification-root-v2'
        m.require(not out.exists(), 'immutable lint output')
        out.mkdir()
        r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
                 controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), main_review_sha256=m.sha(review_path),
                 source_base=review['base_revision'], source_pins=review['source_pins'], current_files=files,
                 baseline_files=baseline_files, new_files=sorted(set(files) - set(baseline_files)),
                 reused_commands=[],
                 gpu_executed=False, model_executed=False, adopted=False, comparisons={},
                 raw_log_budget_bytes=64 << 20, total_wall_budget_seconds=3600,
                 scope='Actual tools/lint/run.sh in free and contrib-llvm modes; new diagnostics rejected, original baseline source unchanged')
        def save():
            (out / 'record.json').write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n')
        owner = m.Owner(out / 'commands', save)
        r['commands'] = owner.commands
        begin = time.monotonic()
        save()
        def run(label, argv, cwd=W, env=ENV, wall=60):
            left = 3600 - (time.monotonic() - begin)
            m.require(left > 0, 'whole lint deadline')
            c, so, se = owner.run(label, argv, env, cwd, wall=min(wall, left), cpu=max(20, int(wall) * 4),
                                  file_cap=64 << 20, text_cap=64 << 20, total_cap=64 << 20, rss_cap=16 << 30)
            m.require(strong(c), 'anchored closure ' + label)
            return c, so.read_text(), se.read_text()
        def git(label, args, cwd=W):
            c, so, _ = run(label, ['/usr/bin/git', *args], cwd=cwd)
            m.require(c['exit_code'] == 0, 'successful ' + label)
            return so.strip()
        def require_tools(so):
            m.require('skipped (' not in so and 'possible secret' not in so, 'all mandatory tools ran')
            for tool in ['clang-tidy', 'cppcheck', 'cmake-lint', 'codespell', 'reuse']:
                m.require('== ' + tool in so, 'actual ' + tool)
            m.require(not so.split('== codespell\n', 1)[1].split('== reuse', 1)[0].strip(), 'codespell clean')
            m.require(not so.split('== reuse\n', 1)[1].strip(), 'reuse clean')
        def diagnostics(text, root, maps=None):
            found = []
            patterns = [r'^(.+?):(\d+):(\d+): (warning|error): (.+)$',
                        r'^(.+?):(\d+)(?:,([^:]+))?: (\[[^\]]+\]) (.+)$']
            for pattern in patterns:
                for path, line, _column, severity, message in re.findall(pattern, text, re.M):
                    p = Path(path)
                    if p.is_absolute():
                        p = p.resolve()
                        path = str(p)
                        if p.is_relative_to(root.resolve()):
                            rel = p.relative_to(root.resolve())
                            if rel.parts[0] in ('src', 'include', 'tools', 'third_party') or str(rel) == 'CMakeLists.txt':
                                path = str(rel)
                    n = int(line)
                    if maps is not None and path in maps:
                        n = maps[path].get(n)
                    found.append((path, n, severity, message))
            return found
        try:
            m.require(git('head', ['rev-parse', 'HEAD']) == review['base_revision'], 'candidate base')
            actual = git('changed', ['diff', '--name-only', 'HEAD']).splitlines()
            actual += git('new', ['ls-files', '--others', '--exclude-standard']).splitlines()
            m.require(set(actual) == set(files), 'only reviewed changes')
            m.require(git('parent-head', ['rev-parse', 'HEAD'], P) == review['base_revision'], 'clean committed baseline')
            m.require(not git('parent-status', ['status', '--porcelain'], P), 'baseline remains clean')
            for f in ['tools/lint/run.sh', 'tools/lint/clang-tidy.yaml', 'tools/lint/codespell.cfg', 'tools/lint/cmake-format.py', 'tools/lint/records.txt']:
                m.require(m.sha(P / f) == m.sha(W / f), 'same lint config ' + f)
            link = W / '.lint'
            m.require(link.is_symlink() and link.resolve() == LINT.resolve(), 'same ignored tools link from original actual lint')
            r['tools_link'] = dict(path=str(link), target=str(LINT), ignored_only=True)
            maps = {}
            for f in baseline_files:
                maps[f] = {}
                for a, z, count in difflib.SequenceMatcher(None, (P / f).read_text().splitlines(), (W / f).read_text().splitlines(), autojunk=False).get_matching_blocks():
                    for i in range(count):
                        maps[f][z + i + 1] = a + i + 1
            r['baseline_pins'] = {f: m.sha(P / f) for f in baseline_files}
            for mode in ['free', 'contrib-llvm']:
                p = B / ('xe-compact-moe-' + mode + '-qualification-root-v2/record.json')
                proof = json.loads(p.read_text())
                m.require(proof['passed'] and proof['complete'] and not proof['active'] and all(strong(c) for c in proof['commands']), 'qualified build ' + mode)
                m.require(proof['source_pins'] == review['source_pins'], 'same mode source')
                db = out / ('baseline-db-' + mode)
                db.mkdir()
                current_db = W / 'build' / mode / 'compile_commands.json'
                rows = json.loads(current_db.read_text())
                def adapt(value):
                    for area in ['src', 'include', 'third_party', 'tools']:
                        value = re.sub(re.escape(str(W) + '/' + area) + r'(?=/|$)', str(P) + '/' + area, value)
                    return value
                for row in rows:
                    row['file'] = adapt(row['file'])
                    row['arguments'] = [adapt(v) for v in row.get('arguments', shlex.split(row.get('command', '')))]
                    row.pop('command', None)
                (db / 'compile_commands.json').write_text(json.dumps(rows, indent=2) + '\n')
                if mode == 'free':
                    prior_path = B / 'xe-compact-moe-lint-qualification-root-v1/record.json'
                    prior = json.loads(prior_path.read_text())
                    m.require(not prior['active'] and not prior['passed'] and
                              prior['error'] == "RuntimeError('no new diagnostic occurrences free')" and
                              prior['source_pins'] == r['source_pins'] and prior['main_review_sha256'] == r['main_review_sha256'],
                              'actual closed original failed comparison, same unchanged source')
                    m.require(all(strong(c) for c in prior['commands']), 'all original lint owners strongly closed')
                    v = prior['comparisons']['free']
                    m.require(v['build_receipt_sha256'] == m.sha(p), 'unchanged actual free build proof')
                    be = next(c for c in prior['commands'] if c['label'] == 'baseline-free')
                    ce = next(c for c in prior['commands'] if c['label'] == 'candidate-free')
                    m.require(be['argv'] == ['/bin/bash', 'tools/lint/run.sh', *baseline_files] and
                              ce['argv'] == ['/bin/bash', 'tools/lint/run.sh', *files], 'same actually executed full mandatory free lint')
                    bso, bse, cso, cse = (v[k] for k in ('baseline_stdout', 'baseline_stderr', 'candidate_stdout', 'candidate_stderr'))
                    for label, raw in [('baseline-free.stdout', bso), ('baseline-free.stderr', bse),
                                       ('candidate-free.stdout', cso), ('candidate-free.stderr', cse)]:
                        raw_path = prior_path.parent / 'commands' / label
                        m.require(raw_path.read_text() == raw, 'original actual complete output ' + label)
                    r['reused_commands'] = [be, ce]
                    r['original_failed_comparison'] = dict(path=str(prior_path), sha256=m.sha(prior_path), original_status='FAILED',
                        correction='Canonical compiler paths stay external in both arms; normalize only repository source paths; include CMake no-column diagnostics',
                        free_lints_reexecuted=False)
                else:
                    be, bso, bse = run('baseline-' + mode, ['/bin/bash', 'tools/lint/run.sh', *baseline_files], cwd=P,
                                       env=dict(ENV, STRATA_LINT_BUILD=str(db)), wall=900)
                    ce, cso, cse = run('candidate-' + mode, ['/bin/bash', 'tools/lint/run.sh', *files],
                                       env=dict(ENV, STRATA_LINT_BUILD=str(W / 'build' / mode)), wall=900)
                require_tools(bso)
                require_tools(cso)
                baseline = diagnostics(bso + '\n' + bse, P)
                current = diagnostics(cso + '\n' + cse, W, maps)
                new = list((collections.Counter(current) - collections.Counter(baseline)).elements())
                r['comparisons'][mode] = dict(baseline_exit_code=be['exit_code'], candidate_exit_code=ce['exit_code'],
                    baseline_diagnostics=baseline, candidate_diagnostics=current, new_findings=new,
                    baseline_stdout=bso, baseline_stderr=bse, candidate_stdout=cso, candidate_stderr=cse,
                    compile_db_sha256=m.sha(current_db), baseline_compile_db_sha256=m.sha(db / 'compile_commands.json'),
                    build_receipt_sha256=m.sha(p))
                save()
                m.require(not new, 'no new diagnostic occurrences ' + mode)
                m.require(ce['exit_code'] == 0 or (be['exit_code'] == 1 and ce['exit_code'] == 1), 'only matched baseline lint failure allowed')
                m.require(not any('error:' in line and '[unknownMacro]' not in line for line in (bse + '\n' + cse).splitlines()), 'no unclassified tool errors')
            m.require(not git('parent-final-status', ['status', '--porcelain'], P), 'parent unchanged')
            m.require(all(m.sha(W / f) == h for f, h in r['source_pins'].items()), 'candidate source stable')
            r['complete'] = True
        except BaseException as error:
            r['error'] = repr(error)
        finally:
            r['active'] = owner.active is not None
            r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
            r['finished_utc'] = m.utc()
            save()
        print(json.dumps(dict(passed=r['passed'], error=r.get('error'),
              comparisons={k: dict(baseline_exit_code=v['baseline_exit_code'],
                                   candidate_exit_code=v['candidate_exit_code'],
                                   new_findings=v['new_findings']) for k, v in r['comparisons'].items()})))
        return 0 if r['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
