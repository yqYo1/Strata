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
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
P = W.parent / 'bench-xestrata-ea-serve-lint-20261011'
REV = 'ea395598c8e967c632a955dc43b77a9291286d49'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
LLVM = W / '.tools/intel-llvm-v711/install'
LINT = W / '.lint'
BASE = dict(PATH=str(LINT / 'venv/bin') + ':' + str(LINT / 'bin') + ':' + str(LINT / 'cppcheck/usr/bin') + ':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', LD_LIBRARY_PATH=str(LINT / 'cppcheck/usr/lib/x86_64-linux-gnu') + ':' + str(LLVM / 'lib') + ':/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib')
FILE = 'src/program/generate.cpp'

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-e32-lints-baseline-compare-root-v5'
    m.require(not out.exists(), 'immutable output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(), controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), gpu_executed=False, model_executed=False, baseline_revision=REV, scope='Actual mandatory lint of only changed generate.cpp in free and contrib-llvm, against exact committed ea baseline; no diagnostic suppression or raw-log reuse', total_wall_budget_seconds=2400, raw_log_budget_bytes=64 << 20)

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    begin = time.monotonic()
    save()

    def run(label, args, cwd=W, env=BASE, wall=60):
        remaining = 2400 - (time.monotonic() - begin)
        m.require(remaining > 0, 'whole lint deadline')
        e, so, se = owner.run(label, args, env, cwd, wall=min(wall, remaining), cpu=int(wall) * 4, text_cap=64 << 20, total_cap=64 << 20, rss_cap=16 << 30)
        m.require(m.closed(e), 'closed ' + label)
        return e, so.read_text(), se.read_text()

    def git(label, args, cwd=W):
        e, so, se = run(label, ['/usr/bin/git', *args], cwd=cwd)
        m.require(e['exit_code'] == 0, 'successful git ' + label)
        return so.strip()

    def check_tools(so, se):
        m.require('skipped (' not in so and 'possible secret' not in so, 'all mandatory tools executed; no secret findings')
        for tool in ['clang-tidy', 'cppcheck', 'codespell', 'reuse']:
            m.require('== ' + tool in so, 'required tool ' + tool)
        spelling = so.split('== codespell\n', 1)[1].split('== reuse', 1)[0]
        m.require(not spelling.strip(), 'no codespell findings')
        m.require(not so.split('== reuse\n', 1)[1].strip(), 'whole repository reuse clean')
        m.require(not any('error:' in x and '[unknownMacro]' not in x for x in se.splitlines()), 'no unclassified tool errors')

    def diagnostics(text, root, line_map=None):
        rows = []
        for path, line, column, severity, message in re.findall(r'^(.+?):(\d+):(\d+): (warning|error): (.+)$', text, re.M):
            p = Path(path)
            if p.is_absolute() and p.is_relative_to(root):
                path = str(p.relative_to(root))
            elif p.is_absolute() and p.is_relative_to(W):
                path = str(p.relative_to(W))
            n = int(line)
            if line_map is not None and path == FILE:
                n = line_map.get(n)
            rows.append((path, n, severity, message))
        return rows

    try:
        review_path = B / 'xe-serialized-serve-admission-main-review-v1.json'
        review = json.loads(review_path.read_text())
        m.require(review['main_reviewed'] and review['base_revision'] == REV, 'reviewed source correction')
        m.require(git('source-head', ['rev-parse', 'HEAD']) == REV, 'current source base')
        m.require(git('modified-paths', ['diff', '--name-only', 'HEAD']).splitlines() == [FILE], 'only reviewed generate source modified')
        m.require(not git('untracked-paths', ['ls-files', '--others', '--exclude-standard']), 'no unreviewed source')
        r['source_review_sha256'] = m.sha(review_path)
        r['source_pins'] = {FILE: m.sha(W / FILE)}
        m.require(r['source_pins'] == review['source_pins'], 'exact reviewed source bytes')
        r['build_proofs'] = {}
        for mode in ['free', 'contrib-llvm']:
            proof = B / ('xe-e32-' + mode + '-full-build-root-v10/record.json')
            v = json.loads(proof.read_text())
            m.require(v['passed'] and not v['active'] and v['complete'] and all(m.closed(c) for c in v['commands']), 'closed full build ' + mode)
            m.require(all(m.sha(W / f) == h for f, h in v['source_pins'].items()), 'same full source pins ' + mode)
            r['build_proofs'][mode] = m.sha(proof)
        if not P.exists():
            git('add-baseline', ['worktree', 'add', '-b', 'bench/xestrata-ea-serve-lint-20261011', str(P), REV])
        m.require(git('baseline-head', ['rev-parse', 'HEAD'], P) == REV, 'exact committed baseline')
        m.require(not git('baseline-status', ['status', '--porcelain', '--untracked-files=no'], P), 'baseline unmodified')
        for f in ['tools/lint/run.sh', 'tools/lint/clang-tidy.yaml', 'tools/lint/codespell.cfg', 'tools/lint/records.txt']:
            m.require(m.sha(P / f) == m.sha(W / f), 'same lint source/settings ' + f)
        old = (P / FILE).read_text().splitlines()
        new = (W / FILE).read_text().splitlines()
        line_map = {}
        for a, z, count in difflib.SequenceMatcher(None, old, new, autojunk=False).get_matching_blocks():
            for i in range(count):
                line_map[z + i + 1] = a + i + 1
        r['baseline_source_sha256'] = m.sha(P / FILE)
        r['comparisons'] = {}
        r['compile_db_proofs'] = {}
        for mode in ['free', 'contrib-llvm']:
            db = out / ('baseline-db-' + mode)
            db.mkdir()
            source = W / 'build' / mode / 'compile_commands.json'
            rows = json.loads(source.read_text())

            def replace(value):
                for area in ['src', 'include', 'third_party', 'tools']:
                    value = re.sub(re.escape(str(W) + '/' + area) + r'(?=/|$)', str(P) + '/' + area, value)
                return value

            for row in rows:
                row['file'] = replace(row['file'])
                row['arguments'] = [replace(a) for a in row.get('arguments', shlex.split(row.get('command', '')))]
                row.pop('command', None)
                m.require(all(not re.search(re.escape(str(W)) + r'/(?:src|include|third_party|tools)(?=/|$)', a) for a in row['arguments']), 'no current project headers mixed into baseline')
            (db / 'compile_commands.json').write_text(json.dumps(rows, indent=2) + '\n')
            r['compile_db_proofs'][mode] = dict(current_sha256=m.sha(source), baseline_adapted_sha256=m.sha(db / 'compile_commands.json'), adaptation='Only source/include/third_party/tools prefixes; installed compiler and all compile/check flags unchanged')
            be, bso, bse = run('baseline-' + mode, ['/bin/bash', 'tools/lint/run.sh', FILE], cwd=P, env=dict(BASE, STRATA_LINT_BUILD=str(db)), wall=540)
            ce, cso, cse = run('current-' + mode, ['/bin/bash', 'tools/lint/run.sh', FILE], env=dict(BASE, STRATA_LINT_BUILD=str(W / 'build' / mode)), wall=540)
            check_tools(bso, bse)
            check_tools(cso, cse)
            baseline = diagnostics(bso + '\n' + bse, P)
            current = diagnostics(cso + '\n' + cse, W, line_map)
            findings = list((collections.Counter(current) - collections.Counter(baseline)).elements())
            r['comparisons'][mode] = dict(baseline_exit_code=be['exit_code'], current_exit_code=ce['exit_code'], baseline_diagnostics=baseline, current_diagnostics=current, new_findings=findings)
            save()
            m.require(not findings, 'no new diagnostic occurrences ' + mode)
            m.require(ce['exit_code'] == 0 or (ce['exit_code'] == 1 and be['exit_code'] == 1), 'only actual source-proven baseline lint failure allowed')
        m.require(not git('baseline-final-status', ['status', '--porcelain', '--untracked-files=no'], P), 'baseline still unchanged')
        r['source_stable'] = all(m.sha(W / p) == h for p, h in r['source_pins'].items())
        m.require(r['source_stable'], 'source stable')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ['passed', 'error', 'source_stable']}))
    raise SystemExit(0 if r['passed'] else 1)
