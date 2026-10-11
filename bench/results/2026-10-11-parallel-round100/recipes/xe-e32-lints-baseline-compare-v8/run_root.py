"""Actual baseline/current mandatory lints of the KV selection atomic accesses."""
import collections
import difflib
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import shlex
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
P = W.parent / 'bench-xestrata-32222-kv-lint-20261011'
REV = '32222cb535eefa0bd7b6cd2e670f81a8013dd1ba'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
LLVM = W / '.tools/intel-llvm-v711/install'
LINT = W / '.lint'
ENV = dict(PATH=str(LINT/'venv/bin')+':'+str(LINT/'bin')+':'+str(LINT/'cppcheck/usr/bin')+':/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8', LD_LIBRARY_PATH=str(LINT/'cppcheck/usr/lib/x86_64-linux-gnu')+':'+str(LLVM/'lib')+':/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib')
FILES = ['src/kernels/xe/kv_stream.cpp']

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B/'xe-e32-lints-baseline-compare-root-v8'
    m.require(not out.exists(), 'immutable output')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(), controller_sha256=m.sha(__file__), owner_sha256=m.sha(O), baseline_revision=REV, gpu_executed=False, model_executed=False, total_wall_budget_seconds=3600, raw_log_budget_bytes=64<<20, scope='Actual baseline and current run.sh on the changed KV selection file, both source-LLVM modes; no previous diagnostic reuse or suppression')
    def save():
        (out/'record.json').write_text(json.dumps(r, indent=2)+'\n')
    owner = m.Owner(out/'commands', save)
    r['commands'] = owner.commands
    begin = time.monotonic()
    save()
    def run(label, argv, cwd=W, env=ENV, wall=60):
        left = 3600-(time.monotonic()-begin)
        m.require(left > 0, 'whole lint deadline')
        e, so, se = owner.run(label, argv, env, cwd, wall=min(wall,left), cpu=int(wall)*4, text_cap=64<<20, total_cap=64<<20, rss_cap=16<<30)
        m.require(m.closed(e), 'normal closed '+label)
        return e, so.read_text(), se.read_text()
    def git(label, argv, cwd=W):
        e, so, _ = run(label, ['/usr/bin/git', *argv], cwd=cwd)
        m.require(e['exit_code'] == 0, 'successful '+label)
        return so.strip()
    def tools(so, se):
        m.require('skipped (' not in so and 'possible secret' not in so, 'all mandatory tools executed')
        for name in ['clang-tidy', 'cppcheck', 'codespell', 'reuse']:
            m.require('== '+name in so, 'required '+name)
        m.require(not so.split('== codespell\n',1)[1].split('== reuse',1)[0].strip(), 'codespell clean')
        m.require(not so.split('== reuse\n',1)[1].strip(), 'reuse clean')
        m.require(not any('error:' in x and '[unknownMacro]' not in x for x in se.splitlines()), 'no unclassified tool errors')
    def diagnostics(text, root, maps=None):
        results = []
        for path, line, column, severity, message in re.findall(r'^(.+?):(\d+):(\d+): (warning|error): (.+)$', text, re.M):
            p = Path(path)
            if p.is_absolute() and p.is_relative_to(root):
                path = str(p.relative_to(root))
            elif p.is_absolute() and p.is_relative_to(W):
                path = str(p.relative_to(W))
            n = int(line)
            if maps is not None and path in maps:
                n = maps[path].get(n)
            results.append((path, n, severity, message))
        return results
    try:
        q = B/'xe-kv-resolve-atomic-main-review-v1.json'
        review = json.loads(q.read_text())
        m.require(review['main_reviewed'] and review['base_revision'] == REV, 'reviewed source')
        m.require(git('source-head', ['rev-parse','HEAD']) == REV, 'current source base')
        m.require(git('paths', ['diff','--name-only','HEAD']).splitlines() == FILES, 'exact reviewed files')
        m.require(not git('untracked',['ls-files','--others','--exclude-standard']), 'no unreviewed source')
        r['source_pins'] = {f:m.sha(W/f) for f in FILES}
        m.require(r['source_pins'] == review['source_pins'], 'exact reviewed bytes')
        r['source_review_sha256'] = m.sha(q)
        if not P.exists():
            git('add-baseline',['worktree','add','-b','bench/xestrata-32222-kv-lint-20261011',str(P),REV])
        m.require(git('baseline-head',['rev-parse','HEAD'],P) == REV, 'committed baseline')
        m.require(not git('baseline-status',['status','--porcelain','--untracked-files=no'],P), 'baseline clean')
        for f in ['tools/lint/run.sh','tools/lint/clang-tidy.yaml','tools/lint/codespell.cfg','tools/lint/records.txt']:
            m.require(m.sha(P/f) == m.sha(W/f), 'same lint configuration '+f)
        maps = {}
        for f in FILES:
            maps[f] = {}
            for a,z,k in difflib.SequenceMatcher(None,(P/f).read_text().splitlines(),(W/f).read_text().splitlines(),autojunk=False).get_matching_blocks():
                for i in range(k):
                    maps[f][z+i+1] = a+i+1
        r['baseline_source_pins'] = {f:m.sha(P/f) for f in FILES}
        r['comparisons'] = {}
        r['build_proofs'] = {}
        r['compile_db_proofs'] = {}
        for mode in ['free','contrib-llvm']:
            q = B/('xe-e32-'+mode+'-full-build-root-v13/record.json')
            v = json.loads(q.read_text())
            m.require(v['passed'] and v['complete'] and not v['active'] and all(m.closed(c) for c in v['commands']), 'closed build '+mode)
            m.require(all(m.sha(W/f) == h for f,h in v['source_pins'].items()), 'matching full source '+mode)
            r['build_proofs'][mode] = m.sha(q)
            db = out/('baseline-db-'+mode)
            db.mkdir()
            source = W/'build'/mode/'compile_commands.json'
            rows = json.loads(source.read_text())
            def replace(value):
                for area in ['src','include','third_party','tools']:
                    value = re.sub(re.escape(str(W)+'/'+area)+r'(?=/|$)',str(P)+'/'+area,value)
                return value
            for row in rows:
                row['file'] = replace(row['file'])
                row['arguments'] = [replace(a) for a in row.get('arguments', shlex.split(row.get('command','')))]
                row.pop('command',None)
                m.require(all(not re.search(re.escape(str(W))+r'/(?:src|include|third_party|tools)(?=/|$)',a) for a in row['arguments']), 'baseline headers isolated')
            (db/'compile_commands.json').write_text(json.dumps(rows,indent=2)+'\n')
            r['compile_db_proofs'][mode] = dict(current_sha256=m.sha(source), baseline_adapted_sha256=m.sha(db/'compile_commands.json'), adaptation='Only project path prefixes; no check/compiler flags changed')
            be,bso,bse = run('baseline-'+mode,['/bin/bash','tools/lint/run.sh',*FILES],cwd=P,env=dict(ENV,STRATA_LINT_BUILD=str(db)),wall=900)
            ce,cso,cse = run('current-'+mode,['/bin/bash','tools/lint/run.sh',*FILES],env=dict(ENV,STRATA_LINT_BUILD=str(W/'build'/mode)),wall=900)
            tools(bso,bse)
            tools(cso,cse)
            baseline = diagnostics(bso+'\n'+bse,P)
            current = diagnostics(cso+'\n'+cse,W,maps)
            new = list((collections.Counter(current)-collections.Counter(baseline)).elements())
            r['comparisons'][mode] = dict(baseline_exit_code=be['exit_code'],current_exit_code=ce['exit_code'],baseline_diagnostics=baseline,current_diagnostics=current,new_findings=new)
            save()
            m.require(not new, 'no new diagnostic occurrences '+mode)
            m.require(ce['exit_code'] == 0 or (be['exit_code'] == 1 and ce['exit_code'] == 1), 'only proven baseline lint failures allowed')
        m.require(not git('baseline-final-status',['status','--porcelain','--untracked-files=no'],P), 'baseline remains clean')
        r['source_stable'] = all(m.sha(W/f) == h for f,h in r['source_pins'].items())
        m.require(r['source_stable'], 'source stable')
        r['complete'] = True
    except BaseException as error:
        r['error'] = repr(error)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ['passed','error','source_stable']}))
    raise SystemExit(0 if r['passed'] else 1)
