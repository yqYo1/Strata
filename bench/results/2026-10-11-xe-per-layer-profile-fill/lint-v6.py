"""Actual lints, with source/flags proven closed v5 generate baseline reuse."""
import collections
import difflib
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import shlex
import time

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
P=W.parent/'bench-xestrata-514-fill-lint-20261011'
PREVIOUS_P=W.parent/'bench-xestrata-ea-serve-lint-20261011'
REV='51436a0fe4c9588b0d3eaae921a7eae12343ce7e'
O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
LLVM=W/'.tools/intel-llvm-v711/install'
LINT=W/'.lint'
BASE=dict(PATH=str(LINT/'venv/bin')+':'+str(LINT/'bin')+':'+str(LINT/'cppcheck/usr/bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LINT/'cppcheck/usr/lib/x86_64-linux-gnu')+':'+str(LLVM/'lib')+':/opt/intel/oneapi/compiler/2026.1/lib:/opt/intel/oneapi/mkl/2026.1/lib')
FILE='src/program/generate.cpp'
SMALL=['src/core/expert_cache.cpp','tests/core/expert_profile_version_test.cpp']
FILES=sorted([FILE,*SMALL])

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-e32-lints-baseline-compare-root-v6'
    m.require(not out.exists(),'immutable lint output')
    out.mkdir()
    r=dict(active=True,passed=False,complete=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),gpu_executed=False,model_executed=False,baseline_revision=REV,total_wall_budget_seconds=2400,raw_log_budget_bytes=64<<20,scope='Actual current run.sh of all three changed files in both modes; actual baseline run.sh for two small files; generate baseline reuses closed source-and-flag-identical v5 current structured diagnostics, remapped to514 lines without raw-log reuse or suppression.')
    def save(): (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    owner=m.Owner(out/'commands',save)
    r['commands']=owner.commands
    begin=time.monotonic()
    save()

    def run(label,args,cwd=W,env=BASE,wall=60):
        left=2400-(time.monotonic()-begin)
        m.require(left>0,'whole lint deadline')
        e,so,se=owner.run(label,args,env,cwd,wall=min(wall,left),cpu=int(wall)*4,text_cap=64<<20,total_cap=64<<20,rss_cap=16<<30)
        m.require(m.closed(e),'closed '+label)
        return e,so.read_text(),se.read_text()
    def git(label,args,cwd=W):
        e,so,_=run(label,['/usr/bin/git',*args],cwd=cwd)
        m.require(e['exit_code']==0,'successful '+label)
        return so.strip()
    def lines(old,new):
        result={}
        for a,z,n in difflib.SequenceMatcher(None,old,new,autojunk=False).get_matching_blocks():
            for i in range(n): result[z+i+1]=a+i+1
        return result
    def check_tools(so,se):
        m.require('skipped (' not in so and 'possible secret' not in so,'all mandatory tools executed')
        for tool in ['clang-tidy','cppcheck','codespell','reuse']:
            m.require('== '+tool in so,'required '+tool)
        m.require(not so.split('== codespell\n',1)[1].split('== reuse',1)[0].strip(),'codespell clean')
        m.require(not so.split('== reuse\n',1)[1].strip(),'reuse clean')
        m.require(not any('error:' in x and '[unknownMacro]' not in x for x in se.splitlines()),'no unclassified tool errors')
    def diagnostics(text,root,maps=None):
        result=[]
        for path,line,col,severity,message in re.findall(r'^(.+?):(\d+):(\d+): (warning|error): (.+)$',text,re.M):
            p=Path(path)
            if p.is_absolute() and p.is_relative_to(root): path=str(p.relative_to(root))
            elif p.is_absolute() and p.is_relative_to(W): path=str(p.relative_to(W))
            n=int(line)
            if maps is not None and path in maps: n=maps[path].get(n)
            result.append((path,n,severity,message))
        return result
    try:
        review_path=B/'xe-per-layer-profile-fill-main-review-v1.json'
        review=json.loads(review_path.read_text())
        m.require(review['main_reviewed'] and review['base_revision']==REV,'reviewed source')
        m.require(git('source-head',['rev-parse','HEAD'])==REV,'current source base')
        m.require(git('modified-paths',['diff','--name-only','HEAD']).splitlines()==FILES,'exact changed files')
        m.require(not git('untracked',['ls-files','--others','--exclude-standard']),'no unreviewed source')
        r['source_pins']={f:m.sha(W/f) for f in FILES}
        m.require(r['source_pins']==review['source_pins'],'exact reviewed bytes')
        r['source_review_sha256']=m.sha(review_path)
        previous_path=B/'xe-e32-lints-baseline-compare-root-v5/record.json'
        previous=json.loads(previous_path.read_text())
        m.require(previous['passed'] and previous['complete'] and not previous['active'] and all(m.closed(c) for c in previous['commands']),'closed actual v5 lint qualification')
        r['previous_lint_receipt_sha256']=m.sha(previous_path)
        if not P.exists(): git('add-baseline',['worktree','add','-b','bench/xestrata-514-fill-lint-20261011',str(P),REV])
        m.require(git('baseline-head',['rev-parse','HEAD'],P)==REV,'exact baseline commit')
        m.require(not git('baseline-status',['status','--porcelain','--untracked-files=no'],P),'baseline unchanged')
        m.require(m.sha(P/FILE)==previous['source_pins'][FILE],'exact514 generate bytes qualified in v5')
        m.require(m.sha(PREVIOUS_P/FILE)==previous['baseline_source_sha256'],'exact previous ea baseline bytes')
        for f in ['tools/lint/run.sh','tools/lint/clang-tidy.yaml','tools/lint/codespell.cfg','tools/lint/records.txt']:
            m.require(m.sha(P/f)==m.sha(W/f)==m.sha(PREVIOUS_P/f),'same lint tool/settings '+f)
        inverse={old:new for new,old in lines((PREVIOUS_P/FILE).read_text().splitlines(),(P/FILE).read_text().splitlines()).items()}
        maps={f:lines((P/f).read_text().splitlines(),(W/f).read_text().splitlines()) for f in FILES}
        r['baseline_source_pins']={f:m.sha(P/f) for f in FILES}
        r['comparisons']={}
        r['build_proofs']={}
        r['compile_db_proofs']={}
        for mode in ['free','contrib-llvm']:
            proof=B/('xe-e32-'+mode+'-full-build-root-v11/record.json')
            build=json.loads(proof.read_text())
            m.require(build['passed'] and build['complete'] and not build['active'] and all(m.closed(c) for c in build['commands']),'closed current build '+mode)
            m.require(all(m.sha(W/f)==h for f,h in build['source_pins'].items()),'current built source '+mode)
            r['build_proofs'][mode]=m.sha(proof)
            db=out/('baseline-db-'+mode)
            db.mkdir()
            source=W/'build'/mode/'compile_commands.json'
            m.require(m.sha(source)==previous['compile_db_proofs'][mode]['current_sha256'],'exact unchanged prior generate compiler/check flags '+mode)
            rows=json.loads(source.read_text())
            def replace(value):
                for area in ['src','include','third_party','tools']:
                    value=re.sub(re.escape(str(W)+'/'+area)+r'(?=/|$)',str(P)+'/'+area,value)
                return value
            for row in rows:
                row['file']=replace(row['file'])
                row['arguments']=[replace(a) for a in row.get('arguments',shlex.split(row.get('command','')))]
                row.pop('command',None)
                m.require(all(not re.search(re.escape(str(W))+r'/(?:src|include|third_party|tools)(?=/|$)',a) for a in row['arguments']),'baseline uses only committed baseline headers')
            (db/'compile_commands.json').write_text(json.dumps(rows,indent=2)+'\n')
            r['compile_db_proofs'][mode]=dict(current_sha256=m.sha(source),baseline_adapted_sha256=m.sha(db/'compile_commands.json'),same_previous_flags=True)
            old=previous['comparisons'][mode]
            generate=[]
            for path,n,severity,message in old['current_diagnostics']:
                if path==FILE:
                    m.require(n in inverse,'unambiguous existing diagnostic line conversion')
                    n=inverse[n]
                generate.append((path,n,severity,message))
            be,bso,bse=run('baseline-small-'+mode,['/bin/bash','tools/lint/run.sh',*SMALL],cwd=P,env=dict(BASE,STRATA_LINT_BUILD=str(db)),wall=540)
            ce,cso,cse=run('current-'+mode,['/bin/bash','tools/lint/run.sh',*FILES],env=dict(BASE,STRATA_LINT_BUILD=str(W/'build'/mode)),wall=720)
            check_tools(bso,bse)
            check_tools(cso,cse)
            baseline=generate+diagnostics(bso+'\n'+bse,P)
            current=diagnostics(cso+'\n'+cse,W,maps)
            findings=list((collections.Counter(current)-collections.Counter(baseline)).elements())
            r['comparisons'][mode]=dict(baseline_small_exit_code=be['exit_code'],baseline_generate_exit_code=old['current_exit_code'],current_exit_code=ce['exit_code'],reused_generate_diagnostics=generate,baseline_diagnostics=baseline,current_diagnostics=current,new_findings=findings)
            save()
            m.require(not findings,'no new lint diagnostic occurrences '+mode)
            m.require(ce['exit_code']==0 or (ce['exit_code']==1 and (be['exit_code']==1 or old['current_exit_code']==1)),'only source-proven baseline lint failure allowed')
        m.require(not git('baseline-final-status',['status','--porcelain','--untracked-files=no'],P),'baseline remains unchanged')
        r['source_stable']=all(m.sha(W/f)==h for f,h in r['source_pins'].items())
        m.require(r['source_stable'],'current source stable')
        r['complete']=True
    except BaseException as error: r['error']=repr(error)
    finally:
        r['active']=owner.active is not None
        r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc']=m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ['passed','error','source_stable']}))
    raise SystemExit(0 if r['passed'] else 1)
