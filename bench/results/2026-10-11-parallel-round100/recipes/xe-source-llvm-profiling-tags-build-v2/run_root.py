"""Main-owned compilation/lint of returned actual-production Event diagnostic."""
import fcntl
import importlib.util
import json
from pathlib import Path
import time

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
ORIGINAL = B/'xe-profiling-tags-diagnostic-source-v1'
S = B/'xe-profiling-tags-diagnostic-source-v2'
LLVM = W/'.tools/intel-llvm-v711/install'
O = B/'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner',O)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','owner pin')
ENV = dict(PATH=str(LLVM/'bin')+':/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8',LD_LIBRARY_PATH=str(LLVM/'lib'))
L = W/'.lint'
LINTENV = dict(ENV,PATH=str(L/'venv/bin')+':'+str(L/'bin')+':'+str(L/'cppcheck/usr/bin')+':'+ENV['PATH'],LD_LIBRARY_PATH=str(L/'cppcheck/usr/lib/x86_64-linux-gnu')+':'+str(LLVM/'lib')+':/opt/intel/oneapi/compiler/2026.1/lib')

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-source-llvm-profiling-tags-build-root-v2'
    m.require(not out.exists(),'immutable output')
    out.mkdir()
    r=dict(active=True,passed=False,complete=False,commands=[],started_utc=m.utc(),controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),gpu_executed=False,source_revision='32222cb535eefa0bd7b6cd2e670f81a8013dd1ba',scope='Actual gpu.cpp/device.cpp and public headers linked directly; source-LLVM same diagnostic in two labels, no BLAS/oneMath proof; build/lint/CLI only',main_reviewed=True,model_executed=False,total_wall_budget_seconds=900,raw_log_budget_bytes=64<<20,binaries={})
    def save():
        (out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
    owner=m.Owner(out/'commands',save)
    r['commands']=owner.commands
    begin=time.monotonic()
    save()
    def run(label,args,env=ENV,wall=120):
        left=900-(time.monotonic()-begin)
        m.require(left>0,'whole build deadline')
        e,so,se=owner.run(label,args,env,W,wall=min(wall,left),cpu=int(wall)*4,text_cap=64<<20,total_cap=64<<20,rss_cap=16<<30)
        m.require(m.closed(e),'closed '+label)
        return e,so.read_text(),se.read_text()
    try:
        q=B/'xe-profiling-interval-guard-source-commit-v1.json'
        publication=json.loads(q.read_text())
        m.require(publication['passed'] and publication['committed_revision']==r['source_revision'],'published qualified production source')
        r['source_publication_sha256']=m.sha(q)
        e,so,se=run('source-head',['/usr/bin/git','rev-parse','HEAD'])
        m.require(e['exit_code']==0 and so.strip()==r['source_revision'],'actual source revision')
        toolchain=json.loads((B/'xe-llvm-v711-source-root-v5/record.json').read_text())
        m.require(toolchain['passed'] and toolchain['complete'] and not toolchain['active'],'qualified compiler')
        m.require(all(m.sha(Path(item['path']))==item['sha256'] for item in toolchain['installed_files'].values()),'actual compiler installation')
        m.require(m.sha(ORIGINAL/'manifest.json')=='cc3f2aebbcc36cd0e571d199ec03f6da4b2ba3f90563f81e3955cf5d7bcba7c8','returned original manifest')
        manifest=json.loads((ORIGINAL/'manifest.json').read_text())
        r['manifest_sha256']=m.sha(ORIGINAL/'manifest.json')
        for item in manifest['artifacts']:
            m.require(m.sha(Path(item['path']))==item['sha256'],'returned artifact '+item['path'])
        review=json.loads((S/'main-review.json').read_text())
        m.require(review['main_reviewed'] and review['derived_probe_sha256']==m.sha(S/'profiling_tags_probe.cpp'),'Main derived probe pin')
        r['probe_review_sha256']=m.sha(S/'main-review.json')
        r['probe_sha256']=m.sha(S/'profiling_tags_probe.cpp')
        r['source_pins']={item['relative_path']:item['sha256'] for item in manifest['production_source_pins']}
        r['source_pins']['src/core/gpu.cpp']='9792966f6b2c84434a1dddafefb3286c3c5fcfa78e23065b455904f9cf6fcfb8'
        m.require(all(m.sha(W/f)==h for f,h in r['source_pins'].items()),'exact independently reviewed actual source and headers')
        for mode in ['free','contrib-llvm']:
            q=B/('xe-e32-'+mode+'-full-build-root-v12/record.json')
            proof=json.loads(q.read_text())
            m.require(proof['passed'] and proof['complete'] and not proof['active'] and all(m.closed(c) for c in proof['commands']),'closed full build '+mode)
            r.setdefault('build_proof_sha256',{})[mode]=m.sha(q)
            defs=['-DSTRATA_PROBE_MODE_FREE='+('1' if mode=='free' else '0')]+(['-DSTRATA_NO_BLAS=1'] if mode=='free' else ['-USTRATA_NO_BLAS'])
            binary=out/('profiling-tags-'+mode)
            compile_args=[str(LLVM/'bin/clang++'),'-std=c++20','-O2','-Wall','-Wextra','-Werror','-fsycl','-fsycl-targets=spir64','-pthread','-I'+str(W/'include'),*defs,str(S/'profiling_tags_probe.cpp'),str(W/'src/core/gpu.cpp'),str(W/'src/core/device.cpp'),'-o',str(binary)]
            e,so,se=run('compile-'+mode,compile_args)
            m.require(e['exit_code']==0,'compile '+mode)
            r['binaries'][mode]=dict(path=str(binary),sha256=m.sha(binary),bytes=binary.stat().st_size)
            db=out/('lint-db-'+mode)
            db.mkdir()
            row=dict(directory=str(W),file=str(S/'profiling_tags_probe.cpp'),arguments=[*compile_args[:compile_args.index(str(S/'profiling_tags_probe.cpp'))],'-c',str(S/'profiling_tags_probe.cpp'),'-o',str(out/('unused-'+mode+'.o'))])
            (db/'compile_commands.json').write_text(json.dumps([row],indent=2)+'\n')
            e,so,se=run('lint-'+mode,['/bin/bash','tools/lint/run.sh',str(S/'profiling_tags_probe.cpp')],env=dict(LINTENV,STRATA_LINT_BUILD=str(db)))
            r.setdefault('lint',{})[mode]=dict(exit_code=e['exit_code'],stdout=so,stderr=se,compile_db_sha256=m.sha(db/'compile_commands.json'))
            m.require(e['exit_code']==0 and 'skipped (' not in so and 'possible secret' not in so,'all mandatory probe lints passed '+mode)
            for name in ['clang-tidy','cppcheck','codespell','reuse']:
                m.require('== '+name in so,'required probe tool '+name)
            e,so,se=run('invalid-cli-'+mode,[str(binary),'unexpected'],wall=10)
            m.require(e['exit_code']==2 and not so and se=='usage: profiling_tags_probe (no arguments)\n','invalid probe CLI refuses before Runtime')
            save()
        m.require(all(m.sha(W/f)==h for f,h in r['source_pins'].items()) and m.sha(S/'profiling_tags_probe.cpp')==r['probe_sha256'],'stable source/probe')
        r['complete']=True
    except BaseException as error:
        r['error']=repr(error)
    finally:
        r['active']=owner.active is not None
        r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc']=m.utc()
        save()
    print(json.dumps({k:r.get(k) for k in ['passed','error','binaries']}))
    raise SystemExit(0 if r['passed'] else 1)
