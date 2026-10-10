import fcntl, importlib.util, json
from pathlib import Path

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
WX=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
OWNER=B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(OWNER)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')
ENV=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')

def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=B/'xe-e32-commit-cpu-root-v1';m.require(not out.exists(),'immutable new output');out.mkdir()
        record=dict(active=True,complete=False,passed=False,commands=[],started_utc=m.utc(),gpu_executed=False,
                    full262144_runtime_qualified=False,scope='Production CommitTransaction helper with event/queue/submission fault injection; plain NDEBUG and ASan/UBSan. No GPU API or full Verifier execution.')
        def save(): (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(out/'commands',save);record['commands']=owner.commands
        def cmd(label,argv,wall=60,env=ENV):
            e,so,se=owner.run(label,argv,env,HERE,wall=wall,text_cap=2<<20,rss_cap=4<<30,cpu=wall,total_cap=8<<20)
            m.require(m.closed(e) and e['exit_code']==0,'closed normal '+label)
            return so.read_text()
        save()
        try:
            files=['include/strata/core/commit_transaction.hpp','include/strata/core/verify.hpp','src/core/verify.cpp','tools/test_commit_transaction.cpp']
            record['controller_sha256']=m.sha(__file__)
            record['source_pins']={f:m.sha(WX/f) for f in files}
            record['compiler_version']=cmd('compiler-version',['/usr/bin/g++','--version'])
            for mode in ['plain-ndebug','asan-ubsan']:
                options=['-O2','-DNDEBUG'] if mode=='plain-ndebug' else ['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
                exe=out/('commit-'+mode)
                cmd('compile-'+mode,['/usr/bin/g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic',*options,'-I'+str(WX/'include'),str(WX/files[3]),'-o',str(exe)])
                env=dict(ENV,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1') if mode!='plain-ndebug' else ENV
                output=cmd('run-'+mode,[str(exe)],wall=20,env=env)
                record.setdefault('results',[]).append(dict(case='production-commit-helper-fault-injection',mode=mode,stdout=output,binary_sha256=m.sha(exe)))
            record['source_stable']=all(m.sha(WX/f)==h for f,h in record['source_pins'].items());m.require(record['source_stable'],'tested source stable')
            record['complete']=True
        except BaseException as exc: record['error']=repr(exc)
        finally:
            record['active']=owner.active is not None;record['passed']=record['complete'] and not record['active'] and not record.get('error');record['finished_utc']=m.utc();save()
        print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'),results=record.get('results'))))
        return 0 if record['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
