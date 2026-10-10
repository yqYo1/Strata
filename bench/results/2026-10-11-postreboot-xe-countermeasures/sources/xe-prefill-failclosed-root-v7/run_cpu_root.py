import fcntl, importlib.util, json
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-b570-prefill-publication-20261011')
HERE=Path(__file__).resolve().parent
OWNER=B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
s=importlib.util.spec_from_file_location('qualified_owner',OWNER);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(OWNER)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')
pins={'src/prefill/prefill.cpp':'59e1c4ea18c38da799314439099bcf14e9a19f0b7ce9f4f00629173a534cf9ce',
      'include/strata/prefill/publication.hpp':'547397243c32126f26caebad4baa39278dcf1471f3bff368af4fc481e5462710',
      'tools/test_prefill_publication.cpp':'205108edebe58650a46e8fa8d6f844cff692c5ae70e0af61d561706b16a7e71f'}
def exact_struct(text,label):
    a=text.index(('class ' if label=='HostVec' else 'struct ')+label+' {');p=text.index('{',a);depth=1;i=p+1
    while depth:
        if text[i]=='{':depth+=1
        if text[i]=='}':depth-=1
        i+=1
    return ('template <class T> ' if label=='HostVec' else '')+text[a:i]+';\n'
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-prefill-failclosed-cpu-root-v7';m.require(not out.exists(),'new output');out.mkdir(mode=0o700)
    record=dict(active=True,complete=False,passed=False,gpu_executed=False,model_executed=False,adopted=False,
                started_utc=m.utc(),source_sha256=pins,controller_sha256=m.sha(__file__),
                harness_sha256=m.sha(HERE/'qualify_stager.cpp'))
    def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    owner=m.Owner(out/'commands',save);record['commands']=owner.commands
    env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
    def run(label,args,wall=15):
        e,so,se=owner.run(label,args,env,W,wall=wall,text_cap=1<<20,rss_cap=1<<30,total_cap=16<<20,cpu=wall)
        m.require(m.closed(e) and e['exit_code']==0,'normal CPU execution '+label)
        return so.read_text()
    save()
    try:
        for f,p in pins.items():m.require(m.sha(W/f)==p,'actual source pin '+f)
        text=(W/'src/prefill/prefill.cpp').read_text()
        for label,name in [('Stager','stager-exact.inc'),('StagerDone','stager-guard-exact.inc'),('HostVec','hostvec-exact.inc')]:
            (out/name).write_text(exact_struct(text,label))
        record['extracted_actual_production']={name:m.sha(out/name) for name in ['stager-exact.inc','stager-guard-exact.inc','hostvec-exact.inc']}
        record['compiler']=dict(path='/usr/bin/g++',sha256=m.sha('/usr/bin/g++'))
        include=['-I',str(W/'include'),'-I',str(out)]
        flags=['/usr/bin/g++','-std=c++20','-O2','-pthread','-Wall','-Wextra']
        run('build-helper',flags+include+[str(W/'tools/test_prefill_publication.cpp'),'-o',str(out/'helper')],30)
        helper=run('qualify-helper',[str(out/'helper')],10)
        m.require(helper=='production publication helper qualification passed\n','complete shared helper contract')
        run('build-actual-stager',flags+include+[str(HERE/'qualify_stager.cpp'),'-o',str(out/'stager')],30)
        result=json.loads(run('qualify-actual-stager',[str(out/'stager')],10))
        m.require(result==dict(passed=True,actual_stager_cases=11,gpu_executed=False),'actual extracted stager cases')
        e,so,se=owner.run('qualify-terminal-cleanup',[str(out/'stager'),'--cleanup-failure'],env,W,wall=10,text_cap=65536,rss_cap=1<<30,total_cap=16<<20,cpu=10)
        m.require(m.closed(e) and e['exit_code']==74,'expected terminal cleanup refusal')
        m.require(so.read_text()=='BEFORE_UNWIND\n' and 'GPU completion unconfirmed' in se.read_text() and 'RELEASED' not in so.read_text(),'no buffer-release destructor after failed wait')
        record['terminal_cleanup_expected_negative']=dict(passed=True,expected_exit_code=74,buffer_release_destructor_executed=False)
        record['result']=result;record['helper_result']=helper.strip()
        for f,p in pins.items():m.require(m.sha(W/f)==p,'stable source '+f)
        record.update(complete=True,passed=True)
    except BaseException as exc:record['error']=repr(exc)
    finally:record['active']=owner.active is not None;record['finished_utc']=m.utc();save()
    print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'))))
    raise SystemExit(0 if record['passed'] else 1)
