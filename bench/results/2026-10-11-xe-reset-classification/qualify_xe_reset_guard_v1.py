"""CPU-only guard tests and replay of the immutable failed driver interval."""
import fcntl
import importlib.util
import json
from pathlib import Path
import types

B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-sycl-health-reset-classification-20261011')
OWNER=B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
spec=importlib.util.spec_from_file_location('qualified_owner',OWNER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(OWNER)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'xe-reset-guard-cpu-root-v1';m.require(not out.exists(),'new output');out.mkdir(mode=0o700)
    files=['sycl/tools/recover-xe.sh','sycl/tools/test_recover_xe.py']
    record=dict(active=True,completed=False,passed=False,gpu_executed=False,recovery_executed=False,
                model_executed=False,source_sha256={p:m.sha(W/p) for p in files},
                controller_sha256=m.sha(__file__),started_utc=m.utc())
    def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    owner=m.Owner(out/'commands',save);record['commands']=owner.commands
    env=dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
    save()
    try:
        e,so,se=owner.run('unit-tests',['/usr/bin/python3','sycl/tools/test_recover_xe.py'],env,W,
                          wall=60,text_cap=2<<20,rss_cap=1<<30,total_cap=16<<20,cpu=60)
        m.require(m.closed(e) and e['exit_code']==0,'finite CPU guard suite')
        text=se.read_text();m.require(text.rstrip().endswith('OK'),'unittest completed')
        record['unittest_summary']=text
        old=B/'postfault-status-health-root-r2/record.json'
        m.require(m.sha(old)=='7d83b26c2bf260004e83d1c3bab81fe6b73d67a419b763cea458a26d25937ea6','original failure pin')
        data=json.loads(old.read_text());m.require(not data['active'] and not data['passed'],'failed interval preserved')
        source=(W/files[0]).read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
        helper=types.ModuleType('guard_replay');exec(compile(source,str(W/files[0]),'exec'),helper.__dict__)
        rows=data['kernel_device_entries']
        matches=[r for r in rows if helper.FAULT.search(r['MESSAGE'])]
        reset=[r for r in matches if 'reset started' in r['MESSAGE']]
        m.require(len(rows)==199 and len(reset)==32,'all observed reset-start messages classified')
        for phrase,count in [('Timedout job',31),('Check job timeout',30),('Timed out wait for G2H',1)]:
            m.require(sum(phrase in r['MESSAGE'] for r in matches)==count,'actual timeout coverage '+phrase)
        record['replay']=dict(original_sha256=m.sha(old),kernel_entries=len(rows),
                              original_regex_matches=len(data['new_fault_messages']),
                              updated_regex_matches=len(matches),reset_started_matches=len(reset),
                              scope='existing failed interval replay only; no new live fault/health test')
        for p in files:m.require(m.sha(W/p)==record['source_sha256'][p],'stable tested source')
        record.update(completed=True,passed=True)
    except BaseException as exc:record['error']=repr(exc)
    finally:
        record['active']=owner.active is not None;record['finished_utc']=m.utc();save()
    print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'))))
    raise SystemExit(0 if record['passed'] else 1)
