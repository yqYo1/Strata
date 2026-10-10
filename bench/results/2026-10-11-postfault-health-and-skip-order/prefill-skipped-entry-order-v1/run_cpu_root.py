"""Finite root-owned CPU qualification of source-extracted skip ordering."""
import fcntl
import importlib.util
import json
from pathlib import Path

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
HERE=Path(__file__).resolve().parent
OWNER=B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
spec=importlib.util.spec_from_file_location('qualified_owner',OWNER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(OWNER)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner source pin')
pins={'prefill.cpp':'e0685a85bc391b6ef916c33ea3f959680088832b8a9a7d706668acba3c47f8d0',
      'extract_contract.py':'01db5837c11d2a872b37f8efc9e129585a61ac26048883d3dc889eafdb4018cb',
      'qualify_order.cpp':'20108c5a18c7967d5a79c647f2a633f0752a52fb3dcadf77f3849befeb3a3e45'}

with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'prefill-skipped-entry-order-cpu-root-r1';m.require(not out.exists(),'new output')
    out.mkdir(mode=0o700)
    record=dict(active=True,complete=False,passed=False,gpu_executed=False,model_executed=False,
                production_SYCL_compiled=False,adopted=False,controller_sha256=m.sha(__file__),
                source_sha256=pins,started_utc=m.utc(),
                scope='actual source-extracted consumer ordering with safe instrumented issuer/event stand-ins')
    def save():
        (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    owner=m.Owner(out/'commands',save);record['commands']=owner.commands
    env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
    def run(label,args,wall=15):
        e,so,se=owner.run(label,args,env,HERE,wall=wall,text_cap=1<<20,rss_cap=2<<30,total_cap=32<<20,cpu=wall)
        m.require(m.closed(e) and e['exit_code']==0,'normal finite CPU command '+label)
        return so.read_text()
    save()
    try:
        for name,pin in pins.items():m.require(m.sha(HERE/name)==pin,'source pin '+name)
        extracted=out/'extracted'
        record['extraction']=json.loads(run('extract',['/usr/bin/python3',str(HERE/'extract_contract.py'),'--output',str(extracted)]))
        compiler=Path('/usr/bin/g++');record['compiler']=dict(path=str(compiler.resolve()),sha256=m.sha(compiler))
        binary=out/'qualify-order'
        run('build',[str(compiler),'-std=c++20','-O2','-Wall','-Wextra','-Werror','-pthread',
                     '-I',str(extracted),str(HERE/'qualify_order.cpp'),'-o',str(binary)],30)
        record['binary']=dict(path=str(binary),sha256=m.sha(binary),bytes=binary.stat().st_size)
        result=json.loads(run('qualify',[str(binary)],20));record['result']=result
        m.require(result==dict(passed=True,cases=112,old_threaded_expected_negative=True,
                              new_publication_order_passed=True,gpu_executed=False),'all bounded ordering cases')
        for name,pin in pins.items():m.require(m.sha(HERE/name)==pin,'unchanged source '+name)
        record.update(complete=True,passed=True)
    except BaseException as exc:record['error']=repr(exc)
    finally:
        record['active']=owner.active is not None;record['finished_utc']=m.utc();save()
    print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'))))
    raise SystemExit(0 if record['passed'] else 1)
