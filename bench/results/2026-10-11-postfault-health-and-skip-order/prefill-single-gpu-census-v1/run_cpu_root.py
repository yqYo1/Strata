"""Root owns the exact-header synthetic reconciliation checks; CPU only."""
import fcntl
import importlib.util
import json
from pathlib import Path

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');HERE=Path(__file__).resolve().parent
OWNER=B/'xestrata-clean-64k-comparison-v4/direct_owner.py'
spec=importlib.util.spec_from_file_location('qualified_owner',OWNER)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.require(m.sha(OWNER)=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686','owner pin')
pins={'qualify_census.cpp':'737ae6ac3f93f610798fa4b234e7acd369bb46da9511ef3dd206d61e8984dd6f',
      'check_receipt.py':'daa39bbe3c0da63fc4d1f05c3bfc8448bf6dcdbb11485727a21fb3c5a8038608',
      'include/strata/prefill_route_census.hpp':'c673382e9f4bf63919489cf00daa71bec793ff7b4584c203291380b827dc110b'}
cases=['valid32','valid64','valid262','collector-bound','off','unsupported','missing-product','unrouted-copy','unknown-branch']
with (B/'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=B/'prefill-single-gpu-census-cpu-root-r1';m.require(not out.exists(),'new output');out.mkdir(mode=0o700)
    record=dict(active=True,complete=False,passed=False,gpu_executed=False,model_executed=False,
                caller_refactor_adopted=False,source_sha256=pins,controller_sha256=m.sha(__file__),
                started_utc=m.utc(),scope='actual unchanged header, synthetic complete single-GPU counts and rejected invalid routes',cases=[])
    def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    owner=m.Owner(out/'commands',save);record['commands']=owner.commands
    env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
    def run(label,args,wall=15):
        e,so,se=owner.run(label,args,env,HERE,wall=wall,text_cap=32<<20,rss_cap=2<<30,total_cap=64<<20,cpu=wall,file_cap=32<<20)
        m.require(m.closed(e) and e['exit_code']==0,'normal CPU command '+label)
        return so,se
    save()
    try:
        for name,pin in pins.items():m.require(m.sha(HERE/name)==pin,'source pin '+name)
        compiler=Path('/usr/bin/g++');record['compiler']=dict(path=str(compiler.resolve()),sha256=m.sha(compiler))
        binary=out/'qualify-census'
        run('build',[str(compiler),'-std=c++20','-O2','-Wall','-Wextra','-Werror','-pthread','-I',str(HERE/'include'),str(HERE/'qualify_census.cpp'),'-o',str(binary)],30)
        record['binary']=dict(path=str(binary),sha256=m.sha(binary),bytes=binary.stat().st_size)
        for case in cases:
            so,se=run(case,[str(binary),case],20)
            check,_=run(case+'-check',['/usr/bin/python3',str(HERE/'check_receipt.py'),case,str(se)],15)
            result=json.loads(check.read_text());m.require(result['passed'] is True and result['case']==case,'closed case checker')
            rows=[json.loads(v) for v in se.read_text().splitlines()]
            receipts=[r for r in rows if r['kind']=='prefill_route_census_receipt']
            record['cases'].append(dict(case=case,check=result,raw_stderr_sha256=m.sha(se),
                                        raw_stderr_bytes=se.stat().st_size,layer_rows=len(rows)-len(receipts),
                                        receipt=receipts[0] if receipts else None))
            save()
        for name,pin in pins.items():m.require(m.sha(HERE/name)==pin,'stable source '+name)
        record.update(complete=True,passed=True,checked=cases)
    except BaseException as exc:record['error']=repr(exc)
    finally:
        record['active']=owner.active is not None;record['finished_utc']=m.utc();save()
    print(json.dumps(dict(record=str(out/'record.json'),passed=record['passed'],error=record.get('error'))))
    raise SystemExit(0 if record['passed'] else 1)
