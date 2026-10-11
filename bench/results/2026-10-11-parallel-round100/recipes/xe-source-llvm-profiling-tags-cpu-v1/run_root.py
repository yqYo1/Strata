import fcntl,importlib.util,json,time
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');T=B/'xe-source-llvm-profiling-tags-v1';O=B/'xe-llvm-v711-source-v5/direct_owner.py'
s=importlib.util.spec_from_file_location('owner',O);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.require(m.sha(O)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f','qualified owner')
with (B/'owned-v0141-measurement.lock').open('a') as l:
 fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=B/'xe-source-llvm-profiling-tags-cpu-root-v1';m.require(not out.exists(),'immutable output');out.mkdir()
 r=dict(active=True,passed=False,complete=False,commands=[],controller_sha256=m.sha(__file__),owner_sha256=m.sha(O),source_pins={f:m.sha(T/f) for f in ['run_root.py','protocol.py','protocol_selftest.py']},gpu_executed=False,model_executed=False,started_utc=m.utc(),scope='CPU parser tests only; no runtime/timestamp proof')
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 owner=m.Owner(out/'commands',save);r['commands']=owner.commands;save()
 try:
  for f in r['source_pins']:compile((T/f).read_text(),str(T/f),'exec')
  e,so,se=owner.run('protocol-selftest',['/usr/bin/python3',str(T/'protocol_selftest.py')],dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8'),T,wall=30,cpu=30,text_cap=1<<20,total_cap=1<<20,rss_cap=1<<30)
  m.require(m.closed(e) and e['exit_code']==0 and not se.read_text(),'closed passing parser selftest')
  r['selftest']=json.loads(so.read_text());m.require(r['selftest']['passed'] and r['selftest']['positive_cases']==2 and r['selftest']['negative_cases']==16,'expected positive/negative tests')
  m.require(all(m.sha(T/f)==h for f,h in r['source_pins'].items()),'stable recipes');r['complete']=True
 except BaseException as e:r['error']=repr(e)
 finally:
  r['active']=owner.active is not None;r['passed']=bool(r['complete'] and not r['active'] and not r.get('error'));r['finished_utc']=m.utc();save()
 print(json.dumps({k:r.get(k) for k in ['passed','error','selftest']}));raise SystemExit(0 if r['passed'] else 1)
