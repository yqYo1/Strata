import fcntl,hashlib,importlib.util,json,sys
from pathlib import Path
B=Path(__file__).parent;Q=B/'xestrata-clean-64k-comparison-v4';OUT=B/'xestrata-clean64k-v4-protocol-root-v1'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
with (B/'owned-v0141-measurement.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not OUT.exists();OUT.mkdir(mode=0o700)
 assert sha(Q/'run_clean.py')=='b809c8baba4264898b1b7901c15d7f29a486d4e61faa833da2d090518bc29580'
 assert sha(Q/'direct_owner.py')=='8b40cc573b9abd069f8386deb6af65f0a55f7992bcda2a6a62b51a5c9a47b686'
 sys.path.insert(0,str(Q));from direct_owner import Owner,closed
 r=dict(active=True,passed=False,complete=False,gpu_executed=False,model_executed=False,controller_sha256=sha(Q/'run_clean.py'),owner_sha256=sha(Q/'direct_owner.py'),tests=[])
 def save():(OUT/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 o=Owner(OUT/'processes',save);r['commands']=o.commands;save()
 env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1')
 try:
  e,so,se=o.run('syntax',['/usr/bin/python3','-c','import pathlib,sys\nfor p in sys.argv[1:]: compile(pathlib.Path(p).read_text(),p,"exec")',str(Q/'run_clean.py'),str(Q/'direct_owner.py'),str(Q/'test_contract.py')],env,Q,wall=10,text_cap=1<<20,rss_cap=2<<30,cpu=10)
  assert closed(e)and e['exit_code']==0;r['tests'].append(dict(case='syntax-3-files',passed=True))
  e,so,se=o.run('synthetic-contract',['/usr/bin/python3',str(Q/'test_contract.py')],env,Q,wall=20,text_cap=1<<20,rss_cap=2<<30,cpu=20)
  assert closed(e)and e['exit_code']==0;r['tests'].append(dict(case='4K8K-geometry-earlystop-negative-and-synthetic-gates',passed=True,scope='4 unittest groups; synthetic receipt/hash mocks cannot admit real GPU'))
  import run_clean as c
  actual=B/'owned-clean64k-v3-baseline-r1/record.json';d=json.loads(actual.read_text());v=c.validate(d['request'],65536,8192)
  assert v['prefill_tps']==d['sample']['prefill_tps']and v['decode_performance_eligible']is False and v['decode_tps']is None
  r['tests'].append(dict(case='actual64K-baseline-prefill-valid-decode3-ineligible',passed=True,source_receipt=str(actual),source_sha256=sha(actual),validation=v,original_receipt_unchanged=True))
  for name,chunk in [('owned-xestrata39-hostusm32k-diagnostic-v7-r1',8192),('owned-xestrata39-hostusm4k-code32k-diagnostic-v8-r1',4096)]:
   p=B/name/'record.json'
   try:c.load_gate(p,sha(p),'xe',chunk)
   except RuntimeError as ex:r['tests'].append(dict(case='actual-failed-fork-not-admitted-'+str(chunk),passed=True,source_receipt=str(p),source_sha256=sha(p),rejection=str(ex)))
   else:raise AssertionError('real failed fork admitted')
  admission=Q.parent/'xestrata-clean-64k-comparison-v3/baseline-math-execution-admission.json'
  c.load_gate(admission,sha(admission),'baseline')
  r['tests'].append(dict(case='real-baseline32K-math-only-admission-originalFAILED-retained',passed=True,source_sha256=sha(admission)))
  r.update(passed=True,complete=True)
 except BaseException as ex:r['error']=repr(ex)
 finally:r['active']=o.active is not None;save()
 print(json.dumps(dict(passed=r['passed'],record=str(OUT/'record.json'),gpu_executed=False)))
 raise SystemExit(0 if r['passed']else 1)
