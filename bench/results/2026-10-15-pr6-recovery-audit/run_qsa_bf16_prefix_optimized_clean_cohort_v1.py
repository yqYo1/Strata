#!/usr/bin/env python3
"""Main serial repeated official bench.sh cohort; each child owns native supervision."""
import hashlib,json,subprocess,sys,datetime
from pathlib import Path
A=Path(__file__).resolve().parent;P=A.parents[2]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 out=A/'qsa-bf16-prefix-optimized-clean-cohort-v1.json';assert not out.exists()
 prior=A/'own-qsa-bf16-prefix-optimized-diagnostic-bf16-prefix-optimized-r0-v1.json';v=json.loads(prior.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults']
 child=A/'run_qsa_bf16_prefix_optimized_long_v1.py';child_sha=sha(child);prior_sha=sha(prior)
 order=[('bf16-prefix',0),('bf16-prefix-optimized',0),('bf16-prefix-optimized',1),('bf16-prefix',1),('bf16-prefix',2),('bf16-prefix-optimized',2)]
 rec=dict(schema='strata-bf16-prefix-clean-cohort-v1',passed=False,complete=False,source=sha(__file__),child=child_sha,order=order,completed=[],active_controller=None,performance_adopted=False,quality_accepted=False,physical256K_model_qualified=False,RESTORE_qualified=False,scope='Separate native processes, official bench.sh actual32K+ input, thinking ON/24 output cap; baseline versus score-pitch33 attention, same BF16 boundaries and4byte-aligned matrix arrays. Six individual samples retained. No parallel test/build/cleanup, no retry or failed sample substitution.')
 def save():out.write_text(json.dumps(rec,indent=2)+'\n')
 save()
 try:
  for arm,n in order:
   assert sha(child)==child_sha and sha(prior)==prior_sha and sha(__file__)==rec['source']
   target=A/f'own-qsa-bf16-prefix-optimized-clean-{arm}-r{n}-v1.json';assert not target.exists()
   argv=[sys.executable,str(child),'clean',arm,str(n),'--prior',str(prior),'--prior-sha256',prior_sha]
   print(json.dumps(dict(start=arm,repetition=n)),flush=True)
   proc=subprocess.Popen(argv,cwd=P);rec['active_controller']=dict(pid=proc.pid,argv=argv);save();code=proc.wait();rec['active_controller']=None
   item=dict(arm=arm,repetition=n,controller_exit=code,path=str(target));rec['completed'].append(item);save();assert code==0 and target.is_file()
   v=json.loads(target.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults'];assert v['binary_sha256']==json.loads(prior.read_text())['binary_sha256']
   item.update(sha256=sha(target),prompt_tps=v['benchmark_result']['results']['prefill_only']['prompt_tps']);prior,prior_sha=target,sha(target);save()
  rec.update(passed=True,complete=True)
 except BaseException as e:rec['error']=repr(e)
 finally:rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 print(json.dumps(dict(passed=rec['passed'],complete=rec['complete'],receipt=str(out))),flush=True)
 return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
