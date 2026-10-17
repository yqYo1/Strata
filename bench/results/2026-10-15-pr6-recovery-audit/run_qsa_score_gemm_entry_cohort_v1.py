#!/usr/bin/env python3
"""Sequential Main actual-entry qualification; never overlap native processes."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
P=A.parents[2]

def main():
 out=A/'qsa-score-gemm-entry-cohort-v1.json';assert not out.exists()
 build=A/'qsa-score-gemm-entry-engine-build-v2.json';b=json.loads(build.read_text());assert b['passed'] and b['complete'] and b['ownership_closed'] and not b['active'] and b['fresh_faults']==[]
 jobs=[('gemm','small'),('head','small'),('gemm','suite'),('head','suite'),('gemm','fault'),('gemm','zero'),('gemm','full'),('gemm','tail'),('head','full')]
 rec=dict(schema=1,passed=False,complete=False,active=True,build_sha256=hashlib.sha256(build.read_bytes()).hexdigest(),jobs=[])
 def save():out.write_text(json.dumps(rec,indent=2)+'\n')
 save()
 try:
  for backend,mode in jobs:
   dst=A/f'qsa-score-gemm-entry-{backend}-{mode}-r0-v1.json';assert not dst.exists()
   argv=[sys.executable,str(A/'run_qsa_score_gemm_entry_v1.py'),mode,'0',backend]
   item=dict(backend=backend,mode=mode,argv=argv);rec['jobs'].append(item);save()
   r=subprocess.run(argv,cwd=P);item['exit_code']=r.returncode
   if dst.exists():
    d=json.loads(dst.read_text());item.update(receipt=dst.name,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),passed=d['passed'],complete=d['complete'],ownership_closed=d['ownership_closed'],fresh_faults=d['fresh_faults'])
   save();assert r.returncode==0 and item.get('passed') and item.get('complete') and item.get('ownership_closed') and item.get('fresh_faults')==[],item
  rec.update(passed=True,complete=True)
 except BaseException as e:
  rec['error']=repr(e)
 finally:
  rec['active']=False;rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 print(json.dumps(dict(receipt=str(out),passed=rec['passed'],complete=rec['complete'],error=rec.get('error'))),flush=True)
 return 0 if rec['passed'] else 1
if __name__=='__main__':sys.exit(main())
