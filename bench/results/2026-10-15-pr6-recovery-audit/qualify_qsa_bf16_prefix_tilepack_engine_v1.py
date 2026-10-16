#!/usr/bin/env python3
"""Main-only sequential engine-linked candidate/control structural checks."""
from pathlib import Path
import hashlib,json,subprocess,sys
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent;P=A.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 dst=A/'qsa-bf16-prefix-tilepack-engine-qualification-v1.json';assert not dst.exists()
 child=A/'run_qsa_bf16_prefix_tilepack_engine_component_v1.py';digest=sha(child)
 r=dict(schema='strata-tilepack-engine-qualification-v1',passed=False,complete=False,active_controller=None,completed=[],child_sha256=digest,source_sha256=sha(Path(__file__)))
 def save():dst.write_text(json.dumps(r,indent=2)+'\n')
 try:
  for mode in ['smoke','component','base-smoke','base-component']:
   assert sha(child)==digest
   argv=[sys.executable,'-B',str(child),mode];p=subprocess.Popen(argv,cwd=P);r['active_controller']={'pid':p.pid,'argv':argv};save();code=p.wait();r['active_controller']=None
   q=A/f'own-qsa-bf16-prefix-tilepack-engine-{mode}-v1.json';assert code==0 and q.exists();v=json.loads(q.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['active'] and v['fresh_faults']==[]
   r['completed'].append({'mode':mode,'path':str(q),'sha256':sha(q),'cases':len(v['cases'])});save();print(json.dumps({'closed':mode,'cases':len(v['cases'])}),flush=True)
  r.update(passed=True,complete=True)
 except BaseException as e:r['error']=repr(e);raise
 finally:save()
if __name__=='__main__':main()
