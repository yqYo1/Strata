#!/usr/bin/env python3
"""Main-only sequential supervision of both actual engine-linked component arms."""
import hashlib,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
A=Path(__file__).resolve().parent
out=A/'qsa-bf16-prefix-optimized-qualification-v1.json'
assert not out.exists()
child=A/'run_qsa_bf16_prefix_optimized_engine_component_v1.py'
digest=hashlib.sha256(child.read_bytes()).hexdigest()
r=dict(passed=False,complete=False,order=['smoke','component','base-smoke','base-component'],completed=[],active_controller=None,child_sha256=digest)
def save():out.write_text(json.dumps(r,indent=2)+'\n')
try:
 for mode in r['order']:
  assert hashlib.sha256(child.read_bytes()).hexdigest()==digest
  p=subprocess.Popen([sys.executable,'-B',str(child),mode]);r['active_controller']=p.pid;save();code=p.wait();r['active_controller']=None
  q=A/f'own-qsa-bf16-prefix-optimized-engine-{mode}-v1.json';v=json.loads(q.read_text())
  r['completed'].append(dict(mode=mode,exit_code=code,path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest()));save()
  assert code==0 and v['passed'] and v['complete'] and v['ownership_closed'] and not v['active'] and not v['fresh_faults']
 r.update(passed=True,complete=True)
except BaseException as e:r['error']=repr(e)
finally:save()
print(json.dumps(r),flush=True)
sys.exit(0 if r['passed'] else 1)
