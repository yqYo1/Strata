#!/usr/bin/env python3
"""Confirm the screened tile8 against original at short/target/full prefix."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent
out=A/'qsa-score-querytile-cohort-v1.json';assert not out.exists()
order=['0-78453-0','8-78453-0']
for ctx in [32768,78453,262144]:
 for rep in range(3):
  for tile in ([0,8] if rep%2==0 else [8,0]):
   name=f'{tile}-{ctx}-{rep}'
   if name not in order:order.append(name)
rec=dict(schema='strata-qsa-score-querytile-cohort-v1',passed=False,complete=False,active_sample=None,samples=[],order=order,
         scope='Selected tile8 screening winner versus same-binary original; three fresh processes per context/arm, ten stage samples each. Same seeded inputs and whole-score/ID bit identity; no model PP claim.')
def save():out.write_text(json.dumps(rec,indent=2)+'\n')
try:
 for name in order:
  dst=A/f'qsa-score-querytile-{name}-v1.json'
  if not dst.exists():
   rec['active_sample']=name;save()
   r=subprocess.run(['/usr/bin/python3',str(A/'run_qsa_score_querytile_v1.py'),name],cwd=A.parents[2]);assert r.returncode==0
  v=json.loads(dst.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults'] and not v['active']
  rec['samples'].append(dict(mode=name,path=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()));rec['active_sample']=None;save()
 rec.update(passed=True,complete=True)
except BaseException as e:
 import traceback
 rec.update(error=repr(e),traceback=traceback.format_exc())
finally:rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps(dict(receipt=str(out),passed=rec['passed'],error=rec.get('error'))),flush=True)
sys.exit(0 if rec['passed'] else 1)
