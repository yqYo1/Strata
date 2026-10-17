#!/usr/bin/env python3
"""Serialize all actual-entry qualification cases before a model run."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent
out=A/'qsa-score-prefill-qualification-v1.json';assert not out.exists()
rec=dict(schema='strata-qsa-score-prefill-qualification-v1',passed=False,complete=False,active_sample=None,samples=[],order=['suite-default','suite-0','suite-8','full-0','full-8','tail-8'],scope='Actual prefll API whole-score/ID baseline parity, identity and sparse boundary, queries1..9, physical262144 with256 and255queries; not full model/RESTORE qualification.')
def save():out.write_text(json.dumps(rec,indent=2)+'\n')
try:
 for mode in rec['order']:
  p=A/f'qsa-score-prefill-{mode}-v1.json'
  if not p.exists():
   rec['active_sample']=mode;save()
   r=subprocess.run(['/usr/bin/python3',str(A/'run_qsa_score_prefill_v1.py'),mode],cwd=A.parents[2]);assert r.returncode==0,(mode,r.returncode)
  v=json.loads(p.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['active'] and v['fresh_faults']==[] and not v['driver_dump_entries']
  rec['samples'].append(dict(mode=mode,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()));rec['active_sample']=None;save()
 rec.update(passed=True,complete=True)
except BaseException as e:
 import traceback
 rec.update(error=repr(e),traceback=traceback.format_exc())
finally:rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps(dict(receipt=str(out),passed=rec['passed'],error=rec.get('error'))),flush=True)
sys.exit(0 if rec['passed'] else 1)
