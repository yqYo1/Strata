#!/usr/bin/env python3
"""First clean screen, same binary and seed, one process for each explicit tile."""
import datetime,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent
out=A/'qsa-score-querytile-screen-v1.json';assert not out.exists()
rec=dict(schema='strata-qsa-score-querytile-screen-v1',passed=False,complete=False,active_sample=None,samples=[],order=['0-78453-0','2-78453-0','4-78453-0','8-78453-0'],scope='Screen only; no production adoption or model performance claim.')
def save():out.write_text(json.dumps(rec,indent=2)+'\n')
try:
 for name in rec['order']:
  dst=A/f'qsa-score-querytile-{name}-v1.json';assert not dst.exists()
  rec['active_sample']=name;save()
  r=subprocess.run(['/usr/bin/python3',str(A/'run_qsa_score_querytile_v1.py'),name],cwd=A.parents[2]);assert r.returncode==0
  v=json.loads(dst.read_text());assert v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults']
  rec['samples'].append(dict(path=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()));rec['active_sample']=None;save()
 rec.update(passed=True,complete=True)
except BaseException as e:
 import traceback
 rec.update(error=repr(e),traceback=traceback.format_exc())
finally:rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps(dict(receipt=str(out),passed=rec['passed'],error=rec.get('error'))),flush=True)
sys.exit(0 if rec['passed'] else 1)
