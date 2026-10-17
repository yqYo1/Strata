#!/usr/bin/env python3
"""Serialize fresh-process QSA stage samples; never overwrite an existing run."""
import datetime, hashlib, json, subprocess, sys
from pathlib import Path
A=Path(__file__).resolve().parent
out=A/'qsa-select-stage-cohort-v1.json'
assert not out.exists()
record=dict(schema='strata-qsa-select-stage-cohort-v1',passed=False,complete=False,active_sample=None,samples=[],
            order=['78453-0','32768-0','131072-0','262144-0','262144-1','131072-1','32768-1','78453-1','78453-2','32768-2','131072-2','262144-2'],
            scope='Ten per-stage observations in each of three fresh processes per geometry; rotated stage order per repetition. Synthetic component evidence only.')
def save():out.write_text(json.dumps(record,indent=2)+'\n')
try:
 for mode in record['order']:
  dst=A/f'qsa-select-stage-{mode}-v1.json'
  if not dst.exists():
   record['active_sample']=mode;save()
   run=subprocess.run(['/usr/bin/python3',str(A/'run_qsa_select_stage_v1.py'),mode],cwd=A.parents[2],check=False)
   assert run.returncode==0,(mode,run.returncode)
  value=json.loads(dst.read_text())
  assert value['passed'] and value['complete'] and value['ownership_closed'] and not value['active'] and value['fresh_faults']==[]
  assert not value['driver_dump_entries'] and value['boot_unchanged']
  record['samples'].append(dict(mode=mode,path=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
  record['active_sample']=None;save()
 record.update(passed=True,complete=True)
except BaseException as e:
 import traceback
 record.update(error=repr(e),traceback=traceback.format_exc())
finally:
 record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps(dict(receipt=str(out),passed=record['passed'],error=record.get('error'))),flush=True)
sys.exit(0 if record['passed'] else 1)
