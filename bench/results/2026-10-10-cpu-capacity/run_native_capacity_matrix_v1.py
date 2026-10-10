from pathlib import Path
import subprocess,json,hashlib,datetime
B=Path(__file__).parent
controller=B/'run_native_capacity_v2_controller2.py'
plan=[dict(NT=nt,mode=mode,repeat=r) for r in (1,2,3) for j,nt in enumerate((1,3,2,4)) for mode in (('streaming','hot') if (r+j)%2 else ('hot','streaming'))]
out=B/'native-capacity-v2-matrix-plan-v1.json';assert not out.exists()
d=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),plan=plan,closed=[],active=False,complete=False,passed=False,scope='Serial fresh processes;3 repeats eachNT1–4 eachhot8/streaming192;25 measurement rounds per2cohort per5arms; no model/GPU')
def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2)+'\n');p.replace(out)
save()
for cell in plan:
 receipt=B/f"native-capacity-v2-measure-nt{cell['NT']}-{cell['mode']}-r{cell['repeat']}-controller2/record.json"
 if receipt.exists():
  assert cell==dict(NT=1,mode='streaming',repeat=1)
  r=json.loads(receipt.read_text());assert r['passed'] and r['complete'] and not r['active'];status=0
 else:
  d['active']=True;d['current']=cell;save()
  status=subprocess.run(['/usr/bin/python3',str(controller),'measure',str(cell['NT']),cell['mode'],str(cell['repeat'])]).returncode
  r=json.loads(receipt.read_text());d['active']=False
 d['closed'].append(dict(**cell,returncode=status,path=str(receipt),sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),passed=r['passed'],complete=r['complete'],active=r['active']));save()
 if status or not r['passed'] or not r['complete'] or r['active']:raise SystemExit('Rejected matrix cell; no retry or gate change')
d['complete']=True;d['passed']=True;d['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();print('MATRIX_CLOSED_PASS',flush=True)
