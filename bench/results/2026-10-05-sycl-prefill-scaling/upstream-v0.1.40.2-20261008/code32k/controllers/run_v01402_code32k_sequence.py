from pathlib import Path
import json,subprocess,sys,time,datetime,hashlib
b=Path(__file__).parent
out=b/'v01402-code32k-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'steps':[],'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'minimum_comparison_tokens':32768,'ordering':'logged pure and patched controls, then clean pure/patch/patch/pure (ABBA); no diagnostic time included in clean comparison'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
 deadline=time.monotonic()+900
 while True:
  d=json.loads((b/'owned-v01402-code32k-pure-diagnostic-r1/record.json').read_text())
  if not d['active']:
   assert d['healthy'];break
  assert time.monotonic()<deadline
  time.sleep(1)
 for mode,phase,rep in [('patched-default','diagnostic',1),('pure','clean',1),('patched-default','clean',1),('patched-default','clean',2),('pure','clean',2)]:
  argv=[sys.executable,str(b/'run_owned_v01402_code32k_v2.py'),mode,phase,str(rep)]
  step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START',mode,phase,rep,flush=True)
  log=out/f'{mode}-{phase}-r{rep}.stdout'
  with log.open('w') as f:rc=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT).returncode
  step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(log));save();print('FINISH',mode,phase,rep,rc,flush=True)
  assert rc==0
 record['passed']=True
except BaseException as e:
 record['error']=repr(e)
finally:
 record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
if not record['passed']:raise SystemExit(1)
