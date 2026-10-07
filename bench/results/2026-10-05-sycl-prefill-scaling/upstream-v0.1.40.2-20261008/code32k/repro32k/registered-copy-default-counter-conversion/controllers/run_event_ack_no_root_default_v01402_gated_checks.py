from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
base=Path(__file__).parent
out=base/'event-ack-no-root-default-v01402-gated-checks'
out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'scope':'Wait for clean scheduling ABBA to complete, then logged-first32K and two fresh32K full state/head checks on the same no-root registered-copy binary, changing only implicit counter conversion from forced-off to default. No concurrent GPU jobs, clean timing or full256K claim.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
 deadline=time.monotonic()+2400
 while True:
  current=json.loads((base/'registered-vs-no-root-v01402-clean-sequence/record.json').read_text())
  if not current['active']:break
  assert time.monotonic()<deadline
  time.sleep(1)
 assert current['passed'],'clean32K comparison failed; do not launch the default-backend test'
 commands=[
  [sys.executable,str(base/'run_owned_event_ack_no_root_default_v01402_code32k.py'),'event-ack-no-root-prefill-default','diagnostic','1'],
  [sys.executable,str(base/'run_event_ack_no_root_default_v01402_state_sequence.py')],
 ]
 for index,argv in enumerate(commands):
  step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save()
  print('START DEFAULT CHECK',index,flush=True)
  with (out/f'check-{index}.stdout').open('w') as stream:
   rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=2400).returncode
  step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
 sequence=json.loads((base/'event-ack-no-root-prefill-default-v01402-state-sequence/record.json').read_text())
 assert sequence['passed'] and not sequence['active']
 record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
