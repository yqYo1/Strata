from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time
base=Path(__file__).parent
out=base/'cb-off-v01402-clean-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'steps':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Wait for three complete32K state/head controls; run two fresh32K processes without API logging, validation, extra waits or state/head dumps. Require all64 IDs and protocol logprobs to match the captured control. No upstream parity/full256K proof.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
 gate=base/'cb-off-v01402-state-sequence/record.json';deadline=time.monotonic()+2400
 while True:
  current=json.loads(gate.read_text())
  if not current['active']:break
  assert time.monotonic()<deadline;time.sleep(1)
 assert current['passed'],'state/head gate failed; do not run timing jobs'
 for rep in [1,2]:
  argv=[sys.executable,str(base/'run_owned_cb_off_v01402_code32k_clean.py'),'patched-default','clean',str(rep)]
  step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START CLEAN',rep,flush=True)
  with (out/f'clean-r{rep}.stdout').open('w') as stream:rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1050).returncode
  step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();print('FINISH CLEAN',rep,rc,flush=True);assert rc==0
 rows=[json.loads((base/f'owned-cb-off-v01402-code32k-clean-r{i}/record.json').read_text()) for i in [1,2]]
 assert all(row['healthy'] and not row['active'] and not row['new_fault_messages'] for row in rows)
 record['clean_mean']={key:sum(row['requests'][0]['measurement'][key] for row in rows)/2 for key in ['prefill_tok_s','decode_tok_s']}
 record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
