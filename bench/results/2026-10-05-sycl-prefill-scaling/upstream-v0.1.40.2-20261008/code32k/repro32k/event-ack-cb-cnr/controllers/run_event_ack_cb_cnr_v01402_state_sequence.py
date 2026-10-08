from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time
base=Path(__file__).parent
out=base/'event-ack-cb-cnr-v01402-state-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'steps':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Logged private candidate32K first use followed by two fresh32K state/head captures without API logging. Compare all66 state parts, all248320 first-head float bytes, generated IDs and protocol logprobs. No speed/full256K claim.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def request(path):
 r=json.loads(path.read_text());assert not r['active'] and r['healthy'] and not r['new_fault_messages'];assert r['cleanup']['inferior_survived']==r['cleanup']['gdb_survived']==False;assert len(r['requests'])==1
 return r['requests'][0]
save()
try:
 first=base/'owned-event-ack-cb-cnr-v01402-code32k-diagnostic-r1/record.json'
 deadline=time.monotonic()+1000
 while True:
  r=json.loads(first.read_text())
  if not r['active']:break
  assert time.monotonic()<deadline;time.sleep(1)
 baseline=request(first)
 for rep in [1,2]:
  argv=[sys.executable,str(base/'run_owned_event_ack_cb_cnr_v01402_code32k.py'),'event-ack-cb-cnr','state',str(rep)]
  step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START STATE',rep,flush=True)
  with (out/f'state-r{rep}.stdout').open('w') as stream:rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1050).returncode
  step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
  current=request(base/f'owned-event-ack-cb-cnr-v01402-code32k-state-r{rep}/record.json')
  differences=[i for i,(a,b) in enumerate(zip(baseline['prefill_state']['parts'],current['prefill_state']['parts'])) if a['bytes']!=b['bytes'] or a['sha256']!=b['sha256']]
  comparison={'prefill_state_different_parts':differences,'first_head_equal':current['first_head']['sha256']==baseline['first_head']['sha256'],'ids_equal':current['ids']==baseline['ids'],'logprobs_equal':current['logprobs']==baseline['logprobs']}
  step['comparison_to_logged']=comparison;save();print('FINISH STATE',rep,comparison,flush=True)
  assert not differences and comparison['first_head_equal'] and comparison['ids_equal'] and comparison['logprobs_equal'],'candidate state/head not reproducible; do not use its timings for tuning'
 record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
