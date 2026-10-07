from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
base=Path(__file__).parent
out=base/'host-profile32k-v01402-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'scope':'First logged/validated host-only unitrace32K check, then quiet host-only profile and matched unprofiled32K control. Every state/head/token/logprob must match the completed default-counter control. No device event instrumentation, clean throughput or full256K claim.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def complete(path):
 r=json.loads(path.read_text());assert not r['active'] and r['healthy'] and r['completed'] and not r['new_fault_messages']
 assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
 assert len(r['requests'])==1 and r['requests'][0]['measurement']['prompt_tokens']==32768
 c=r['requests'][0]['comparison_to_default_counter_control'];assert not c['different_prefill_state_parts'] and all(c[key] for key in ['first_head_equal','ids_equal','logprobs_equal'])
 return r
save()
try:
 first=base/'owned-event-ack-no-root-host-profile-v01402-code32k-diagnostic-r1/record.json'
 deadline=time.monotonic()+1200
 while True:
  r=json.loads(first.read_text())
  if not r['active']:break
  assert time.monotonic()<deadline;time.sleep(1)
 complete(first)
 for phase in ['state','control']:
  argv=[sys.executable,str(base/'run_owned_host_profile32k_v01402.py'),'event-ack-no-root-host-profile',phase,'1']
  step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START',phase,flush=True)
  with (out/(phase+'.stdout')).open('w') as stream:rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1100).returncode
  step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
  r=complete(base/f'owned-event-ack-no-root-host-profile-v01402-code32k-{phase}-r1/record.json')
  step['measurement']=r['requests'][0]['measurement'];save();print('FINISH',phase,flush=True)
 record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
