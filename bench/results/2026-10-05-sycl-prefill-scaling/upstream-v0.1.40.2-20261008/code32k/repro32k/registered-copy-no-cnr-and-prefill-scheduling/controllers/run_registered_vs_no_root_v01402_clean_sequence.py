from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time

base=Path(__file__).parent
out=base/'registered-vs-no-root-v01402-clean-sequence'
out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'ordering':'registered/no-root/no-root/registered (ABBA)','scope':'Two fresh32K timing jobs per private candidate, only after three full state/head matches per arm. No debug logs, validation, state/head dumps, transfer timing or extra waits. Normal MTP4 generates64tokens; reject any output mismatch or xe fault. These jobs do not validate default counter conversion or full262144-cell serving.'}
def save():
 (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
 for mode in ['event-ack-registered-copy-cb','event-ack-no-root-prefill-cb']:
  gate=base/f'{mode}-v01402-state-sequence/record.json'
  deadline=time.monotonic()+2400
  while True:
   r=json.loads(gate.read_text())
   if not r['active']:break
   assert time.monotonic()<deadline
   time.sleep(1)
  assert r['passed'],f'{mode} full state/head gate failed; reject before timing'
 order=[('event-ack-registered-copy-cb',1),('event-ack-no-root-prefill-cb',1),('event-ack-no-root-prefill-cb',2),('event-ack-registered-copy-cb',2)]
 for mode,rep in order:
  controller=base/('run_owned_event_ack_registered_copy_cb_v01402_code32k_clean.py' if mode=='event-ack-registered-copy-cb' else 'run_owned_event_ack_no_root_prefill_cb_v01402_code32k_clean.py')
  argv=[sys.executable,str(controller),mode,'clean',str(rep)]
  step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  record['steps'].append(step);save();print('START CLEAN',mode,rep,flush=True)
  with (out/f'{mode}-r{rep}.stdout').open('w') as stream:
   rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1050).returncode
  step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
  print('FINISH CLEAN',mode,rep,rc,flush=True);assert rc==0
  r=json.loads((base/f'owned-{mode}-v01402-code32k-clean-r{rep}/record.json').read_text())
  assert not r['active'] and r['healthy'] and r['completed'] and not r['new_fault_messages']
  assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
  assert len(r['requests'])==1 and r['requests'][0]['measurement']['prompt_tokens']==32768
  request=r['requests'][0]
  assert request['measurement']['generated_tokens']==64 and all(request['comparison_to_logged_control'].values())
  assert not any(key.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) or key in ['STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE','STRATA_PREFILL_SYNC','STRATA_PREFILL_TRANSFER_TIMING'] for key in r['environment'])
  step['measurement']=request['measurement'];save()
 record['clean_mean']={mode:{key:sum(step['measurement'][key] for step in record['steps'] if step['argv'][2]==mode)/2 for key in ['prefill_tok_s','decode_tok_s']} for mode in ['event-ack-registered-copy-cb','event-ack-no-root-prefill-cb']}
 a=record['clean_mean']['event-ack-registered-copy-cb'];b=record['clean_mean']['event-ack-no-root-prefill-cb']
 record['relative_change']={key:b[key]/a[key]-1 for key in a}
 record['passed']=True
except BaseException as error:
 record['error']=repr(error)
finally:
 record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
