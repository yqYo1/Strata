from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time

base=Path(__file__).parent
out=base/'qsa-batch128-layout1-v01402-clean-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'ordering':'default/batch128-layout1/batch128-layout1/default (ABBA)','scope':'Four fresh32K clean timing jobs only after full-state/head/output parity. Same executable/model/environment/normal MTP4, no debug validation/dumps/profiler/extra waits. Every output must match the logged control.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    gate=base/'qsa-batch128-layout1-v01402-state-sequence/record.json'
    deadline=time.monotonic()+2400
    while True:
        r=json.loads(gate.read_text())
        if not r['active']:break
        assert time.monotonic()<deadline
        time.sleep(1)
    assert r['passed'],'Full-state/head gate failed; reject timing'
    order=[('qsa-default',1),('qsa-batch128-layout1',1),('qsa-batch128-layout1',2),('qsa-default',2)]
    for mode,rep in order:
        argv=[sys.executable,str(base/'run_owned_qsa_batch128_layout1_v01402_code32k_clean.py'),mode,'clean',str(rep)]
        step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START CLEAN',mode,rep,flush=True)
        with (out/f'{mode}-r{rep}.stdout').open('w') as stream:
            rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1100).returncode
        step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
        r=json.loads((base/f'owned-{mode}-v01402-code32k-clean-r{rep}/record.json').read_text())
        assert r['healthy'] and not r['active'] and r['completed'] and not r['new_fault_messages']
        assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
        request=r['requests'][0];mm=request['measurement']
        assert mm['prompt_tokens']==32768 and mm['generated_tokens']==64
        assert all(request['comparison_to_logged_control'].values())
        env=r['environment']
        assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_','UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','MKL_CBWR','EnableImplicitConvertionToCounterBasedEvents','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE','STRATA_PREFILL_SYNC','STRATA_PREFILL_TRANSFER_TIMING'] for k in env)
        step['measurement']=mm;save();print('FINISH CLEAN',mode,rep,flush=True)
    record['clean_mean']={mode:{key:sum(s['measurement'][key] for s in record['steps'] if s['argv'][2]==mode)/2 for key in ['prefill_tok_s','decode_tok_s']} for mode in ['qsa-default','qsa-batch128-layout1']}
    a=record['clean_mean']['qsa-default'];b=record['clean_mean']['qsa-batch128-layout1']
    record['relative_change']={key:b[key]/a[key]-1 for key in a}
    record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
