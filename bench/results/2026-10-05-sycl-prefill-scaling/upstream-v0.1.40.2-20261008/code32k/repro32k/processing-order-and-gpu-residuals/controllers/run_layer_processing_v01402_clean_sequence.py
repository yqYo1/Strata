from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time

base=Path(__file__).parent
out=base/'layer-processing-v01402-clean-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'ordering':'default/RAM/GPU/GPU/RAM/default (ABCCBA)','scope':'Six fresh32K clean timing jobs after nine complete state/head/output controls. Same executable and normal MTP4, all64 outputs match. No debug logging, validation, payload checks, state/head dumps, profiler, transfer profiling or extra waits. No full256K/adoption claim.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    gate=base/'layer-gpu-release-v01402-state-sequence/record.json';deadline=time.monotonic()+3600
    while True:
        r=json.loads(gate.read_text())
        if not r['active']:break
        assert time.monotonic()<deadline
        time.sleep(1)
    assert r['passed'],'Full-state/head/release gate failed; reject timing'
    modes=['processing-default','layer2-ram-layout1','layer2-gpu-release-layout1']
    order=[(modes[0],1),(modes[1],1),(modes[2],1),(modes[2],2),(modes[1],2),(modes[0],2)]
    for mode,rep in order:
        argv=[sys.executable,str(base/'run_owned_layer_processing_v01402_code32k_clean.py'),mode,'clean',str(rep)]
        step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START CLEAN',mode,rep,flush=True)
        with (out/f'{mode}-r{rep}.stdout').open('w') as stream:
            rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1100).returncode
        step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
        r=json.loads((base/f'owned-{mode}-v01402-code32k-clean-r{rep}/record.json').read_text())
        assert r['healthy'] and not r['active'] and r['completed'] and not r['new_fault_messages']
        assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
        req=r['requests'][0];mm=req['measurement']
        assert mm['prompt_tokens']==32768 and mm['generated_tokens']==64 and all(req['comparison_to_logged_control'].values())
        assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_','UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','MKL_CBWR','EnableImplicitConvertionToCounterBasedEvents','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE','STRATA_PREFILL_SYNC','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_DRAFT_VERIFY'] for k in r['environment'])
        step['measurement']=mm;save();print('FINISH CLEAN',mode,rep,flush=True)
    record['clean_mean']={mode:{key:sum(s['measurement'][key] for s in record['steps'] if s['argv'][2]==mode)/2 for key in ['prefill_tok_s','decode_tok_s']} for mode in modes}
    a=record['clean_mean'][modes[0]]
    record['relative_to_default']={mode:{key:record['clean_mean'][mode][key]/a[key]-1 for key in a} for mode in modes[1:]}
    record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
