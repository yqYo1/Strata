from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time

base=Path(__file__).parent
out=base/'main-cache-release-v01402-clean-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'ordering':'kept/RAMhalf/RAMfull/snapshotfull/snapshotfull/RAMfull/RAMhalf/kept (ABCDDCBA)','scope':'Eight fresh clean32K jobs after twelve main-state/head/output captures and payload checks. Same64MiB segment allocation and e82fc5 binary. No diagnostics, validation, byte verification, state/head dumps, profiler, transfer timing or extra waits. No full256K/adoption claim.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    gate=base/'main-cache-release-v01402-state-sequence/record.json';deadline=time.monotonic()+7200
    while True:
        r=json.loads(gate.read_text())
        if not r['active']:break
        assert time.monotonic()<deadline
        time.sleep(1)
    assert r['passed'],'Full-state/head/main-cache/MTP gates failed; reject timing'
    modes=['main-vmm-kept-ram','main-vmm-half-ram','main-vmm-full-ram','main-vmm-full-snapshot']
    order=[(mode,1) for mode in modes]+[(mode,2) for mode in reversed(modes)]
    for mode,rep in order:
        argv=[sys.executable,str(base/'run_owned_main_cache_release_v01402_code32k_clean.py'),mode,'clean',str(rep)]
        step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START CLEAN',mode,rep,flush=True)
        with (out/f'{mode}-r{rep}.stdout').open('w') as stream:rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1100).returncode
        step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
        r=json.loads((base/f'owned-{mode}-v01402-code32k-clean-r{rep}/record.json').read_text());assert r['healthy'] and not r['active'] and r['completed'] and not r['new_fault_messages'];assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
        req=r['requests'][0];mm=req['measurement'];assert mm['prompt_tokens']==32768 and mm['generated_tokens']==64 and all(req['comparison_to_logged_control'].values())
        assert not any(k.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_','UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','MKL_CBWR','EnableImplicitConvertionToCounterBasedEvents','STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE','STRATA_PREFILL_SYNC','STRATA_PREFILL_TRANSFER_TIMING','STRATA_PREFILL_DRAFT_VERIFY','STRATA_PREFILL_CACHE_VERIFY'] for k in r['environment'])
        step['measurement']=mm;save();print('FINISH CLEAN',mode,rep,flush=True)
    record['clean_mean']={mode:{key:sum(s['measurement'][key] for s in record['steps'] if s['argv'][2]==mode)/2 for key in ['prefill_tok_s','decode_tok_s']} for mode in modes};a=record['clean_mean'][modes[0]]
    record['relative_to_kept']={mode:{key:record['clean_mean'][mode][key]/a[key]-1 for key in a} for mode in modes[1:]};record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
