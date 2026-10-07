from pathlib import Path
import datetime, hashlib, json, subprocess, sys

base=Path(__file__).parent
out=base/'main-cache-release-v01402-state-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'scope':'Four main-cache lease arms with identical64MiB segment allocation. Each first use has flushedUR/LevelZero logs/validation, then two fresh32K complete main-state/head/output controls; immutable RAM or snapshot payload verification. Captured timings excluded. No full256K/adoption or hang-prevention claim.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    parent=json.loads((base/'layer-processing-v01402-clean-sequence/record.json').read_text());assert parent['passed'] and not parent['active']
    modes=['main-vmm-kept-ram','main-vmm-half-ram','main-vmm-full-ram','main-vmm-full-snapshot']
    for mode in modes:
        for phase,rep in [('diagnostic',1),('state',1),('state',2)]:
            argv=[sys.executable,str(base/'run_owned_main_cache_release_v01402_code32k.py'),mode,phase,str(rep)]
            step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START',mode,phase,rep,flush=True)
            with (out/f'{mode}-{phase}-r{rep}.stdout').open('w') as stream: rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1100).returncode
            step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
            d=base/f'owned-{mode}-v01402-code32k-{phase}-r{rep}';r=json.loads((d/'record.json').read_text());assert r['healthy'] and r['completed'] and not r['active'] and not r['new_fault_messages']
            assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
            req=r['requests'][0];c=req['comparison_to_default_counter_control'];assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal']);assert req['measurement']['prompt_tokens']==32768 and req['measurement']['generated_tokens']==64
            lines=(d/'project-messages.txt').read_text().splitlines()
            main=[line for line in lines if line.startswith('strata prefill cache release:')];assert len(main)==1
            verify=[line for line in lines if line.startswith('strata prefill cache verify:')];assert len(verify)==1
            if mode=='main-vmm-kept-ram':assert '0 physical bytes, 0 restored bytes' in main[0] and 'source ram' in main[0]
            else:assert 'same_address 1' in main[0] and ('source snapshot' if mode.endswith('snapshot') else 'source ram') in main[0]
            releases=[line for line in lines if line.startswith('strata mtp decode release:')];restores=[line for line in lines if line.startswith('strata mtp decode restore:')];assert len(releases)==len(restores)==1 and all('verified=1' in line for line in releases+restores)
            step.update(measurement=req['measurement'],main_cache=main[0],main_verify=verify[0],mtp_release=releases[0],mtp_restore=restores[0]);save();print('FINISH',mode,phase,rep,flush=True)
        gate=base/f'{mode}-v01402-state-sequence';gate.mkdir(mode=0o700)
        (gate/'record.json').write_text(json.dumps({'active':False,'passed':True,'scope':record['scope'],'parent_controller_sha256':record['controller_sha256'],'steps':record['steps'][-3:],'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    record['passed']=True
except BaseException as error: record['error']=repr(error)
finally: record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
