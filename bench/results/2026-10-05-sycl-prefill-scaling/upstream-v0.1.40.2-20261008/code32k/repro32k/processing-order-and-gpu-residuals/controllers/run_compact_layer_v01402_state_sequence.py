from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time

base=Path(__file__).parent
out=base/'compact-layer-v01402-state-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'scope':'Three fresh32K full-state/head/output controls of compact2/layout1, then three controls adding layer-major2 with RAM residuals. Any mismatch/fault rejects timing; no release/restore/full256K or speed claim.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def complete(path):
    r=json.loads(path.read_text())
    assert r['healthy'] and not r['active'] and r['completed'] and not r['new_fault_messages']
    assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
    assert len(r['requests'])==1 and r['requests'][0]['measurement']['prompt_tokens']==32768
    c=r['requests'][0]['comparison_to_default_counter_control']
    assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal'])
    return r
save()
try:
    first=base/'owned-compact2-layout1-v01402-code32k-diagnostic-r1/record.json'
    deadline=time.monotonic()+1200
    while True:
        r=json.loads(first.read_text())
        if not r['active']:break
        assert time.monotonic()<deadline
        time.sleep(1)
    complete(first)
    for mode in ['compact2-layout1','layer2-ram-layout1']:
        gate=base/f'{mode}-v01402-state-sequence';gate.mkdir(mode=0o700)
        state={'active':True,'passed':False,'mode':mode,'minimum_comparison_tokens':32768,'records':[],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            phases=[('state',1),('state',2)] if mode=='compact2-layout1' else [('diagnostic',1),('state',1),('state',2)]
            if mode=='compact2-layout1':state['records'].append(str(first))
            (gate/'record.json').write_text(json.dumps(state,indent=2)+'\n')
            for phase,rep in phases:
                argv=[sys.executable,str(base/'run_owned_compact_layer_v01402_code32k.py'),mode,phase,str(rep)]
                step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START',mode,phase,rep,flush=True)
                with (out/f'{mode}-{phase}-r{rep}.stdout').open('w') as stream:
                    rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1100).returncode
                step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
                p=base/f'owned-{mode}-v01402-code32k-{phase}-r{rep}/record.json'
                r=complete(p);step['measurement']=r['requests'][0]['measurement'];state['records'].append(str(p));save();print('FINISH',mode,phase,rep,flush=True)
            assert len(state['records'])==3
            state['passed']=True
        except BaseException as error:
            state['error']=repr(error);raise
        finally:
            state.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (gate/'record.json').write_text(json.dumps(state,indent=2)+'\n')
    record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
