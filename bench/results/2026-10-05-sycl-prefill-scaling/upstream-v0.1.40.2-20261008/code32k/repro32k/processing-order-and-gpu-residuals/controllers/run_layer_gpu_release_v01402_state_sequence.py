from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time

base=Path(__file__).parent
out=base/'layer-gpu-release-v01402-state-sequence';out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'scope':'Wait for the completed compact/RAM-row32K exact-state gates, then logged32K and two fresh32K full-state/head/output controls of all-GPU residuals with MTP release/immutable-RAM restoration and payload verification. No clean speed/full256K claim.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
    gate=base/'compact-layer-v01402-state-sequence/record.json';deadline=time.monotonic()+2400
    while True:
        r=json.loads(gate.read_text())
        if not r['active']:break
        assert time.monotonic()<deadline
        time.sleep(1)
    assert r['passed'],'RAM-row gate failed; reject release experiment'
    for phase,rep in [('diagnostic',1),('state',1),('state',2)]:
        argv=[sys.executable,str(base/'run_owned_layer_gpu_release_v01402_code32k.py'),'layer2-gpu-release-layout1',phase,str(rep)]
        step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save();print('START',phase,rep,flush=True)
        with (out/f'{phase}-r{rep}.stdout').open('w') as stream:
            rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1100).returncode
        step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
        r=json.loads((base/f'owned-layer2-gpu-release-layout1-v01402-code32k-{phase}-r{rep}/record.json').read_text())
        assert r['healthy'] and not r['active'] and r['completed'] and not r['new_fault_messages']
        assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
        req=r['requests'][0];c=req['comparison_to_default_counter_control']
        assert not c['different_prefill_state_parts'] and all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal'])
        assert req['measurement']['prompt_tokens']==32768 and req['measurement']['generated_tokens']==64
        messages=(base/f'owned-layer2-gpu-release-layout1-v01402-code32k-{phase}-r{rep}/project-messages.txt').read_text()
        releases=[line for line in messages.splitlines() if line.startswith('strata mtp decode release:')]
        restores=[line for line in messages.splitlines() if line.startswith('strata mtp decode restore:')]
        assert len(releases)==len(restores)==1 and all('verified=1' in line for line in releases+restores)
        assert '32767 GPU rows, 8192 reused scratch rows' in messages
        step.update(measurement=req['measurement'],release=releases[0],restore=restores[0]);save();print('FINISH',phase,rep,flush=True)
    record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
