from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time

base=Path(__file__).parent
out=base/'qsa-batch128-layout1-v01402-state-sequence'
out.mkdir(mode=0o700)
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'steps':[],'minimum_comparison_tokens':32768,'scope':'Logged32K batch128/layout1 correctness run, then two fresh32K state/head/output repeats. Any mismatch stops the sequence and rejects clean timing. No speed/full256K claim.'}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def complete(path):
    r=json.loads(path.read_text())
    assert not r['active'] and r['healthy'] and r['completed'] and not r['new_fault_messages']
    assert not any(r['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
    assert len(r['requests'])==1 and r['requests'][0]['measurement']['prompt_tokens']==32768
    c=r['requests'][0]['comparison_to_default_counter_control']
    assert not c['different_prefill_state_parts'] and all(c[key] for key in ['first_head_equal','ids_equal','logprobs_equal'])
    return r
save()
try:
    first=base/'owned-qsa-batch128-layout1-v01402-code32k-diagnostic-r1/record.json'
    deadline=time.monotonic()+1200
    while True:
        r=json.loads(first.read_text())
        if not r['active']:break
        assert time.monotonic()<deadline
        time.sleep(1)
    complete(first)
    for rep in [1,2]:
        argv=[sys.executable,str(base/'run_owned_qsa_batch128_layout1_v01402_code32k.py'),'qsa-batch128-layout1','state',str(rep)]
        step={'argv':argv,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        record['steps'].append(step);save();print('START STATE',rep,flush=True)
        with (out/f'state-r{rep}.stdout').open('w') as stream:
            rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=1100).returncode
        step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
        r=complete(base/f'owned-qsa-batch128-layout1-v01402-code32k-state-r{rep}/record.json')
        step['measurement']=r['requests'][0]['measurement'];save();print('FINISH STATE',rep,flush=True)
    record['passed']=True
except BaseException as error:record['error']=repr(error)
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
