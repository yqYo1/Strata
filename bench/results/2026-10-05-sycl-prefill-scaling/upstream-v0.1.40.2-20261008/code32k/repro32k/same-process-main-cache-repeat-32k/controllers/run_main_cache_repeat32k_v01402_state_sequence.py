"""Retain logged repeated check, then two fresh capture-only processes."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys
base=Path(__file__).parent
out=base/'main-cache-repeat32k-v01402-state-sequence';out.mkdir(mode=0o700)
mode='main-vmm-full-ram-repeat32k'
controller=base/'run_owned_main_cache_repeat32k_v01402.py'
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Three normal-MTP32768-token full rereads per process. One already-completed logged/validated process and two fresh complete-state/head captures. All raw66 state parts, every head float,64IDs/logprobs and MTP43/66 must match the hard control; no captured duration enters speed comparison. No full256K/adoption claim.',
        'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model_controller_sha256':hashlib.sha256(controller.read_bytes()).hexdigest(),
        'minimum_comparison_tokens':32768,'steps':[]}
def save():(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def check(phase,repetition):
    p=base/f'owned-{mode}-v01402-code32k-{phase}-r{repetition}'
    d=json.loads((p/'record.json').read_text())
    assert d['healthy'] and d['completed'] and d['math_gate_passed'] and not d['active']
    assert d['exit_code']==0 and not d['new_fault_messages'] and not any(d['cleanup'].values())
    assert len(d['requests'])==3
    for req in d['requests']:
        c=req['comparison_to_default_counter_control']
        assert not c['different_prefill_state_parts']
        assert all(c[k] for k in ['first_head_equal','ids_equal','logprobs_equal'])
        assert req['math_gate_passed'] and req['mtp_counts']==[43,66] and req['no_prompt_reuse']
        assert req['measurement']['prompt_tokens']==32768 and req['measurement']['generated_tokens']==64
    lines=(p/'project-messages.txt').read_text().splitlines()
    for prefix in ['strata prefill cache release:','strata prefill cache verify:','strata mtp decode release:','strata mtp decode restore:']:
        matching=[s for s in lines if s.startswith(prefix)];assert len(matching)==3
        if prefix.startswith('strata mtp'):assert all('verified=1' in s for s in matching)
    return {'record_file':str(p/'record.json'),'record_sha256':hashlib.sha256((p/'record.json').read_bytes()).hexdigest(),
            'measurements':[q['measurement'] for q in d['requests']],
            'all_raw_state_head_outputs_counts_exact':True}
save()
try:
    record['steps'].append({'phase':'diagnostic','repetition':1,'already_completed':True,**check('diagnostic',1)});save()
    for repetition in [1,2]:
        argv=[sys.executable,str(controller),mode,'state',str(repetition)]
        step={'argv':argv,'phase':'state','repetition':repetition,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};record['steps'].append(step);save()
        print('START state',repetition,flush=True)
        with (out/f'state-r{repetition}.stdout').open('w') as stream:
            rc=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,timeout=2000).returncode
        step.update(exit_code=rc,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();assert rc==0
        step.update(check('state',repetition));save();print('FINISH state',repetition,flush=True)
    record['passed']=True
except BaseException as error:
    record['error']=repr(error)
finally:
    record.update(active=False,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
if not record['passed']:raise SystemExit(1)
