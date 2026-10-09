"""Run six order-balanced independent decode blocks, one owned model at a time."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

b=Path(__file__).parent
observer=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(observer/'sycl/tools'))
from owned_gdb import process_identity

def digest(path):
    return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()

controller=b/'run_owned_stager_affinity_v0141_decode_repeat_v1.py'
expected_controller='f6fd828a668d83d60713cf7ecb635c9c33512488dac8e60747895b13a6a53327'
assert digest(controller)==expected_controller
preparation=b/'stager-affinity-decode-repeat-controller-preparation-v1.json'
prep=json.loads(preparation.read_text());assert prep['passed'] and not prep['gpu_executed']
assert prep['controller_sha256']==expected_controller
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
diag_path=b/'owned-stager-affinity-v0141-code32k-affinity-diagnostic-r1/record.json'
diag=json.loads(diag_path.read_text())
def terminal(data):
    assert not data['active'] and data['completed'] and data['healthy'] and data['math_gate_passed']
    assert data['exit_code']==0 and not data['exit_signal'] and not data['new_fault_messages']
    assert not any(data['cleanup'].values()) and data['boot_id']==boot
    for role in ['inferior','debugger']:
        old=data[role];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
terminal(diag)
assert digest(diag_path)=='ff94770a606e18639b6e49c03c10722a74698b733fa32bb507091b9464a5aebc'
assert not diag['performance_eligible'] and len(diag['requests'])==4
assert diag['stager_affinity_startup_gate']['passed']
assert all(x['math_gate_passed'] and all(x['head_and_live_state_comparison'].values()) and all(x['qualified_physical_reference_output_comparison'].values()) for x in diag['requests'])
full_path=b/'owned-native-expert-copy-v0141-full256k-diagnostic-r1/record.json'
assert digest(full_path)=='7f206dc65c01d05192d54211e6ba1f52eb34dd029f25e9a7c2235b1a2567d533'
full=json.loads(full_path.read_text());terminal(full)
saved=next(x for x in full['sessions'] if x['name']=='save-control32k')
checkpoint_identity={'file':saved['file'],'sha256':saved['image']['sha256'],'bytes':saved['bytes'],'tokens':saved['tokens'],'source_receipt_sha256':digest(full_path)}

orders=[['baseline','affinityoff','affinityon'],['affinityoff','affinityon','baseline'],
        ['affinityon','baseline','affinityoff'],['affinityon','affinityoff','baseline'],
        ['affinityoff','baseline','affinityon'],['baseline','affinityon','affinityoff']]
assert set(orders[0])=={'baseline','affinityoff','affinityon'}
assert all(sorted(x)==sorted(orders[0]) for x in orders)
assert all(sum(order[position]==mode for order in orders)==2 for mode in orders[0] for position in range(3))
out=b/'stager-affinity-v0141-decode-repeat-comparison-sequence-v1'
assert not out.exists();out.mkdir(mode=0o700)
started=time.monotonic()
record={'active':True,'passed':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':boot,'orders':orders,'controller_sha256':expected_controller,'sequence_controller_sha256':digest(__file__),
        'first_diagnostic_receipt_sha256':digest(diag_path),'preparation_receipt_sha256':digest(preparation),
        'steps':[],'cpu_preflights':[],'deadline_seconds':10800,
        'measurement_plan':{'independent_processes_per_mode':6,'measured_repeats_per_process':12,'excluded_warm_repeats_per_process':1,'input_tokens':32832,'checkpoint_tokens':32831,'generated_tokens':64,'logprobs':5},
        'adoption_policy':'Evaluate prefill and decode separately. Faster prefill does not compensate for decode regression. A confidence interval including zero is not evidence of equivalence.'}
child=None;child_identity=None
def save():
    record['elapsed_seconds']=time.monotonic()-started
    temp=out/'record.json.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(out/'record.json')
save()
try:
    for mode in orders[0]:
        argv=[sys.executable,str(controller),mode,'clean','1','--cpu-preflight']
        checked=subprocess.run(argv,capture_output=True,text=True,timeout=60)
        item={'mode':mode,'argv':argv,'exit_code':checked.returncode,'stdout':checked.stdout,'stderr':checked.stderr}
        record['cpu_preflights'].append(item);save()
        assert checked.returncode==0,checked.stderr
        result=json.loads(checked.stdout)
        assert result['cpu_preflight_passed'] and not result['gpu_executed']
    for block,order in enumerate(orders,1):
        for position,mode in enumerate(order):
            assert time.monotonic()-started<record['deadline_seconds']
            assert digest(controller)==expected_controller
            receipt=b/f'owned-stager-affinity-v0141-decode-repeat-{mode}-clean-r{block}'/'record.json'
            assert not receipt.parent.exists()
            argv=[sys.executable,str(controller),mode,'clean',str(block)]
            stdout=out/f'block{block}-{position}-{mode}.stdout';stderr=out/f'block{block}-{position}-{mode}.stderr'
            item={'block':block,'position':position,'mode':mode,'argv':argv,'receipt':str(receipt),'stdout':str(stdout),'stderr':str(stderr),'active':True}
            record['steps'].append(item)
            with stdout.open('w') as so,stderr.open('w') as se:
                child=subprocess.Popen(argv,stdout=so,stderr=se,start_new_session=True)
                child_identity=process_identity(child.pid)
                assert child_identity is not None
                item['controller_identity']=child_identity;save()
                began=time.monotonic()
                while child.poll() is None:
                    if time.monotonic()-began>1800 or time.monotonic()-started>record['deadline_seconds']:
                        raise TimeoutError('bounded owned comparison deadline')
                    save();time.sleep(.5)
                item['exit_code']=child.returncode;item['active']=False
            assert child.returncode==0,(mode,block,stderr.read_text())
            assert receipt.is_file();data=json.loads(receipt.read_text());terminal(data)
            assert data['mode']==mode and data['repetition']==block and data['phase']=='clean'
            assert data['performance_eligible'] and data['phase_specific_decode_sequence_completed']
            assert len(data['requests'])==14 and len(data['sessions'])==13
            assert sum(x['timing_eligible'] for x in data['requests'])==12
            assert data['checkpoint_identity']==checkpoint_identity
            assert data['argv'][1:]==diag['argv'][1:]
            expected_env=dict(diag['actual_target_environment'])
            # Diagnostic logging is deliberately absent from performance runs.
            for key in list(expected_env):
                if key.startswith(('UR_LOG_','ZE_ENABLE_','ZEL_')) or key in ['UR_ENABLE_LAYERS','STRATA_TRACE']:
                    expected_env.pop(key)
            for key in ['STRATA_DUMP_FIRST_LOGITS','STRATA_PREFILL_DUMP_STATE','STRATA_STAGER_CPU_LIST']:
                expected_env.pop(key,None)
            if mode=='affinityon':expected_env['STRATA_STAGER_CPU_LIST']='1,2,3'
            assert data['actual_target_environment']==expected_env
            assert data['stager_affinity_startup_gate']['passed'] and data['stager_affinity_startup_gate']['enabled']==(mode=='affinityon')
            item['receipt_sha256']=digest(receipt);item['math_gate_passed']=True
            save();child=None;child_identity=None
    assert len(record['steps'])==18 and all(x['exit_code']==0 and not x['active'] for x in record['steps'])
    record['configuration_equal_except_executable_and_stager_cpu_affinity']=True
    record['passed']=True
except BaseException as error:
    record['error']=repr(error)
finally:
    if child and child.poll() is None:
        now=process_identity(child.pid)
        if now and child_identity and now['start_ticks']==child_identity['start_ticks']:
            os.kill(child.pid,signal.SIGINT)
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:record['controller_survived_after_interrupt']=True
    record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps({'passed':record['passed'],'elapsed_seconds':record['elapsed_seconds'],'steps':len(record['steps']),'error':record.get('error')}))
if not record['passed']:raise SystemExit(1)
