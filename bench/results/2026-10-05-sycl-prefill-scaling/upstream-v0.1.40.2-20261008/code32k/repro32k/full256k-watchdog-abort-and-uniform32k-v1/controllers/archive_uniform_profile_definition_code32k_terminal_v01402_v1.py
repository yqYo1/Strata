"""Append the immutable completed r6 numerical sequence, preserving its metadata error."""
from pathlib import Path
import datetime,hashlib,json,shutil
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k/full256k-watchdog-abort-and-uniform32k-v1'
run=base/'owned-profile-definition-v01402-code32k-diagnostic-r6'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=json.loads((run/'record.json').read_text())
assert not r['active'] and r['completed'] and r['healthy'] and r['math_gate_passed']
assert r['exit_code']==0 and not r['exit_signal'] and not r.get('error') and not r['new_fault_messages']
assert not any(r['cleanup'].values())
assert r['code32k_sequence_completed'] and not r['full_capacity_sequence_completed']
assert len(r['requests'])==6 and all(x['math_gate_passed'] for x in r['requests'])
fresh=[x for x in r['requests'] if x['name'] in ('control32k-before','alternate32k-first','control32k-repeat','alternate32k-repeat')]
assert len(fresh)==4 and all(x['measurement']['prompt_tokens']==32768 and max(x['resume_tokens'])==0 for x in fresh)
assert all(x['passed'] for x in r['sessions']) and {x['name'] for x in r['sessions']}=={'save-control32k','restore-control32k'}
identity=json.loads((run/'actual-executable-identity-v1.json').read_text());assert identity['passed']
assert identity['actual_binary_sha256']=='dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
assert r['argv'][0]==identity['exe'] and r['inferior']['pid']==identity['pid'] and r['inferior']['start_ticks']==identity['start_ticks'] and r['boot_id']==identity['boot_id']
assert all(not Path('/proc',str(r[k]['pid'])).exists() for k in ('inferior','debugger'))
for name in ('record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','debugger/inferior-argv.json','probes/kernel-before.stdout','probes/kernel-after.stdout'):
    a=run/name;b=out/'terminal-uniform32k-r6'/name
    assert not b.exists();b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
derived=dict(active=False,passed=True,adopted=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             scope='Derived complete32K numerical/actual disk-resume/normal owned exit proof; not full physical256K, throughput, profiler coverage or stall-fix proof.',
             raw_receipt_sha256=sha(run/'record.json'),actual_executable_identity_sha256=sha(run/'actual-executable-identity-v1.json'),
             actual_binary_sha256=identity['actual_binary_sha256'],raw_inherited_binary_sha256=r['binary_sha256'],
             four_fresh_reads_tokens=[32768]*4,all_head_used_state_output_logprob_mtp_comparisons_passed=True,
             actual_disk_restore_continuation_passed=True,exit_code=0,owned_survivors=False,new_kernel_fault_messages=[],
             raw_stderr_bytes=r['engine_log_bytes'],raw_stderr_sha256=r['engine_log_sha256'],
             minimum_performance_input_tokens=32768,diagnostic_durations_excluded=True,
             full_capacity_sequence_passed=False,watchdog_pending_event_cause_resolved=False,
             no_reset_rebind_reboot_service_package_global_change=True)
(out/'uniform32k-completed-derived-proof.json').write_text(json.dumps(derived,indent=2)+'\n')
shutil.copy2(Path(__file__),out/'controllers'/Path(__file__).name)
print(json.dumps({'completed':True,'raw_receipt_sha256':derived['raw_receipt_sha256'],'actual_binary_sha256':identity['actual_binary_sha256'],'derived_proof_sha256':sha(out/'uniform32k-completed-derived-proof.json')},indent=2))
