"""Retain terminal r5 evidence and a scoped completed portion of the separate r6 gate."""
from pathlib import Path
import datetime,hashlib,json,shutil
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k'
out=parent/'full256k-watchdog-abort-and-uniform32k-v1';out.mkdir()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(a,b):
    assert not b.exists();b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
run=base/'owned-full-kv-access-v01402-full256k-diagnostic-r5'
r=json.loads((run/'record.json').read_text())
assert sha(run/'record.json')=='e7e541f91ef053acebaa6b9ed2bc3139010e269f0b671d23f423869e2c7a882f'
assert not r['active'] and not r['healthy'] and 'SIGABRT' in r['error']
for name in ('record.json','protocol.stdout.raw','events.jsonl','debugger/failure.mi.txt','debugger/inferior-argv.json','probes/kernel-before.stdout','probes/kernel-after.stdout'):
    copy(run/name,out/'terminal-r5'/name)
for name in ('full256k-watchdog-abort-v01402-review-v1','post-full256k-abort-v01402-health-v1'):
    for p in sorted((base/name).iterdir()):
        if p.is_file():copy(p,out/'checks'/name/p.name)
live=base/'owned-profile-definition-v01402-code32k-diagnostic-r6'
lr=json.loads((live/'record.json').read_text())
identity=json.loads((live/'actual-executable-identity-v1.json').read_text())
assert identity['passed'] and identity['actual_binary_sha256']=='dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
first=next(x for x in lr['requests'] if x['name']=='control32k-before')
continuation=next(x for x in lr['requests'] if x['name']=='resume32k-reference')
saved=next(x for x in lr['sessions'] if x['name']=='save-control32k')
assert first['math_gate_passed'] and continuation['math_gate_passed'] and saved['passed']
assert first['measurement']['prompt_tokens']==32768
snapshot={k:lr[k] for k in ('started_utc','boot_id','argv','environment','inferior','debugger','uniform_header_build_receipt_sha256','uniform_header_source_review_sha256','uniform_header_linked_review_sha256')}
snapshot.update(active=False,completed=True,adopted=False,
                captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='First completed fresh32768 control, SAVE and live continuation only; not disk restore, repeated-input, complete sequence, capacity, speed or stall-fix proof.',
                binary_sha256=identity['actual_binary_sha256'],raw_parent_binary_sha256=lr['binary_sha256'],
                actual_identity_sha256=sha(live/'actual-executable-identity-v1.json'),
                parent_receipt_at_capture_sha256=sha(live/'record.json'),
                requests=[first,continuation],sessions=[saved])
(out/'uniform32k-first-completed-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
copy(live/'actual-executable-identity-v1.json',out/'uniform32k-actual-executable-identity-v1.json')
for name in ('check_health_after_full256k_abort_v01402_v1.py','audit_full256k_watchdog_abort_v01402_v1.py',
             'prepare_owned_profile_definition_code32k_v01402_v1.py','prepare_owned_profile_definition_code32k_v01402_v2.py',
             'run_owned_profile_definition_code32k_v01402_v1.py','run_owned_profile_definition_code32k_v01402_v2.py',
             'capture_profile_definition_running_identity_v01402_v1.py',Path(__file__).name):
    copy(base/name,out/'controllers'/name)
summary=dict(active=False,completed=True,adopted=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             r5_first_full_passed=True,r5_full_capacity_sequence_passed=False,
             r5_terminal='Host watchdog SIGABRT after60s without progress, second full read layer33/chunk196608',
             r5_new_kernel_fault_messages=[],post_exit_device_runtime_health_passed=True,
             underlying_pending_event_cause_resolved=False,
             uniform_header_first32k_math_passed=True,uniform_header_complete_sequence_passed=False,
             raw_r6_metadata_error='binary_sha256 inherited7f054 before binary reselection; actual argv and live /proc/PID/exe independently prove dd5; raw receipt preserved and correction attached.',
             minimum_performance_input_tokens=32768,diagnostic_durations_excluded=True,
             pending=['separate uniform-header repeated32K/disk restore/normal exit numerical gate',
                      'complete physical256K/repeated full/disk roundtrip/clipped restore/refusal/later32K before adoption',
                      'actual profiler device timestamps and graph coverage','matched quiet>=32K speed comparison'],
             no_reset_rebind_reboot_service_package_global_change=True,production_and_main_unchanged=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'archive':str(out),'r5_terminal':True,'uniform_header_first32k_passed':True},indent=2))
