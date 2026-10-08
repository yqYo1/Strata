"""Retain the failed >=32K timestamp experiment without speed attribution."""
from pathlib import Path
import datetime,hashlib,json,re

base=Path(__file__).parent
run=base/'owned-device-profile-v01402-code32k-diagnostic-r1'
out=base/'device-profile-watchdog-abort-v01402-review-v1';out.mkdir(mode=0o700)
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
receipt=run/'record.json';r=json.loads(receipt.read_text())
assert not r['active'] and not r['completed'] and not r['healthy']
assert 'SIGABRT' in r['error'] and not r['new_fault_messages']
assert not any(r['cleanup'][k] for k in ['inferior_survived','gdb_survived'])
assert r['binary_sha256']=='dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
mi=run/'debugger/failure.mi.txt'
decoded=''.join(json.loads(line[1:]) for line in mi.read_text().splitlines() if line.startswith('~'))
stacks={}
for tid in (82,80,79,78,1):
    match=re.search(r'\nThread '+str(tid)+r' .*?(?=\nThread |\Z)',decoded,re.S);assert match
    stacks[str(tid)]=match.group().split('\nrax ')[0]
assert 'main::{lambda(std::stop_token)#1}' in stacks['82'] and '__GI_abort' in stacks['82']
assert 'Stager::wait' in stacks['1']
assert 'queryCounterBasedEventStatus' in stacks['80'] and 'queryCounterBasedEventStatus' in stacks['78']
(out/'selected-stacks.txt').write_text('\n'.join(stacks.values()))
stderr=run/'debugger/inferior.stderr';size=stderr.stat().st_size;assert size==3849093051
span=json.loads((run/'last-submissions-source-span-v1.json').read_text())
spans=[('last-submissions',span['start'],span['end']),('watchdog-report',3830426805,3830432805),('terminal-status',size-32768,size)]
excerpts=[]
with stderr.open('rb') as stream:
    for name,start,end in spans:
        stream.seek(start);data=stream.read(end-start);assert len(data)==end-start
        path=out/(name+'.raw.txt');path.write_bytes(data)
        excerpts.append({'name':name,'start':start,'end':end,'bytes':len(data),'sha256':sha(path)})
watchdog=(out/'watchdog-report.raw.txt').read_text(errors='replace')
assert 'no progress for 60 s' in watchdog and 'layer 26' in watchdog and '16384' in watchdog
submissions=(out/'last-submissions.raw.txt').read_text(errors='replace')
assert '<--- urEnqueueUSMMemcpy(' in submissions and 'UR_RESULT_SUCCESS' in submissions
copies=[s for s in submissions.splitlines() if '<--- urEnqueueUSMMemcpy(' in s]
(out/'last-copy-submissions.json').write_text(json.dumps(copies,indent=2)+'\n')
health=base/'post-device-profile-abort-v01402-health-v1/record.json';hr=json.loads(health.read_text())
assert hr['healthy'] and not hr['active'] and hr['terminal_controller_receipt_sha256']==sha(receipt)
trace=run/f"strata.{r['inferior']['pid']}.json";assert trace.is_file() and trace.stat().st_size==0
record={'active':False,'passed':True,'adopted':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Evidence extraction only. Timestamp-enabled32768 run aborted before numerical completion; all its durations and incomplete/empty trace are rejected for performance analysis.',
        'controller_sha256':sha(__file__),'terminal_receipt_sha256':sha(receipt),'debugger_failure_sha256':sha(mi),
        'raw_stderr_bytes':size,'raw_stderr_full_hash_pending':True,'raw_excerpts':excerpts,'health_receipt_sha256':sha(health),
        'observed':{'actual_binary_sha256':r['binary_sha256'],'input_tokens':32768,'completed_prompt_progress_tokens':16384,
                    'abort_thread':82,'abort_call':'main watchdog -> abort','watchdog_seconds':60,
                    'stage':'first32K prefill, chunk16384, layer26','main_thread':'Stager::wait',
                    'staging_threads':'Three application copy-completion polling workers; native counter events remain NOT_READY',
                    'pending_native_events':['0x6089678','0x608af58','0x6087d98'],
                    'last_copy_size_bytes':1971200,'gpu_trace_bytes':0,'complete_numerical_gate_passed':False,
                    'new_kernel_fault_messages':[],'post_exit_exact_word_gpu_runtime_check_passed':True,'owned_survivors':False},
        'limits':['No successful32K inference result, throughput, transfer/kernel breakdown or CPU utilization can be derived from this run.',
                  'DD5 passed four unprofiled fresh32K reads and actual disk restoration, but an earlier unprofiled full256K run also stalled on pending copy events. This failure does not establish that the profiler alone caused it.',
                  'The source-audited profiler alters event and append handling; source review did not guarantee runtime compatibility. Its GPU timestamp configuration remains rejected.',
                  'Expected/current native counter values and a complete GPU command timeline were not captured; the underlying event/dependency/lifetime cause remains unresolved.',
                  'Uniform DPCT definitions do not prove the pending-event problem is fixed.'],
        'minimum_performance_input_tokens':32768,'diagnostic_timing_excluded':True,
        'no_reset_rebind_reboot_service_package_global_change':True}
(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'passed':True,'record_sha256':sha(out/'record.json'),'health_passed':True,'timestamp_configuration_accepted':False,'underlying_cause_resolved':False},indent=2))
