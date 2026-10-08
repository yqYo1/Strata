"""Archive the rejected GPU timestamp configuration and completed polling gate."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k'
failed=base/'owned-device-profile-v01402-code32k-diagnostic-r1'
run=base/'owned-poll-backoff-v01402-code32k-diagnostic-r1'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def copy(a,b):
    assert not b.exists();b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
f=json.loads((failed/'record.json').read_text())
r=json.loads((run/'record.json').read_text())
assert not f['active'] and not f['completed'] and not f['healthy'] and not f['new_fault_messages']
assert not r['active'] and r['completed'] and r['healthy'] and r['math_gate_passed']
assert r['code32k_sequence_completed'] and not r['full_capacity_sequence_completed']
assert r['exit_code']==0 and not r['exit_signal'] and not r.get('error') and not r['new_fault_messages']
assert not any(r['cleanup'].values())
assert r['binary_sha256']=='44a274d8c699829020abb8a85a331a638f40efdf5d6c959a076a58e818f75391'
assert len(r['requests'])==6 and all(v['math_gate_passed'] for v in r['requests'])
dd5_path=base/'owned-profile-definition-v01402-code32k-diagnostic-r6/record.json'
dd5=json.loads(dd5_path.read_text())
assert dd5['healthy'] and dd5['math_gate_passed'] and not dd5['active']
by_name={v['name']:v for v in dd5['requests']}
for result in r['requests']:
    reference=by_name[result['name']]
    assert result['ids']==reference['ids'] and result['logprobs']==reference['logprobs']
    assert result['mtp_counts']==reference['mtp_counts']
    if 'first_head' in result:
        assert result['first_head']['sha256']==reference['first_head']['sha256']
names=['control32k-before','alternate32k-first','control32k-repeat','alternate32k-repeat']
fresh=[v for v in r['requests'] if v['name'] in names]
assert len(fresh)==4 and all(v['input_tokens']==32768 and v['resume_tokens'] and all(n==0 for n in v['resume_tokens']) for v in fresh)
assert all(v['passed'] for v in r['sessions']) and {v['name'] for v in r['sessions']}=={'save-control32k','restore-control32k'}
assert all(not Path('/proc',str(d[k]['pid'])).exists() for d in [f,r] for k in ['inferior','debugger'])
out=parent/'device-profile-rejection-and-poll32k-v1';out.mkdir()
copy(base/'archive-device-profile-poll32k-cpu-check-v2.json',out/'checks/archive-device-profile-poll32k-cpu-check-v2.json')
for d,dest,files in [
    (failed,'terminal-device-profile',('record.json','protocol.stdout.raw','events.jsonl','debugger/failure.mi.txt','debugger/inferior-argv.json','probes/kernel-before.stdout','probes/kernel-after.stdout','clock-calibration-during-v1.json','last-submissions-source-span-v1.json')),
    (run,'terminal-poll32k',('record.json','protocol.stdout.raw','events.jsonl','project-messages.txt','debugger/inferior-argv.json','probes/kernel-before.stdout','probes/kernel-after.stdout'))]:
    for name in files:copy(d/name,out/dest/name)
for name in ['unitrace-device-options-v01402-v1','device-profile-watchdog-abort-v01402-review-v1','post-device-profile-abort-v01402-health-v1','prefill-poll-backoff-v01402-relink-v3']:
    for p in sorted((base/name).iterdir()):
        if p.is_file() and p.name!='options':copy(p,out/'checks'/name/p.name)
for name in ['audit_unitrace_device_options_v01402_v1.py','prepare_owned_device_profile32k_v01402_v1.py','run_owned_device_profile32k_v01402_v1.py',
             'check_health_after_device_profile_abort_v01402_v1.py','audit_device_profile_abort_v01402_v1.py','relink_prefill_poll_backoff_v01402_v3.py',
             'prepare_owned_poll_backoff_code32k_v01402_v1.py','run_owned_poll_backoff_code32k_v01402_v1.py',
             'archive_device_profile_abort_and_poll32k_v01402_v1.py','prepare_archive_device_profile_poll32k_v01402_v2.py',Path(__file__).name]:
    copy(base/name,out/'controllers'/name)
poll=json.loads((base/'prefill-poll-backoff-v01402-relink-v3/record.json').read_text())
copy(Path(poll['actual_shadow_header']),out/'compiled-source/polling/strata/host_wait.hpp')
copy(Path(poll['actual_dependency_file']),out/'compiled-source/polling/prefill.cpp.o.d')
for name in ['record.json','stderr','stdout']:
    path=base/'vtune-smoke/gpu-collect'/name
    if path.is_file():copy(path,out/'vtune-prior-startup-failure'/name)
stderr=failed/'debugger/inferior.stderr'
hash_record={'active':False,'passed':True,'scope':'Additional full-file identity after failed model/debugger exit. Raw terminal receipt and historical evidence extraction receipt remain unchanged.',
             'terminal_receipt_sha256':sha(failed/'record.json'),'raw_stderr_bytes':stderr.stat().st_size,'raw_stderr_sha256':sha(stderr),
             'empty_gpu_trace_bytes':(failed/f"strata.{f['inferior']['pid']}.json").stat().st_size,
             'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/'device-profile-full-log-identity.json').write_text(json.dumps(hash_record,indent=2)+'\n')
summary={'active':False,'completed':True,'adopted':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
         'device_timestamp_configuration_accepted':False,'device_profile_math_gate_passed':False,
         'device_profile_terminal_receipt_sha256':sha(failed/'record.json'),
         'device_profile_terminal':'Host watchdog SIGABRT after60s without progress, first32768 prompt at layer26/chunk16384',
         'post_exit_exact_word_gpu_runtime_health_passed':True,'new_kernel_fault_messages':[],
         'underlying_pending_event_cause_resolved':False,'vtune_prior_xpu_startup_failure':'Analysis not applicable; VTune cannot recognize processor/microarchitecture, before GPU helper execution.',
         'poll_build_receipt_sha256':sha(base/'prefill-poll-backoff-v01402-relink-v3/record.json'),
         'poll_actual_binary_sha256':r['binary_sha256'],'poll_completed_terminal_receipt_sha256':sha(run/'record.json'),
         'poll_four_fresh32768_reads_passed':True,'poll_same_dd5_base_state_head_ids_logprobs_mtp_equal':True,
         'poll_actual_disk_restore_continuation_passed':True,'poll_exit_code':0,'poll_forced_cleanup':False,
         'owned_survivors':False,'physical256k_sequence_passed':False,
         'minimum_performance_input_tokens':32768,'all_diagnostic_and_profiled_durations_excluded':True,
         'pending':['Quiet matched>=32768 full reads with first and repeated measurements separated',
                    'Physical262144 cells, repeated fresh full prefill, actual disk restore, clipped speculative tail, refusals and later valid fresh32K before adoption',
                    'A validated profiler GPU timeline and native pending-counter/dependency diagnosis'],
         'no_reset_rebind_reboot_service_package_global_change':True,'production_and_main_unchanged':True}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'README.md').write_text('''# GPU timestamp rejection and the polling candidate's32K numerical gate

Measurements and diagnostics use the existing Ryzen 5 5600X, 128 GiB RAM and Arc B570 with 10 GiB VRAM, kernel 7.0.0-38, NEO 26.31.39395.14 and oneAPI 2026.1.1. Every fresh input in these gates has 32768 tokens. Diagnostic and instrumented durations are excluded from performance comparisons.

The earlier VTune 2026.4 XPU collection actually exited before the GPU helper ran: the analysis was not applicable because VTune could not recognize the processor/microarchitecture. Disabling CPU sampling and characterization had already been tried. The retained raw receipt/error output corrects the later incomplete assessment that only capability help had been checked.

The pinned Intel PTI unitrace source is 6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033. A CPU option/source check confirms that device timing and Chrome device tracing activate kernel timestamp instrumentation without API tracing, metrics, KMD collection, sampling or the forked supervisor branch. Collection stays enabled from startup through shutdown. The source supports the modern LaunchKernelWithArguments API; counter-event pools bypass timestamp flag rewriting, while in-order counter signals use a profiler timestamp event followed by the original signal. This review did not guarantee runtime compatibility.

The first logged timestamp experiment used the independently qualified DD5 binary and the same context capacity of 262144, chunk size of 8192, int8 KV with 32768 resident tokens and normal MTP with four draft tokens. It stopped after reporting 16384 prompt tokens. The application watchdog aborted after 60 seconds without progress at layer 26 of chunk 16384. The main thread was in Stager::wait; three staging workers were polling copy completion, with native counter events returning NOT_READY. Its GPU trace file is empty and no inference request completed. This configuration is rejected: neither the partial progress nor its durations establish PCIe dominance or a valid performance comparison. Raw controller/protocol/MI evidence, exact-offset excerpts and the additional full stderr identity are retained.

Owned model and debugger cleanup completed. No new xe/kernel fault was recorded, and a subsequent exact-word GPU/runtime health check passed without reset, rebind or reboot. An earlier unprofiled full256K run also stalled on pending copy events, so this result does not establish that the profiler alone caused the problem. Expected/current native counter values and the underlying dependency/lifetime cause remain unresolved.

The separate polling candidate is 44a274d8c699829020abb8a85a331a638f40efdf5d6c959a076a58e818f75391, based on DD5. Only prefill.cpp.o changes: after 32 failed readiness checks, request a 10 us host sleep instead of another scheduler yield. Completion queries, retained events, generation checks, acquire/release ordering, cancellation priority and the timeout policy are unchanged. The CPU tests were retained from the original polling build. Actual Ninja dependencies cover 116 linked translation units and confirm that only prefill includes the changed template header. Every prefill archive member and unchanged baseline link input was checked. The compiled shadow header's old comment about driver calls is retained exactly as built; the earlier review explains why that comment is inaccurate.

The logged candidate gate completed four fresh 32768-token reads using two alternating inputs, all first-head floats, all used state parts, 64 output IDs/logprobs and MTP-count comparisons. All six requests' output/head/MTP evidence also matches the completed DD5 run. Actual SAVE/RESTORE continuation matched, all saved main indexer spare rows were canonical, and QUIT exited normally with no forced cleanup, survivors or new kernel fault. Fresh repeated reads report RESUME 0. Full native event values and a validated GPU timeline are still unavailable.

This is a numerical gate, not a speed gain or an adoption. Quiet matched inputs of at least 32768 tokens, with first and repeated reads reported separately, remain pending. Physical occupancy through cell 262143, repeated fresh full prefill, actual full disk restoration, a clipped speculative tail, capacity refusals and a later valid fresh 32K read must all pass before adoption. Production and main remain unchanged; the embedding service remains stopped. No reset, rebind, reboot, service, package or global configuration action was performed.
''')
print(json.dumps({'archive':str(out),'poll32k_passed':True,'device_timestamp_configuration_accepted':False,'summary_sha256':sha(out/'summary.json')},indent=2))
