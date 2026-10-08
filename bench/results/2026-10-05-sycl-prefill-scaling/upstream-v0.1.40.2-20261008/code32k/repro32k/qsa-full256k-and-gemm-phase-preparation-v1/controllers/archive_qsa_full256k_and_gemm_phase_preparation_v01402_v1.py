"""Archive terminal QSA qualification and separately scoped unexecuted preparations."""
from pathlib import Path
import ast, datetime, hashlib, json, shutil, subprocess

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
old=base/'archive_uniform_full256k_and_qsa_source_v01402_v1.py'
source=old.read_text()
# Reuse the complete physical-capacity verifier, never the old source-only status.
checks=source[:source.index('\ndef sha(path):')]
checks=checks.replace("run = base / 'owned-profile-definition-v01402-full256k-diagnostic-r7'", "run = base / 'owned-qsa-reduce12-sg32-v01402-full256k-diagnostic-r1'")
checks=checks.replace("== 'dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'", "== '26c29c1353c64cef0e81c18bc1f651c9bcc25f977429d71b18052c5d13ecd34e'")
exec(compile(checks,str(old),'exec'),globals())
assert sessions['save-full256k-first']['all_saved_state_and_kv_bytes_equal_dd5']
assert r['qualified_dd5_full256k_receipt_sha256']=='18436b6361831d41aa19e1374454aed458e746f3e234092a1793fd15b823d2a3'
assert r['qsa_matched_quiet32k_sequence_sha256']==hashlib.sha256((base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1/record.json').read_bytes()).hexdigest()
assert r['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def copy(a,b):
    a,b=Path(a),Path(b)
    assert a.is_file() and not a.is_symlink() and not b.exists()
    assert a.stat().st_size<=32*1024**2
    b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)

out=parent/'qsa-full256k-and-gemm-phase-preparation-v1'
assert not out.exists();out.mkdir()
for p in sorted(run.rglob('*')):
    if p.is_file() and p.stat().st_size<=32*1024**2:
        copy(p,out/'terminal-qsa-full256k-r1'/p.relative_to(run))
controllers=['prepare_owned_qsa_reduce12_full256k_v01402_v1.py','run_owned_qsa_reduce12_full256k_v01402_v1.py',
             'prepare_gemm_host_scalars_v01402_v1.py','prepare_build_gemm_host_scalars_v01402_v1.py',
             'build_gemm_host_scalars_v01402_v1.py','prepare_build_gemm_host_scalars_v01402_v2.py',
             'build_gemm_host_scalars_v01402_v2.py','prepare_owned_gemm_host_scalars_code32k_v01402_v1.py',
             'run_owned_gemm_host_scalars_code32k_v01402_v1.py','prepare_owned_dd5_phase32k_v01402_v1.py',
             'run_owned_dd5_phase32k_v01402_v1.py','prepare_owned_dd5_phase32k_v01402_v2.py',
             'run_owned_dd5_phase32k_v01402_v2.py',Path(__file__).name]
for name in controllers: copy(base/name,out/'controllers'/name)
checks=['prepare-owned-qsa-reduce12-full256k-v01402-v1.json',
        'gemm-host-scalars-api-contract-review-v01402-v1.json','gemm-host-scalars-builder-early-guard-cpu-v01402-v1.json',
        'gemm-host-scalars-build-recipe-review-v01402-v2.json','prepare-build-gemm-host-scalars-v01402-v1.json',
        'prepare-build-gemm-host-scalars-v01402-v2.json','prepare-owned-gemm-host-scalars-code32k-v01402-v1.json',
        'existing-prefill-profiler-feasibility-v01402-source-review-v1.json',
        'prepare-owned-dd5-phase32k-v01402-v1.json','prepare-owned-dd5-phase32k-v01402-v2.json']
for name in checks: copy(base/name,out/'checks'/name)
for name in ['gemm-host-scalars-v01402-source-v1','gemm-host-scalars-ninja-command-review-v01402-v1']:
    for p in sorted((base/name).rglob('*')):
        if p.is_file():copy(p,out/'preparation'/name/p.relative_to(base/name))
gemm=json.loads((base/'gemm-host-scalars-v01402-source-v1/record.json').read_text())
assert gemm['prepared'] and not gemm['compiled'] and not gemm['gpu_tested'] and not gemm['adopted']
assert not (root/'build-sycl-gemm-host-scalars-v2-20261008').exists()
assert not (base/'gemm-host-scalars-v01402-build-v2').exists()
assert not (base/'owned-dd5-phase-v2-v01402-code32k-diagnostic-r1').exists()
phase=json.loads((base/'prepare-owned-dd5-phase32k-v01402-v2.json').read_text())
assert phase['prepared'] and not phase['gpu_launched'] and len(phase['CPU_validation_cases'])==11
quiet=json.loads((parent/'qsa-reduce12-build-numerical-and-quiet32k-v1/summary.json').read_text())
summary={'active':False,'passed':True,'adopted':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'scope':'Complete QSA physical256K lifecycle qualification; separate source/preparation-only GEMM scalar dispatch and existing compute-phase diagnostics. No new speed, PCIe dominance or stall-cause conclusion.',
         'raw_receipt_sha256':sha(run/'record.json'),'actual_binary_sha256':r['binary_sha256'],
         'qualified_dd5_full256k_receipt_sha256':r['qualified_dd5_full256k_receipt_sha256'],
         'elapsed_diagnostic_seconds':r['elapsed_seconds'],'fresh32k_reads':4,'fresh_full_reads':2,
         'full_input_tokens':262140,'physical_cells_per_layer':262144,'last_physical_cell':262143,'kv_layers':13,
         'all_head_used_state_output_logprob_mtp_gates_passed':True,
         'all_saved_state_and_kv_bytes_equal_qualified_dd5':True,
         'all_saved_state_and_kv_tensor_bytes_equal_repeat_and_disk_roundtrip':True,'ignored_saved_kv_tensor_bytes':0,
         'actual32k_disk_restore_continuation_passed':True,'actual_full_disk_restore_and_clipped_two_token_verification_passed':True,
         'both_capacity_refusals_before_gpu_work_and_later_fresh32k_passed':True,
         'normal_exit_code':0,'owned_survivors':False,'new_kernel_fault_messages':[],
         'raw_stderr_path':str(run/'debugger/inferior.stderr'),'raw_stderr_bytes':r['engine_log_bytes'],
         'raw_stderr_sha256':r['engine_log_sha256'],'raw_stderr_digest_computed_by_terminal_controller_not_rehashed_by_archiver':True,
         'minimum_performance_input_tokens':32768,'diagnostic_durations_excluded':True,
         'underlying_pending_native_event_cause_resolved':False,
         'QSA_32k_measurement_summary_sha256':sha(parent/'qsa-reduce12-build-numerical-and-quiet32k-v1/summary.json'),
         'gemm_engine_compiled':False,'gemm_GPU_tested':False,'gemm_adopted':False,
         'gemm_candidate_source_sha256':gemm['candidate_source_sha256'],
         'gemm_scalar_api_source_review_only':True,'gemm_ninja_recursive_command_assumption_rejected_before_compile':True,
         'existing_compute_phase_profile_executed':False,'phase_report_CPU_validation_cases':11,
         'no_reset_rebind_reboot_service_package_global_change':True,'production_and_main_unchanged':True,
         'pending':['Initial logged DD5 existing compute-phase numerical/instrumentation gate on >=32768 inputs',
                    'Separate no-API-log profile if numerical instrumentation succeeds; intervals include host/copy gaps',
                    'Compile GEMM one-object builder v2 and initial logged >=32768 numerical/disk gate',
                    'Any new performance candidate needs its own clean >=32768 first/later comparison and physical256K lifecycle']}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'README.md').write_text('''# Complete QSA physical 256K qualification and next preparations

The QSA reduction binary (`26c29c1353c64cef0e81c18bc1f651c9bcc25f977429d71b18052c5d13ecd34e`) completed the capacity and state sequence on Arc B570 10 GiB, Ryzen 5 5600X and 128 GiB system RAM, with NEO 26.31.39395.14, IGC 2.41.5 and oneAPI 2026.1.1. Context was 262144, chunk 8192, int8 KV with 32768 resident cells, requested expert cache 128, five CPU workers and MTP 4. Actual argv, environment, executable identity and owned PID/start ticks are in the [terminal receipt](terminal-qsa-full256k-r1/record.json).

Two fresh 262140-token inputs plus four generated tokens reached physical cell 262143. A fresh 32768-token input separated the full reads; both full reads reported RESUME 0 and REUSED 0. Head, used state, output IDs, logprobs and visible MTP counts matched the qualified DD5 reference and each other. All 13 saved KV layers contain 262144 physical cells. Every saved state and KV tensor byte matched DD5, the repeated full save, and an actual disk RESTORE/SAVE round trip. No saved tensor bytes were ignored.

The restored 262142-token input plus two outputs resumed at 262139 and clipped the final speculative verification window to two tokens, matching the earlier clipped request. A separate 32K disk restoration reproduced live continuation. Requests of 262144 plus one output and 262142 plus three outputs were refused before GPU verification or token output. A later fresh 32768-token input matched the reference. All twelve request gates and six session operations passed; QUIT returned zero, no forced cleanup or owned process remained, and no new kernel GPU fault was recorded. No reset, rebind, reboot, service, package or global setting changed.

This qualifies this QSA binary for the tested physical lifecycle. It does not resolve the earlier pending native-event cause. Flushed UR/ZE/ZEL/Strata logs were enabled, so all durations here are diagnostic and excluded from performance comparisons. The complete terminal stderr byte count and SHA-256 are in [summary.json](summary.json); large stderr, state and session files remain private. The archiver uses the terminal controller's completed hash rather than rehashing the large stderr.

The [earlier clean 32K comparison](../qsa-reduce12-build-numerical-and-quiet32k-v1/README.md) remains the performance evidence: sixteen fresh 32768-token reads in DD5/QSA/QSA/DD5 process order, with first process reads separate from later full reads. Later pooled prefill was 427.237 versus 432.984 tok/s (+1.345%); decode was 16.730 versus 16.410 tok/s (-1.916%). Two processes per mode do not establish a decode benefit or regression. That archive's pending-capacity statement describes its earlier boundary; this later archive supplies QSA's own completed capacity result. The main and production binaries remain unchanged.

The next GEMM candidate is separately source-only. It passes BF16/F16 GEMM alpha/beta as host values to the typed oneMKL call, retaining the installed DPCT wrapper's compute mode, queue, dimensions, transpose and strides. The old host scalar path performs pointer classification and a host dereference; it does not copy a scalar or wait for the GPU. The possible benefit is less host dispatch work, with no measured benefit or stall-cause claim. The installed headers and official GEMM/scalar/compute-mode contracts are reviewed in `checks`. The first builder's assumption that `ninja -t commands target` returns one command was rejected before compilation: it returned 28 commands including prerequisites. The prepared v2 uses `ninja -t commands -s target` and replaces only the prefill GEMM archive member. The live-job early guard was exercised on the CPU and prevented build directories, compiler processes and large input reads. No GEMM engine has been compiled or run at this archive boundary.

The existing compute-phase profiler is also unexecuted at this boundary. Source review shows the benchmark passes a separately created compute queue, so DPCT's default-queue global-wait branch does not apply here. Marker barriers, two extra compute waits per chunk and timestamps between markers alter execution and include host/copy gaps. They are not pure GPU busy time or a clean throughput comparison. Transfer timing would enable profiling when its separate copy queue is created; it is disabled in this prepared first diagnostic. No unitrace interposer is retried. The prepared logged controller compares four fresh 32768-token A/B reads and actual disk continuation against DD5, then requires complete finite nonnegative phase reports, correct prefix length, consistent totals, plausible wall bounds and no profiling errors. Eleven CPU cases exercised valid reports and malformed, missing, incomplete, nonfinite, negative, inconsistent, oversized and API-error rejection. Instrumentation rejection follows normal QUIT. Native profiling still needs its initial logged numerical/instrumentation gate.

Every performance comparison uses at least 32768 input tokens and separates first/later full reads. Prepared source and CPU checks do not qualify another binary, inherit this capacity proof or demonstrate GPU performance.
''')
print(json.dumps({'archive':str(out),'QSA_full256k_passed':True,'gemm_compiled':False,'phase_profile_executed':False,'raw_receipt_sha256':summary['raw_receipt_sha256']}))
