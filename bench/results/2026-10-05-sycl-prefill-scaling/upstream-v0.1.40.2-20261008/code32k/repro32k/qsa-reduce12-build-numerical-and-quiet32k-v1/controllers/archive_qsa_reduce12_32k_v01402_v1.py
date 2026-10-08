"""Require all sixteen fresh32K gates, then summarize and archive quiet ABBA."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess, sys

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0, str(root/'sycl/tools'))
from owned_gdb import process_identity
parent = root/'bench/results/2026-10-05-sycl-prefill-scaling'
upstream = parent/'upstream-v0.1.40.2-20261008'
code = upstream/'code32k'
repro = code/'repro32k'
out = repro/'qsa-reduce12-build-numerical-and-quiet32k-v1'
sequence_dir = base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1'
sequence = json.loads((sequence_dir/'record.json').read_text())
assert sequence['passed'] and not sequence['active'] and len(sequence['steps']) == 4
assert sequence['minimum_comparison_input_tokens'] == 32768
assert sequence['job_controller_sha256'] == 'f83dcd4fcdc38731602986d1e83804b22544d4da27134594a0e215f1c3750cc6'
assert sequence['controller_sha256'] == '8a0977339226ccb2e2f8c83dbf9426aca9997ad8a48635f14c9b2c3f14e920a4'
reference_path = base/'owned-profile-definition-v01402-code32k-diagnostic-r6/record.json'
reference = json.loads(reference_path.read_text())
assert reference['healthy'] and reference['math_gate_passed'] and not reference['active']
references = {r['name']: r for r in reference['requests']}
names = ['control32k-first', 'alternate32k-first', 'control32k-repeat', 'alternate32k-repeat']
reference_names = ['control32k-before', 'alternate32k-first', 'control32k-repeat', 'alternate32k-repeat']
expected_plan = [('dd5-control', 1), ('qsa-reduce12', 1), ('qsa-reduce12', 2), ('dd5-control', 2)]
binary_ids = {'dd5-control': 'dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34',
              'qsa-reduce12': '26c29c1353c64cef0e81c18bc1f651c9bcc25f977429d71b18052c5d13ecd34e'}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

qsa_numerical_path = base/'owned-qsa-reduce12-sg32-v01402-code32k-diagnostic-r1/record.json'
qsa_numerical = json.loads(qsa_numerical_path.read_text())
assert sha(qsa_numerical_path) == '02c8e8abfccca07aaf1767c8d0ed737c8f0087de08208a6e14e377aff7195d33'
assert not qsa_numerical['active'] and qsa_numerical['completed'] and qsa_numerical['healthy'] and qsa_numerical['math_gate_passed']
assert qsa_numerical['code32k_sequence_completed'] and not qsa_numerical['full_capacity_sequence_completed']
assert qsa_numerical['exit_code'] == 0 and not qsa_numerical['exit_signal'] and not qsa_numerical.get('error')
assert not qsa_numerical['new_fault_messages'] and not any(qsa_numerical['cleanup'].values())
assert len(qsa_numerical['requests']) == 6 and all(r['math_gate_passed'] for r in qsa_numerical['requests'])
assert len(qsa_numerical['sessions']) == 2 and all(r['passed'] for r in qsa_numerical['sessions'])
assert qsa_numerical['actual_executable_identity']['sha256'] == binary_ids['qsa-reduce12']
qsa_references = {r['name']: r for r in qsa_numerical['requests']}
for role in ['inferior', 'debugger']:
    identity = qsa_numerical[role]; now = process_identity(identity['pid'])
    assert not now or now['start_ticks'] != identity['start_ticks']
build_path = base/'qsa-reduce12-sg32-v01402-build-v1/record.json'
build = json.loads(build_path.read_text())
assert sha(build_path) == qsa_numerical['qsa_build_receipt_sha256']
assert build['passed'] and not build['active'] and build['baseline_inputs_unchanged']
assert build['only_qsa_decode_archive_member_replaced'] and not build['adopted']
assert sha(build['candidate_binary']) == build['candidate_binary_sha256'] == binary_ids['qsa-reduce12']
assert [r['member'] for r in build['archive_members'] if r['replaced']] == ['qsa_decode_attn.dp.cpp.o']
assert all(r['before_sha256'] == r['after_sha256'] for r in build['archive_members'] if not r['replaced'])
assert qsa_numerical['engine_log_bytes'] == (qsa_numerical_path.parent/'debugger/inferior.stderr').stat().st_size
jobs = []
rows = []
for step, (mode, repetition) in zip(sequence['steps'], expected_plan):
    assert (step['mode'], step['repetition']) == (mode, repetition) and step['exit_code'] == 0
    path = Path(step['receipt'])
    assert sha(path) == step['receipt_sha256']
    d = json.loads(path.read_text())
    assert not d['active'] and d['completed'] and d['healthy'] and d['math_gate_passed']
    assert d['quiet_four_fresh_reads_completed'] and not d['full_capacity_sequence_completed']
    assert d['exit_code'] == 0 and not d['exit_signal'] and not d.get('error')
    assert not d['new_fault_messages'] and not any(d['cleanup'].values())
    assert d['binary_sha256'] == binary_ids[mode] and len(d['requests']) == 4
    assert d['argv'][1:] == reference['argv'][1:]
    env = d['actual_target_environment']
    assert not any(k.startswith(('UR_LOG_', 'ZE_ENABLE_', 'ZEL_', 'UNITRACE_', 'XPTI_')) or
                   k in ['LD_PRELOAD', 'ZET_ENABLE_METRICS', 'STRATA_TRACE', 'STRATA_DUMP_FIRST_LOGITS',
                         'STRATA_PREFILL_DUMP_STATE', 'STRATA_PREFILL_SYNC', 'STRATA_PREFILL_TRANSFER_TIMING'] for k in env)
    for key in ['inferior', 'debugger']:
        identity = d[key]
        now = process_identity(identity['pid'])
        assert not now or now['start_ticks'] != identity['start_ticks']
    for i, (r, name, reference_name) in enumerate(zip(d['requests'], names, reference_names)):
        expected = references[reference_name]
        assert r['name'] == name and r['process_read'] == i+1 and r['first_process_read'] == (i == 0)
        assert r['math_gate_passed'] and all(r['comparison'].values())
        assert r['input_tokens'] == 32768 and r['resume_tokens'] == [0, 0]
        assert len(r['ids']) == len(r['logprobs']) == 64
        assert r['ids'] == expected['ids'] and r['logprobs'] == expected['logprobs'] and r['mtp_counts'] == expected['mtp_counts']
        qr = qsa_references[reference_name]
        assert r['ids'] == qr['ids'] and r['logprobs'] == qr['logprobs'] and r['mtp_counts'] == qr['mtp_counts']
        prompt = Path(r['prompt_file'])
        assert sha(prompt) == r['prompt_sha256'] and len(prompt.read_text().split()) == 32768
        mm = r['measurement']
        assert mm['prompt_tokens'] == 32768 and mm['generated_tokens'] == 64
        assert mm['prompt_ms'] > 0 and mm['decode_ms'] > 0
        assert abs(mm['prefill_tok_s']-32768000/mm['prompt_ms']) < 1e-9
        assert abs(mm['decode_tok_s']-64000/mm['decode_ms']) < 1e-9
        rows.append({'ordinal': step['ordinal'], 'mode': mode, 'repetition': repetition,
                     'input': 'A' if i in [0, 2] else 'B', 'name': name, 'process_read': i+1,
                     'first_process_read': i == 0, 'input_tokens': 32768,
                     'ids_logprobs_mtp_and_fresh_read_equal': True, **mm})
    jobs.append({'ordinal': step['ordinal'], 'mode': mode, 'repetition': repetition,
                 'receipt': str(path), 'receipt_sha256': sha(path), 'binary_sha256': d['binary_sha256'],
                 'exit_code': 0, 'forced_cleanup': False, 'new_kernel_fault_messages': []})
assert len(rows) == 16

def aggregate(selected):
    assert selected
    return {'requests': len(selected), 'input_tokens_per_request': 32768,
            'total_prompt_ms': sum(r['prompt_ms'] for r in selected),
            'total_decode_ms': sum(r['decode_ms'] for r in selected),
            'prefill_tok_s': sum(r['prompt_tokens'] for r in selected)*1000/sum(r['prompt_ms'] for r in selected),
            'decode_tok_s': sum(r['generated_tokens'] for r in selected)*1000/sum(r['decode_ms'] for r in selected),
            'prefill_range': [min(r['prefill_tok_s'] for r in selected), max(r['prefill_tok_s'] for r in selected)],
            'decode_range': [min(r['decode_tok_s'] for r in selected), max(r['decode_tok_s'] for r in selected)]}

groups = {}
for mode in binary_ids:
    groups[mode] = {}
    for name, predicate in [('first_process_read', lambda r: r['first_process_read']),
                            ('subsequent_full_reads', lambda r: not r['first_process_read'])]:
        groups[mode][name] = aggregate([r for r in rows if r['mode'] == mode and predicate(r)])
    groups[mode]['per_read_slot'] = {str(i): aggregate([r for r in rows if r['mode'] == mode and r['process_read'] == i]) for i in range(1, 5)}
effects = {}
for name in ['first_process_read', 'subsequent_full_reads']:
    c, p = groups['dd5-control'][name], groups['qsa-reduce12'][name]
    effects[name] = {metric+'_change_percent': (p[metric]/c[metric]-1)*100 for metric in ['prefill_tok_s', 'decode_tok_s']}
source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
summary = {'active': False, 'completed': True, 'passed': True, 'adopted': False,
           'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_commit': source_commit,
           'scope': 'Four fresh processes in DD5/QSA/QSA/DD5 order, four full32768 inputs A/B/A/B and64 generated tokens each. Only quiet complete gated jobs are compared.',
           'minimum_performance_input_tokens': 32768, 'fresh_full_reads': 16,
           'all_ids_logprobs_mtp_equal_and_resume_reused_zero': True,
           'same_arguments_after_binary': True, 'all_normal_quit_no_forced_cleanup_no_new_kernel_fault': True,
           'first_process_read_separated': True, 'first_read_is_not_cold_disk_system_benchmark': True,
           'rates_computed_from_total_tokens_over_total_engine_time': True,
           'groups': groups, 'effects': effects, 'jobs': jobs, 'measurements': rows,
           'sequence_receipt_sha256': sha(sequence_dir/'record.json'), 'reference_dd5_receipt_sha256': sha(reference_path),
           'interpretation': 'The QSA SG32/transposed reduction improves prefill in all later read slots in this matched32K sample. Decode is slower in the pooled sample; two processes per mode do not establish a decode benefit or regression. Physical256K qualification is still required before adoption.',
           'cpu_utilization_or_pcie_dominance_measured': False, 'underlying_pending_counter_cause_resolved': False,
           'physical256k_sequence_passed': False,
           'pending_before_adoption': ['Physical262144 cells, repeated fresh full prefill, actual disk restore, clipped speculative tail, capacity refusals and later valid fresh32K', 'Continue source/lifetime checks and retain owned failure capture; no claim of resolving the earlier native-counter cause'],
           'no_reset_rebind_reboot_service_package_global_change': True,
           'production_and_main_unchanged': True}
summary.update(qsa_initial_numerical_receipt_sha256=sha(qsa_numerical_path),
               qsa_build_receipt_sha256=sha(build_path), candidate_engine_compiled=True,
               qsa_initial_four_fresh32k_and_actual_disk_continuation_passed=True,
               qsa_source_sha256=build['candidate_source_sha256'],
               only_qsa_decode_archive_member_replaced=True,
               numerical_diagnostic_durations_excluded=True,
               numerical_engine_log_bytes=qsa_numerical['engine_log_bytes'],
               numerical_engine_log_sha256=qsa_numerical['engine_log_sha256'],
               underlying_stall_cause_resolved=False)
out.mkdir()
def copy(a, b):
    a, b = Path(a), Path(b)
    assert not b.exists() and a.is_file() and not a.is_symlink()
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a, b)
for job in jobs:
    directory = Path(job['receipt']).parent
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            assert path.stat().st_size <= 64*1024**2
            copy(path, out/'quiet-runs'/directory.name/path.relative_to(directory))
for path in sorted(sequence_dir.iterdir()):
    if path.is_file(): copy(path, out/'sequence'/path.name)
for path in sorted(build_path.parent.iterdir()):
    if path.is_file(): copy(path, out/'build'/path.name)
for path in sorted(qsa_numerical_path.parent.rglob('*')):
    if path.is_file() and path.stat().st_size <= 32*1024**2:
        copy(path, out/'initial-numerical32k'/path.relative_to(qsa_numerical_path.parent))
for name in ['prepare_owned_qsa_reduce12_code32k_v01402_v1.py', 'run_owned_qsa_reduce12_code32k_v01402_v1.py',
             'prepare_owned_qsa_reduce12_code32k_v01402_v2.py', 'run_owned_qsa_reduce12_code32k_v01402_v2.py',
             'build_qsa_reduce12_sg32_v01402_v1.py', 'prepare_owned_qsa_reduce12_matched_quiet32k_v01402_v1.py',
             'run_owned_qsa_reduce12_matched_quiet32k_v01402_v1.py', 'run_qsa_reduce12_matched_quiet32k_v01402_sequence_v1.py',
             'capture_owned_native_counter_v01402_v1.py', 'prepare_archive_qsa_reduce12_32k_v01402_v1.py', Path(__file__).name]:
    copy(base/name, out/'controllers'/name)
for name in ['qsa-reduce12-code32k-preflight-rejection-v1.json', 'prepare-owned-qsa-reduce12-code32k-v01402-v2.json',
             'prepare-owned-qsa-reduce12-matched-quiet32k-v01402-v1.json', 'qsa-reduce12-matched-quiet32k-v01402-source-review-v1.json',
             'hc-and-dense-knob-applicability-v01402-source-review-v1.json']:
    copy(base/name, out/'checks'/name)
copy(build['candidate_source'], out/'source/qsa_decode_attn.dp.cpp')
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
table = []
for label, name in [('First process read', 'first_process_read'), ('Subsequent full reads', 'subsequent_full_reads')]:
    c, p = groups['dd5-control'][name], groups['qsa-reduce12'][name]
    table.append(f"| {label} | {c['requests']} per mode | {c['prefill_tok_s']:.3f} | {p['prefill_tok_s']:.3f} | {effects[name]['prefill_tok_s_change_percent']:+.3f}% | {c['decode_tok_s']:.3f} | {p['decode_tok_s']:.3f} | {effects[name]['decode_tok_s_change_percent']:+.3f}% |")
(out/'README.md').write_text("""# QSA twelve-head reduction: build, numerical gate and matched 32K timing

Measured on Ryzen 5 5600X, 128 GiB RAM and Arc B570 with 10 GiB VRAM; kernel 7.0.0-38, NEO 26.31.39395.14, IGC 2.41.5 and oneAPI 2026.1.1. Both binaries use the integrated v0.1.40.2 engine with context 262144, chunk 8192, int8 KV, 32768 resident cells, cache 128, five CPU workers and MTP 4. Main and production binaries were unchanged.

The candidate replaces only `qsa_decode_attn.dp.cpp.o` in the qualified DD5 kernel archive. The SG32/transposed arm computes the same twelve-head dot expressions and XOR reduction tree using sixteen subgroup exchanges per lane, compared with sixty in the previous source. Actual subgroup lane/group identifiers route the head and cell work. The earlier [CPU arithmetic/routing review and complete DD5 lifecycle](../uniform-full256k-and-qsa-source-v1/README.md) remain scoped to their own archive boundary. The actual one-object compile, archive replacement and link took 29.461 seconds and passed header/dependency, archive-member and other link-input checks. Exact build commands and warnings are retained under `build`.

The candidate's initial logged GPU gate completed four fresh 32768-token inputs A/B/A/B. Head, used state, all output IDs, logprobs and visible MTP counts matched the qualified references. Actual disk SAVE/RESTORE continuation matched the live continuation. All six request gates and both session operations passed, followed by normal QUIT, no forced cleanup or surviving owned process, and no new kernel fault. Flushed UR/ZE/ZEL/Strata logs were enabled, so all of these diagnostic times are excluded from speed comparisons. The 13437101076-byte raw stderr and large state/session files remain private, with complete terminal hashes and comparisons in the receipt. The earlier numerical launcher v1 was rejected on the CPU before creating a run directory or GPU process because its inherited repetition guard still required 7; the corrected v2 and separate negative receipt are preserved.

For timing, run order was DD5, QSA, QSA, DD5. Each fresh process read A/B/A/B, with 32768 input and 64 generated tokens each time. All sixteen requests reported RESUME 0 and REUSED 0, and output IDs, logprobs and visible MTP counts matched both completed numerical gates. Every process exited normally. API/validation logs, profiler/preload, phase trace, state/head/payload dumps and extra waits were disabled. No other model, GPU health test, build or heavy local analysis ran concurrently. The executed controller checked the owned PID/start ticks, executable path and SHA-256 through `/proc/PID/exe` after READY; argument and environment checks are retained in the source and receipts.

First process reads are separate from later full reads. They include process JIT/capture warmup and are not a cold disk/system benchmark. Rates use total tokens divided by total engine time. Per-read-slot values and raw ranges are in [summary.json](summary.json).

| Reads | Count | DD5 PP tok/s | QSA PP tok/s | PP change | DD5 TG tok/s | QSA TG tok/s | TG change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
""" + '\n'.join(table) + """

In this sample all later QSA prefill reads were faster than all later matched controls. The pooled later full-read gain is about 1.35%, or 1.018 seconds per 32K prefill. Decode was slower in the pooled sample; two processes per mode do not establish a decode benefit or regression. The changed arm is the transposed prefill path; the nontransposed decode arithmetic remains unchanged. This result does not measure PCIe dominance, resolve earlier native-event failures or reach the performance target.

The QSA binary is not adopted. It still requires its own complete physical 256K sequence: all 262144 cells through cell 262143, repeated fresh full prefill, all saved state/KV tensor bytes, actual disk restoration, clipped speculative tail, both capacity refusals and a later valid fresh 32768-token input. It cannot inherit DD5's completed capacity proof. No reset, rebind, reboot, service, package or global setting changed.

The separate HC/dense knob review is source-only: existing `gr_upmix` and calibrated `bf16_hcd_exact` fast paths are HIP implementations with no SYCL equivalent enabled by their knobs. No new flags or performance conclusions result from that review. The archive manifest identifies copied files; paths inside raw records retain the original machine locations. The large diagnostic stderr is represented by the complete terminal hash rather than copied or rehashed here.
""")
print(json.dumps({'archive':str(out), 'groups':groups, 'effects':effects},indent=2))
