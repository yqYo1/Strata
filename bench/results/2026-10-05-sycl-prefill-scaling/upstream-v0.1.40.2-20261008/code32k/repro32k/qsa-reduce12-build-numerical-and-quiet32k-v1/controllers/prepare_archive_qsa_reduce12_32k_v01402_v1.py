"""Derive the gated quiet analyzer; retain all executed QSA build/numerical records."""
from pathlib import Path
import ast, hashlib

base = Path(__file__).parent
old = base / 'archive_poll_matched_quiet32k_v01402_v1.py'
s = old.read_text()
assert hashlib.sha256(s.encode()).hexdigest()
s = s[:s.index('out.mkdir()')]
replacements = {
    "out = repro/'poll-matched-quiet32k-v1'": "out = repro/'qsa-reduce12-build-numerical-and-quiet32k-v1'",
    "sequence_dir = base/'poll-matched-quiet32k-v01402-sequence-v1'": "sequence_dir = base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1'",
    'e681237f10cdb6cfd27e6a6ef848ac26519def4809a74125584054f56ecc3ff6': 'f83dcd4fcdc38731602986d1e83804b22544d4da27134594a0e215f1c3750cc6',
    '8c0686547eb7945ca5a74cd9eedfb6ceab4a7e5a426a5c46258e2b1288d4b021': '8a0977339226ccb2e2f8c83dbf9426aca9997ad8a48635f14c9b2c3f14e920a4',
    'dd5-poll-backoff': 'qsa-reduce12',
    '44a274d8c699829020abb8a85a331a638f40efdf5d6c959a076a58e818f75391': '26c29c1353c64cef0e81c18bc1f651c9bcc25f977429d71b18052c5d13ecd34e',
    'Four fresh processes in DD5/poll/poll/DD5 order': 'Four fresh processes in DD5/QSA/QSA/DD5 order',
    'No useful prefill improvement in this32K sample; no polling speed adoption. Two processes per mode are insufficient to establish a decode speed benefit or regression.': 'The QSA SG32/transposed reduction improves prefill in all later read slots in this matched32K sample. Decode is slower in the pooled sample; two processes per mode do not establish a decode benefit or regression. Physical256K qualification is still required before adoption.',
    "'Resolve or safely avoid pending native copy counter stalls'": "'Continue source/lifetime checks and retain owned failure capture; no claim of resolving the earlier native-counter cause'",
}
for a, b in replacements.items():
    assert a in s, a
    s = s.replace(a, b)
needle = "jobs = []\n"
insert = '''qsa_numerical_path = base/'owned-qsa-reduce12-sg32-v01402-code32k-diagnostic-r1/record.json'
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
'''
assert s.count(needle) == 1
s = s.replace(needle, insert)
needle = "        assert r['ids'] == expected['ids'] and r['logprobs'] == expected['logprobs'] and r['mtp_counts'] == expected['mtp_counts']\n"
assert s.count(needle) == 1
s = s.replace(needle, needle + "        qr = qsa_references[reference_name]\n        assert r['ids'] == qr['ids'] and r['logprobs'] == qr['logprobs'] and r['mtp_counts'] == qr['mtp_counts']\n")
s += '''summary.update(qsa_initial_numerical_receipt_sha256=sha(qsa_numerical_path),
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
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\\n')
table = []
for label, name in [('First process read', 'first_process_read'), ('Subsequent full reads', 'subsequent_full_reads')]:
    c, p = groups['dd5-control'][name], groups['qsa-reduce12'][name]
    table.append(f"| {label} | {c['requests']} per mode | {c['prefill_tok_s']:.3f} | {p['prefill_tok_s']:.3f} | {effects[name]['prefill_tok_s_change_percent']:+.3f}% | {c['decode_tok_s']:.3f} | {p['decode_tok_s']:.3f} | {effects[name]['decode_tok_s_change_percent']:+.3f}% |")
(out/'README.md').write_text(\"\"\"# QSA twelve-head reduction: build, numerical gate and matched 32K timing

Measured on Ryzen 5 5600X, 128 GiB RAM and Arc B570 with 10 GiB VRAM; kernel 7.0.0-38, NEO 26.31.39395.14, IGC 2.41.5 and oneAPI 2026.1.1. Both binaries use the integrated v0.1.40.2 engine with context 262144, chunk 8192, int8 KV, 32768 resident cells, cache 128, five CPU workers and MTP 4. Main and production binaries were unchanged.

The candidate replaces only `qsa_decode_attn.dp.cpp.o` in the qualified DD5 kernel archive. The SG32/transposed arm computes the same twelve-head dot expressions and XOR reduction tree using sixteen subgroup exchanges per lane, compared with sixty in the previous source. Actual subgroup lane/group identifiers route the head and cell work. The earlier [CPU arithmetic/routing review and complete DD5 lifecycle](../uniform-full256k-and-qsa-source-v1/README.md) remain scoped to their own archive boundary. The actual one-object compile, archive replacement and link took 29.461 seconds and passed header/dependency, archive-member and other link-input checks. Exact build commands and warnings are retained under `build`.

The candidate's initial logged GPU gate completed four fresh 32768-token inputs A/B/A/B. Head, used state, all output IDs, logprobs and visible MTP counts matched the qualified references. Actual disk SAVE/RESTORE continuation matched the live continuation. All six request gates and both session operations passed, followed by normal QUIT, no forced cleanup or surviving owned process, and no new kernel fault. Flushed UR/ZE/ZEL/Strata logs were enabled, so all of these diagnostic times are excluded from speed comparisons. The 13437101076-byte raw stderr and large state/session files remain private, with complete terminal hashes and comparisons in the receipt. The earlier numerical launcher v1 was rejected on the CPU before creating a run directory or GPU process because its inherited repetition guard still required 7; the corrected v2 and separate negative receipt are preserved.

For timing, run order was DD5, QSA, QSA, DD5. Each fresh process read A/B/A/B, with 32768 input and 64 generated tokens each time. All sixteen requests reported RESUME 0 and REUSED 0, and output IDs, logprobs and visible MTP counts matched both completed numerical gates. Every process exited normally. API/validation logs, profiler/preload, phase trace, state/head/payload dumps and extra waits were disabled. No other model, GPU health test, build or heavy local analysis ran concurrently. The executed controller checked the owned PID/start ticks, executable path and SHA-256 through `/proc/PID/exe` after READY; argument and environment checks are retained in the source and receipts.

First process reads are separate from later full reads. They include process JIT/capture warmup and are not a cold disk/system benchmark. Rates use total tokens divided by total engine time. Per-read-slot values and raw ranges are in [summary.json](summary.json).

| Reads | Count | DD5 PP tok/s | QSA PP tok/s | PP change | DD5 TG tok/s | QSA TG tok/s | TG change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
\"\"\" + '\\n'.join(table) + \"\"\"

In this sample all later QSA prefill reads were faster than all later matched controls. The pooled later full-read gain is about 1.35%, or 1.018 seconds per 32K prefill. Decode was slower in the pooled sample; two processes per mode do not establish a decode benefit or regression. The changed arm is the transposed prefill path; the nontransposed decode arithmetic remains unchanged. This result does not measure PCIe dominance, resolve earlier native-event failures or reach the performance target.

The QSA binary is not adopted. It still requires its own complete physical 256K sequence: all 262144 cells through cell 262143, repeated fresh full prefill, all saved state/KV tensor bytes, actual disk restoration, clipped speculative tail, both capacity refusals and a later valid fresh 32768-token input. It cannot inherit DD5's completed capacity proof. No reset, rebind, reboot, service, package or global setting changed.

The separate HC/dense knob review is source-only: existing `gr_upmix` and calibrated `bf16_hcd_exact` fast paths are HIP implementations with no SYCL equivalent enabled by their knobs. No new flags or performance conclusions result from that review. The archive manifest identifies copied files; paths inside raw records retain the original machine locations. The large diagnostic stderr is represented by the complete terminal hash rather than copied or rehashed here.
\"\"\")
print(json.dumps({'archive':str(out), 'groups':groups, 'effects':effects},indent=2))
'''
ast.parse(s)
target = base / 'archive_qsa_reduce12_32k_v01402_v1.py'
assert not target.exists()
target.write_text(s)
print(target, hashlib.sha256(target.read_bytes()).hexdigest())
