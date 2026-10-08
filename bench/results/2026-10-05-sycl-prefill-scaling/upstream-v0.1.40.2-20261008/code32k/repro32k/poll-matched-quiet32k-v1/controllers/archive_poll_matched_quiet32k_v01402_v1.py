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
out = repro/'poll-matched-quiet32k-v1'
sequence_dir = base/'poll-matched-quiet32k-v01402-sequence-v1'
sequence = json.loads((sequence_dir/'record.json').read_text())
assert sequence['passed'] and not sequence['active'] and len(sequence['steps']) == 4
assert sequence['minimum_comparison_input_tokens'] == 32768
assert sequence['job_controller_sha256'] == 'e681237f10cdb6cfd27e6a6ef848ac26519def4809a74125584054f56ecc3ff6'
assert sequence['controller_sha256'] == '8c0686547eb7945ca5a74cd9eedfb6ceab4a7e5a426a5c46258e2b1288d4b021'
reference_path = base/'owned-profile-definition-v01402-code32k-diagnostic-r6/record.json'
reference = json.loads(reference_path.read_text())
assert reference['healthy'] and reference['math_gate_passed'] and not reference['active']
references = {r['name']: r for r in reference['requests']}
names = ['control32k-first', 'alternate32k-first', 'control32k-repeat', 'alternate32k-repeat']
reference_names = ['control32k-before', 'alternate32k-first', 'control32k-repeat', 'alternate32k-repeat']
expected_plan = [('dd5-control', 1), ('dd5-poll-backoff', 1), ('dd5-poll-backoff', 2), ('dd5-control', 2)]
binary_ids = {'dd5-control': 'dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34',
              'dd5-poll-backoff': '44a274d8c699829020abb8a85a331a638f40efdf5d6c959a076a58e818f75391'}

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

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
    c, p = groups['dd5-control'][name], groups['dd5-poll-backoff'][name]
    effects[name] = {metric+'_change_percent': (p[metric]/c[metric]-1)*100 for metric in ['prefill_tok_s', 'decode_tok_s']}
source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
summary = {'active': False, 'completed': True, 'passed': True, 'adopted': False,
           'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_commit': source_commit,
           'scope': 'Four fresh processes in DD5/poll/poll/DD5 order, four full32768 inputs A/B/A/B and64 generated tokens each. Only quiet complete gated jobs are compared.',
           'minimum_performance_input_tokens': 32768, 'fresh_full_reads': 16,
           'all_ids_logprobs_mtp_equal_and_resume_reused_zero': True,
           'same_arguments_after_binary': True, 'all_normal_quit_no_forced_cleanup_no_new_kernel_fault': True,
           'first_process_read_separated': True, 'first_read_is_not_cold_disk_system_benchmark': True,
           'rates_computed_from_total_tokens_over_total_engine_time': True,
           'groups': groups, 'effects': effects, 'jobs': jobs, 'measurements': rows,
           'sequence_receipt_sha256': sha(sequence_dir/'record.json'), 'reference_dd5_receipt_sha256': sha(reference_path),
           'interpretation': 'No useful prefill improvement in this32K sample; no polling speed adoption. Two processes per mode are insufficient to establish a decode speed benefit or regression.',
           'cpu_utilization_or_pcie_dominance_measured': False, 'underlying_pending_counter_cause_resolved': False,
           'physical256k_sequence_passed': False,
           'pending_before_adoption': ['Physical262144 cells, repeated fresh full prefill, actual disk restore, clipped speculative tail, capacity refusals and later valid fresh32K', 'Resolve or safely avoid pending native copy counter stalls'],
           'no_reset_rebind_reboot_service_package_global_change': True,
           'production_and_main_unchanged': True}
out.mkdir()

def copy(a, b):
    assert not b.exists()
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a, b)

for job in jobs:
    directory = Path(job['receipt']).parent
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            assert path.stat().st_size <= 64*1024**2 and not path.is_symlink()
            copy(path, out/'runs'/directory.name/path.relative_to(directory))
for path in sorted(sequence_dir.iterdir()):
    if path.is_file():
        copy(path, out/'sequence'/path.name)
for name in ['prepare_owned_poll_matched_quiet32k_v01402_v1.py', 'run_owned_poll_matched_quiet32k_v01402_v1.py',
             'run_poll_matched_quiet32k_v01402_sequence_v1.py', Path(__file__).name]:
    copy(base/name, out/'controllers'/name)
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
table = []
for label, name in [('First process read', 'first_process_read'), ('Subsequent full reads', 'subsequent_full_reads')]:
    c, p = groups['dd5-control'][name], groups['dd5-poll-backoff'][name]
    table.append(f"| {label} | {c['requests']} per mode | {c['prefill_tok_s']:.3f} | {p['prefill_tok_s']:.3f} | {effects[name]['prefill_tok_s_change_percent']:+.3f}% | {c['decode_tok_s']:.3f} | {p['decode_tok_s']:.3f} | {effects[name]['decode_tok_s_change_percent']:+.3f}% |")
(out/'README.md').write_text('''# Matched32K quiet comparison of host polling backoff

Ryzen 5 5600X, 128 GiB RAM and Arc B570 with 10 GiB VRAM; kernel 7.0.0-38, NEO 26.31.39395.14 and oneAPI 2026.1.1. Both binaries use the integrated v0.1.40.2 engine with context capacity 262144, chunk8192, int8 KV, 32768 resident tokens, cache128, five CPU workers and MTP4. The candidate only changes the prefill host wait after32 failed event readiness checks to request a10 us sleep. The [prior numerical and source/link gate](../device-profile-rejection-and-poll32k-v1/README.md) describes the exact build.

Run order was DD5 control, polling candidate, polling candidate, DD5 control. Each fresh process read A/B/A/B, every input32768 tokens and each output64 tokens. All sixteen reads reported RESUME0 and REUSED0, and every output ID, logprob and visible MTP count matched the completed DD5 gate. All processes exited through normal QUIT, with no forced cleanup, surviving owned process or new kernel fault. Runtime validation logs, profiler/preload, state/head/payload dumps and extra phase waits were disabled; model/debugger protocol records were retained. No other model, GPU health test, build or heavy local analysis ran concurrently.

Each first process read is reported separately from its later full reads. It includes process JIT/capture warmup and is not a cold disk/system measurement. A later first use of input B belongs to the already-running process and is retained as its own read slot in summary.json. Rates below use total tokens divided by total engine time, not the arithmetic mean of rates.

| Reads | Count | Control PP tok/s | Poll PP tok/s | PP change | Control TG tok/s | Poll TG tok/s | TG change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(table)+'''

This small sample shows no useful prefill gain from the polling change. The raw decode range and per-read-slot results are retained in summary.json; two processes per mode do not establish a decode benefit or regression. CPU utilization and GPU/PCIe overlap were not measured. This is not evidence that transfer waits dominate, that the pending native counter problem is fixed, or that the performance target is reached. The candidate is not adopted.

The complete physical256K sequence still has to pass before adoption: all262144 cells, repeated fresh full prefill, actual disk restoration, clipped speculative tail, capacity refusal, then a valid fresh32K input. Earlier incomplete or instrumented runs remain excluded from these speed figures. Main and production binaries were unchanged; no reset, rebind, reboot, service, package or global setting was changed.

The exact terminal receipts, actual argument/environment identities, raw protocol/MI/engine logs, input token files, supervisor and executed versioned controllers are included. Archived paths retain the original machine paths; the manifest identifies the copied files. Historical controller comments describing the diagnostic ancestor are not evidence of instrumentation: the quiet controller's explicit environment checks and terminal receipts describe this execution.
''')
print(json.dumps({'archive': str(out), 'groups': groups, 'effects': effects}, indent=2))
