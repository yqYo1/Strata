"""Archive a terminal full-capacity baseline and separate source-only candidate."""
from pathlib import Path
import datetime, hashlib, json, shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent = root / 'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k'
run = base / 'owned-profile-definition-v01402-full256k-diagnostic-r7'
r = json.loads((run / 'record.json').read_text())
assert not r['active'] and r['completed'] and r['healthy'] and r['math_gate_passed']
assert r['physical256k_sequence_completed'] and r['capacity_sequence_completed']
assert r['exit_code'] == 0 and not r['exit_signal'] and not r.get('error')
assert not r['new_fault_messages'] and not any(r['cleanup'].values())
assert r['actual_executable_identity']['sha256'] == r['binary_sha256'] == 'dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
for role in ['inferior', 'debugger']:
    row = r[role]
    try:
        stat = (Path('/proc') / str(row['pid']) / 'stat').read_text()
    except FileNotFoundError:
        continue
    assert int(stat[stat.rindex(')') + 2:].split()[19]) != row['start_ticks']
assert len(r['requests']) == 12 and all(x['math_gate_passed'] for x in r['requests'])
requests = {x['name']: x for x in r['requests']}
fresh_names = ['control32k-before', 'control32k-between-full-reads', 'control32k-before-restore', 'control32k-after-refusals']
for name in fresh_names:
    req = requests[name]
    assert req['input_tokens'] == 32768 and max(req['resume_tokens']) == 0
    assert all(req['comparison'].values())
for name in ['full256k-first', 'full256k-repeat']:
    req = requests[name]
    assert req['input_tokens'] == 262140 and req['max_new'] == 4 and max(req['resume_tokens']) == 0
    assert all(req['comparison'].values())
    assert max(start + count - 1 for start, count in req['verify_windows']) == 262143
for name in ['full256k-clipped-reference', 'full256k-clipped-restored']:
    req = requests[name]
    assert req['input_tokens'] == 262142 and req['max_new'] == 2
    assert req['clipped_verify_tail'] == 2 and max(req['resume_tokens']) == 262139
assert all(requests['full256k-clipped-restored']['comparison'].values())
assert all(requests['resume32k-restored']['comparison'].values())
for name in ['no-room', 'one-too-many']:
    req = requests[name]
    assert not req['allowed'] and not req['ids'] and not req.get('verify_windows')
    assert any(line.startswith('ERR ') for line in req['protocol'])
assert len(r['sessions']) == 6 and all(x['passed'] for x in r['sessions'])
sessions = {x['name']: x for x in r['sessions']}
for name in ['save-full256k-first', 'save-full256k-repeat', 'save-full256k-restored']:
    gate = sessions[name]['full_capacity_gate']
    assert gate == {'passed': True, 'layers': 13, 'physical_cells_per_layer': 262144, 'last_physical_cell': 262143, 'ignored_tensor_bytes': 0}
assert sessions['save-full256k-repeat']['all_saved_state_and_kv_bytes_equal']
assert sessions['save-full256k-restored']['all_saved_state_and_kv_bytes_equal']
assert sessions['restore-full256k']['engine_checksum_and_compatibility_validated']
assert sessions['restore-control32k']['engine_checksum_and_compatibility_validated']
assert (run / 'debugger/inferior.stderr').stat().st_size == r['engine_log_bytes']

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def copy(source, target):
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

out = parent / 'uniform-full256k-and-qsa-source-v1'
out.mkdir()
for name in ['record.json', 'protocol.stdout.raw', 'events.jsonl', 'project-messages.txt',
             'debugger/inferior-argv.json', 'debugger/gdb-mi.stdout', 'debugger/engine-and-gdb.stderr',
             'probes/kernel-before.stdout', 'probes/kernel-after.stdout']:
    copy(run / name, out / 'terminal-full256k-r7' / name)
for name in ['prepare_owned_uniform_full256k_v01402_v1.py', 'prepare_owned_uniform_full256k_v01402_v2.py',
             'prepare_owned_uniform_full256k_v01402_v3.py', 'run_owned_uniform_full256k_v01402_v1.py',
             'capture_owned_native_counter_v01402_v1.py', 'test_owned_native_counter_v01402_v1.py',
             'capture_b70_reference_sources_v01402_v1.py',
             'prepare_qsa_reduce12_sg32_v01402_v1.py', 'prepare_qsa_reduce12_sg32_v01402_v2.py', 'prepare_qsa_reduce12_sg32_v01402_v3.py',
             'test_qsa_reduce12_sg32_float_v01402_v1.py', 'test_qsa_reduce12_sg32_float_v01402_v2.py',
             'build_qsa_reduce12_sg32_v01402_v1.py', 'prepare_owned_qsa_reduce12_code32k_v01402_v1.py',
             'run_owned_qsa_reduce12_code32k_v01402_v1.py', Path(__file__).name]:
    copy(base / name, out / 'controllers' / name)
for name in ['prepare-owned-uniform-full256k-v01402-v3.json', 'prepare-owned-uniform-full256k-negative-v1.json',
             'prepare-owned-uniform-full256k-negative-v2.json', 'prepare-owned-qsa-reduce12-code32k-v01402-v1.json',
             'b70-attention-hc-applicability-v01402-review-v1.json']:
    copy(base / name, out / 'checks' / name)
source_dirs = ['native-counter-fields-v01402-source-v1', 'b70-attention-hc-reference-v01402-source-v1',
               'qsa-reduce12-sg32-v01402-source-v1', 'qsa-reduce12-sg32-v01402-source-v2', 'qsa-reduce12-sg32-v01402-source-v3']
for name in source_dirs:
    for source in sorted((base / name).rglob('*')):
        if source.is_file():
            copy(source, out / 'source-review' / name / source.relative_to(base / name))
for name in ['owned-native-counter-v01402-cpu-v1', 'qsa-reduce12-sg32-float-cpu-v01402-v1', 'qsa-reduce12-sg32-float-cpu-v01402-v2']:
    for source in sorted((base / name).rglob('*')):
        if source.is_file() and source.name not in ['fixture', 'fixture.o']:
            with source.open('rb') as stream:
                if stream.read(4) == b'\x7fELF':
                    continue
            assert source.stat().st_size < 32*1024*1024
            copy(source, out / 'cpu-checks' / name / source.relative_to(base / name))

candidate = json.loads((base / 'qsa-reduce12-sg32-v01402-source-v3/record.json').read_text())
assert candidate['prepared'] and not candidate['compiled'] and not candidate['gpu_tested'] and not candidate['adopted']
assert candidate['cpu_bitwise_equal_outputs'] == 82176 and candidate['cpu_semantic_routing_cases'] == 1024
summary = {
    'active': False, 'passed': True, 'adopted': False,
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Complete DD5 physical256K numerical/lifecycle qualification, and separate source-only QSA candidate. Diagnostic timing excluded, no new speed/stall-cause claim.',
    'raw_receipt_sha256': sha(run / 'record.json'), 'actual_binary_sha256': r['binary_sha256'],
    'elapsed_diagnostic_seconds': r['elapsed_seconds'],
    'fresh32k_reads': 4, 'fresh_full_reads': 2, 'full_input_tokens': 262140,
    'physical_cells_per_layer': 262144, 'last_physical_cell': 262143, 'kv_layers': 13,
    'all_head_used_state_output_logprob_mtp_gates_passed': True,
    'all_saved_state_and_kv_tensor_bytes_equal_on_repeat_and_disk_roundtrip': True,
    'ignored_saved_kv_tensor_bytes': 0,
    'actual32k_disk_restore_continuation_passed': True,
    'actual_full_disk_restore_and_clipped_two_token_verification_passed': True,
    'both_capacity_refusals_before_gpu_work_and_later_fresh32k_passed': True,
    'normal_exit_code': 0, 'owned_survivors': False, 'new_kernel_fault_messages': [],
    'raw_stderr_path': str(run / 'debugger/inferior.stderr'),
    'raw_stderr_bytes': r['engine_log_bytes'], 'raw_stderr_sha256': r['engine_log_sha256'],
    'raw_stderr_digest_computed_by_terminal_controller_not_rehashed_by_archiver': True,
    'minimum_performance_input_tokens': 32768, 'diagnostic_durations_excluded': True,
    'underlying_pending_native_event_cause_resolved': False,
    'native_counter_capture_cpu_cases': 3, 'actual_neo_counter_field_capture_tested': False,
    'qsa_candidate_source_sha256': candidate['candidate_source_sha256'],
    'qsa_actual_helper_cpu_bitwise_equal_outputs': 82176,
    'qsa_abstract_subgroup_routing_cpu_cases': 1024,
    'qsa_engine_compiled': False, 'qsa_gpu_tested': False, 'qsa_speed_measured': False,
    'qsa_full_capacity_pass_inherited_from_dd5': False,
    'pending': ['Build the one-object QSA candidate, then initial logged>=32K numerical/session gate',
                'Clean>=32K comparison with first and later full reads separate',
                'Complete physical256K lifecycle qualification of the new candidate before adoption'],
    'no_reset_rebind_reboot_service_package_global_change': True,
    'production_and_main_unchanged': True}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({'archive': str(out), 'DD5_full256k_passed': True, 'qsa_compiled_or_gpu_tested': False,
                  'raw_receipt_sha256': summary['raw_receipt_sha256']}))
