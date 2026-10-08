"""Prepare quiet QSA/DD5 comparisons; no build or GPU launch occurs here."""
from pathlib import Path
import ast, datetime, hashlib, json

base = Path(__file__).parent
quiet_parent = base / 'run_owned_poll_matched_quiet32k_v01402_v1.py'
numerical_parent = base / 'run_owned_qsa_reduce12_code32k_v01402_v2.py'
sequence_parent = base / 'run_poll_matched_quiet32k_v01402_sequence_v1.py'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(quiet_parent) == 'e681237f10cdb6cfd27e6a6ef848ac26519def4809a74125584054f56ecc3ff6'
assert sha(numerical_parent) == 'f16ec6dfc63c749e9812ece21a4183a475a955eea43d338be09926ca854fe825'
assert sha(sequence_parent) == '8c0686547eb7945ca5a74cd9eedfb6ceab4a7e5a426a5c46258e2b1288d4b021'
s = quiet_parent.read_text()
def replace(old, new):
    global s
    assert s.count(old) == 1, (old, s.count(old))
    s = s.replace(old, new)
replace("assert mode in ['dd5-control','dd5-poll-backoff'] and phase=='quiet' and repetition in [1,2]",
        "assert mode in ['dd5-control','qsa-reduce12'] and phase=='quiet' and repetition in [1,2]")
old = "out = base/f'owned-poll-matched-{mode}-v01402-code32k-{phase}-r{repetition}';out.mkdir(mode=0o700)"
guard = '''qualified_path=base/'owned-profile-definition-v01402-full256k-diagnostic-r7/record.json'
qualified=json.loads(qualified_path.read_text())
assert not qualified['active'] and qualified['healthy'] and qualified['completed'] and qualified['math_gate_passed']
assert qualified['physical256k_sequence_completed'] and qualified['capacity_sequence_completed']
numerical_path=base/'owned-qsa-reduce12-sg32-v01402-code32k-diagnostic-r1/record.json'
numerical=json.loads(numerical_path.read_text())
assert not numerical['active'], 'initial QSA numerical job is still active'
assert numerical['healthy'] and numerical['completed'] and numerical['math_gate_passed']
assert numerical['code32k_sequence_completed'] and not numerical['full_capacity_sequence_completed']
assert numerical['exit_code']==0 and not numerical['exit_signal'] and not numerical['new_fault_messages'] and not any(numerical['cleanup'].values())
assert len(numerical['requests'])==6 and all(req['math_gate_passed'] for req in numerical['requests'])
for prior_run in [qualified,numerical]:
    assert prior_run['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for role in ['inferior','debugger']:
        old=prior_run[role];now=process_identity(old['pid'])
        assert not now or now['start_ticks']!=old['start_ticks']
qsa_path=base/'qsa-reduce12-sg32-v01402-build-v1/record.json'
qsa_build=json.loads(qsa_path.read_text())
assert qsa_build['passed'] and not qsa_build['active'] and qsa_build['baseline_inputs_unchanged']
assert qsa_build['only_qsa_decode_archive_member_replaced']
assert qsa_build['baseline_full256k_receipt_sha256']==hashlib.sha256(qualified_path.read_bytes()).hexdigest()
assert numerical['qsa_build_receipt_sha256']==hashlib.sha256(qsa_path.read_bytes()).hexdigest()
assert numerical['actual_executable_identity']['sha256']==numerical['binary_sha256']==qsa_build['candidate_binary_sha256']=='26c29c1353c64cef0e81c18bc1f651c9bcc25f977429d71b18052c5d13ecd34e'
out = base/f'owned-qsa-reduce12-matched-{mode}-v01402-code32k-{phase}-r{repetition}';out.mkdir(mode=0o700)'''
replace(old, guard)
start = s.index("    poll_path=base/'prefill-poll-backoff-v01402-relink-v3/record.json'\n")
end = s.index("    record['adopted']=False\n", start)
num = numerical_parent.read_text()
qsa_start = num.index("    source_review_path=base/'qsa-reduce12-sg32-v01402-source-v3/record.json'\n")
qsa_end = num.index("    record['physical256k_sequence_completed']=False\n", qsa_start)
selection = num[qsa_start:qsa_end]
selection += '''    assert numerical['argv'][0]==str(qsa_binary)
    if mode=='dd5-control':binary=Path(uniform['candidate_binary'])
    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert record['binary_sha256']==(uniform['candidate_binary_sha256'] if mode=='dd5-control' else qsa_build['candidate_binary_sha256'])
    record['qsa_full32k_numerical_gate_sha256']=hashlib.sha256(numerical_path.read_bytes()).hexdigest()
    record['unprofiled_same_base_numerical_gate_sha256']=hashlib.sha256(completed_path.read_bytes()).hexdigest()
    record['profiler_failure_receipt_sha256']=hashlib.sha256(failed_path.read_bytes()).hexdigest()
    record['single_variable']='Only qsa_decode_attn.dp.cpp.o differs between DD5 and QSA reduce12/actual-subgroup routing; other actual link inputs, model arguments, environment and request order are identical.'
    record['candidate_build_receipt_sha256']=record['uniform_header_build_receipt_sha256'] if mode=='dd5-control' else record['qsa_build_receipt_sha256']
    from capture_owned_native_counter_v01402_v1 import capture as capture_native_counter
    counter_cpu_path=base/'owned-native-counter-v01402-cpu-v1/record.json'
    counter_cpu=json.loads(counter_cpu_path.read_text());assert counter_cpu['passed'] and not counter_cpu['active']
    counter_controller=base/'capture_owned_native_counter_v01402_v1.py'
    assert hashlib.sha256(counter_controller.read_bytes()).hexdigest()==counter_cpu['capture_controller_sha256']
    record['native_counter_capture_controller_sha256']=counter_cpu['capture_controller_sha256']
    for p in sorted(base.glob('owned-qsa-reduce12-matched-*-v01402-code32k-quiet-r*/record.json')):
        if p.parent==out:continue
        old=json.loads(p.read_text());assert not old['active'] and old['healthy'] and old['math_gate_passed']
        assert old['exit_code']==0 and not old['new_fault_messages'] and not any(old['cleanup'].values())
        for role in ['inferior','debugger']:
            prior=old[role];now=process_identity(prior['pid'])
            assert not now or now['start_ticks']!=prior['start_ticks']

'''
s = s[:start] + selection + s[end:]
old_scope = "    record['scope']='Matched quiet DD5 versus polling backoff, four complete32768-token reads A/B/A/B per process. Same262144 capacity/int8/resident32768/chunk8192/cache128/workers5/pcie0/MTP4/GEN64. First process read is reported separately from later reads. No API/validation logs, trace/profiler, state/head/payload dump, additional phase waits or prefix reuse. All64IDs/logprobs/MTP counts must equal completed DD5 numerical gate. No physical256K or adoption proof.'"
new_scope = "    record['scope']='Matched quiet DD5 versus QSA reduce12/actual-subgroup routing, four complete32768-token reads A/B/A/B per process. Same262144 capacity/int8/resident32768/chunk8192/cache128/workers5/pcie0/MTP4/GEN64. First process read separate from later reads. No API/validation logs, trace/profiler, state/head/payload dump, additional phase waits or prefix reuse. All64IDs/logprobs/MTP counts equal qualified DD5 and QSA numerical gates. No candidate physical256K or adoption proof.'"
replace(old_scope, new_scope)
old_history = "    record['previous_goal_turn']='Progress: both DD5 and44a completed four fresh32K state/head/output/logprob/MTP and actual disk-continuation gates. Unitrace GPU timestamps rejected after watchdog abort; post-exit device health and later44a math gate passed. All evidence archived and pushed in523ecdd0; candidate remains private.'"
replace(old_history, "    record['previous_goal_turn']='Progress: DD5 passed complete repeated physical256K/disk restoration/clipped/refusal/later32K qualification; one-object QSA candidate passed four fresh32K head/used-state/output/logprob/MTP and actual disk-continuation gates. Measure quiet first/later full reads; no inherited candidate capacity or stall-cause proof.'")
old_error = "        try:record['snapshots'].append(g.snapshot('failure',resume=False))\n        except BaseException as inspect:record['snapshot_error']=repr(inspect)"
new_error = "        try:\n            snap=g.snapshot('failure',resume=False);record['snapshots'].append(snap)\n            record['native_counter_at_failure']=capture_native_counter(g,snap,out/'native-counter-at-failure-v1.json')\n        except BaseException as inspect:record['snapshot_error']=repr(inspect)"
replace(old_error, new_error)
needle = "        result['math_gate_passed']=all(result['comparison'].values())\n"
replace(needle, '''        qsa_expected=next(item for item in numerical['requests'] if item['name']==reference_name)
        result['comparison'].update(qsa_ids_equal=result['ids']==qsa_expected['ids'],
                                    qsa_logprobs_equal=result['logprobs']==qsa_expected['logprobs'],
                                    qsa_mtp_counts_equal=result['mtp_counts']==qsa_expected['mtp_counts'])
        result['math_gate_passed']=all(result['comparison'].values())
''')
ast.parse(s)
assert 'poll_build' not in s and 'dd5-poll-backoff' not in s
assert s.index("assert not numerical['active']") < s.index("out = base/f'owned-qsa-reduce12-matched")
assert s.index("if mode=='dd5-control':binary=") < s.index("    argv=[str(binary)]+args")
assert s.count("len(tokens)==32768") == 1
target = base / 'run_owned_qsa_reduce12_matched_quiet32k_v01402_v1.py'
assert not target.exists();target.write_text(s)

seq = sequence_parent.read_text()
seq = seq.replace('run_owned_poll_matched_quiet32k_v01402_v1.py', target.name)
seq = seq.replace(sha(quiet_parent), sha(target))
seq = seq.replace('dd5-poll-backoff', 'qsa-reduce12')
seq = seq.replace('owned-poll-matched-', 'owned-qsa-reduce12-matched-')
seq = seq.replace('poll-matched-quiet32k-v01402-sequence-v1', 'qsa-reduce12-matched-quiet32k-v01402-sequence-v1')
seq = seq.replace('Matched DD5/44a ABBA', 'Matched DD5/QSA reduce12 ABBA')
guard_marker = "out=base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1';out.mkdir(mode=0o700)"
seq_guard = '''numerical_path=base/'owned-qsa-reduce12-sg32-v01402-code32k-diagnostic-r1/record.json'
numerical=json.loads(numerical_path.read_text())
assert not numerical['active'], 'initial numerical sequence remains active'
assert numerical['healthy'] and numerical['completed'] and numerical['math_gate_passed']
assert numerical['exit_code']==0 and not numerical['exit_signal'] and not numerical['new_fault_messages'] and not any(numerical['cleanup'].values())
for role in ['inferior','debugger']:
    old=numerical[role];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
out=base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1';out.mkdir(mode=0o700)'''
assert seq.count(guard_marker) == 1;seq = seq.replace(guard_marker, seq_guard)
ast.parse(seq)
sequence = base / 'run_qsa_reduce12_matched_quiet32k_v01402_sequence_v1.py'
assert not sequence.exists();sequence.write_text(seq)
receipt = {'active':False,'prepared':True,'gpu_launched':False,
           'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'preparer_sha256':sha(__file__),'quiet_parent_sha256':sha(quiet_parent),
           'numerical_parent_sha256':sha(numerical_parent),'sequence_parent_sha256':sha(sequence_parent),
           'job_controller':str(target),'job_controller_sha256':sha(target),
           'sequence_controller':str(sequence),'sequence_controller_sha256':sha(sequence),
           'minimum_performance_input_tokens':32768,'process_order':['DD5','QSA','QSA','DD5'],
           'reads_per_process':['A32768','B32768','A32768','B32768'],
           'first_process_read_separated_from_later_full_reads':True,
           'no_gpu_launch_or_engine_build_by_preparer':True,
           'pending':['Terminal initial QSA numerical/disk restoration sequence','Review then execute finite matched quiet comparison'],
           'candidate_full256k_proof_inherited':False,'adopted':False}
p = base / 'prepare-owned-qsa-reduce12-matched-quiet32k-v01402-v1.json'
assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'prepared':True,'gpu_launched':False,'job_controller_sha256':sha(target),'sequence_controller_sha256':sha(sequence)}))
