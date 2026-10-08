"""Prepare the logged native GEMM numerical gate without building or launching."""
from pathlib import Path
import ast, datetime, hashlib, json

base = Path(__file__).parent
parent = base/'run_owned_qsa_reduce12_code32k_v01402_v2.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(parent) == 'f16ec6dfc63c749e9812ece21a4183a475a955eea43d338be09926ca854fe825'
s = parent.read_text()
s = s.replace(s.splitlines()[0], '"""Logged native GEMM host-scalar32K numerical/disk gate with owned failure capture."""', 1)
s = s.replace("qsa_path=base/'qsa-reduce12-sg32-v01402-build-v1/record.json'", "qsa_path=base/'gemm-host-scalars-v01402-build-v2/record.json'")
s = s.replace("qsa_build['only_qsa_decode_archive_member_replaced']", "qsa_build['only_gemm_archive_member_replaced']")
old_out = "out = base/f'owned-qsa-reduce12-sg32-v01402-code32k-{phase}-r{repetition}';out.mkdir(mode=0o700)"
new_out = '''preceding_path=base/'owned-qsa-reduce12-sg32-v01402-full256k-diagnostic-r1/record.json'
preceding=json.loads(preceding_path.read_text())
assert not preceding['active'] and preceding['completed'] and preceding['healthy'] and preceding['math_gate_passed']
assert preceding['physical256k_sequence_completed'] and preceding['capacity_sequence_completed']
assert preceding['exit_code']==0 and not preceding['exit_signal'] and not preceding.get('error')
assert not preceding['new_fault_messages'] and not any(preceding['cleanup'].values())
assert hashlib.sha256(preceding_path.read_bytes()).hexdigest()==qsa_build['preceding_qsa_full256k_receipt_sha256']
for role in ['inferior','debugger']:
    owned=preceding[role];now=process_identity(owned['pid'])
    assert not now or now['start_ticks']!=owned['start_ticks']
assert preceding['boot_id']==qualified['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
out = base/f'owned-gemm-host-scalars-v01402-code32k-{phase}-r{repetition}';out.mkdir(mode=0o700)'''
assert s.count(old_out) == 1
s = s.replace(old_out,new_out)
begin = s.index("    source_review_path=base/'qsa-reduce12-sg32-v01402-source-v3/record.json'")
end = s.index("    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()",begin)
selection = '''    source_review_path=base/'gemm-host-scalars-v01402-source-v1/record.json'
    assert hashlib.sha256(source_review_path.read_bytes()).hexdigest()==qsa_build['source_review_sha256']
    native_review=json.loads(source_review_path.read_text())
    assert native_review['prepared'] and native_review['changed_default_call_sites']==2
    assert all(native_review['source_contract_review'].values())
    native_source=Path(qsa_build['candidate_source'])
    assert hashlib.sha256(native_source.read_bytes()).hexdigest()==qsa_build['candidate_source_sha256']==native_review['candidate_source_sha256']=='0225c809c1608a59719d84809075b3349bddcf09e5a9bb7aada4bc611aada955'
    for path,expected_sha in native_review['source_inputs_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_sha
    assert qsa_build['baseline_binary_sha256']==uniform['candidate_binary_sha256']
    assert qsa_build['baseline_link_input_sha256']==candidate_inputs==qualified['uniform_candidate_link_input_sha256']
    compile_args=next(step['argv'] for step in qsa_build['steps'] if step['label']=='compile')
    native_dep=Path(compile_args[compile_args.index('-MF')+1])
    native_object=Path(compile_args[compile_args.index('-o')+1])
    assert hashlib.sha256(native_dep.read_bytes()).hexdigest()==qsa_build['actual_dependency_file_sha256']
    assert hashlib.sha256(native_object.read_bytes()).hexdigest()==qsa_build['candidate_object_sha256']
    inventory=qsa_path.parent/'actual-dependency-files.json'
    assert hashlib.sha256(inventory.read_bytes()).hexdigest()==qsa_build['actual_dependency_inventory_sha256']
    dep_rows=json.loads(inventory.read_text())
    assert {item['path'] for item in dep_rows if item['path'].endswith('/dpct/device.hpp')}=={str(Path(qsa_build['shadow_header']).resolve())}
    for item in dep_rows:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    native_binary=Path(qsa_build['candidate_binary'])
    native_archive=native_binary.parent/'libstrata_prefill.a'
    assert hashlib.sha256(native_archive.read_bytes()).hexdigest()==qsa_build['candidate_archive_sha256']
    members=subprocess.check_output(['/usr/bin/ar','t',native_archive],text=True).splitlines()
    assert members==[item['member'] for item in qsa_build['archive_members']]
    assert [item['member'] for item in qsa_build['archive_members'] if item['replaced']]==['gemm.dp.cpp.o']
    for item in qsa_build['archive_members']:
        assert hashlib.sha256(subprocess.check_output(['/usr/bin/ar','p',native_archive,item['member']])).hexdigest()==item['after_sha256']
        if not item['replaced']:assert item['before_sha256']==item['after_sha256']
    binary=native_binary
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==qsa_build['candidate_binary_sha256']
    record['gemm_build_receipt_sha256']=hashlib.sha256(qsa_path.read_bytes()).hexdigest()
    record['gemm_source_review_sha256']=hashlib.sha256(source_review_path.read_bytes()).hexdigest()
    record['qualified_dd5_full256k_receipt_sha256']=hashlib.sha256(qualified_path.read_bytes()).hexdigest()
    record['preceding_qsa_full256k_receipt_sha256']=hashlib.sha256(preceding_path.read_bytes()).hexdigest()
    record['gemm_actual_dependency_inventory_sha256']=qsa_build['actual_dependency_inventory_sha256']
    record['gemm_candidate_source_sha256']=qsa_build['candidate_source_sha256']
    record['gemm_only_prefill_archive_member_replaced']=True
    record['physical256k_sequence_completed']=False
'''
s = s[:begin]+selection+s[end:]
s = s.replace("record['candidate_build_receipt_sha256']=record['qsa_build_receipt_sha256']", "record['candidate_build_receipt_sha256']=record['gemm_build_receipt_sha256']")
old_change = 'Only QSA decode-attention archive member replaced on fully-qualified DD5. New SG32/transposed12-head reduction uses explicit subgroup lane/group IDs; same dot expressions, chunk64, scales, softmax/PV/merge, queues/events and host lifetimes.'
new_change = 'Only gemm.dp.cpp.o in the prefill archive replaced on fully-qualified DD5. Known host FP32 scalars directly select the same typed oneMKL BF16/F16 to FP32 operation, descriptor queue, math mode, dimensions/strides, event dependencies and error wrapper. No extra wait or storage/queue lifetime change.'
assert s.count(old_change) == 1
s = s.replace(old_change,new_change)
old_scope = 'Logged QSA reduce12/actual-subgroup-routing numerical gate on qualified DD5: four fresh32768-token A/B reads, baseline head/used-state/IDs/logprobs/MTP comparison, actual32K SAVE/RESTORE continuation. Diagnostic durations excluded. No full-capacity/speed/stall-fix/adoption claim; source-derived native counter capture only at owned stopped failure.'
new_scope = 'Logged native GEMM known-host-scalar numerical gate on qualified DD5: four fresh32768-token A/B reads, qualified head/used-state/IDs/logprobs/MTP comparisons, actual32K disk continuation. Diagnostic times excluded. No speed, own physical256K, adoption or stall-cause claim; owned stopped native-counter capture only on failure.'
assert s.count(old_scope) == 1
s = s.replace(old_scope,new_scope)
old_history = "    record['previous_goal_turn']='Progress: sixteen matched quiet32768 reads passed full output/logprob/MTP/freshness/normal-exit gates. Polling speed gain rejected; evidence committed/pushed0f103c4c. DD5 full physical gate remains required.'"
assert s.count(old_history) == 1
s = s.replace(old_history,"    record['previous_goal_turn']='GEMM source/one-object builder/native API contract prepared during the owned QSA full sequence. Its builder and this gate require that preceding job to be terminal. This candidate is compared against DD5 separately and does not inherit a different binary capacity proof.'")
# Retain historical qsa_gate/qsa-batch128 paths; rename only this build's local aliases.
import re
s = re.sub(r'\bqsa_path\b','gemm_path',s)
s = re.sub(r'\bqsa_build\b','gemm_build',s)
ast.parse(s)
assert 'repetition==1' in s and 'repetition==7' not in s
assert s.index("assert not preceding['active']") < s.index("out = base/f'owned-gemm-host-scalars")
assert s.index('    binary=native_binary') < s.index("    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()")
assert s.count("request('control32k-before'") == 1 and s.count("request('alternate32k-repeat'") == 1
assert s.count("session('RESTORE','restore-control32k'") == 1
assert "dd5_alternate=next(item for item in uniform_numerical['requests']" in s
assert 'qsa_only_decode_archive_member_replaced' not in s and 'cpu_semantic_routing_cases' not in s
assert s.count('capture_native_counter(g,snap,out') == 1
target = base/'run_owned_gemm_host_scalars_code32k_v01402_v1.py'
assert not target.exists()
target.write_text(s)
receipt = {'active':False,'prepared':True,'engine_built_by_preparer':False,'gpu_launched':False,
           'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preparer_sha256':sha(__file__),
           'parent_numerical_controller_sha256':sha(parent),'controller':str(target),'controller_sha256':sha(target),
           'source_review_sha256':sha(base/'gemm-host-scalars-v01402-source-v1/record.json'),
           'checks':{'AST':True,'preceding_full_terminal_owned_absence_before_output_directory':True,
                     'actual_gemm_source_dependencies_member_link_identity':True,'binary_identity_after_selection_and_proc':True,
                     'four_fresh32768_A_B_qualified_numerical_comparisons':True,'actual32k_disk_continuation':True,
                     'diagnostic_environment_and_failure_only_owned_counter_capture':True},
           'minimum_performance_input_tokens':32768,
           'scope':'Prepared only; no GEMM engine compile/GPU numerical result/performance/own256K qualification or adoption.'}
p = base/'prepare-owned-gemm-host-scalars-code32k-v01402-v1.json'
assert not p.exists()
p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'prepared':True,'controller_sha256':sha(target),'gpu_launched':False}))
