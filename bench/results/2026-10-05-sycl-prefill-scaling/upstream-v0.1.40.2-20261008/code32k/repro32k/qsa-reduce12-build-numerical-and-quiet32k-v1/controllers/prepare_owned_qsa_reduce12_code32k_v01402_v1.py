"""Prepare a logged numerical gate without building or launching a GPU job."""
from pathlib import Path
import ast, datetime, hashlib, json

base = Path(__file__).parent
full_source = base / 'run_owned_uniform_full256k_v01402_v1.py'
short_source = base / 'run_owned_profile_definition_code32k_v01402_v2.py'
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert digest(full_source) == '01838505aba1ce78f4c43159f2f350dfdc25fe248cf49ab7446612c44ae09b83'
assert digest(short_source) == '597c6022a603c1902f0d75052e183940b20177f1f3f66d5439af7c75961a826b'
s = full_source.read_text()
old = "out = base/f'owned-profile-definition-v01402-full256k-{phase}-r{repetition}';out.mkdir(mode=0o700)"
new = '''assert mode=='full-kv-access' and phase=='diagnostic', 'logged numerical gate only'
qualified_path=base/'owned-profile-definition-v01402-full256k-diagnostic-r7/record.json'
qualified=json.loads(qualified_path.read_text())
assert not qualified['active'], 'full256K job remains active'
assert qualified['healthy'] and qualified['completed'] and qualified['math_gate_passed']
assert qualified['physical256k_sequence_completed'] and qualified['capacity_sequence_completed']
assert qualified['exit_code']==0 and not qualified['exit_signal'] and not qualified['new_fault_messages']
assert not any(qualified['cleanup'].values())
for role in ['inferior','debugger']:
    owned=qualified[role];now=process_identity(owned['pid'])
    assert not now or now['start_ticks']!=owned['start_ticks']
qsa_path=base/'qsa-reduce12-sg32-v01402-build-v1/record.json'
qsa_build=json.loads(qsa_path.read_text())
assert qsa_build['passed'] and not qsa_build['active'] and not qsa_build['gpu_tested'] and not qsa_build['adopted']
assert qsa_build['baseline_full256k_receipt_sha256']==hashlib.sha256(qualified_path.read_bytes()).hexdigest()
assert qsa_build['baseline_inputs_unchanged'] and qsa_build['only_qsa_decode_archive_member_replaced']
out = base/f'owned-qsa-reduce12-sg32-v01402-code32k-{phase}-r{repetition}';out.mkdir(mode=0o700)'''
assert s.count(old) == 1
s = s.replace(old, new)
marker = "    record['physical256k_sequence_completed']=False\n"
selection = '''    source_review_path=base/'qsa-reduce12-sg32-v01402-source-v3/record.json'
    assert hashlib.sha256(source_review_path.read_bytes()).hexdigest()==qsa_build['source_review_sha256']
    qsa_source_review=json.loads(source_review_path.read_text())
    assert qsa_source_review['prepared'] and qsa_source_review['actual_subgroup_lane_and_group_ids_used']
    assert qsa_source_review['cpu_helper_proof_reused_by_exact_body_identity'] and qsa_source_review['cpu_semantic_routing_cases']==1024
    qsa_source=Path(qsa_build['candidate_source'])
    assert hashlib.sha256(qsa_source.read_bytes()).hexdigest()==qsa_build['candidate_source_sha256']==qsa_source_review['candidate_source_sha256']=='c1b0d58f46f0807af10d310320ebe4f9f2b4ad62f1a584abf72a597d7d5f1da1'
    assert qsa_build['baseline_binary_sha256']==uniform['candidate_binary_sha256']
    assert qsa_build['baseline_link_input_sha256']==candidate_inputs==qualified['uniform_candidate_link_input_sha256']
    compile_args=next(step['argv'] for step in qsa_build['steps'] if step['label']=='compile')
    qsa_dep=Path(compile_args[compile_args.index('-MF')+1])
    qsa_object=Path(compile_args[compile_args.index('-o')+1])
    assert hashlib.sha256(qsa_dep.read_bytes()).hexdigest()==qsa_build['actual_dependency_file_sha256']
    assert hashlib.sha256(qsa_object.read_bytes()).hexdigest()==qsa_build['candidate_object_sha256']
    inventory=qsa_path.parent/'actual-dependency-files.json'
    assert hashlib.sha256(inventory.read_bytes()).hexdigest()==qsa_build['actual_dependency_inventory_sha256']
    for item in json.loads(inventory.read_text()):
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
    qsa_binary=Path(qsa_build['candidate_binary'])
    qsa_archive=qsa_binary.parent/'libstrata_kernels.a'
    assert hashlib.sha256(qsa_archive.read_bytes()).hexdigest()==qsa_build['candidate_archive_sha256']
    members=subprocess.check_output(['/usr/bin/ar','t',qsa_archive],text=True).splitlines()
    assert members==[item['member'] for item in qsa_build['archive_members']]
    assert [item['member'] for item in qsa_build['archive_members'] if item['replaced']]==['qsa_decode_attn.dp.cpp.o']
    for item in qsa_build['archive_members']:
        assert hashlib.sha256(subprocess.check_output(['/usr/bin/ar','p',qsa_archive,item['member']])).hexdigest()==item['after_sha256']
        if not item['replaced']:assert item['before_sha256']==item['after_sha256']
    binary=qsa_binary
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==qsa_build['candidate_binary_sha256']
    record['qsa_build_receipt_sha256']=hashlib.sha256(qsa_path.read_bytes()).hexdigest()
    record['qsa_source_review_sha256']=hashlib.sha256(source_review_path.read_bytes()).hexdigest()
    record['qualified_dd5_full256k_receipt_sha256']=hashlib.sha256(qualified_path.read_bytes()).hexdigest()
    record['qsa_actual_dependency_inventory_sha256']=qsa_build['actual_dependency_inventory_sha256']
    record['qsa_candidate_source_sha256']=qsa_build['candidate_source_sha256']
    record['qsa_only_decode_archive_member_replaced']=True
    record['physical256k_sequence_completed']=False
'''
assert s.count(marker) == 1
s = s.replace(marker, selection)
assert s.index('    binary=qsa_binary') < s.index("    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()")
marker = '    args=list(reference[\'args\'])\n'
new = '''    record['candidate_build_receipt_sha256']=record['qsa_build_receipt_sha256']
    record['compatibility_change']='Only QSA decode-attention archive member replaced on fully-qualified DD5. New SG32/transposed12-head reduction uses explicit subgroup lane/group IDs; same dot expressions, chunk64, scales, softmax/PV/merge, queues/events and host lifetimes.'
    args=list(reference['args'])
'''
assert s.count(marker) == 1
s = s.replace(marker, new)
old_scope = "    record['scope']='DD5 uniform DPCT definition repair on the canonical indexer/visible-output/KV baseline. Logged physical262144 occupancy through cell262143, repeated fresh full prefill, full disk SAVE/RESTORE, clipped speculative tail, capacity refusals, actual32K continuation and later valid fresh32K. All performance timings excluded; no polling/profiler/global changes or adoption claim. Owned read-only native-counter fields are captured only after failure stop.'"
new_scope = "    record['scope']='Logged QSA reduce12/actual-subgroup-routing numerical gate on qualified DD5: four fresh32768-token A/B reads, baseline head/used-state/IDs/logprobs/MTP comparison, actual32K SAVE/RESTORE continuation. Diagnostic durations excluded. No full-capacity/speed/stall-fix/adoption claim; source-derived native counter capture only at owned stopped failure.'"
assert s.count(old_scope) == 1
s = s.replace(old_scope, new_scope)
start_marker = '    # A mismatch leaves the engine waiting for input, so QUIT can close it normally.\n'
end_marker = "    current=None;send(b'QUIT\\n');event('quit')\n"
short = short_source.read_text()
short_plan = short[short.index(start_marker):short.index(end_marker)]
old_alt = "        alternate_first=request('alternate32k-first',alternate,64,fresh=True)"
new_alt = "        dd5_alternate=next(item for item in uniform_numerical['requests'] if item['name']=='alternate32k-first')\n        alternate_first=request('alternate32k-first',alternate,64,dd5_alternate,fresh=True)"
assert short_plan.count(old_alt) == 1
short_plan = short_plan.replace(old_alt, new_alt)
s = s[:s.index(start_marker)] + short_plan + s[s.index(end_marker):]
old_gate = "record['math_gate_passed']=math_ok and bool(record.get('capacity_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])"
new_gate = "record['math_gate_passed']=math_ok and bool(record.get('code32k_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])"
assert s.count(old_gate) == 1
s = s.replace(old_gate, new_gate)
ast.parse(s)
assert s.count('capture_native_counter(g,snap,out') == 1
assert s.count("request('control32k-before'") == 1 and s.count("request('alternate32k-repeat'") == 1
assert s.index("assert not qualified['active']") < s.index("out = base/f'owned-qsa-reduce12")
assert "request('full256k-repeat'" not in s
target = base / 'run_owned_qsa_reduce12_code32k_v01402_v1.py'
assert not target.exists()
target.write_text(s)
receipt = {
    'active': False, 'prepared': True, 'gpu_launched': False, 'engine_built': False,
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'preparer_sha256': digest(__file__), 'full_parent_sha256': digest(full_source),
    'numerical_plan_parent_sha256': digest(short_source),
    'controller': str(target), 'controller_sha256': digest(target),
    'minimum_performance_input_tokens': 32768,
    'checks': {'four_fresh32k_reads': True, 'A_and_B_compare_qualified_baselines': True,
               'live_and_actual_disk32k_continuation': True, 'full_job_terminal_guard_before_output_creation': True,
               'single_qsa_object_sources_dependencies_archive_link_guard': True,
               'binary_metadata_assigned_after_candidate_selection': True,
               'actual_exe_pid_start_boot_environment_capture': True,
               'diagnostic_logs_and_owned_stopped_native_counter_capture': True},
    'limitations': ['Prepared only. Candidate build and initial GPU numerical execution remain pending.',
                    'Four fresh32K reads are a numerical gate, not clean speed or physical256K capacity proof.']}
p = base / 'prepare-owned-qsa-reduce12-code32k-v01402-v1.json'
assert not p.exists()
p.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'prepared': True, 'gpu_launched': False, 'controller_sha256': receipt['controller_sha256']}))
