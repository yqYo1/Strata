"""Prepare the QSA candidate's own complete capacity gate; launch nothing."""
from pathlib import Path
import ast, datetime, hashlib, json

base = Path(__file__).parent
short_path = base/'run_owned_qsa_reduce12_code32k_v01402_v2.py'
full_path = base/'run_owned_uniform_full256k_v01402_v1.py'
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert digest(short_path) == 'f16ec6dfc63c749e9812ece21a4183a475a955eea43d338be09926ca854fe825'
assert digest(full_path) == '01838505aba1ce78f4c43159f2f350dfdc25fe248cf49ab7446612c44ae09b83'
s = short_path.read_text()
s = s.replace(s.splitlines()[0], '"""Logged QSA physical256K gate with owned read-only native-counter failure capture."""', 1)
marker = "out = base/f'owned-qsa-reduce12-sg32-v01402-code32k-{phase}-r{repetition}';out.mkdir(mode=0o700)"
guards = '''numerical_path=base/'owned-qsa-reduce12-sg32-v01402-code32k-diagnostic-r1/record.json'
assert hashlib.sha256(numerical_path.read_bytes()).hexdigest()=='02c8e8abfccca07aaf1767c8d0ed737c8f0087de08208a6e14e377aff7195d33'
numerical=json.loads(numerical_path.read_text())
assert not numerical['active'] and numerical['completed'] and numerical['healthy'] and numerical['math_gate_passed']
assert numerical['code32k_sequence_completed'] and numerical['exit_code']==0 and not numerical['exit_signal']
assert not numerical.get('error') and not numerical['new_fault_messages'] and not any(numerical['cleanup'].values())
assert numerical['actual_executable_identity']['sha256']==qsa_build['candidate_binary_sha256']
assert numerical['qsa_build_receipt_sha256']==hashlib.sha256(qsa_path.read_bytes()).hexdigest()
quiet_path=base/'qsa-reduce12-matched-quiet32k-v01402-sequence-v1/record.json'
quiet_terminal=json.loads(quiet_path.read_text())
assert not quiet_terminal['active'] and quiet_terminal['passed'] and len(quiet_terminal['steps'])==4
assert [(x['mode'],x['repetition']) for x in quiet_terminal['steps']]==[('dd5-control',1),('qsa-reduce12',1),('qsa-reduce12',2),('dd5-control',2)]
assert quiet_terminal['minimum_comparison_input_tokens']==32768
for step in quiet_terminal['steps']:
    path=Path(step['receipt']);assert hashlib.sha256(path.read_bytes()).hexdigest()==step['receipt_sha256']
    d=json.loads(path.read_text())
    assert not d['active'] and d['completed'] and d['healthy'] and d['math_gate_passed'] and d['quiet_four_fresh_reads_completed']
    assert d['exit_code']==0 and not d['exit_signal'] and not d.get('error') and not d['new_fault_messages'] and not any(d['cleanup'].values())
    assert d['qsa_full32k_numerical_gate_sha256']==hashlib.sha256(numerical_path.read_bytes()).hexdigest()
    assert d['argv'][1:]==numerical['argv'][1:]
    assert len(d['requests'])==4 and all(x['input_tokens']==32768 and x['resume_tokens']==[0,0] and x['math_gate_passed'] and all(x['comparison'].values()) for x in d['requests'])
    for role in ['inferior','debugger']:
        owned=d[role];now=process_identity(owned['pid'])
        assert not now or now['start_ticks']!=owned['start_ticks']
for role in ['inferior','debugger']:
    owned=numerical[role];now=process_identity(owned['pid'])
    assert not now or now['start_ticks']!=owned['start_ticks']
assert numerical['boot_id']==qualified['boot_id']==quiet_terminal['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert shutil.disk_usage(base).free>160*1024**3, 'space for bounded diagnostic log and full sessions'
out = base/f'owned-qsa-reduce12-sg32-v01402-full256k-{phase}-r{repetition}';out.mkdir(mode=0o700)'''
assert s.count(marker) == 1
s = s.replace(marker, guards)
old_scope = "Logged QSA reduce12/actual-subgroup-routing numerical gate on qualified DD5: four fresh32768-token A/B reads, baseline head/used-state/IDs/logprobs/MTP comparison, actual32K SAVE/RESTORE continuation. Diagnostic durations excluded. No full-capacity/speed/stall-fix/adoption claim; source-derived native counter capture only at owned stopped failure."
new_scope = "Logged QSA reduce12/actual-subgroup-routing physical262144 qualification: two fresh full262140+4 reads through cell262143, exact DD5 head/used-state/output/MTP and all saved state/KV bytes, actual full disk restore, clipped speculative tail, both capacity refusals, actual32K disk continuation and later valid fresh32K. Diagnostic times excluded; no adoption or stall-cause claim. Owned native counter capture only on stopped failure."
assert s.count(old_scope) == 1
s = s.replace(old_scope, new_scope)
start = '    # A mismatch leaves the engine waiting for input, so QUIT can close it normally.\n'
end = "    current=None;send(b'QUIT\\n');event('quit')\n"
full = full_path.read_text()
plan = full[full.index(start):full.index(end)]
old_reference = "previous_full=next(x for x in rejected['requests'] if x['name']=='full256k-first')"
assert plan.count(old_reference) == 1
plan = plan.replace(old_reference,"previous_full=next(x for x in qualified['requests'] if x['name']=='full256k-first')")
marker = "        full_saved['full_capacity_gate']=full_kv_gate(full_saved['image']);save()\n"
extra = '''        full_saved['full_capacity_gate']=full_kv_gate(full_saved['image'])
        dd5_full_saved=next(x for x in qualified['sessions'] if x['name']=='save-full256k-first')
        full_saved['all_saved_state_and_kv_bytes_equal_dd5']=full_saved['image']['semantic_sha256']==dd5_full_saved['image']['semantic_sha256']
        math_ok=math_ok and full_saved['all_saved_state_and_kv_bytes_equal_dd5'];save()
        if not math_ok:reject()
'''
assert plan.count(marker) == 1
plan = plan.replace(marker, extra)
s = s[:s.index(start)] + plan + s[s.index(end):]
old = "record['math_gate_passed']=math_ok and bool(record.get('code32k_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])"
assert s.count(old) == 1
s = s.replace(old, "record['math_gate_passed']=math_ok and bool(record.get('capacity_sequence_completed')) and bool(record.get('physical256k_sequence_completed')) and all(req['math_gate_passed'] for req in record['requests'])")
marker = "    record['physical256k_sequence_completed']=False\n"
assert s.count(marker) == 1
s = s.replace(marker, marker+"    record['qsa_initial32k_numerical_receipt_sha256']=hashlib.sha256(numerical_path.read_bytes()).hexdigest()\n    record['qsa_matched_quiet32k_sequence_sha256']=hashlib.sha256(quiet_path.read_bytes()).hexdigest()\n")
old = "    record['previous_goal_turn']='Progress: sixteen matched quiet32768 reads passed full output/logprob/MTP/freshness/normal-exit gates. Polling speed gain rejected; evidence committed/pushed0f103c4c. DD5 full physical gate remains required.'"
assert s.count(old) == 1
s = s.replace(old,"    record['previous_goal_turn']='QSA one-object build, four fresh32K numerical/disk gates and sixteen quiet matched32K reads passed. Later PP +1.345%; pooled TG slower with two processes per mode. QSA full physical lifecycle remains required; DD5 full result cannot be inherited.'")
ast.parse(s)
assert "repetition==1" in s and "repetition==7" not in s
assert s.count("request('full256k-first'") == 1 and s.count("request('full256k-repeat'") == 1
assert s.count('capture_native_counter(g,snap,out') == 1
assert s.index("assert not numerical['active']") < s.index("out = base/f'owned-qsa-reduce12-sg32-v01402-full256k")
assert s.index('    binary=qsa_binary') < s.index("    record['binary_sha256']=")
target = base/'run_owned_qsa_reduce12_full256k_v01402_v1.py'
assert not target.exists()
target.write_text(s)
receipt = {'active':False,'prepared':True,'gpu_launched':False,'engine_built_by_preparer':False,
           'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preparer_sha256':digest(__file__),
           'numerical_controller_parent_sha256':digest(short_path),'full_plan_parent_sha256':digest(full_path),
           'controller':str(target),'controller_sha256':digest(target),'minimum_performance_input_tokens':32768,
           'checks':{'AST':True,'repetition_guard_1':True,'complete_numerical_quiet_terminal_guards_before_directory':True,
                     'owned_pid_start_absence_and_same_boot':True,'one_object_dependencies_archive_link_checks':True,
                     'two_fresh_full_reads_all_physical_cells':True,'DD5_full_math_and_saved_semantic_comparison':True,
                     'actual_full_disk_restore_every_KV_byte':True,'clipped_tail_refusals_later_fresh32k':True,
                     'diagnostic_environment_and_failure_only_owned_counter_capture':True},
           'scope':'Source preparation only. Candidate full execution and result are still pending; no adoption or performance claim.'}
p = base/'prepare-owned-qsa-reduce12-full256k-v01402-v1.json'
assert not p.exists()
p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
