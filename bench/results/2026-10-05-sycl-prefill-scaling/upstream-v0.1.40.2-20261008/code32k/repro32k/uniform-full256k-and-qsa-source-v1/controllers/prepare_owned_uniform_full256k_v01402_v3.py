"""Prepare full DD5 qualification, preserving executed historical controllers."""
from pathlib import Path
import ast, hashlib, json, re

base = Path(__file__).parent
old = base/'run_owned_full_indexer_spare_v01402_v5.py'
old_receipt = base/'owned-full-kv-access-v01402-full256k-diagnostic-r5/record.json'
terminal = json.loads(old_receipt.read_text())
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(old_receipt) == 'e7e541f91ef053acebaa6b9ed2bc3139010e269f0b671d23f423869e2c7a882f'
assert sha(old) == terminal['source_sha256'][str(old)]
assert not terminal['active'] and not terminal['healthy'] and terminal['exit_code'] is None
uniform_ancestor = base/'run_owned_profile_definition_code32k_v01402_v2.py'
ancestor_receipt = json.loads((base/'owned-profile-definition-v01402-code32k-diagnostic-r6/record.json').read_text())
assert ancestor_receipt['healthy'] and ancestor_receipt['math_gate_passed'] and not ancestor_receipt['active']
assert sha(uniform_ancestor) == ancestor_receipt['source_sha256'][str(uniform_ancestor)]
uniform = uniform_ancestor.read_text()
uniform_block = uniform[uniform.index("    uniform_path=base/'dpct-profile-definition-v01402-build-v1/record.json'"):uniform.index("    final=json.loads((base/'owned-upstream-e8ca-refresh-short-final-float-storage/record.json').read_text())")]
s = old.read_text()
def replace(a, b):
    global s
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b)
replace('"""First logged upstream-refresh model check through owned GDB/PTY; no performance/capacity claim."""',
        '"""Logged DD5 physical256K qualification with owned read-only native-counter failure capture."""')
replace("assert mode=='full-kv-access' and phase=='diagnostic' and repetition==5", "assert mode=='full-kv-access' and phase=='diagnostic' and repetition==7")
replace("out = base/f'owned-{mode}-v01402-full256k-{phase}-r{repetition}'", "out = base/f'owned-profile-definition-v01402-full256k-{phase}-r{repetition}'")
health_assignment = re.search(r"^    health_path=.*$", s, flags=re.M)
assert health_assignment
replace(health_assignment[0], "    health_path=base/'post-device-profile-abort-v01402-health-v1/record.json'")
old_health = re.search(r"assert hashlib\.sha256\(health_path\.read_bytes\(\)\)\.hexdigest\(\)=='([0-9a-f]{64})'", s)
assert old_health
replace(old_health[0], "assert hashlib.sha256(health_path.read_bytes()).hexdigest()=='27eda22ac64827a042db7ef90c204a644e8dbc1d56ed2c4132aae57768f91cff'")
cpu = base/'owned-native-counter-v01402-cpu-v1/record.json'
cpu_data = json.loads(cpu.read_text())
assert cpu_data['passed'] and not cpu_data['active'] and len(cpu_data['cases']) == 3
quiet = base/'poll-matched-quiet32k-v01402-sequence-v1/record.json'
checkpoint = json.loads((base/'poll-matched-quiet32k-terminal-continuation-20261008-v1.json').read_text())
assert sha(quiet) == checkpoint['sequence_receipt_sha256']
identity = base/'owned-profile-definition-v01402-code32k-diagnostic-r6/actual-executable-identity-v1.json'
guards = f'''
    uniform_numerical_path=base/'owned-profile-definition-v01402-code32k-diagnostic-r6/record.json'
    assert hashlib.sha256(uniform_numerical_path.read_bytes()).hexdigest()=='a96f3686903b63138a827c173e0ea5b1a8beebbee8ef9cdae46459bf9f2c9699'
    uniform_numerical=json.loads(uniform_numerical_path.read_text())
    assert not uniform_numerical['active'] and uniform_numerical['completed'] and uniform_numerical['healthy'] and uniform_numerical['math_gate_passed']
    assert Path(uniform_numerical['argv'][0]).resolve()==binary.resolve()
    identity_path=base/'owned-profile-definition-v01402-code32k-diagnostic-r6/actual-executable-identity-v1.json'
    assert hashlib.sha256(identity_path.read_bytes()).hexdigest()=='{sha(identity)}'
    quiet_path=base/'poll-matched-quiet32k-v01402-sequence-v1/record.json'
    assert hashlib.sha256(quiet_path.read_bytes()).hexdigest()=='{sha(quiet)}'
    quiet=json.loads(quiet_path.read_text());assert quiet['passed'] and not quiet['active'] and len(quiet['steps'])==4
    for step in quiet['steps']:
        p=Path(step['receipt']);assert hashlib.sha256(p.read_bytes()).hexdigest()==step['receipt_sha256']
        d=json.loads(p.read_text());assert not d['active'] and d['healthy'] and d['completed'] and d['math_gate_passed']
        assert d['exit_code']==0 and not d['exit_signal'] and not d['new_fault_messages'] and not any(d['cleanup'].values())
        for key in ['inferior','debugger']:
            old_id=d[key];now=process_identity(old_id['pid']);assert not now or now['start_ticks']!=old_id['start_ticks']
    capture_cpu_path=base/'owned-native-counter-v01402-cpu-v1/record.json'
    assert hashlib.sha256(capture_cpu_path.read_bytes()).hexdigest()=='{sha(cpu)}'
    capture_cpu=json.loads(capture_cpu_path.read_text());assert capture_cpu['passed'] and not capture_cpu['active']
    capture_path=base/'capture_owned_native_counter_v01402_v1.py'
    assert hashlib.sha256(capture_path.read_bytes()).hexdigest()==capture_cpu['capture_controller_sha256']
    assert hashlib.sha256((observer/'sycl/tools/owned_gdb.py').read_bytes()).hexdigest()==capture_cpu['owned_helper_sha256']
    from capture_owned_native_counter_v01402_v1 import capture as capture_native_counter
    record['native_counter_capture_controller_sha256']=capture_cpu['capture_controller_sha256']
    record['native_counter_capture_cpu_receipt_sha256']=hashlib.sha256(capture_cpu_path.read_bytes()).hexdigest()
    record['uniform_complete_numerical_receipt_sha256']=hashlib.sha256(uniform_numerical_path.read_bytes()).hexdigest()
    record['uniform_numerical_actual_identity_sha256']=hashlib.sha256(identity_path.read_bytes()).hexdigest()
    record['uniform_quiet32k_sequence_sha256']=hashlib.sha256(quiet_path.read_bytes()).hexdigest()
    record['physical256k_sequence_completed']=False
'''
replace("    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()", uniform_block+guards+"    record['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()")
replace("+list(base.glob('owned-full-kv-access-*/record.json')):", "+list(base.glob('owned-full-kv-access-*/record.json'))+list(base.glob('owned-profile-definition-*/record.json'))+list(base.glob('owned-device-profile-*/record.json'))+list(base.glob('owned-poll-backoff-*/record.json'))+list(base.glob('owned-poll-matched-*/record.json')):")
replace("        if previous_path.parent.name in known_failures:", "        known_failures['owned-full-kv-access-v01402-full256k-diagnostic-r5']='e7e541f91ef053acebaa6b9ed2bc3139010e269f0b671d23f423869e2c7a882f'\n        known_failures['owned-device-profile-v01402-code32k-diagnostic-r1']='53cd3cebbc27b9d50d525e44cbc5dbc372a91beb0f9eb0e122b6150e43fd898e'\n        if previous_path.parent.name in known_failures:")
replace("    os.close(slave);slave=None;save()", """    os.close(slave);slave=None
    actual_identity=process_identity(g.inferior['pid'])
    assert actual_identity and actual_identity['start_ticks']==g.inferior['start_ticks']
    assert Path(os.readlink('/proc/'+str(actual_identity['pid'])+'/exe')).resolve()==binary.resolve()
    assert hashlib.sha256(Path('/proc',str(actual_identity['pid']),'exe').read_bytes()).hexdigest()==record['binary_sha256']
    record['actual_executable_identity']={'inferior':actual_identity,'sha256':record['binary_sha256'],'boot_id':record['boot_id']}
    actual=dict(item.decode().split('=',1) for item in Path('/proc',str(actual_identity['pid']),'environ').read_bytes().split(b'\\0') if b'=' in item)
    assert not any(k.startswith(('UNITRACE_','XPTI_')) or k in ['LD_PRELOAD','ZET_ENABLE_METRICS'] for k in actual)
    record['actual_target_environment']={k:v for k,v in actual.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['LD_LIBRARY_PATH','NEOReadDebugKeys','EnableDirectSubmission']}
    assert argv[1:]==uniform_numerical['argv'][1:]
    save()
""")
replace("        record['capacity_sequence_completed']=True", "        record['capacity_sequence_completed']=True\n        record['physical256k_sequence_completed']=True")
replace("        try:record['snapshots'].append(g.snapshot('failure',resume=False))", """        try:
            snap=g.snapshot('failure',resume=False);record['snapshots'].append(snap)
            if snap:
                record['native_counter_at_failure']=capture_native_counter(g,snap,out/'native-counter-at-failure-v1.json')""")
for key, value in {
    'scope': 'DD5 uniform DPCT definition repair on the canonical indexer/visible-output/KV baseline. Logged physical262144 occupancy through cell262143, repeated fresh full prefill, full disk SAVE/RESTORE, clipped speculative tail, capacity refusals, actual32K continuation and later valid fresh32K. All performance timings excluded; no polling/profiler/global changes or adoption claim. Owned read-only native-counter fields are captured only after failure stop.',
    'previous_goal_turn': 'Progress: sixteen matched quiet32768 reads passed full output/logprob/MTP/freshness/normal-exit gates. Polling speed gain rejected; evidence committed/pushed0f103c4c. DD5 full physical gate remains required.',
    'configuration_note': 'Context262144, chunk8192, int8 KV resident32768, cache128, workers5, pcie0, MTP4. PC1/ckpt1/every262139/root0/turn-token-1 preserves qualified32K geometry and actual disk restoration. Fresh reads require RESUME0/REUSED0. Flushed diagnostic UR/ZE/ZEL/Strata logs retained; no profiler, additional phase waits, cache retirement or prefetch. Expected/current native counters are read through source-derived fields only at an owned stopped failure.'
}.items():
    pattern = r"^    record\['"+key+r"'\]=.*$"
    s, count = re.subn(pattern, "    record['"+key+"']="+repr(value), s, flags=re.M)
    assert count == 1
old_plan = old.read_text().split("    # A mismatch leaves the engine waiting for input, so QUIT can close it normally.\n",1)[1].split("    current=None;send(b'QUIT",1)[0]
new_plan = s.split("    # A mismatch leaves the engine waiting for input, so QUIT can close it normally.\n",1)[1].split("    current=None;send(b'QUIT",1)[0]
assert new_plan.replace("        record['physical256k_sequence_completed']=True\n", '') == old_plan
ast.parse(s)
target = base/'run_owned_uniform_full256k_v01402_v1.py'
assert not target.exists()
target.write_text(s)
receipt = {'active': False, 'passed': True, 'old_full_controller_sha256': sha(old),
           'uniform_numerical_controller_sha256': sha(uniform_ancestor), 'candidate_controller_sha256': sha(target),
           'same_full_request_session_math_plan': True, 'executable_metadata_after_dd5_selection': True,
           'native_capture_cpu_passed': True, 'gpu_launched': False}
(base/'prepare-owned-uniform-full256k-v01402-v3.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt, indent=2))
