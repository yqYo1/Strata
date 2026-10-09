"""Prepare physical262144 proof after the complete native-copy quiet comparison."""
from pathlib import Path
import ast
import datetime
import hashlib
import json

B = Path(__file__).parent
OLD = B / 'run_owned_upstream_v0141_full256k_v2.py'
OUT = B / 'run_owned_native_expert_copy_v0141_full256k_v1.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD) == 'a34cb70ed479a4bea5f9727cb19461d32014d9679d3c4642a6e9cc380d2651a1'
quiet_path = B / 'native-expert-copy-v0141-quiet32k-comparison-sequence-v1/record.json'
quiet = json.loads(quiet_path.read_text())
assert not quiet['active'] and quiet['passed'] and len(quiet['steps']) == 6
assert quiet['configuration_equal_except_explicit_native_copy_flag']
for step in quiet['steps']:
    p = Path(step['receipt'])
    assert sha(p) == step['receipt_sha256']
    d = json.loads(p.read_text())
    assert not d['active'] and d['completed'] and d['healthy'] and d['math_gate_passed']
    assert d['native_copy_queue_startup_gate_passed'] and len(d['requests']) == 4
    assert d['exit_code'] == 0 and not d['exit_signal'] and not d['new_fault_messages'] and not any(d['cleanup'].values())
assert quiet['summary']['nativeon']['later_fresh_reads']['prefill_tok_s'] > 1.01 * max(
    quiet['summary'][mode]['later_fresh_reads']['prefill_tok_s'] for mode in ('baseline', 'nativeoff'))
assert not OUT.exists()
original = OLD.read_text()
text = original

def replace(before, after):
    global text
    assert text.count(before) == 1, before[:160]
    text = text.replace(before, after)

replace('Logged integrated upstream v0.1.41 physical256K gate', 'Logged native expert-copy v0.1.41 physical256K gate')
replace("root = observer.parent/'sync-upstream-v0.1.41-20261009'", "root = observer.parent/'perf-sycl-prefill-native-copy-v0141-20261009'")
replace("assert mode=='integrated' and phase=='diagnostic' and repetition==2",
        "assert mode=='nativeon' and phase=='diagnostic' and repetition==1\nassert sys.argv[4:] in [[], ['--cpu-preflight']]")
replace("sequence_path=base/'upstream-v0141-matched32k-comparison-sequence-v1/record.json'",
        "sequence_path=base/'native-expert-copy-v0141-quiet32k-comparison-sequence-v1/record.json'\nassert hashlib.sha256(sequence_path.read_bytes()).hexdigest()=='" + sha(quiet_path) + "'")
replace('updated32K comparison still active; no full-capacity process', 'native-copy32K comparison still active; no full-capacity process')
replace("assert sequence['passed'] and len(sequence['steps'])==8", "assert sequence['passed'] and len(sequence['steps'])==6")
replace("assert sequence['paired_requested_configuration']", "assert sequence['configuration_equal_except_explicit_native_copy_flag']")
start = text.index("build_path=base/'integrated-upstream-v0.1.41-20261009-v2/build-record.json'")
stop = text.index('# Preserve terminal failed-history evidence', start)
bindings = '''reference_build_path=base/'integrated-upstream-v0.1.41-20261009-v2/build-record.json'
reference_build=json.loads(reference_build_path.read_text())
assert reference_build['passed'] and not reference_build['active'] and reference_build['commit']=='1eb89482a4afd20277ae0405780ed4f8eb98eb20'
assert reference_build['binary_sha256']=='86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8'
build_path=base/'native-expert-copy-v0141-private-build-v1/record.json'
assert hashlib.sha256(build_path.read_bytes()).hexdigest()=='74b48ad90996c942fd74fa826d22dfecf76145d8dec82a40cfa0d35dc9b65fad'
build=json.loads(build_path.read_text())
assert build['passed'] and not build['active'] and build['compiled_engine'] and not build['adopted']
assert build['source_head']=='6caa1421f9212750a425fe1729139ffdde6e9f9a'
binary=Path(build['binary'])
assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binary_sha256']=='2721f8ef417456c8a549cf438292ce34955a078d30974ed0200e1115ab8a3f5f'
candidate_source=root/'sycl/src/prefill/prefill.cpp'
candidate_header=root/'sycl/include/dpct/device.hpp'
assert hashlib.sha256(candidate_source.read_bytes()).hexdigest()=='6399d65feea884f932a5a1079a91c80228293bfc3be371732b48413977229f84'
assert hashlib.sha256(candidate_header.read_bytes()).hexdigest()=='8805874ecb1d58d7acea8ca931154a38d0ad3d623f7c5ee8d599fa2ba56a841b'
flags_path=base/'native-expert-copy-v0141-uniform-build-flags-v1.json'
assert hashlib.sha256(flags_path.read_bytes()).hexdigest()=='54a1a491a7f00874a90af1b7776d0654b8b001e9aea3da99f6e0b1dfe0a83e53'
flags=json.loads(flags_path.read_text());assert flags['passed'] and not flags['active']
assert flags['old_compile_count']==flags['candidate_compile_count']==114 and not flags['missing_sources'] and not flags['different_flags']
assert flags['build_receipt_sha256']==hashlib.sha256(build_path.read_bytes()).hexdigest()
first_path=base/'owned-native-expert-copy-v0141-code32k-diagnostic-r1/record.json'
assert hashlib.sha256(first_path.read_bytes()).hexdigest()=='83670ee10b75d702c621e0ce029c5be1a4701ac6684d08b4f22de905704ae908'
first=json.loads(first_path.read_text())
assert not first['active'] and first['completed'] and first['healthy'] and first['math_gate_passed']
assert first['native_copy_queue_startup_gate_passed'] and all(x==1 for x in first['native_copy_queue_ordinals'])
assert first['exit_code']==0 and not first['exit_signal'] and not first['new_fault_messages'] and not any(first['cleanup'].values())
assert first['binary_sha256']==build['binary_sha256'] and first['reference_binary_sha256']==reference_build['binary_sha256']
assert len(first['requests'])==4 and all(x['math_gate_passed'] for x in first['requests'])
assert all(x['first_head']['finite'] and x['first_head']['floats']==248320 and len(x['prefill_state']['parts'])==66 for x in first['requests'])
for role in ['inferior','debugger']:
    old=first[role];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
baseline_full_path=base/'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json'
assert hashlib.sha256(baseline_full_path.read_bytes()).hexdigest()=='168d7896cbe3a60dfc604a08b6fe310f16d823a17d13d8a013260bfc7facb0a4'
baseline_full=json.loads(baseline_full_path.read_text())
assert not baseline_full['active'] and baseline_full['completed'] and baseline_full['healthy'] and baseline_full['math_gate_passed']
assert baseline_full['physical256k_sequence_completed'] and baseline_full['capacity_sequence_completed']
assert baseline_full['binary_sha256']==reference_build['binary_sha256']
for role in ['inferior','debugger']:
    old=baseline_full[role];now=process_identity(old['pid'])
    assert not now or now['start_ticks']!=old['start_ticks']
assert first['boot_id']==baseline_full['boot_id']==qualified['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert hashlib.sha256(Path('/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0.12.0').read_bytes()).hexdigest()=='bfdc0f26bf88bd0bccd66fc25302a4b22559f8fe30e499be18529be3ed610e60'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==build['source_head']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
assert shutil.disk_usage(base).free>170*1024**3
'''
text = text[:start] + bindings + text[stop:]
replace("assert subset['binary_sha256']==build['binary_sha256'] and subset['source_commit']==build['commit']",
        "assert subset['binary_sha256']==reference_build['binary_sha256'] and subset['source_commit']==reference_build['commit']")
replace("lock=(base/'owned-v0141-measurement.lock').open('a')",
        "if sys.argv[4:]==['--cpu-preflight']:\n    print(json.dumps({'cpu_preflight_passed':True,'gpu_executed':False,'binary_sha256':build['binary_sha256'],'quiet_sequence_sha256':hashlib.sha256(sequence_path.read_bytes()).hexdigest(),'full_reference_sha256':hashlib.sha256(baseline_full_path.read_bytes()).hexdigest()}))\n    sys.exit(0)\nlock=(base/'owned-v0141-measurement.lock').open('a')")
replace("out = base/f'owned-upstream-v0141-integrated-full256k-{phase}-r{repetition}'", "out = base/f'owned-native-expert-copy-v0141-full256k-{phase}-r{repetition}'")
text = text.replace('New integrated v0.1.41 physical262144 qualification with exactly the qualified DD5 request history.',
                    'Native copy-only queue272 physical262144 qualification with exactly the qualified DD5 request history.')
replace("    env.update(STRATA_PREFILL_LAYER_MAJOR='0',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0')",
        "    env.update(STRATA_PREFILL_LAYER_MAJOR='0',STRATA_KV_STAGE_OWN='1',STRATA_KV_PREFETCH='0',STRATA_PREFILL_COPY_ENGINE='1',STRATA_DECODE_TIMING='1')\n    env.pop('STRATA_PREFILL_HOST_TIMING',None)\n    assert 'STRATA_VERIFY_PROFILE' not in env")
replace("    record['compatibility_change']='Updated upstream engine plus transferred qualified completion/state/KV corrections; all server changes included. This binary requires new full-capacity proof. No QSA reduce12 or GEMM scalar experiment.'",
        "    record['compatibility_change']='Only opt-in native expert-copy queue selection differs from qualified869 math. Candidate272 was uniformly rebuilt with all114 compiler commands matched. Four-fresh32K heads/live-state/output pass; this new binary still needs full-capacity proof.'\n    record['decode_host_counters']='STRATA_DECODE_TIMING prints existing unconditional host counters once per request. No STRATA_VERIFY_PROFILE, GPU timestamps, added synchronization or GPU profiler. Logged diagnostic times are excluded from speed comparisons.'\n    record['first_native_copy_diagnostic_receipt_sha256']=hashlib.sha256(first_path.read_bytes()).hexdigest()\n    record['baseline_full_capacity_receipt_sha256']=hashlib.sha256(baseline_full_path.read_bytes()).hexdigest()\n    record['uniform_flags_receipt_sha256']=hashlib.sha256(flags_path.read_bytes()).hexdigest()\n    record['root']=str(root);record['source_head']=build['source_head']\n    record['performance_eligible']=False;record['adopted']=False;record['full_lifecycle_passed']=False")
replace("    record['previous_goal_turn']='Updated upstream raw/integrated/control same-day32K sequence must finish first. Its output-only checks do not prove new intermediate state or whole-context disk restoration.'",
        "    record['previous_goal_turn']='Native-copy first4fresh32K full-head/live-state proof and all24 quiet reads must finish before this full-capacity process. Baseline869 full qualification does not qualify candidate272.'")
replace("observer/'sycl/tools/owned_gdb.py',root/'sycl/src/prefill/prefill.cpp']}",
        "observer/'sycl/tools/owned_gdb.py',root/'sycl/src/prefill/prefill.cpp',candidate_header,flags_path]}")
replace("    record['engine_log_bytes']=engine_log.stat().st_size",
        """    queue_lines=[value for value in project if value.startswith('strata prefill copy queue: ')]
    assert queue_lines, 'native copy-only queue creation required'
    prefix='strata prefill copy queue: native copy-only, ordinal '
    suffix=', index0, in-order, profiling0'
    ordinals=[]
    for value in queue_lines:
        assert value.startswith(prefix) and value.endswith(suffix)
        ordinal=value[len(prefix):-len(suffix)]
        assert ordinal.isdecimal() and int(ordinal)==1
        ordinals.append(int(ordinal))
    record['native_copy_queue_ordinals']=ordinals
    record['native_copy_queue_startup_gate_passed']=True
    record['engine_log_bytes']=engine_log.stat().st_size""")
replace("    if mode=='integrated':", "    if mode=='nativeon':")
replace("==build['commit']\n        record['source_status_after']", "==build['source_head']\n        record['source_status_after']")
replace("    record['active']=False;record['finished_utc']", "    record['full_lifecycle_passed']=record['healthy'] and bool(record.get('math_gate_passed')) and bool(record.get('native_copy_queue_startup_gate_passed'))\n    record['active']=False;record['finished_utc']")

# Every old function, including nested numerical/state readers and requests,
# remains byte-for-AST identical. Changes are bindings, metadata and admission.
defs = lambda s: [(n.name, ast.dump(n, include_attributes=False)) for n in ast.walk(ast.parse(s)) if isinstance(n, ast.FunctionDef)]
assert defs(original) == defs(text)
compile(text, str(OUT), 'exec')
OUT.write_text(text)
record = {'active': False, 'prepared': True, 'gpu_executed': False,
          'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'original_full_controller_sha256': sha(OLD), 'controller_sha256': sha(OUT),
          'terminal_quiet_sequence_sha256': sha(quiet_path),
          'all_old_function_ASTs_unchanged': [name for name, _ in defs(original)],
          'scope': 'Same twelve-request qualified history, including two fresh full reads through cell262143, all saved bytes, actual restoration, clipped tails, refusals and later32K. Existing host decode counters are printed with GPU profiling disabled. First diagnostics logged; no performance/adoption claim. CPU preflight required before GPU launch.'}
p = B / 'native-expert-copy-v0141-full-controller-preparation-v1.json'
assert not p.exists()
p.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
