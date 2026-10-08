"""Prepare a private one-object GEMM builder with an early live-GPU-job guard."""
from pathlib import Path
import ast, datetime, hashlib, json

base = Path(__file__).parent
parent = base/'build_qsa_reduce12_sg32_v01402_v1.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(parent) == '99729b6dcf9c05bb91af023023549a7f39699a9f07047fd5bbbfdfe4ce3a8e54'
s = parent.read_text()
s = s.replace('Build exactly one private QSA object', 'Build exactly one private GEMM object', 1)
guard = '''current_path = base/'owned-qsa-reduce12-sg32-v01402-full256k-diagnostic-r1/record.json'
current = json.loads(current_path.read_text())
assert not current['active'], 'active QSA full256K job: no compile or large input reads'
assert current['completed'] and current['healthy'] and current['math_gate_passed']
assert current['physical256k_sequence_completed'] and current['capacity_sequence_completed']
assert current['exit_code'] == 0 and not current['exit_signal'] and not current.get('error')
assert not current['new_fault_messages'] and not any(current['cleanup'].values())
assert current['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for role in ['inferior', 'debugger']:
    row = current[role]
    try:
        stat = (Path('/proc')/str(row['pid'])/'stat').read_text()
    except FileNotFoundError:
        continue
    assert int(stat[stat.rindex(')')+2:].split()[19]) != row['start_ticks'], 'owned QSA process remains live'

'''
needle = "full_path = base / 'owned-profile-definition-v01402-full256k-diagnostic-r7/record.json'"
assert s.count(needle) == 1
s = s.replace(needle,guard+needle)
s = s.replace("source_dir = base / 'qsa-reduce12-sg32-v01402-source-v3'", "source_dir = base / 'gemm-host-scalars-v01402-source-v1'")
old = "assert prepared['cpu_helper_proof_reused_by_exact_body_identity'] and prepared['cpu_semantic_routing_cases'] == 1024"
assert s.count(old) == 1
s = s.replace(old,"assert prepared['changed_default_call_sites'] == 2 and all(prepared['source_contract_review'].values())")
s = s.replace('c1b0d58f46f0807af10d310320ebe4f9f2b4ad62f1a584abf72a597d7d5f1da1', '0225c809c1608a59719d84809075b3349bddcf09e5a9bb7aada4bc611aada955')
old = "assert digest(prepared['original_source']) == prepared['original_source_sha256']"
new = '''original_source = root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/prefill/gemm.dp.cpp'
assert digest(original_source) == prepared['source_inputs_sha256'][str(original_source)] == '11a2c943ccc323c328b58d99119d77b3b4ed95c92a0f680480bf5ede96bcb705'
assert all(digest(path) == expected for path,expected in prepared['source_inputs_sha256'].items())'''
assert s.count(old) == 1
s = s.replace(old,new)
s = s.replace("candidate = root / 'build-sycl-qsa-reduce12-sg32-v1-20261008'", "candidate = root / 'build-sycl-gemm-host-scalars-v1-20261008'")
s = s.replace("out = base / 'qsa-reduce12-sg32-v01402-build-v1'", "out = base / 'gemm-host-scalars-v01402-build-v1'")
s = s.replace('replace one qsa_decode_attn.dp.cpp.o member', 'replace one gemm.dp.cpp.o member')
old = "    original_archive = baseline.parent / 'libstrata_kernels.a'"
new = '''    prefill_paths = [Path(path) for path in inputs if Path(path).name == 'libstrata_prefill.a']
    assert len(prefill_paths) == 1
    original_archive = prefill_paths[0]
    assert digest(original_archive) == '1c4a8d644ce2d5d74d0466e59e109ea902b337bee084151f862b215026b8db1f' '''
assert s.count(old) == 1
s = s.replace(old,new.rstrip())
s = s.replace("private_archive = candidate / 'libstrata_kernels.a'", "private_archive = candidate / 'libstrata_prefill.a'")
s = s.replace('qsa_decode_attn.dp.cpp.o', 'gemm.dp.cpp.o')
s = s.replace('CMakeFiles/strata_kernels.dir/src/kernels/cuda/gemm.dp.cpp.o', 'CMakeFiles/strata_prefill.dir/src/prefill/gemm.dp.cpp.o')
s = s.replace("assert Path(argv[argv.index('-c') + 1]) == Path(prepared['original_source'])", "assert Path(argv[argv.index('-c') + 1]) == original_source")
old = "    expected_original_object = next(row for row in uniform['archive_members'] if row['member'] == object_path.name)\n    assert before_members[object_path.name] == expected_original_object['after_sha256']"
new = '''    original_object = build/target
    assert original_object.is_file()
    assert before_members[object_path.name] == digest(original_object)
    record['original_gemm_object_sha256'] = digest(original_object)'''
assert s.count(old) == 1
s = s.replace(old,new)
s = s.replace('only_qsa_decode_archive_member_replaced', 'only_gemm_archive_member_replaced')
needle = "    'baseline_full256k_receipt_sha256': digest(full_path),"
assert s.count(needle) == 1
s = s.replace(needle,"    'preceding_qsa_full256k_receipt_sha256': digest(current_path),\n"+needle)
ast.parse(s)
assert s.index("assert not current['active']") < s.index('def digest') < s.index('candidate.mkdir()') < s.index("run('compile', argv)")
assert 'qsa_decode_attn.dp.cpp.o' not in s
assert "original_archive = baseline.parent / 'libstrata_kernels.a'" not in s
assert "'CMakeFiles/strata_prefill.dir/src/prefill/gemm.dp.cpp.o'" in s
target = base/'build_gemm_host_scalars_v01402_v1.py'
assert not target.exists()
target.write_text(s)
receipt = {'active':False,'prepared':True,'engine_built_by_preparer':False,'gpu_launched':False,
           'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preparer_sha256':sha(__file__),
           'parent_builder_sha256':sha(parent),'controller':str(target),'controller_sha256':sha(target),
           'source_record_sha256':sha(base/'gemm-host-scalars-v01402-source-v1/record.json'),
           'checks':{'AST':True,'active_qsa_guard_before_large_reads_or_mutation':True,
                     'preceding_qsa_full_normal_terminal_and_owned_absence':True,'DD5_full_qualified_baseline':True,
                     'actual_prefill_archive_and_original_member_identity':True,'exact_original_Ninja_compile_flags':True,
                     'same_uniform_dpct_header_first_include':True,'only_gemm_member_and_link_input_replacement':True,
                     'actual_dependency_inventory_and_input_unchanged_checks':True,'no_engine_or_GPU_execution':True},
           'minimum_performance_input_tokens':32768,
           'scope':'Builder source preparation and AST checks only. Native oneMKL compile/device correctness/performance/own256K proof remain pending.'}
p = base/'prepare-build-gemm-host-scalars-v01402-v1.json'
assert not p.exists()
p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'prepared':True,'controller_sha256':sha(target),'engine_built':False}))
