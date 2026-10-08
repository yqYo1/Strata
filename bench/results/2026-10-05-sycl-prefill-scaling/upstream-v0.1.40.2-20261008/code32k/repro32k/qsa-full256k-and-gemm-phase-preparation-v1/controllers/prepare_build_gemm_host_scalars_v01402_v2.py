"""Correct Ninja enumeration without changing the CPU-proven early GPU guard."""
from pathlib import Path
import ast, datetime, hashlib, json

base = Path(__file__).parent
parent = base/'build_gemm_host_scalars_v01402_v1.py'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(parent) == 'f60996974a7ee8b67bf3006c72efcc1a3440af04eedbb4e3eec3206e3340c81c'
old = parent.read_text()
s = old
a = "['/usr/bin/ninja', '-t', 'commands', target]"
assert s.count(a) == 1
s = s.replace(a,"['/usr/bin/ninja', '-t', 'commands', '-s', target]")
s = s.replace("candidate = root / 'build-sycl-gemm-host-scalars-v1-20261008'", "candidate = root / 'build-sycl-gemm-host-scalars-v2-20261008'")
s = s.replace("out = base / 'gemm-host-scalars-v01402-build-v1'", "out = base / 'gemm-host-scalars-v01402-build-v2'")
ast.parse(s)
before = old[:old.index('def digest')]
after = s[:s.index('def digest')]
assert before == after
negative = json.loads((base/'gemm-host-scalars-builder-early-guard-cpu-v01402-v1.json').read_text())
assert negative['passed'] and negative['controller_sha256'] == sha(parent)
assert not negative['heavy_or_mutating_traps_reached']
review = json.loads((base/'gemm-host-scalars-ninja-command-review-v01402-v1/record.json').read_text())
assert review['passed'] and review['recursive_command_count'] == 28 and review['single_target_command_count'] == 1
target = base/'build_gemm_host_scalars_v01402_v2.py'
assert not target.exists()
target.write_text(s)
record = {'active':False,'prepared':True,'engine_built_by_preparer':False,'gpu_launched':False,
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preparer_sha256':sha(__file__),
          'parent_builder_sha256':sha(parent),'controller':str(target),'controller_sha256':sha(target),
          'early_guard_prefix_sha256':hashlib.sha256(after.encode()).hexdigest(),
          'CPU_early_guard_proof_reused_by_exact_prefix_identity':True,
          'CPU_early_guard_receipt_sha256':sha(base/'gemm-host-scalars-builder-early-guard-cpu-v01402-v1.json'),
          'actual_Ninja_enumeration_review_sha256':sha(base/'gemm-host-scalars-ninja-command-review-v01402-v1/record.json'),
          'checks':{'AST':True,'only_single_target_Ninja_command_enumerated':True,
                    'unchanged_early_live_GPU_guard':True,'separate_v2_output_and_candidate_paths':True},
          'minimum_performance_input_tokens':32768,
          'scope':'Preparation only. V1 single-command assumption was rejected during CPU recipe review; no engine compile, GPU launch or device/performance result.'}
p = base/'prepare-build-gemm-host-scalars-v01402-v2.json'
assert not p.exists()
p.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
