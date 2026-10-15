#!/usr/bin/env python3
"""Write compact provenance receipts only; does not run tests/builds/Git/GPU."""
from pathlib import Path
import hashlib
import json
P=Path('/home/yayoi/orca/workspaces/Strata/sycl')
A=P/'bench/results/2026-10-15-pr6-recovery-audit/int8-prompt-head-split-v1'
R='bench/results/2026-10-15-pr6-recovery-audit/'
paths=[
 'AGENTS.md', '.claude/skills/strata-upstream-sync/SKILL.md',
 '.claude/skills/strata-upstream-sync/references/evaluation.md',
 '.claude/skills/strata-upstream-sync/references/integration.md',
 'docs/ARTIFACT_RETENTION.md',
 'src/kernels/sycl/prompt_attention.cpp',
 'include/strata/kernels/qsa_prompt_attn.hpp',
 'include/strata/sycl/launch.hpp',
 'CMakeLists.txt', 'cmake/sycl_backend.cmake',
 'tests/sycl/decode_attention.cpp', 'tools/sycl/kv_copy_groups_host.cpp',
 R+'research-index.json',
 R+'research-profiled-prompt-attention-round600-v1.txt',
 R+'research-int8-attention-tiling-round605-v1.txt',
 R+'prompt-direct-profile.ja.md',
 R+'matrix-direct-probe-v1/implementation.txt',
 R+'matrix-direct-probe-v1/product.patch',
 R+'matrix-direct-probe-v1/files/tools/sycl/matrix_direct_probe.cpp',
]
def receipt(path, root):
 data=path.read_bytes()
 return {'path':str(path.relative_to(root)), 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()}
def write(name,data):
 (A/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
write('input-sha256.json', {
 'schema':'source-proposal-input-sha256-v1',
 'date_jst':'2026-10-15', 'workspace':str(P),
 'git_revision':None, 'git_revision_reason':'Git commands prohibited during parent-owned benchmark; Main pins revision later',
 'scope':'Read-only inputs for proposal generation/review; not model runtime input captures',
 'inputs':[receipt(P/name,P) for name in paths],
})
write('proposal-status.json', {
 'date_jst':'2026-10-15', 'status':'UNAPPLIED_UNBUILT_UNTESTED_GPU_UNQUALIFIED',
 'write_root':str(A), 'actual_product_edits':False, 'actual_cmake_edits':False,
 'actual_existing_fixture_edits':False, 'actual_build_test_gpu_server_profiler_git_cleanup':False,
 'operations_performed':['Read project source/reports/policy','Author proposal files under write_root','Run source/probe-text generators only','Compute source/artifact SHA256 receipts'],
 'candidate':{'mode':1, 'direct_exact':'1', 'selector':'STRATA_SYCL_PROMPT_HEAD_SPLIT',
              'values_exact':['32','64'],'wg':64,'local_rows':8,'head_tiles':[8,4],
              'common_q_scale_source_heads':12, 'existing_launcher_preserved':True,
              'legacy_dispatch_preserved_unless_exact_opt_in':True,
              'ch32_bitwise':'Expected for finite inputs; UNPROVEN',
              'ch64_gate':'Tolerance and model quality required; UNMEASURED',
              'packed_local_bytes':{'32':31088,'64':53872},'dynamic_slm_measured':False},
 'fixture':{'status':'SOURCE_PROPOSAL_ONLY','correctness_cases':28,'isolation_pairs':23,
            'oracle':'FP64 scalar independent attention','oracle_and_ch64_pair_abs_rel':1e-5,
            'all_24_heads':True,'tails':[0,1,31,32,33,63,64,65,2051],
            'logical_cap':262144,'selected_logical_id':262143,'high_cap_active_width':65,
            'full_physical_256k_proved':False,'restore_proved':False,'model_quality_proved':False},
 'prior_profile':{'source':'../prompt-direct-profile.ja.md','calls_per_arm':108,
                  'indirect_device_aggregate_seconds':44.066063496,
                  'direct_device_aggregate_seconds':29.805022972,
                  'same_generated_ids':128,'clean_benchmark':False,
                  'head_split_measurement':False,
                  'parent_clean_direct_benchmark':'Active/incomplete at task assignment; not reclassified here'},
 'retention':{'consumer':'Main serial head-split review/build/component/model gates',
              'next_review':'First compile/host/device result or source rejection',
              'raw_captures_created':False,'deletions':[]},
})
outputs=[]
for path in sorted(A.rglob('*')):
 if path.is_file() and path.name!='artifact-sha256.json':
  outputs.append(receipt(path,A))
write('artifact-sha256.json',{
 'schema':'source-proposal-output-sha256-v1','status':'UNAPPLIED_UNBUILT_UNTESTED_GPU_UNQUALIFIED',
 'scope':'Generated source/diff/docs/scripts/input receipts only; excludes this self-manifest',
 'artifacts':outputs,
 'total_artifact_bytes_excluding_this_manifest':sum(item['bytes'] for item in outputs),
})
print(json.dumps({'write_root':str(A),'artifact_count':len(outputs),
                  'artifact_bytes_excluding_manifest':sum(item['bytes'] for item in outputs),
                  'base_product_sha256':hashlib.sha256((P/'src/kernels/sycl/prompt_attention.cpp').read_bytes()).hexdigest()},ensure_ascii=False))
