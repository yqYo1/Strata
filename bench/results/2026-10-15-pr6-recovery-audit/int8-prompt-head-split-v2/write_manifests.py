#!/usr/bin/env python3
"""Generate provenance text only. Never import/execute controller, comparator or probe."""
from pathlib import Path
import hashlib
import json
P=Path('/home/yayoi/orca/workspaces/Strata/sycl')
R=P/'bench/results/2026-10-15-pr6-recovery-audit'
A=R/'int8-prompt-head-split-v2'
V1=R/'int8-prompt-head-split-v1'
def item(path,root):
 data=path.read_bytes()
 return {'path':str(path.relative_to(root)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def write(name,data):
 (A/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
inputs=[P/'AGENTS.md',P/'.claude/skills/strata-upstream-sync/SKILL.md',
 P/'docs/ARTIFACT_RETENTION.md',P/'src/kernels/sycl/prompt_attention.cpp',
 P/'include/strata/kernels/qsa_prompt_attn.hpp',P/'include/strata/kernels/qsa.hpp',
 P/'include/strata/sycl/launch.hpp',P/'include/strata/core/mtp_expert_source.hpp',
 P/'CMakeLists.txt',P/'cmake/sycl_backend.cmake',P/'tests/sycl/decode_attention.cpp',
 V1/'source.diff',V1/'files/src/kernels/sycl/prompt_attention.cpp',
 V1/'files/tools/sycl/int8_prompt_head_split_probe.cpp',V1/'design-audit.ja.md',
 V1/'main-controller-plan.ja.md',V1/'proposal-status.json',V1/'input-sha256.json',V1/'artifact-sha256.json',
 R/'research-profiled-prompt-attention-round600-v1.txt',R/'research-int8-attention-tiling-round605-v1.txt',
 R/'prompt-direct-profile.ja.md']
write('input-sha256.json',{'schema':'source-proposal-input-sha256-v2','date_jst':'2026-10-15',
 'workspace':str(P),'git_revision':None,'git_reason':'All Git commands prohibited; Main pins revision after freeze release',
 'scope':'Read-only source/report inputs; no model/tensor capture',
 'inputs':[item(p,P) for p in inputs]})
write('proposal-status.json',{'schema':'head-split-proposal-status-v2','date_jst':'2026-10-15',
 'status':'UNAPPLIED_UNBUILT_UNTESTED_UNEXECUTED_GPU_UNQUALIFIED','write_root':str(A),
 'v1_original_preserved':True,'product_cmake_existing_fixture_binary_changed':False,
 'build_test_gpu_server_profiler_git_cleanup_run':False,
 'controller_comparator_help_or_syntax_checks_run':False,
 'operations_performed':['Read source and previous proposal','Author new v2 text/source snapshots only','Generate SHA256 receipts'],
 'kernel':{'origin':'v1 fork-derived kernel proposal','source_diff_sha256':item(A/'source.diff',A)['sha256'],
           'v1_source_diff_sha256':item(V1/'source.diff',V1)['sha256'],
           'snapshot_sha256':item(A/'files/src/kernels/sycl/prompt_attention.cpp',A)['sha256'],
           'v1_snapshot_sha256':item(V1/'files/src/kernels/sycl/prompt_attention.cpp',V1)['sha256'],
           'old_launcher_changed':False,'mode':1,'direct_exact':'1','split_values_exact':['32','64'],
           'ch32_bitwise':'REQUIRED_BUT_UNPROVEN','ch64_abs_rel_tolerance':1e-5,
           'packed_local_bytes':{'32':31088,'64':53872},'dynamic_slm_measured':False,
           'legacy_INT_MAX_loop_issue':'Separate issue; existing body preserved; does not arise at cap262144'},
 'probe':{'one_process_one_arm':True,'process_environment_writers':False,
          'pre_runtime_route_validation':'Read-only exact expected exec environment',
          'deterministic_full_cases':28,'isolation_pairs':23,'isolation_native_processes':46,
          'all_f32_outputs_and_guards_captured':True,'all_f64_references_captured':True,
          'all_input_array_sha256':True,'capture_default_bytes':8388608,'capture_hard_max_bytes':16777216,
          'overwrite_allowed':False,'actual_captures_created':False},
 'comparison':{'separate_process':True,'normal_exit_retirement_and_fresh_fault_receipts_required':True,
               'same_source_binary_fixture_and_inputs_required':True,'all_values_oracle':True,
               'ch32_bitwise':True,'ch64_abs_rel':1e-5,'missing_declined_incomplete':'FAIL_CLOSED',
               'existing_environment_switch_test_planned':False},
 'controller':{'source_only':True,'executed':False,'existing_shared_lock_required':True,
               'freeze_requires_both_clean_direct0_direct1_same92acf71e_binary':True,
               'all_benchmark_owners_retired_required':True,'mutates_parent_os_environ':False,
               'child_route_set_only_at_exec':True,'main_readonly_fault_adapter_required':True,
               'fault_adapter_implemented_here':False,'whole_deadline_seconds':21600,'whole_byte_budget':67108864},
 'limits':{'observed_route_proved':False,'runtime_fault_gate_run':False,'model_measurement_run':False,
           'model_minimum_actual_input_required':32768,'physical_256k_proved':False,'restore_proved':False,
           'source_build_binary_frozen_until_both_benchmark_arms_close':True},
 'retention':{'consumer':'Main v2 source/controller/comparator review and first serial validation',
              'full_array_consumer':'Future exact offline pair comparison and first numerical-failure investigation',
              'review_point':'After comparisons and required diagnostic decisions; preserve last unresolved fixture',
              'deletions':[]}})
artifacts=[item(p,A) for p in sorted(A.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json']
write('artifact-sha256.json',{'schema':'source-proposal-output-sha256-v2',
 'status':'UNAPPLIED_UNBUILT_UNTESTED_UNEXECUTED_GPU_UNQUALIFIED',
 'scope':'Proposal source/diff/scripts/docs/receipts only; excludes self-manifest',
 'artifacts':artifacts,'total_bytes_excluding_manifest':sum(p['bytes'] for p in artifacts)})
print(json.dumps({'write_root':str(A),'artifact_count_excluding_self':len(artifacts),
 'artifact_bytes_excluding_self':sum(p['bytes'] for p in artifacts),
 'v1_kernel_diff_sha256':item(V1/'source.diff',V1)['sha256'],
 'v2_kernel_diff_sha256':item(A/'source.diff',A)['sha256'],
 'actual_product_sha256':item(P/'src/kernels/sycl/prompt_attention.cpp',P)['sha256']}))
