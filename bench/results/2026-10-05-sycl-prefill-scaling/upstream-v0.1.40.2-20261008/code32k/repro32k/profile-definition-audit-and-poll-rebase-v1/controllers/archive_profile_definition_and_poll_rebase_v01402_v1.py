"""Freeze first repaired full image plus CPU-only profiling preparation."""
from pathlib import Path
import datetime,hashlib,json,shutil
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k'
out=parent/'profile-definition-audit-and-poll-rebase-v1';out.mkdir()
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(a,b):
    b.parent.mkdir(parents=True,exist_ok=True);assert not b.exists();shutil.copy2(a,b)
live_path=base/'owned-full-kv-access-v01402-full256k-diagnostic-r5/record.json';live=json.loads(live_path.read_text());assert live['active'] and not live.get('error') and live['active_request']=='full256k-repeat'
first=next(x for x in live['requests'] if x['name']=='full256k-first');between=next(x for x in live['requests'] if x['name']=='control32k-between-full-reads');saved=next(x for x in live['sessions'] if x['name']=='save-full256k-first')
assert first['math_gate_passed'] and first['last_executed_physical_cell']==262143 and between['math_gate_passed']
assert saved['passed'] and saved['all_saved_main_indexer_spares_equal_live_dead'] and saved['full_capacity_gate']['ignored_tensor_bytes']==0
snapshot={k:live[k] for k in ['started_utc','boot_id','binary_sha256','argv','environment','inferior','debugger','indexer_spare_build_receipt_sha256','indexer_spare_source_sha256','indexer_spare_source_review_sha256']}
snapshot.update(active=False,completed=True,adopted=False,captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Completed repaired first full physical-capacity read/image and following fresh32K control only. Parent remains active in its second full read; no repeated-full/disk-roundtrip/exit/adoption proof.',requests=[first,between],sessions=[saved],parent_live_receipt=str(live_path))
(out/'full-first-v5-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
names=['dpct-profile-definition-v01402-source-review-v1','dpct-profile-definition-v01402-source-review-v2','dpct-profile-definition-v01402-source-review-v3','dpct-profile-definition-linked-tu-audit-v1','dpct-profile-definition-linked-tu-audit-v2','dpct-profile-definition-linked-tu-audit-v3','dpct-profile-definition-v01402-build-v1','prefill-poll-backoff-v01402-relink-v2','vtune-xpu-host-capabilities-v1']
for name in names:
    receipt=json.loads((base/name/'record.json').read_text());assert not receipt['active']
    for p in sorted((base/name).iterdir()):
        if p.is_file() and (p.suffix in ['.json','.diff'] or (name!='vtune-xpu-host-capabilities-v1' and p.suffix in ['.txt','.stdout','.stderr'] and p.name!='ninja-deps.txt')):
            copy(p,out/'checks'/name/p.name)
for name in ['audit_dpct_profile_definition_v01402_v1.py','audit_dpct_profile_definition_v01402_v2.py','audit_dpct_profile_definition_v01402_v3.py','audit_linked_dpct_tus_v01402_v2.py','audit_linked_dpct_tus_v01402_v3.py','build_dpct_profile_definition_v01402_v1.py','relink_prefill_poll_backoff_v01402_v2.py',Path(__file__).name]:copy(base/name,out/'controllers'/name)
header=root/'build-sycl-dpct-profile-definition-v3-20261008/include/dpct/device.hpp';copy(header,out/'source/candidate-device.hpp');copy(root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/include/dpct/device.hpp',out/'source/compiled-original-device.hpp')
build=json.loads((base/'dpct-profile-definition-v01402-build-v1/record.json').read_text());assert build['passed'] and not build['active']
for item in build['objects']:copy(Path(item['dependency']),out/'dependencies'/Path(item['object']).name.replace('.o','')/'object.d')
poll=json.loads((base/'prefill-poll-backoff-v01402-relink-v2/record.json').read_text());assert poll['passed'] and not poll['active']
private=[]
for p in [Path(first['first_head']['file']),Path(first['prefill_state']['file']),Path(saved['file']),Path(build['candidate_binary']),Path(poll['candidate_binary'])]:private.append(dict(file=str(p),bytes=p.stat().st_size,sha256=digest(p)))
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
summary=dict(active=False,completed=True,adopted=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),first_repaired_full_gate_passed=True,first_full_reference_math_equal=True,first_full_saved_main_spares_canonical=True,full_capacity_sequence_passed=False,current_parent_phase='second full read active; no other GPU experiment',profile_definition_candidate={'binary_sha256':build['candidate_binary_sha256'],'gpu_tested':False,'replaced_objects':7,'actual_header_users':67,'all_linked_compile_commands':116,'profiled_helper_tokens_unchanged':True,'explicit_copy_queue_property_factory_unchanged':True},poll_rebase={'binary_sha256':poll['candidate_binary_sha256'],'baseline_binary_sha256':poll['baseline_binary_sha256'],'gpu_tested':False,'only_prefill_link_input_replaced':True},vtune={'installed_version':'2026.4.0 build632893','help_only':True,'gpu_support_and_collection_unproven':True,'planned_analysis':'xpu-offload, no CPU sampling or hardware metrics, programming API tracing'},negative_cpu_preflights=['v1 excerpt parser expected a typedef rather than actual sycl::event*','v2 single-header include prefix did not override dpct.hpp relative include; full shadow tree plus -H proves v3 selection','linked v1 wrong Ninja delimiter; linked v2 rejects a custom CPU GCC object without Ninja deps; v3 explicitly validates that sole exception'],minimum_performance_input_tokens=32768,diagnostic_durations_excluded_from_speed=True,no_reset_rebind_reboot_service_package_global_change=True,pending=['current full repeat/disk roundtrip/clipped/refusal/later32K/normal owned exit','separate uniform-header logged32K numerical/exit gate before profiling','actual profiler device timestamp and graph coverage','matched quiet>=32768 performance','complete capacity gate before adopting any candidate'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(archive=str(out),header_binary_sha256=build['candidate_binary_sha256'],poll_binary_sha256=poll['candidate_binary_sha256'],first_full_saved_bytes=saved['bytes']),indent=2))
