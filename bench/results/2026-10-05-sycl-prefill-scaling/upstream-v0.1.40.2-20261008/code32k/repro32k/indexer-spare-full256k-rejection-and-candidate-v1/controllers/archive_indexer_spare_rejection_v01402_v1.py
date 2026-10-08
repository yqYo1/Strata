"""Freeze the normal full-capacity rejection and its private kernel candidate."""
from pathlib import Path
import datetime,hashlib,json,shutil
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008/code32k/repro32k'
out=parent/'indexer-spare-full256k-rejection-and-candidate-v1';out.mkdir()
def copy(a,b):
    b.parent.mkdir(parents=True,exist_ok=True);assert not b.exists();shutil.copy2(a,b)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
run=base/'owned-full-kv-access-v01402-full256k-diagnostic-r4'
r=json.loads((run/'record.json').read_text())
assert r['healthy'] and r['completed'] and not r['active'] and not r['math_gate_passed']
assert r['exit_code']==0 and not r['new_fault_messages'] and not r['cleanup']['forced']
assert not r['cleanup']['inferior_survived'] and not r['cleanup']['gdb_survived']
assert digest(run/'record.json')=='3ba1f08a4e0d11b81a90aa7119b5f1dc2a4c3cf31eb2b5d95b314f1fa9271b95'
for name in ['record.json','project-messages.txt','protocol.stdout.raw']:
    copy(run/name,out/'full256k-r4'/name)
for name in ['indexer-spare-commit-v01402-source-review-v1','indexer-spare-commit-v01402-build-v1','post-indexer-spare-rejection-v01402-health-v1']:
    receipt=json.loads((base/name/'record.json').read_text());assert not receipt['active'] and (receipt.get('passed') or receipt.get('healthy'))
    for p in sorted((base/name).iterdir()):
        if p.is_file() and p.suffix in ['.json','.diff','.stdout','.stderr']:
            copy(p,out/'checks'/name/p.name)
for name in ['run_owned_full_visible_commit_v01402_v4.py','build_indexer_spare_commit_v01402_v1.py','run_owned_full_indexer_spare_v01402_v5.py','check_health_after_indexer_spare_rejection_v01402_v1.py',Path(__file__).name]:
    copy(base/name,out/'controllers'/name)
candidate=root/'build-sycl-indexer-spare-commit-v1-20261008'
copy(candidate/'source/native_qsa_indexer.dp.cpp',out/'source/candidate-native_qsa_indexer.dp.cpp')
copy(candidate/'native_qsa_indexer.dp.cpp.o.d',out/'source/native_qsa_indexer.dp.cpp.o.d')
copy(root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp',out/'source/compiled-original-native_qsa_indexer.dp.cpp')
artifacts=[]
paths={Path(s['file']) for s in r['sessions'] if s['command']=='SAVE'}
for req in r['requests']:
    for key in ['first_head','prefill_state']:
        if key in req:paths.add(Path(req[key]['file']))
paths.add(candidate/'strata')
for p in sorted(paths):artifacts.append(dict(file=str(p),bytes=p.stat().st_size,sha256=digest(p)))
(out/'private-artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
summary=dict(active=False,completed=True,adopted=False,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_capacity_sequence_passed=False,full_read_repeat_passed=True,full_repeat_saved_state_and_kv_all_bytes_equal=True,full_physical_cells=262144,kv_layers=13,ignored_full_saved_tensor_bytes=0,disk32k_resume_output_parity_passed=True,full_disk_roundtrip_passed=False,full_disk_roundtrip_difference='Main12 layers only: last pooled row65535,512bytes per layer; restored row equals live idx_dead, original saved row does not. All other pooled bytes and all other state/KV parts match.',owned_exit_code=0,forced_cleanup=False,new_fault_messages=[],candidate_binary_sha256=digest(candidate/'strata'),candidate_gpu_gate='separate r5 controller; no completed hardware/adoption proof in this frozen archive',pending=['new32K complete numerical control','full256K repeat, all-byte image and actual disk roundtrip','restored clipped tail','capacity refusal and later fresh32K','owned normal exit and fault audit','matched quiet inputs>=32768 for performance'],minimum_performance_input_tokens=32768,diagnostic_durations_excluded_from_speed=True,no_reset_rebind_reboot_service_package_global_change=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(archive=str(out),private_artifacts=len(artifacts),summary=summary),indent=2))
