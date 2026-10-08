"""Freeze the completed first full read and an unexecuted-GPU polling candidate."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
repro=parent/'code32k/repro32k'
out=repro/'visible-output-commit-full-first-and-poll-backoff-v2';out.mkdir()
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(p,q):
    q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();shutil.copy2(p,q);assert digest(p)==digest(q)
path=base/'owned-full-kv-access-v01402-full256k-diagnostic-r4/record.json'
live=json.loads(path.read_text());assert live['active'] and live['active_request']=='full256k-repeat' and not live.get('error')
first=next(v for v in live['requests'] if v['name']=='full256k-first')
between=next(v for v in live['requests'] if v['name']=='control32k-between-full-reads')
saved=next(v for v in live['sessions'] if v['name']=='save-full256k-first')
assert first['math_gate_passed'] and first['resume_tokens']==[0,0] and first['last_executed_physical_cell']==262143
assert between['math_gate_passed'] and saved['passed'] and saved['full_capacity_gate']['ignored_tensor_bytes']==0
snapshot={k:live[k] for k in ['started_utc','boot_id','environment','argv','binary_sha256','inferior','debugger']}
snapshot.update(active=False,completed=True,scope='Completed first full physical-capacity read and image evidence, with exact interposed32K control. Parent process is still performing the second full read; no whole-process completion/cleanup or repeated-state/disk-restore proof.',
                requests=[first,between],sessions=[saved],captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_live_receipt=str(path))
(out/'full-first-v4-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
private=[]
for p in [Path(first['first_head']['file']),Path(first['prefill_state']['file']),Path(saved['file'])]:
    private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
for name in ['prefill-poll-backoff-v01402-build-v1','prefill-poll-backoff-v01402-source-review-v1']:
    record=json.loads((base/name/'record.json').read_text());assert record['passed'] and not record['active']
    for p in sorted((base/name).iterdir()):
        if p.is_file() and p.suffix in ['.json','.cpp','.diff','.stdout','.stderr']:
            copy(p,out/'checks'/name/p.name)
copy(base/'build_prefill_poll_backoff_v01402_v1.py',out/'controllers/build_prefill_poll_backoff_v01402_v1.py')
copy(Path(__file__),out/'controllers'/Path(__file__).name)
binary=root/'build-sycl-prefill-poll-backoff-v1-20261008/strata'
copy(binary.parent/'include/strata/host_wait.hpp',out/'source/candidate-host_wait.hpp')
orig=root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/include/strata/host_wait.hpp'
copy(orig,out/'source/compiled-original-host_wait.hpp')
copy(binary.parent/'prefill.cpp.o.d',out/'source/prefill.cpp.o.d')
private.append({'file':str(binary),'bytes':binary.stat().st_size,'sha256':digest(binary)})
summary={'active':False,'completed':True,'adopted':False,'full_capacity_sequence_passed':False,
         'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'hardware':'ArcB57010GiB, Ryzen5600X,128GiB; pinned oneAPI2026.1.1/UR0.12/NEO26.31.39395.14',
         'first_full_read':{'input_tokens':262140,'output_tokens':4,'resume_tokens':first['resume_tokens'],
             'verify_windows':first['verify_windows'],'last_physical_cell':262143,
             'saved_consumed_tokens':saved['tokens'],'saved_bytes':saved['bytes'],'kv_layers':13,
             'physical_cells_per_saved_layer':262144,'ignored_saved_tensor_bytes':0,
             'exact_saved_consumed_ids':saved['saved_ids_match_exact_consumed_visible_prefix']},
         'interposed_fresh32k_complete_math_gate_passed':True,
         'parent_status_at_capture':'second fresh full256K read active, no error; no restart/other GPU experiment',
         'pending':['repeated full-image byte equality','actual disk32K/full restore','clipped-tail parity','refusals','later32K parity','owned normal exit and fault audit'],
         'polling_candidate':{'binary_sha256':digest(binary),'gpu_tested':False,'adopted':False,
             'cpu_sanitizer_ownership_deadline_cancellation_checks':True,
             'actual_shadow_header_dependency_verified':True,
             'only_prefill_object_replaced':True,'readiness_query_generation_lifetime_unchanged':True,
             'policy':'first32 failures yield, then request10us host sleep; standard completion query still required'},
         'minimum_performance_input_tokens':32768,'diagnostic_durations_excluded_from_speed':True,
         'source_authority':'Actual compiled overlay plus pinned private layer/generate object replacements. Shadow header applies only to the next prefill TU.',
         'limits':['First full read proves physical bounds and readable full image, not cross-engine numerical accuracy or repeated/disk restoration.',
                   'Synthetic polling counts do not measure CPU utilization, DMA or inference speed.',
                   'Host deadline cannot preempt a blocked runtime status query. The public comment is corrected, with no behavior change to the running binary.'],
         'no_reset_rebind_reboot_service_package_global_change':True}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
print(json.dumps({'archive':str(out),'first_full_read':summary['first_full_read'],'poll_binary_sha256':digest(binary)},indent=2))
