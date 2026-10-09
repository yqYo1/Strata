"""Root reviewed completed reports only; actual live snapshot, not inferred from files."""
from pathlib import Path
import datetime,fcntl,hashlib,json,shutil
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-10-parallel-round35'
CPU=[('native-service-cohort-fit-holdout-first-cell-round82.txt','Exact22/20NT1 first 192/192 routed cohort selection and production shape limitations'),('native-gu22-down20-nt1-external-kernel-candidates-round83.txt','Exact-format NT1 external kernel candidates and known-mechanism limits'),('native-pool-host-pin-construction-order-production-audit-round84.txt','Parent host pin/pool construction order audit; root SYCL-path correction'),('iq2s-nt1-paired-gu-exact-order-prototype-contract-round85.txt','Pinned IQ2S NT1 pair independent integer/FMA/reduction schedule'),('zen3-native-pool-heterogeneous-tail-candidates-round86.txt','Heterogeneous NT task-grain novelty screen; no additional candidate'),('iq2s-nt1-actual-build-isa-static-admission-round87.txt','Actual build static ISA admission; root linked ISA closes remaining gap'),('iq2s-nt1-q8k-legal-input-integer-bounds-round88.txt','Production-quantized ordinary Q8K input domain and saturation/int32 bounds'),('zen3-iq2s-nt1-decoder-instruction-alternatives-round89.txt','External decoder instructions narrowed to known mechanisms; no measured candidate'),('native-q8k-live-domain-aggregate-observation-contract-round90.txt','Diagnostic finite-input domain aggregate contract; root corrects alternate-checkout provenance')]
GPU=[('gdn-chunkwise-wy-config-parity-and-workspace-round82.txt','Actual head mapping/36GDN layers and WY physical-context workspace'),('gdn-wy-bounded-sycl-kernel-feasibility-round83.txt','Bounded WY chunk/persistent-kernel feasibility; not parity or speed'),('gdn-serial-subgroup-remap-exact-order-screen-round84.txt','Known HIP quad pattern adapted to exact-order SYCL subgroup32 candidate'),('gdn-wy-equation-oracle-independent-review-round85.txt','Independent source derivation/review of separate CPU WY oracle'),('gdn-sycl-subgroup32-admission-fallback-contract-round86.txt','Exact named-kernel pre-submit admission and no post-submit fallback'),('xe2-gdn-grf-compile-tuning-evidence-round87.txt','Per-kernel GRF controls/counter evidence; target support unverified'),('gdn-sycl-quad-exact-order-semantic-review-round88.txt','Quad ownership/collective/staging proof and launch attribute correction'),('gdn-quad-existing-parity-coverage-audit-round89.txt','Actual existing keyhead parity gaps for candidate/tails/carry/fullcontext'),('xe2-gdn-quad-profiler-discriminator-evidence-round90.txt','Primary unitrace/VTune discriminators with B570 counter support unknown'),('xe2-llamacpp-b580-new-implementation-transfer-screen-round91.txt','Pinned B580 implementer token-block GDN transfer screen; no speed transfer'),('gdn-token-block-staging-exact-order-transfer-contract-round92.txt','Quad plus token-block staging coverage/SLM/register/race contract'),('xe2-long-context-full-attention-transfer-screen-round93.txt','Long-context full-attention transfer screen; no new exact candidate'),('gdn-quad-v1-independent-source-safety-review-round94.txt','Independent source safety review found failed-drain USM teardown defect; root source fix committed')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prev=R/'report-registry-v47.json';assert sha(prev)=='1aaa7ccf01cb74d3c8441a9467cb7136a8e583b3d9ab8c8ac3767dfc564ef839'
  d=json.loads(prev.read_text());assert d['research_completed']==165
  now=datetime.datetime.now(datetime.timezone.utc).isoformat();existing={Path(x.get('path',x.get('report',''))).name for x in d['reports']}
  for agent,batch in [('/root/research_dense_mirror_phase_ownership',CPU),('/root/research_prefill_algorithms_blogs',GPU)]:
   for name,scope in batch:
    assert name not in existing;src=R/name;target=A/'research'/name;assert src.is_file() and not target.exists()
    shutil.copyfile(src,target);assert sha(src)==sha(target)
    d['reports'].append(dict(path=str(src),sha256=sha(src),bytes=src.stat().st_size,agent=agent,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True))
  d.update(created_utc=now,research_completed=187,previous_committed_registry=dict(path=str(prev),sha256=sha(prev),storage_commit='3acff3b348900f8e9c14910fb3738888286b8194'))
  scopes=[dict(agent='/root/research_dense_mirror_phase_ownership',model='gpt-6-luna',status='running-at-snapshot',round=91,scope='Single exact-order IQ2S NT1 decoder candidate contract and R90 checkout provenance correction',report=str(R/'iq2s-nt1-single-decoder-candidate-contract-round91.txt')),dict(agent='/root/research_prefill_algorithms_blogs',model='gpt-6-luna',status='running-at-snapshot',round=95,scope='Independent review of root drain fix, host async memcpy lifetimes and host-only initialization',report=str(R/'gdn-quad-root-fix-host-lifetime-review-round95.txt'))]
  implementation_handoff=dict(agent='/root/implement_repeat_capture_reader_hardening',model='gpt-6.1-sol',status='completed-source-only',report=str(R/'implementation-gdn-quad-pipeline-v1.txt'),sha256=sha(R/'implementation-gdn-quad-pipeline-v1.txt'),worktree='/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-quad-pipeline-20261010',source_commit='61d2f1a62ee680ba1150f9e52c3c2e71c4612ac4',root_fix_commit='6d67ba1972475cf59c940b48990d95f566abba80',built=False,gpu_executed=False,root_owns_all_build_test_gpu=True)
  d['completed_implementation_handoffs_v48']=[implementation_handoff]
  d['pending_scopes']=scopes
  d['live_agent_snapshot']=dict(time_utc='2026-10-09T22:56:41.889530+00:00',source='collaboration.list_agents directly checked by root before this archive',states=[*[dict(agent=x['agent'],status='running') for x in scopes],dict(agent=implementation_handoff['agent'],status='completed-source-only')],snapshot_only=True)
  d.setdefault('source_and_result_commits',{}).update(corrected_native_CPU_calibration='af12591ca840deb4649b703cbb7469fbc2417fc9',gdn_wy_equation_oracle='e4959ac36d4f84f0e49e4d2f6974fe9ea2e83e70',gdn_quad_source_only='61d2f1a62ee680ba1150f9e52c3c2e71c4612ac4',gdn_quad_root_drain_source_fix='6d67ba1972475cf59c940b48990d95f566abba80')
  evidence={}
  for name in ['native-service-calibration-cpu-build-v3','native-service-calibration-cpu-build-v4','native-service-calibration-22-20-nt1-correctness-tasks6-batch6-r1','native-service-calibration-22-20-nt1-correctness-tasks6-batch6-r2','native-service-calibration-22-20-nt1-correctness-tasks6-batch6-r3',*[f'native-service-calibration-22-20-nt1-timing-tasks{task}-batch6-r{rep}' for task,rep in [(6,1),(0,1),(0,2),(6,2),(6,3),(0,3),(0,4),(6,4)]],'gdn-wy-equation-oracle-cpu-validation-v1','gdn-wy-equation-oracle-cpu-validation-v2','native-iq2s-nt1-linked-isa-audit-v1']:
   p=B/name/'record.json';v=json.loads(p.read_text());assert not v['active'] and not v.get('cleanup',[]) and not v.get('survivors',[])
   evidence[name]=dict(path=str(p),sha256=sha(p),original_passed=v['passed'],active=False,elapsed_seconds=v.get('elapsed_seconds'),timing_interpretation_invalid=('timing-' in name and name[-2:] in ('r1','r2')),gpu_or_model_performance_evidence=False)
  d['root_new_closed_evidence']=evidence
  d['root_decisions_v48']=dict(old_timing_all_workers_cpu0_interpretation_invalid=True,corrected_384_ids_passed=True,corrected_4_fresh_processes_actual_distinct_cpu_placement=True,R46_fit_gates_not_all_passed=True,microbench_neither_task_policy_nor_route_weight_adopted=True,wy_double_synthetic_100_release_and_sanitized_pass=True,wy_does_not_emulate_gpu_fp_or_validate_model=True,new_gpu_run=False,new_full262144_lifecycle_pass=False,quad_source_implementation_pending=False,quad_root_build_and_runtime_validation_pending=True,quad_host_transfer_lifetime_independent_review_pending=True,token_block_staging_next_separate_candidate=True,production_math_thresholds_unchanged=True)
  corrections=[dict(report='native-pool-host-pin-construction-order-production-audit-round84.txt',correction='Report src path alone is insufficient for active SYCL. Root read actual sycl generator: current pool4185/scratch5666, retained869 pool4161/scratch5629; AdaptJobs I/O pin1347 is not main host pin. Source ordering audit only.'),dict(report='gdn-sycl-subgroup32-admission-fallback-contract-round86.txt',correction='R88 direct launch inspection resolves attribute confusion: ordinary pipeline has no reqd SG attr; out norm32/tuned keyhead16. Do not copy adjacent attr assumption.'),dict(report='gdn-chunkwise-wy-config-parity-and-workspace-round82.txt',correction='48 total model layers,36 GDN layers; use persistent36-layer sizing confirmedR83.'),dict(report='gdn-wy-bounded-sycl-kernel-feasibility-round83.txt',correction='Current ordinary pipeline has5 barriers/token; roughly4 wording is inexact, checkedR84/R88.'),dict(report='iq2s-nt1-actual-build-isa-static-admission-round87.txt',correction='Root exact linked3382d73a dot disassembly contains AVX2 pair products/scaled sums/YMM FMA; closes emitted ISA scope only, not full paired operation proof.'),dict(report='xe2-llamacpp-b580-new-implementation-transfer-screen-round91.txt',correction='Token blocking already exists in Strata keyhead TB8. R92 narrows novel experiment to quad gathers plus token-block staging, without copying upstreamSG16 arithmetic.')]
  corrections.append(dict(report='native-q8k-live-domain-aggregate-observation-contract-round90.txt',correction='R90 reports different hashes from an alternate checkout, not a mutation of the assigned pinned dependency. Root rechecked /home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned clean HEAD3cf03257f219afbe7334045ff7c6a06ac68c627d: ggml/src/ggml-quants.c07143d7068936ae46b3c528b2f3d4bbb666e74d88992165716174d243573965d; ggml/src/ggml-cpu/ggml-cpu.c7b451e6528cec3bc27bfd812a1314859c92d22a00c2db371d543cde5b7d5ca94; x86/quants.c3c489fdc77e3ab484a5f188bff9c60c0e9483c951ed65c1a562a04c249d5628e. Original R90 preserved, followup asks exact alternate path.'))
  target=R/'report-registry-v48.json';assert not target.exists();target.write_text(json.dumps(d,indent=2)+'\n');shutil.copyfile(target,A/target.name)
  root_review=dict(time_utc=now,registry_sha256=sha(target),new_completed_research_reports=len(CPU)+len(GPU),total_completed_research_reports=187,actual_live_snapshot=d['live_agent_snapshot'],pending_scopes=scopes,completed_implementation_handoff=implementation_handoff,evidence=evidence,decisions=d['root_decisions_v48'],corrections=corrections,root_owns_all_serialized_validation=True,shared_lock=str(B/'owned-v0141-measurement.lock'),no_mid_task_intervention=True,report_retention='Compact source/results in commits; no model blobs; successful raw duplicate logs reviewed after closed run')
  (A/'root-parallel-review-v48.json').write_text(json.dumps(root_review,indent=2)+'\n')
  shutil.copyfile(__file__,A/Path(__file__).name)
  p=A/'REPORT.md';p.write_text(p.read_text()+'''

## Reviewed batch v48 and parallel continuation

Root fully reviewed22 new Luna reports (CPU82–90, GPU82–94);187 are now
completed and archived with model, scope, hashes and status. Every assignment
shared the whole past-report directory and the then-latest committed v47. At
this boundary root directly checked two Luna researchers running on distinct
new CPU single-decoder contract / GPU root-fix and host-lifetime review scopes.
The separate Sol source-only quad implementation has completed its handoff. These are timestamped live
snapshots, not permanent claims that a saved assignment is still running.

Main work continued in parallel: a frozen384 real-weight ID cohort passed
native/GGML correctness. Root invalidated four original timings because the
fixture pinned the parent before pool creation and all workers inheritedCPU0;
fixed the order, added actual singleton placement guards, and collected four
fresh25-round-per-arm/cohort processes on six distinct physical CPUs. The
homogeneous22/20NT1 microbench tended to favor18 over6 tasks, but fit/holdout
gates did not all pass and production callback shapes differ. No engine task
policy, route-service cost model or speed claim was adopted. Every original
status, failure, individual sample, invalidation and bootstrap result is
committed in native-role worktree af12591c. Root also closed the emitted-ISA
question by inspecting the exact current IQ2S NT1 linked dot.

A separate independent GDN/WY equation oracle passed100 synthetic cases in
both release and ASan/UBSan (GNU13.3), with unchanged math bounds. The first
sanitizer compile's per-file limit failure and the corrected compiler-only
budget remain separate receipts. Source, complete case results, actual compiler
flags and root supervisors are in oracle commit e4959ac3. This does not emulate
GPU native math or validate model/full262144 lifecycle. No GPU ran in this batch.

The source-only Sol completed an opt-in SG32 quad pipeline that preserves
ascending32-row FMA partials and ordered four-way sums. Root will own its build,
small logged diagnostic, exact GPU parity, long-prompt/repeat/physical262144
and quiet repeated timing gates. Token-block input staging is a separate later
candidate, not folded into the quad comparison. Existing Strata keyhead already
stages8 tokens; external SG16 arithmetic is not Strata bitwise proof. No source
adoption or production numerical threshold/default has changed.

Quad source handoff is committed at61d2f1a; R94 independent review identified
explicit USM release after failed queue wait despite unknown command completion.
Root source fix6d67ba19 skips that release, preserves failure, and adds a host-only
pre-queue contract mode. These have not been built or run. Followup R95 reviews
host-side asynchronous memcpy staging/receive lifetimes too before GPU execution.
R90's alternate GGML checkout hashes do not describe the actual pinned dependency:
root verified its clean3cf03257 and original hashes; original report is preserved.
''')
  identity=A/'archive-file-identities.json'
  original=json.loads(identity.read_text());assert isinstance(original,dict) and 'files' not in original
  entries={}
  for p in sorted(A.rglob('*')):
   if p.is_file() and p!=identity:entries[str(p.relative_to(A))]=dict(sha256=sha(p),bytes=p.stat().st_size)
  assert set(original)<=set(entries)
  for n,v in original.items():
   if n!='REPORT.md':assert v==entries[n],n
  identity.write_text(json.dumps(entries,indent=2)+'\n')
  print(json.dumps(dict(registry=str(target),sha256=sha(target),research_completed=187,archive_files=len(entries),live_researchers=2,live_implementer=0,completed_source_implementations=1)))

if __name__=='__main__':main()
