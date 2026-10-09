"""Root reviewed reports; preserve historical status and capture live state separately."""
from pathlib import Path
import datetime, fcntl, hashlib, json, shutil
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-10-parallel-round35'
CPU=[
('iq2s-nt1-single-decoder-candidate-contract-round91.txt','Index-only candidate contract, scalar grid reads and independent arithmetic'),
('iq2s-index-spread-lane-and-language-proof-round92.txt','Byte/index lane and C++ intrinsic/language proof'),
('zen3-iq2s-index-only-codegen-counter-contract-round93.txt','Zen3 static codegen and minimal model-specific counter contract'),
('iq2s-prototype-three-arm-attribution-review-round94.txt','Baseline/control/index comparison attribution and shared adaptations'),
('zen3-local-pmu-metadata-admission-round95.txt','Observed5600X/PMU metadata, perf policy and correct Model21 scope'),
('iq2s-mixed-lane-proof-and-offline-codegen-round96.txt','Bijective mixed-lane/alignment coverage and offline codegen contract'),
('iq2s-precise-fp-and-ub-qualification-round97.txt','Precise contraction, MXCSR, admitted integer bounds and nonfinite distinction'),
('iq2s-index-only-real-cohort-reader-contract-round98.txt','Existing384-ID actual GU reader and IQ4NL Down640 format correction'),
('iq2s-index-only-actual-linked-codegen-round99.txt','Actual index stack materialization and extra vector stack work; timing hold'),
('iq2s-actual-linked-q8k-quantizer-fp-order-round100.txt','Actual reference quantizer multiply/add order versus custom production quantizer gap'),
('iq2s-register-index-independent-source-review-round101.txt','Independent exact register-return source/ISA/lane and intrinsic store caveat review'),
('iq2s-intelllvm-intrinsic-store-alias-contract-round102.txt','Exact IntelLLVM packed may_alias intrinsic-store contract closes compiler-extension alias caveat')]
GPU=[
('gdn-quad-root-fix-host-lifetime-review-round95.txt','Independent review found ordinary host vectors unwinding before queue completion'),
('gdn-quad-fail-stop-wrapper-independent-review-round96.txt','Independent review of root no-unwind host-endpoint fix'),
('gdn-quad-production-selector-phase-contract-round97.txt','Source production prefill selector versus independent decode step paths'),
('iq2s-index-only-prototype-independent-source-review-round98.txt','Independent standalone CPU source review and mixed-byte gap'),
('gdn-quad-observed-spill-grf256-contract-round99.txt','Actual named spill5056/reference0, runtime-cost and target-support unknown'),
('bmg-grf256-target-and-native-artifact-support-round100.txt','BMG general feasibility versus exact installed-property support gap'),
('gdn-quad-exact-order-spill-source-variants-round101.txt','Exact-order liveness source variants and corrected v2 target-image inventory'),
('gdn-quad-quiet-prefix-independent-review-round102.txt','Independent new quiet timing lifetime/pairing/service-interval review'),
('bmg-installed-igc-neo-register-contract-round103.txt','Installed compiler path ambiguity and exact target property support gap'),
('gdn-quad-quiet-order-and-admission-discriminator-round104.txt','Actual quiet order effect and bounded host admission/device event discriminator'),
('gdn-executable-bundle-phase-reuse-contract-round105.txt','Bundle reuse ownership contract; live source pins were concurrent edits, not claimed commit')]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prev=R/'report-registry-v48.json'; assert sha(prev)=='18a8e56913a3970f278210e7b6176c95ad82913c5d617368d6229c13e43539ac'
  d=json.loads(prev.read_text()); assert d['research_completed']==187
  now=datetime.datetime.now(datetime.timezone.utc).isoformat()
  existing={Path(x.get('path',x.get('report',''))).name for x in d['reports']}
  for agent,batch in [('/root/research_dense_mirror_phase_ownership',CPU),('/root/research_prefill_algorithms_blogs',GPU)]:
   for name,scope in batch:
    assert name not in existing; p=R/name; target=A/'research'/name; assert p.is_file() and not target.exists()
    shutil.copyfile(p,target); assert sha(p)==sha(target)
    d['reports'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,agent=agent,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True))
  supplement=R/'native-q8k-live-domain-observation-provenance-correction-round90-addendum.txt'
  shutil.copyfile(supplement,A/'research'/supplement.name)
  d['supplements_v49']=[dict(path=str(supplement),sha256=sha(supplement),counted_as_new_research_report=False)]
  snapshot=json.loads((B/'parallel-live-snapshot-v49.json').read_text())
  d.update(created_utc=now,research_completed=210,previous_committed_registry=dict(path=str(prev),sha256=sha(prev),storage_commit='dbd17e2387dc452a1ce7dae5e8f7746a019a0a7d'),live_agent_snapshot=snapshot)
  d['pending_scopes']=snapshot['assignments']
  d['registry_version']=49
  d['snapshot_utc']=snapshot['time_utc']
  d['running_assignments']=snapshot['assignments']
  d['root_current_work']='Fresh CPU register-return release/sanitizer/codegen gate; archived quiet32768 GPU component comparison rejects current quad as slower; all execution remains root-serialized'
  d['historical_field_notice']='Older round/live/receipt fields are retained history; only v49 live_agent_snapshot and pending_scopes describe this timestamp.'
  d['completed_implementation_handoffs_v49']=[
   dict(model='gpt-6.1-sol',report=str(R/'implementation-iq2s-nt1-index-spread-v1.txt'),sha256=sha(R/'implementation-iq2s-nt1-index-spread-v1.txt'),source_commit='a9c09660304bc5dc3deb8cf8da10f1f431a9ffe1',root_test_result_commit='1e35586ce7cf2c9f493cd7a34a44137bd500a34b'),
   dict(model='gpt-6.1-sol',report=str(R/'implementation-gdn-quad-quiet-prefix-timing-v1.txt'),sha256=sha(R/'implementation-gdn-quad-quiet-prefix-timing-v1.txt'),source_commit='fdf4b3c52a68f8d925afb3a6519be4ce0b74156f',original_report_status='source-only; owner evidence separately recorded',root_result_commit='a581ebb05ff52530ba41107224a0308c4c7b9fd1'),
   dict(model='gpt-6.1-sol',report=str(R/'implementation-iq2s-register-index-extraction-v1.txt'),sha256=sha(R/'implementation-iq2s-register-index-extraction-v1.txt'),source_commit='22064c0b26232b0964afd70e7d9fe47535b39224',original_report_status='source-only; fresh root release/sanitizer/codegen pending separately')]
  evidence={}
  for name in ['gdn-quad-cpu-build-v1','gdn-quad-cpu-build-v2','gdn-quad-initial-gates-v1','gdn-quad-prefix-carry-gates-v1','iq2s-index-spread-cpu-validation-v1','iq2s-index-spread-cpu-validation-v2','gdn-quad-quiet-cpu-build-v1','gdn-quad-quiet-gates-v1','iq2s-register-index-cpu-validation-v1','iq2s-register-index-cpu-validation-v2']:
   p=B/name/'record.json'
   if not p.exists():continue
   v=json.loads(p.read_text());
   if name.startswith('iq2s-register-index-cpu-validation-') and v['active']:continue
   assert not v['active'] and not v['survivors']
   evidence[name]=dict(path=str(p),sha256=sha(p),original_complete=v['complete'],original_passed=v['passed'],cleanup=v['cleanup'],survivors=v['survivors'],performance_eligible=v.get('performance_eligible',False),component_timing_validated=v.get('component_timing_validated',False),model_opened=v.get('model_opened',False),adopted=v.get('adopted',False))
  d['root_new_closed_evidence_v49']=evidence
  corrections=[
   dict(report='iq2s-index-spread-lane-and-language-proof-round92.txt',correction='Final second-half low read is qs-relative28..35, not qs-relative30; block-relative offset30 includes the fp16 prefix.'),
   dict(report='iq2s-prototype-three-arm-attribution-review-round94.txt',correction='Only control vs index isolates index source; trait baseline vs control includes ABI, width guard and sign memcpy, not a call-only change.'),
   dict(report='zen3-local-pmu-metadata-admission-round95.txt',correction='ActualCPU5600X family19h/model21h/B0; Model00h-0Fh errata do not apply. perf paranoid4/caps0 is metadata, not actual perf_event_open test.'),
   dict(report='iq2s-index-only-prototype-independent-source-review-round98.txt',correction='Root first actual build found cstdint first inclusion within namespace created nested std; global preinclude fixed it. Original failed receipt preserved, source-only review did not discover compile defect.'),
   dict(report='bmg-grf256-target-and-native-artifact-support-round100.txt',correction='R101 corrects inventory: real v2 executable has packaged SYCL target image; no standalone final native ISA. Public table omission is not proof BMG hardware cannot support256.'),
   dict(report='gdn-quad-exact-order-spill-source-variants-round101.txt',correction='Output FMA and state update share a source loop; changing its pragma does not isolate output-only unroll without another structural change.'),
   dict(report='gdn-quad-quiet-prefix-independent-review-round102.txt',correction='DPCT check() is no-op; wait_and_throw is actual completion/error boundary. Root corrected README wording and failed-v1 recipe path.'),
   dict(report='iq2s-index-only-real-cohort-reader-contract-round98.txt',correction='Root task incorrectly shorthand DownK8192. Actual selected Down20 is IQ4NL/Q8_0 K640, Gate/Up22 IQ2S K2560;8192 only standalone synthetic width.')]
  corrections.append(dict(report='iq2s-actual-linked-q8k-quantizer-fp-order-round100.txt',correction='Complete reference quantizer capture was closed but not committed when researcher wrote committed; first archived at22064c0b source handoff. Custom Strata AVX2 and production worker MXCSR remain unproved.'))
  corrections.append(dict(report='bmg-installed-igc-neo-register-contract-round103.txt',correction='Root actual quiet second-process maps and post-close inode/hash pins identify loaded /usr/local IGC2.41.5, igdfcl2.41.5, clang22.1 for that process only. Target GRF256 property support remains unknown.'))
  corrections.append(dict(report='gdn-executable-bundle-phase-reuse-contract-round105.txt',correction='Reported kernels15bd.../headerd9b... pins are live concurrent Sol diagnostic edits, not fdf4b3c5 committed blobs. Bundle design reasoning may be useful, but report is not exact committed-source audit. Root froze git blobs kernelsaf16.../header044d... in research-source-snapshots/gdn-quad-quiet-fdf4b3c5 and followup must use these; original report preserved.'))
  d['root_decisions_v49']=dict(quad_short_72_calls_and_synthetic32768_262144_prefix_passed=True,model_full_context_lifecycle_passed=False,model_performance_or_default_adoption=False,quad_spill5056_reference0_runtime_cost_not_inferred=True,unsupported_grf_property_disabled=True,iq2s_release_and_intelllvm_sanitized_96_profiles_passed=True,iq2s_actual_linked_codegen_captured=True,iq2s_extra_materialized_index_and_accumulator_stack_work_observed=True,iq2s_performance_adoption_held=True,root_owns_all_serialized_validation=True,quad_quiet32768_pairs=10,quad_all10_pairs_slower=True,quad_paired_ratio_min=1.6751428814599496,quad_paired_ratio_max=2.9606863005395985,quad_order_stratified_median_ratios={'legacy-first':1.7020588536027552,'quad-first':2.8866668995177736},quad_current_candidate_adoption_rejected=True,quad_service_includes_per_chunk_admission_and_norm_and_completion=True,quad_order_effect_cause_unresolved=True,quad_component_not_model_timing=True,iq2s_register_new_release_and_sanitizer_passed=True,iq2s_register_all96profile_rows_identical=True,iq2s_register_new_linked_candidate_frame_bytes=32,iq2s_register_control_frame_bytes=40,iq2s_register_old_index_stack_roundtrips_removed_observed_by_root=True,iq2s_register_performance_unmeasured=True)
  target=R/'report-registry-v49.json'; assert not target.exists(); target.write_text(json.dumps(d,indent=2)+'\n');shutil.copyfile(target,A/target.name)
  review=dict(time_utc=now,registry_sha256=sha(target),new_completed_research_reports=23,total_completed_research_reports=210,actual_live_snapshot=snapshot,evidence=evidence,corrections=corrections,decisions=d['root_decisions_v49'],source_result_commits=dict(quad_diagnostic_qualification='247025fc576abfe764f797da1a1c78035497ee17',quad_quiet_source='fdf4b3c52a68f8d925afb3a6519be4ce0b74156f',iq2s_qualification='1e35586ce7cf2c9f493cd7a34a44137bd500a34b',quad_quiet_actual_result='a581ebb05ff52530ba41107224a0308c4c7b9fd1',iq2s_register_source='22064c0b26232b0964afd70e7d9fe47535b39224'))
  shutil.copyfile(B/'research-source-snapshots/gdn-quad-quiet-fdf4b3c5/manifest.json',A/'gdn-quiet-committed-source-snapshot-manifest.json')
  (A/'root-parallel-review-v49.json').write_text(json.dumps(review,indent=2)+'\n');shutil.copyfile(__file__,A/Path(__file__).name)
  p=A/'REPORT.md';p.write_text(p.read_text()+'''

## Reviewed batch v49: parallel research and root validation

Root fully read23 further Luna reports (CPU91–102/GPU95–105);210 reports are archived. Original source-only reports and failed receipts remain unchanged. Live status is a separately timestamped direct collaboration snapshot, not inferred from report files. Every new assignment shares the past-report directory and the latest committed registry. Separate Sol handed off CPU index-only and quiet GDN fixtures; root reviews and owns their serialized tests.

The GDN short72 positive calls and synthetic32768/262144 carried prefixes passed bitwise full state/FP32/FP16 and guards with normal closure. This is not actual model/full-context lifecycle. The candidate's compiler spill5056 versus reference0 is a resource observation; experimental undocumented GRF256 remains off. Quiet timing uses >=32768 prefixes, full warmups, alternating paired orders, admission-inclusive host service and complete paired checks. Both fresh32768 processes completed with five paired samples each. All10 candidate samples were slower: paired quad/reference ratio1.6751–2.9607, order-stratified median1.7021 (legacy-first) and2.8867 (quad-first). Current quad adoption is rejected. The pronounced order effect and host admission versus device-work cause remain unresolved; service includes per-chunk admission and norm/completion, so it is not kernel-only or whole-model timing. Actual result commit a581ebb05ff52530ba41107224a0308c4c7b9fd1 includes individual CSV values and loaded JIT library pins.

The CPU source's first root build found a header namespace defect missed by static reviews. Root fixed global cstdint inclusion, added mixed/alignment coverage and MXCSR logging, and passed release plus IntelLLVM ASan/UBSan:96 profiles/1536GU pairs/9216dots,4194304 extra mixed index comparisons and1048576 vector checks. Offline linked symbols expose extra stack materialization/accumulator traffic in the index candidate; correctness does not justify adopting it as a performance improvement. Independent R99 confirmed the emitted-code finding; separate Sol register-return revision is source commit22064c0b26232b0964afd70e7d9fe47535b39224 and has now passed fresh release and IntelLLVM ASan/UBSan with all96 profile rows unchanged. Root fully read the new candidate/control instructions: candidate frame32/control40, old materialized-index roundtrips and accumulator spills absent, intentional32B sign scratch retained. Independent new-codegen review and performance/actual-weight/model gates remain separate. First fresh qualification was interrupted by a /proc process-exit observation race during sanitizer build; its failed receipt and owned cleanup remain unchanged, corrected fresh run passed80.447s. R100 resolves reference quantizer multiply/add contraction in the old linked binary, but custom production quantizer and worker MXCSR remain outside that evidence.

The source cohort contract corrected root's Down width assumption: selected Down20 is IQ4NL/Q8_0 K640; Gate/Up22 is IQ2S K2560. Synthetic IQ2S width8192 is not the actual Down geometry. Nonfinite results and live/ordinary input scope remain separate. Production defaults/math thresholds, previous invalid timing classifications and original failure statuses remain unchanged. Root continued actual builds, tests and measurement preparation while the separate researchers worked.
''')
  identity=A/'archive-file-identities.json';old=json.loads(identity.read_text());assert isinstance(old,dict) and 'files' not in old
  new={str(p.relative_to(A)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(A.rglob('*')) if p.is_file() and p!=identity}
  assert set(old)<=set(new)
  for n,v in old.items():
   if n!='REPORT.md':assert v==new[n],n
  identity.write_text(json.dumps(new,indent=2)+'\n')
  print(json.dumps(dict(registry=str(target),sha256=sha(target),completed=210,archive_files=len(new))))

if __name__=='__main__':main()
