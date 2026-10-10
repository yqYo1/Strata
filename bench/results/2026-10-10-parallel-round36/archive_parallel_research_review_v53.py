"""Closed round36 registry: preserve new reports, reference previous history once."""
from pathlib import Path
import datetime, fcntl, hashlib, json, shutil, subprocess
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A=W/'bench/results/2026-10-10-parallel-round36'
T=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-actual-cohort-timing-20261010')
def ident(p):
    p=Path(p)
    return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2)+'\n')
with (B/'owned-v0141-measurement.lock').open('a')as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prev=R/'report-registry-v52.json'
    assert ident(prev)['sha256']=='2f573bc46f79e78c7fa231a786cdf9ab39fd7ed62781f7349e5b8f83f9dfb900'
    old=json.loads(prev.read_text());assert len(old['reports'])==221
    assert subprocess.check_output(['git','log','-1','--format=%H'],cwd=T,text=True).strip()=='bd01b72ed35e89dd5de9b41abeeaab120ee65c9e'
    assert not A.exists();A.mkdir(parents=True)
    entries=[
      (115,'actual-linked-q8k-contract','CPU','Actual linked custom-Q8K FP/dispatch admission'),
      (116,'actual-linked-iq2s-dot-contract','PP','IQ2_S linked contract and conditional next instruction candidate'),
      (117,'external-cpu-nt1-next-candidate','CPU','External CPU NT1 format-preserving candidates'),
      (118,'external-prefill-next-candidate','PP','External exact-output prefill candidate search'),
      (119,'actual-weight-runtime-admission','CPU','Closed actual-weight Release/ASan admission review'),
      (120,'iq2s-phase-integration-boundary','PP','Explicit prefill/decode selection and physical-context gates'),
      (121,'iq2s-timing-source-admission','CPU','Immutable timing source source-only contract and compile blocker'),
      (122,'iq2s-timing-design','PP','Immutable timing source paired comparison and replication design')]
    reports=list(old['reports']);new=[]
    for n,name,kind,scope in entries:
        p=R/f'round{n}-{name}.txt';x=ident(p)
        dest=A/'research'/p.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest);assert ident(dest)==x
        entry=dict(path=str(p),**x,agent='/root/research_dense_mirror_phase_ownership'if kind=='CPU'else'/root/research_prefill_algorithms_blogs',model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True)
        reports.append(entry);new.append(entry)
    sol=R/'implementation-iq2s-real-cohort-timing-v1.txt';shutil.copyfile(sol,A/sol.name)
    ggml=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned/ggml/src/ggml-quants.c')
    frozen=B/'research-source-snapshots/iq2s-custom-quantizer-d68516f4/pinned-ggml/ggml/src/ggml-quants.c'
    assert ident(ggml)==ident(frozen)
    corrections={
      'R113_R115_R117_R119_hash_transcriptions':dict(reason='Original report hashes for ggml-quants.c are invalid/mistranscribed. Do not copy report text as an identity.',actual_path=str(ggml),frozen_path=str(frozen),identity=ident(ggml),method='Root SHA256 directly from identical current and immutable file bytes'),
      'R122_expert_shuffle':dict(reason='Expert order is fixed; seed rotates only arm offset. R122 sentence claiming expert shuffle is incorrect.',source=str(T/'sycl/tools/native-service-calibration/native_service_calibration.cpp'),source_identity=ident(T/'sycl/tools/native-service-calibration/native_service_calibration.cpp')),
      'R121_compile_blocker':dict(reason='R121 independently found the vector-vs-array order_hash mismatch in immutable v1; root preserved the first failed build then fixed the private helper in 156bef6d. Fresh Release and ASan/UBSan builds pass on corrected immutable v2.'),
      'Sol_timing_handoff':dict(reason='Trait vec_dot has eight arguments; switch is inside the timed region. Original handoff preserved. Root source comments/readme corrected before the source commit.')}
    write(A/'provenance-corrections.json',corrections)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    decisions=dict(register_index_candidate='REJECT after three fresh CPU processes; paired time ratio to trait median across process/cells 1.229..1.318',production_unchanged=True,physical_256K_lifecycle_passed=False,model_inference_run=False,gpu_work_submitted=False,actual_weight_numerical_admission='Release, full ASan/UBSan, default-OFF original correctness/pool, exact bits pass',new_timing_source='Separate Sol source-only; root full review, compile fix, all builds/tests/captures and three serialized samples',next_research='New closed-result CPU instruction hypothesis and exact same-buffer prefill-exp legal/numerical admission gap; avoid repeating rejected register-index candidate or unqualified broad backend switches')
    snapshot=dict(time_utc=now,method='Direct collaboration.list_agents at 2026-10-10 01:04 UTC: both Luna investigators and separate Sol source implementer completed; all latest reports fully read.',agents=[dict(agent='/root/research_dense_mirror_phase_ownership',model='gpt-6-luna',status='completed',latest_round=121),dict(agent='/root/research_prefill_algorithms_blogs',model='gpt-6-luna',status='completed',latest_round=122),dict(agent='/root/implement_repeat_capture_reader_hardening',model='gpt-6.1-sol',status='completed-source-only')],notice='Snapshot at this closed boundary; saved running entries never establish live status. New assignments follow committed evidence, while root continues independent work.')
    registry=dict(registry_version=53,created_utc=now,research_completed=len(reports),reports=reports,previous_committed_registry=dict(path=str(prev),**ident(prev),storage_commit='12a3e1a6297a2815aeb4ff8eda86d5079f26a241'),scope='Four renewed parallel research waves, separate timing source implementation, root actual-weight runtime/linked qualification and three-process rejected-candidate screen',historical_field_policy='Prior registries preserve historical assignment details. This schema carries report inventory and current closed evidence, without repeating stale running/decision fields.',new_reports=new,implementation_handoff=dict(path=str(sol),**ident(sol),model='gpt-6.1-sol',status='completed-source-only',root_full_report_reviewed=True),live_agent_snapshot=snapshot,current_root_decisions=decisions,report_provenance_corrections=corrections,timing_result_commit='bd01b72ed35e89dd5de9b41abeeaab120ee65c9e',qualification_result_commit='906cac75a6f3f48a85e059c64c45393b96a2edc4',recurring_research_policy=dict(models='Multiple gpt-6-luna researchers, distinct scopes; separate gpt-6.1-sol implementation if useful',share='Past R directory, latest committed registry, immutable source pins and prior root decisions in every assignment',intervention='No midtask intervention; fully read completed reports before follow-ups',execution='Root owns all builds, tests, CPU/GPU work, profiling, Git and cleanup under shared lock; research never executes these',repeat='Assign new reviewed-result or source/qualification gaps while root continues independent tuning. Record concrete dependency when there is no new admissible gap.'))
    target=R/'report-registry-v53.json';assert not target.exists();write(target,registry);shutil.copyfile(target,A/target.name)
    write(A/'root-parallel-review-v53.json',dict(created_utc=now,new_reports=8,total_reports=len(reports),registry_identity=ident(target),decisions=decisions,snapshot=snapshot))
    (A/'REPORT.md').write_text('''# Parallel research resumed; actual-weight CPU candidate rejected

The recurring research loop had stopped after the previous assignments completed. Root acknowledged that gap and resumed two read-only gpt-6-luna investigators on distinct questions, four waves /eight new reports R115–R122. A separate gpt-6.1-sol agent implemented a private actual-weight caller-dot timing mode. Researchers received past reports, the latest committed registry, prior decisions and immutable source snapshots. Root continued production CPU build, linked-code review, numerical and sanitizer admission, the compile fix and serialized measurement while research ran. No researcher executed tests or GPU work, and no in-progress assignment was interrupted.

Root fully read all eight reports and the implementation handoff. R115/R116 closed source/linked FP and packed-layout questions; R117/R118 found no new qualified external candidate; R119 independently reviewed returned runtime admission; R120 identified the explicit caller phase policy and actual full-context gates for any future integration. R121 independently found the timing source vector-vs-array hash compile blocker, also caught by the first root build; the failure is preserved and the root private-helper fix passes fresh Release and full ASan/UBSan builds. R122 supplies paired analysis constraints. Report errors are recorded in provenance-corrections.json; originals remain unchanged.

Three fresh CPU timing processes rejected the current register-index candidate: across two selected-expert splits and stream192/hot8 cells, process median paired elapsed-time ratios to the production trait are 1.229–1.318. The direct control also costs more than the trait. These are selected actual model weights with synthetic inputs, not model prefill/decode measurements. Numerical and sanitizer gates pass, but production integration is unnecessary for this slower candidate. There was no GPU/model inference or reset this batch. The full physical 262,144-token lifecycle remains a separate mandatory adoption gate.

The 229-report inventory is retained in registry v53. Earlier report bytes already committed in round35 are referenced instead of copied again, and obsolete historical running fields are left in historical registries. The timestamped agent snapshot at this closed boundary says completed; it does not claim completed agents are running. Recurring follow-ups use new closed-result instruction/algorithm questions and exact prefill buffer/FP qualification gaps while the main agent proceeds independently.
''')
    shutil.copyfile(__file__,A/Path(__file__).name)
    write(A/'archive-file-identities.json',{str(p.relative_to(A)):ident(p) for p in sorted(A.rglob('*')) if p.is_file()})
    print(json.dumps(dict(registry=str(target),**ident(target),completed=len(reports),archive_files=sum(p.is_file()for p in A.rglob('*')))))
