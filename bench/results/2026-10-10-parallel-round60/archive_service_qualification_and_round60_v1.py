from pathlib import Path
from collections import Counter
import datetime,fcntl,hashlib,json,re,shutil,subprocess
B=Path(__file__).parent;R=B/'research-20261009'
M=Path('/home/yayoi/ghq/github.com/yqYo1/Strata')
W=M/'.worktree/docs-sycl-storage-retention-20261009'
Q=M/'.worktree/diag-sycl-prefill-service-qualification-20261010'
N=M/'.worktree/diag-sycl-native-service-capacity-20261010'
A=W/'bench/results/2026-10-10-parallel-round60'
C=W/'bench/results/2026-10-10-hardware-concurrency'
QA=Q/'bench/results/2026-10-10-prefill-service-device-qualification'
def ident(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def copy(p,target):
 assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert ident(p)==ident(target)
def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def commit(root,paths,message):
 for args in [('diff','--check'),('add',*paths),('diff','--cached','--check'),('commit','-m',message),('push',),('rev-parse','HEAD')]:subprocess.run(['git',*args],cwd=root,check=True)
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert git(W,'rev-parse','HEAD')=='66ce79d37eef5e339438c728e29388730ce10b48' and not git(W,'status','--porcelain')
 assert git(Q,'rev-parse','HEAD')=='7c5ad75a45c7e188ecef7a947b49b829dc5c5710' and not git(Q,'status','--porcelain')
 assert git(N,'rev-parse','HEAD')=='af12591ca840deb4649b703cbb7469fbc2417fc9'
 assert git(N,'status','--porcelain')=='M sycl/tools/native-service-calibration/native_service_calibration.cpp'
 assert ident(N/'sycl/tools/native-service-calibration/native_service_calibration.cpp')['sha256']=='f062f24ea57443c4cd59d09d2b56d498165ea497254ae9fa8998375b062b2ac3'
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for folder,name in [('prefill-service-qualification-cpu-build-v1','build'),('prefill-service-small-device-qualification-v1','device')]:
  p=B/folder;r=json.loads((p/'record.json').read_text());assert r['passed'] and r['complete'] and not r['active']
  for cmd in r['commands']:
   assert cmd['normal_exit'] and cmd['session_empty'] and cmd['observation_complete'] and cmd['exit_code']==0 and not cmd['errors'] and not cmd['cleanup'] and not cmd['survivors']
   for o in cmd['owners']:
    stat=Path('/proc')/str(o['pid'])/'stat'
    if stat.exists():assert int(stat.read_text().rsplit(')',1)[1].split()[19])!=o['start_ticks']
  copy(p/'record.json',QA/(name+'-record.json'))
  if name=='device':
   assert not r['visible_kernel_GPU_entries'] and not r['devcoredump_after']
   copy(p/'qualification.stdout',QA/'qualification.stdout')
   for filename in ['kernel-cursor.stdout','kernel-cursor.stderr','kernel-interval.stdout','kernel-interval.stderr']:copy(p/filename,QA/filename)
   trace=p/'qualification.stderr';text=trace.read_text();lines=text.splitlines()
   ur=Counter(re.findall(r'-> (UR_RESULT_[A-Z_]+)',text));assert ur=={'UR_RESULT_SUCCESS':210}
   errors=re.findall(r'ERROR \(([^)]+)\) in (\w+)',text);assert errors==[('2013265955','zeCommandListIsGraphCaptureEnabledExt')]*2
   header=Path('/usr/include/level_zero/ze_api.h');enum=next(line for line in header.read_text().splitlines() if 'ZE_RESULT_QUERY_FALSE =' in line);assert '0x78000023' in enum and int('78000023',16)==2013265955
   windows=[]
   for i,line in enumerate(lines):
    if 'ERROR (' in line:windows.append(dict(first_line=i,lines=lines[max(0,i-2):i+2]))
   write(QA/'diagnostic-trace-summary.json',dict(original_path=str(trace),**ident(trace),lines=len(lines),UR_results=dict(ur),LevelZero_numeric_queries=errors,local_header=dict(path=str(header),**ident(header),enum_line=enum),
    interpretation='Both logger ERROR lines are ZE_RESULT_QUERY_FALSE graph-capture checks; each corresponding UR query returnsSUCCESS with result0. They are not failed submissions or a GPU fault. No unexpected UR return is present.',
    primary_enum_reference='https://github.com/oneapi-src/level-zero/blob/master/include/ze_api.h',relevant_windows=windows,first_lines=lines[:4],last_lines=lines[-8:]))
 copy(B/'build_prefill_service_qualification_v1.py',QA/'build_prefill_service_qualification_v1.py')
 copy(B/'run_prefill_service_small_qualification_v1.py',QA/'run_prefill_service_small_qualification_v1.py')
 (QA/'README.md').write_text('''# Returned-event SYCL build and small B570 qualification

The root rebuilt the changed prefill and Gemm translation units, replaced exactly two archive members and linked both the private engine and a small qualification against byte-pinned cached numerical libraries. The cached base is7d0105f2; diagnostic source is7c5ad75a. Precise, SG32, per-kernel split and sequential oneMKL flags are recorded in the build receipt. This is an isolated diagnostic rebuild, not a fresh rebuild of every dependency.

The first B570 device check used flushed UR/Level Zero diagnostics, finite ownership/deadline/log bounds and the previously qualified safety flags. It passed full119float output/padding equality for T7,N11,K13,ldy17 with FP16 operands and FP32 output, both returned-event and unchanged f16 wrappers, completion, monotonic submit/start/end, normal0 exit and fresh fault checks. No model was run. All210 observed UR results areSUCCESS. The two numeric Level Zero graph-capture query-false returns are explained with exact header/UR context in the compact trace summary; the logger's wordERROR alone is not a submission error.

The diagnostic's original host-contract v1 remainsFAILED and its corrected v2 remainsPASS. This small device result qualifies layout/type/completion and profiling fields. It does not qualify the complete oneMKL internal-kernel interval, actual32K route service, clean model speed, full physical262144 positions, or an optimization. Those remain separate gates. Successful verbose API history can be retired after this compact evidence is committed.
''')
 write(QA/'archive-file-identities.json',{str(p.relative_to(QA)):ident(p) for p in sorted(QA.rglob('*')) if p.is_file()})
 commit(Q,['bench/results/2026-10-10-prefill-service-device-qualification'],'test(sycl): qualify returned-event GEMM on B570')
 # Source-only native extension is reviewed separately, before any numeric run.
 NA=N/'bench/results/2026-10-10-native-service-capacity-source'
 copy(R/'implementation-native-service-capacity-v2.txt',NA/'implementation-native-service-capacity-v2.txt')
 write(NA/'root-admission.json',dict(created_utc=now,source=ident(N/'sycl/tools/native-service-calibration/native_service_calibration.cpp'),root_changed_source_and_handoff_read=True,
  scope='NT1–4 actual-payload hot/streaming fixture only; production kernels unchanged',tested=False,adopted=False,
  predeclared_calibration_gates=dict(NT1='Existing exact GGML controls',GU=dict(maxabs=1e-4,nrms=1e-5),Down=dict(maxabs=1e-4,nrms=1e-5),full_chain=dict(maxabs=1e-3,nrms=1e-4)),
  gate_rationale='Owner chooses strict bounded fixture error budgets before anyNT2–4 run: stages reject >1e-4 absolute or1e-5 normalized RMS; independently requantized nonlinear full chain rejects >1e-3 absolute or1e-4 normalized RMS. These are calibration admission limits, not a proof of exact equality, model sensitivity or model-adoption tolerances. Do not broaden to fit any failed result; preserve failures and investigate.',
  next='Root owns serial build, parser/formula checks, untimed actual384-payload references and separate fresh timing. No permission round is required for already-authorized read-only measurement.'))
 for args in [('diff','--check'),('add','sycl/tools/native-service-calibration/native_service_calibration.cpp','bench/results/2026-10-10-native-service-capacity-source'),('diff','--cached','--check'),('commit','-m','test(bench): bound native NT1–4 capacity controls'),('push','-u','origin','diag/sycl-native-service-capacity-20261010'),('rev-parse','HEAD')]:subprocess.run(['git',*args],cwd=N,check=True)
 prev=R/'report-registry-v82.json';assert ident(prev)['sha256']=='726b6d305b57b7e13b822180b3aeaa1f08374bfc215b822089d36c2540bf9a64'
 d=json.loads(prev.read_text());assert len(d['reports'])==331;new=[]
 for name,agent,scope,digest in [
  ('round225-hardware-concurrency-profile-evidence-audit.txt','research_bottleneck_evidence_audit_v203','Closed raw/compactPTI event, engine and concurrency evidence audit','c4d4cb547bdf628de527fb2cbd92731ff7b038c107314cf2f2282a65c3dfafa2'),
  ('round226-ple-readonly-capacity-probe-contract.txt','research_native_iq4nl_esimd_k640_v202','Source-matched bounded PLE page/storage capacity control',None)]:
  source=R/name;identity=ident(source)
  if digest:assert identity['sha256']==digest
  copy(source,A/'research'/name);entry=dict(path=str(source),**identity,agent='/root/'+agent,model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True)
  d['reports'].append(entry);new.append(entry)
 correction=dict(R225_correction_provenance='Original report names R/round59-root-review-corrections.txt and digest a1554..., but that file is absent. Actual supplied/root-reviewed R223 correction is committed round59/root-review.json and registryv82 root_review_corrections_round59. Preserve the original report and correct this unsupported path/hash attribution; its empirical trace reconciliation is separately supported.',
  R225_precision='Epoch double-microsecond serialization does not retain nanosecond accuracy at this epoch; the root1us guard is only a conservative serialization check, not clock calibration.Observed5.084us minimum gap andzero intersection remain narrow trace observations.',
  R226_one_page_bound='Its roughly2GiB for32K is expressly a one-page-per-row bound.8KiB straddling jobs can exceed that; actual source job count/bytes must be used, no universal2GiB maximum.',
  R226_authorization='Read-only bounded measurements are already authorized by the user. The report wording future separately authorized creates no new permission requirement.')
 d.setdefault('root_decision_history',[]).append(dict(at_utc=now,prior_current_root_decisions=d.get('current_root_decisions')))
 d.update(registry_version=83,created_utc=now,research_completed=333,new_reports=new,
  previous_committed_registry=dict(path=str(prev),**ident(prev),storage_commit='66ce79d37eef5e339438c728e29388730ce10b48'),
  scope='Independent complete capacity trace audit and storage probe contract; root serial returned-event device qualification and native capacity-source review',
  live_agent_snapshot=dict(time_utc=now,method='Actual collaboration.list_agents plus completed Sol handoff',agents=[dict(agent='/root/research_bottleneck_evidence_audit_v203',model='gpt-6-luna',round=225,status='completed'),dict(agent='/root/research_native_iq4nl_esimd_k640_v202',model='gpt-6-luna',round=226,status='completed'),dict(agent='/root/implement_prefill_phase_timer_validity',model='gpt-6.1-sol',status='completed-source-only')]),
  current_root_decisions=dict(copy_GEMM='Independent audit agrees narrow144/3168zero-intersection/recordedengine observation; raw consumer resolved after archival.',
   production_service='Isolated2TU SYCL build and small119float B570 qualificationPASS, no model speed or full oneMKL span/full262K claim.',
   CPU_capacity='Newsource-only NT1–4/hot8fixture committed'+git(N,'rev-parse','HEAD')+'; predeclared numerical gates, compile/run pending.',
   PLE_capacity='Actual reader is blockingpread pool; matched sequential/random/source-page/real-reader controls remain unmeasured.',adopted=False))
 d['root_review_corrections_round60']=correction
 reg=R/'report-registry-v83.json';assert not reg.exists();write(reg,d);copy(reg,A/reg.name)
 write(A/'root-review.json',dict(created_utc=now,reports=new,registry=ident(reg),corrections=correction,qualification_source_archive_commit=git(Q,'rev-parse','HEAD'),native_fixture_source_commit=git(N,'rev-parse','HEAD')))
 copy(Path(__file__),A/Path(__file__).name)
 (A/'README.md').write_text('''# Hardware capacity research review and qualification

R225 independently reconciles the closed PTI trace to the source operation counts and compact event table; it supports only zero observed copy/GEMM intersection under recorded safe settings, not production serialization cause or physical engine saturation. R226 supplies a concrete source-matched read-only PLE capacity control; no SSD result is claimed. Root corrections preserve original reports and fix R225's unsupported correction-file attribution and clarify time precision, straddling pages and existing user authorization.

Root concurrently rebuilt the production returned-event diagnostic and passed a bounded first B570 layout/completion/profiling-field check. The separate Sol source-only native NT1–4/hot-eight fixture was fully reviewed and committed with predeclared calibration gates; compile and actual-payload numerical/timing work remain root-owned. No production optimization or physical262144-context pass follows.
''')
 limits=C/'HARDWARE_LIMITS.md';text=limits.read_text();text+='\n\nThe separate diagnostic source7c5ad75a subsequently passed the root-owned isolated SYCL build and119float B570 layout/completion/profiling-field check. Exact receipts are committed on `diag/sycl-prefill-service-qualification-20261010`; production oneMKL internal-kernel span, matched32K route service and physical262144 positions remain unqualified.\n';limits.write_text(text)
 write(A/'archive-file-identities.json',{str(p.relative_to(A)):ident(p) for p in sorted(A.rglob('*')) if p.is_file()})
 commit(W,['bench/results/2026-10-10-parallel-round60','bench/results/2026-10-10-hardware-concurrency/HARDWARE_LIMITS.md'],'docs(sycl): review capacity evidence and storage measurement contract')
 print(json.dumps(dict(registry=str(reg),**ident(reg),completed=333,native_commit=git(N,'rev-parse','HEAD'),qualification_commit=git(Q,'rev-parse','HEAD'),storage_commit=git(W,'rev-parse','HEAD'))))
