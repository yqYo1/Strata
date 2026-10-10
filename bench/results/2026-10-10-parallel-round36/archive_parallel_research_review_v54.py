"""Preserve returned post-screen/domain/lifecycle reviews and current source/build state."""
from pathlib import Path
import datetime,fcntl,hashlib,json,shutil,subprocess
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');R=B/'research-20261009'
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009');A=W/'bench/results/2026-10-10-parallel-round36'
def ident(p):return dict(bytes=Path(p).stat().st_size,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
with (B/'owned-v0141-measurement.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 prev=R/'report-registry-v53.json';assert ident(prev)['sha256']=='7f7b02911fa73b03788fe75795f9c9e1c4abeffdb1b3eb1af7621c8b6d898cbc';d=json.loads(prev.read_text());assert len(d['reports'])==229
 entries=[(123,'post-rejection-iq2s-next-candidate','CPU','Closed CPU rejection and conditional sign-scratch priority'),(124,'gdn-gate-exp-qualification','PP','Exact prefill factorization producer/all-consumer/scratch/FP admission'),(125,'gdn-fp32-native-exp-domain','CPU','Binary32 storage/materialization and native-exp domain limits'),(126,'gdn-probe-async-usm-lifecycle','PP','USM, async failure, partial submit and process termination contracts'),(128,'gdn-probe-async-source-admission','PP','Corrected immutable private probe lifecycle and actual debug scripts')]
 new=[]
 for n,name,kind,scope in entries:
  p=R/f'round{n}-{name}.txt';origin=p
  if not p.exists():
   origin=B/'R'/p.name;assert origin.is_file();shutil.copyfile(origin,p)
  x=ident(p);assert ident(origin)==x;dest=A/'research'/p.name;assert not dest.exists();shutil.copyfile(p,dest)
  entry=dict(path=str(p),**x,agent='/root/research_dense_mirror_phase_ownership'if kind=='CPU'else'/root/research_prefill_algorithms_blogs',model='gpt-6-luna',scope=scope,status='completed-read-only',root_full_report_reviewed=True)
  if origin!=p:entry['original_agent_output_path']=str(origin)
  d['reports'].append(entry);new.append(entry)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 correction=dict(R123_ratio_ranges='All process/cell paired register/direct medians are1.199..1.230, not20..32% slower than direct in the later scratch paragraph. Direct/trait medians span1.034..1.091;5–6% is only shorthand for many cells.',R126_script_availability='Both pinned parent scripts exist in b7436c79 worktree; R128 directly verifies their bytes and corrects the prior absence claim. Original R126 remains unchanged.',R128_output_path='Agent interpreted shorthand R as a literal B/R directory; root copied identical bytes to the canonical shared research directory. Only one committed report copy is required.')
 write(A/'provenance-corrections-v54.json',correction)
 d.update(registry_version=54,created_utc=now,research_completed=len(d['reports']),previous_committed_registry=dict(path=str(prev),**ident(prev),storage_commit='3ff794d1aa9b0756b436e2fb22eaccf6dea3353b'),new_reports=new,
   live_agent_snapshot=dict(time_utc=now,method='Direct collaboration.list_agents after R128 returned: CPU-source investigator R127 running; async investigator R128 completed; separate Sol private probe returned source-only. Snapshot is a boundary observation, not a persistent running claim.',agents=[dict(agent='/root/research_dense_mirror_phase_ownership',model='gpt-6-luna',status='running',round=127,scope='Immutable corrected private probe numerical/layout source admission'),dict(agent='/root/research_prefill_algorithms_blogs',model='gpt-6-luna',status='completed',round=128,next_dependency='Root bounded runtime/progress evidence or new owner-controller immutable source, avoiding repeated settled lifecycle analysis'),dict(agent='/root/implement_repeat_capture_reader_hardening',model='gpt-6.1-sol',status='completed-source-only',handoff='implementation-gdn-gate-factor-probe-v1.txt')]),
   current_root_decisions=dict(CPU_register_index='REJECT, production unchanged',sign_scratch='Deprioritized: no emitted/isolated evidence for useful gain',GDN_gate_factor='Necessary copied-expression private discriminator only; exact native-exp model domain and production recurrence equivalence not established',private_probe_source_commit='a0d25a1564dc32b4c8f8a9875b172ad44ea5063a',private_probe_snapshot='f47cd7310c43b0ec115487c3606da7361d5aad277ba19e60f40f43d93fa98126',cpu_only_build_receipt=dict(path=str(B/'gdn-gate-factor-probe-cpu-build-v1/record.json'),**ident(B/'gdn-gate-factor-probe-cpu-build-v1/record.json')),gpu_work_this_batch=False,model_inference_this_batch=False,full_physical_256K_lifecycle=False,root_parallel_work='Reviewed reports, separately prepared/corrected probe source and protocol, compiled under whole-session owner limits, committed evidence and retired659200B duplicate parent success logs; sole owner of execution',next_gate='Fully review R127; verify intended device and source/executable/config pins, strict output counts, retained LevelZero diagnostics and external owned wall/resource/log/fault supervision before first necessary GPU check'),additional_provenance_corrections=correction)
 d['implementation_handoffs_current_batch']=[d.pop('implementation_handoff'),dict(path=str(R/'implementation-gdn-gate-factor-probe-v1.txt'),**ident(R/'implementation-gdn-gate-factor-probe-v1.txt'),model='gpt-6.1-sol',status='completed-source-only',root_full_report_reviewed=True,committed_in_probe_source='a0d25a1564dc32b4c8f8a9875b172ad44ea5063a')]
 target=R/'report-registry-v54.json';assert not target.exists();write(target,d);shutil.copyfile(target,A/target.name)
 write(A/'root-parallel-review-v54.json',dict(created_utc=now,new_reports=len(new),total_reports=len(d['reports']),registry_identity=ident(target),decisions=d['current_root_decisions'],snapshot=d['live_agent_snapshot']))
 p=A/'REPORT.md';p.write_text(p.read_text()+'''

## Reviewed follow-ups v54

Five further reports R123–R126 and R128 were fully reviewed and preserved (234 reports total); the seventh parallel wave's numerical-source reviewer R127 is still running at this timestamped boundary. R123 deprioritizes a sign-scratch idea after the actual-weight rejection. R124/R125 require an actual target discriminator for the proposed same-buffer prefill gate factor, distinguish binary32 storage from physical local spills, and leave version-specific native-exp domain unknown. R126 supplies async/USM ownership rules; R128 applies them to the private source and corrects the prior missing-script claim.

A separate Sol returned a standalone source-only probe. Root reviewed every file and corrected incomplete production math/subgroup/JIT options, explicit B570 admission and the callback's nonthrowing sticky bounded diagnostics. The immutable root-v2 source compiled normally in6.253s with full inherited-session limits, required flags and clean closure. Its commit a0d25a1564dc32b4c8f8a9875b172ad44ea5063a contains source, protocol, original handoff, compiler/command/build/executable pins and compact evidence. No GPU or model was executed. Passing this copied-expression probe later would still not qualify real recurrence/state/output/context behavior or speed. R128 requires external intended-device identity, normal0,15cases/75stages/7,970,640comparisons and three equal full8K fingerprints; process termination never proves device cancellation.

Root continued source review, corrections, fresh bounded compilation, result decisions, commits and artifact retirement during both new research waves. Current snapshots distinguish returned agents from active agents and record a concrete evidence dependency for the next lifecycle inquiry. Recurring new assignments use committed past reports and settled decisions rather than reopening rejected candidates. Original reports are retained unchanged; corrections are separate.
''')
 shutil.copyfile(__file__,A/Path(__file__).name)
 identity=A/'archive-file-identities.json';old=json.loads(identity.read_text());updated={str(p.relative_to(A)):ident(p)for p in sorted(A.rglob('*'))if p.is_file()and p!=identity};assert set(old)<=set(updated)
 for rel,x in old.items():
  if rel!='REPORT.md':assert updated[rel]==x,rel
 write(identity,updated);print(json.dumps(dict(registry=str(target),**ident(target),completed=len(d['reports']))))
