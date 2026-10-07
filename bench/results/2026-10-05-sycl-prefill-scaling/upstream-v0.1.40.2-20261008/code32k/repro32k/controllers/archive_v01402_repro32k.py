from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
parent=root/'bench/results/2026-10-05-sycl-prefill-scaling/upstream-v0.1.40.2-20261008'
out=parent/'code32k/repro32k';assert not out.exists();out.mkdir()
def copy(source,destination):
 target=out/destination;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
runs=['owned-v01402-code32k-patched-default-head-r1','owned-v01402-code32k-patched-default-head-r2','owned-event-ack-v01402-code32k-diagnostic-r1','owned-cb-off-v01402-code32k-diagnostic-r1','owned-cb-off-v01402-code32k-state-r1']
private=[]
for name in runs:
 directory=base/name;record=json.loads((directory/'record.json').read_text());assert not record['active']
 assert not any(record['cleanup'][key] for key in ['inferior_survived','gdb_survived'])
 for file in ['record.json','protocol.stdout.raw','events.jsonl','project-messages.txt']:
  p=directory/file
  if p.exists():copy(p,'runs/'+name+'/'+file)
 for file in ['inferior-argv.json','inferior-environment.json','failure.mi.txt','failure.console.txt']:
  p=directory/'debugger'/file
  if p.exists():copy(p,'runs/'+name+'/debugger/'+file)
 for p in (directory/'probes').glob('*'):
  if p.is_file():copy(p,'runs/'+name+'/probes/'+p.name)
 log=directory/'debugger/inferior.stderr'
 with log.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 private.append({'file':str(log),'bytes':log.stat().st_size,'sha256':digest,'reason':'Full diagnostic/engine log retained privately; raw protocol, environment, progress and owned stack snapshots archived.'})
 if not record['healthy']:
  with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-32768));tail=stream.read()
  target=out/'runs'/name/'debugger/inferior.stderr.tail';target.write_bytes(tail)
 for request in record['requests']:
  for field in ['first_head','prefill_state']:
   if field in request:
    p=Path(request[field]['file'])
    with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    private.append({'file':str(p),'bytes':p.stat().st_size,'sha256':digest,'reason':'Large raw state/head retained privately; per-part hashes and first-head receipt are in the run record.'})
for name in ['post-v01402-code32k-head-stall-health','post-event-ack-v01402-stall-health']:
 r=json.loads((base/name/'record.json').read_text());assert r['healthy'] and not r['active']
 copy(base/name/'record.json','health/'+name+'/record.json')
 for p in (base/name/'probes').glob('*'):
  if p.is_file():copy(p,'health/'+name+'/probes/'+p.name)
for name in ['v01402-code32k-head-sequence','event-ack-v01402-state-sequence','cb-off-v01402-state-sequence','cb-off-v01402-clean-sequence']:
 r=json.loads((base/name/'record.json').read_text());assert not r['active'] and not r['passed']
 copy(base/name/'record.json','sequences/'+name+'/record.json')
for file in ['build_event_ack_v01402.py','build_event_ack_v01402_v2.py','prepare_event_ack_v01402_controller.py','run_owned_event_ack_v01402_code32k.py','run_event_ack_v01402_state_sequence.py','prepare_cb_off_v01402_controller.py','run_owned_cb_off_v01402_code32k.py','run_cb_off_v01402_state_sequence.py','prepare_cb_off_v01402_clean_controller.py','run_owned_cb_off_v01402_code32k_clean.py','run_cb_off_v01402_clean_sequence.py','run_owned_v01402_code32k_heads.py','run_v01402_code32k_head_sequence.py','check_post_v01402_code32k_stall_health.py','check_post_event_ack_v01402_stall_health.py','archive_v01402_repro32k.py']:
 copy(base/file,'controllers/'+file)
for name in ['event-ack-v01402-build','event-ack-v01402-v2-build']:
 r=json.loads((base/name/'record.json').read_text());assert not r['active']
 for p in (base/name).glob('*'):
  if p.is_file():copy(p,'private-candidate-build/'+name+'/'+p.name)
for file in ['event_completion.hpp','event_completion_test.cpp']:
 copy(root/'build-sycl-event-ack-v2-20261008/source'/file,'private-candidate-source/'+file)
for file in ['cb-off-v01402-state-analysis/record.json','cb-off-v01402-state-analysis/cnr-research.json','event-ack-driver-source-review/review.json']:
 copy(base/file,'analysis/'+file)
(out/'private-artifacts.json').write_text(json.dumps(private,indent=2)+'\n')
(out/'README.md').write_text('''# 32K output reproducibility and stopped-job follow-up

All model jobs here use the same32,768-token code-review fixture as the parent,
context33024,8192-token chunks, int8 KV, normal MTP4, cache128, five CPU workers,
FIRST0 and RING8 on the same Arc B570 boot. No short-input speed comparison is
used. State/head capture and API logging exclude these jobs from performance
comparisons. No production source/binary was changed by these experiments.

The first fresh production head capture completes64 finite-logprob outputs,
but its first logprob (-1.152473) differs from both previously observed results
(-1.023937 and -1.051243). The next identical head job stops in its first chunk
at layer1. The kernel journal records one CCS engine reset. The main stack waits
in queue::ext_oneapi_submit_barrier, and a runtime host-task thread waits for a
native event in assignKernelEventCompletionData. The code_location fields next
to call return0x59f6b0 identify line4823,column45 of the recorded production
prefill.cpp: compute waits for the staged copy marker. This localizes a wait;
it does not establish which operation caused the engine reset.

A private candidate replaces only expert/PLE host_task acknowledgements with
owned events returned by the actual DMA and status queries before host-buffer
reuse. Ring ownership, event lifetime, failed-query handling, monotonic generations
and concurrent CPU readers pass normally and under ASan/UBSan. Its first logged32K
job nevertheless stops in layer0 while Stager::wait waits for readiness. Three
staging threads are inside queryCounterBasedEventStatus ->
synchronizeTimestampCompletionWithTimeout -> assignKernelEventCompletionData.
No new xe fault is recorded. The candidate is not adopted; CPU tests do not
prove runtime completion or GPU correctness. Its initial relocated-source compile
failure and corrected private build are both retained.

Both stopped jobs are cleaned up with neither owned inferior nor debugger
remaining. Fresh logged H2D/kernel/D2H checks pass on the same boot after each
failure. No reset, rebind, reboot, service or package change is performed.
The private candidate trace uses immediate command lists and successful native
appends; SUBMITTED status alone is not evidence of unflushed regular batches.

The installed NEO source version26.31.39395.14 places both observed native waits
in its profiling counter-event completion path. A process-local experiment with
EnableImplicitConvertionToCounterBasedEvents=0 keeps the production executable,
arithmetic and queue profiling unchanged. Its logged and unlogged state/head32K
jobs both complete normally without xe faults, but their states and outputs differ.
Two successful jobs do not prove prevention, and this setting does not fix the
output discrepancy. Full trace/state/head files stay private with receipts.

The post-prefill captures contain66parts, taken after32767 batched input tokens
and before the held-out input token/first verifier window. PLE history and the
first three QSA K/V/scales/pooled states match completely. The first differing
GDN storage ordinal is11 (model layer14); all preceding GDN rows match. The first
different QSA ordinal is3 (layer15), with the first K/V byte difference at token8200,
in the second8K chunk. Parts1,3 and21..65 differ; dead/block positions match.
All248,320 first-head floats differ (maximum absolute difference1.692817,
RMS0.200383). This proves a discrepancy exists before decode, without identifying
its first producing operation or excluding legal BLAS reduction variation.

The state/head gate therefore stops before a second state repeat and before
either clean timing job is launched. No rate from these jobs is accepted for
tuning. The rejected private candidate and failed equality gate remain explicit;
neither is a fix. Full262,144-cell normal-MTP validation and PP1000/TG70 remain open.
An additional documented oneMKL GPU CNR experiment is being evaluated separately.

Primary references: [NEO event completion](https://github.com/intel/compute-runtime/blob/26.31.39395.14/level_zero/core/source/event/event_impl.inl),
[NEO event conversion](https://github.com/intel/compute-runtime/blob/26.31.39395.14/level_zero/core/source/cmdlist/cmdlist_hw.inl),
[SYCL events](https://github.khronos.org/SYCL_Reference/iface/event.html),
[oneMKL GPU reproducibility conditions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/reproducibility-conditions.html).
Source URLs/hashes and observations versus inferences are recorded in analysis.
''')
note='''

Later32K state/head follow-up localizes an output discrepancy to the prefill
state, before the first verifier window. An identical head job records a CCS
reset; a rejected private event-ack candidate stops without a new xe fault.
Both are cleaned up and subsequent same-boot logged GPU health passes. Disabling
implicit counter-event conversion permits two completed32K state captures but
does not make their states/outputs equal. The equality gate prevents clean timing
jobs from launching. See [full records and limits](repro32k/README.md).
'''
with (parent/'code32k/README.md').open('a') as stream:stream.write(note)
def manifest(directory):
 path=directory/'manifest.json';meta=json.loads(path.read_text()) if path.exists() else {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
 meta.update(revised_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision_reason='Append completed32K reproducibility/state investigations and rejected candidate; do not change earlier raw records.',files={str(p.relative_to(directory)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(directory.rglob('*')) if p.is_file() and p!=path})
 path.write_text(json.dumps(meta,indent=2)+'\n')
 for name,item in meta['files'].items():
  p=directory/name;assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
 print(directory.name,len(meta['files']))
manifest(out);manifest(parent/'code32k');manifest(parent)
