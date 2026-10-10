from pathlib import Path
import datetime,fcntl,hashlib,json,shutil,csv
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007');W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-quad-pipeline-20261010');A=W/'bench/results/2026-10-10-gdn-host-event-diagnostic'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not A.exists();A.mkdir(parents=True)
 br=B/'gdn-host-event-cpu-build-v1';rr=B/'gdn-host-event-runtime-v1'
 for p in (br/'record.json',rr/'record.json'):
  r=json.loads(p.read_text());assert r['passed'] and r['complete'] and not r['active'] and not r['cleanup'] and not r['survivors']
  shutil.copyfile(p,A/(p.parent.name+'-record.json'))
 build=Path(json.loads((br/'record.json').read_text())['build'])
 for n in ['compile_commands.json','CMakeCache.txt']:
  shutil.copyfile(build/n,A/n)
 for n in ['build_gdn_host_event_cpu_v1.py','run_gdn_host_event_runtime_v1.py','make_gdn_host_event_gate_v1.py']:
  shutil.copyfile(B/n,A/n)
 for n in ['diagnostic-legacy-first.stdout','diagnostic-quad-first.stdout','gpu-short.stdout','host-contract.stdout','host-contract.stderr','host-fail-stop.stderr','diagnostic-invalid-length.stderr','diagnostic-dirty-environment.stderr']:
  assert (rr/n).stat().st_size<32768;shutil.copyfile(rr/n,A/n)
 records=[]
 for parent in [br,rr]:
  for p in sorted(parent.iterdir()):
   if not p.is_file() or p.name=='record.json':continue
   ident=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),retention='compact evidence plus receipt hashes')
   if p.stat().st_size and p.suffix in ('.stdout','.stderr','.trace') and not (A/p.name).exists() and p.name!='toolchain-env.stdout':
    raw=p.read_text(errors='replace').splitlines();name=parent.name+'-'+p.name+'.excerpt.txt';excerpt=raw[:12]+(['... bounded excerpt ...'] if len(raw)>24 else [])+raw[12:] if len(raw)<=24 else raw[:12]+['... bounded excerpt ...']+raw[-12:]
    (A/name).write_text('\n'.join(excerpt)+'\n');ident['excerpt']=name
   records.append(ident)
 (A/'compact-log-identities.json').write_text(json.dumps(records,indent=2)+'\n')
 r=json.loads((rr/'record.json').read_text());out=[]
 for process in r['diagnostic_component_samples']:
  for row in process['aggregate_rows']:
   out.append({'process_starting_order':process['starting_order'],**row})
 with (A/'aggregate-intervals.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
 decision={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'cd353f30a47dfd42818aace6f464bd227084e779','instrumented_only':True,'clean_timing_or_model_adoption':False,'complete_normal_exit':True,'no_owned_survivors':True,'visible_kernel_gpu_entries':r['visible_kernel_gpu_entries'],'journal_observation_limit':'User-visible journal only; absence of visible entry is not comprehensive driver proof','rows':192,'fresh_processes':2,'recorded_pairs_per_process':1,'warmup_pairs_per_process':2,'full_state_and_guarded_outputs_bitwise':True,'digests_repeat_across_every_pair_and_process':True,'diagnostic_clock_definition':'Host preparation/query/submit intervals are distinct from backend command start/end. Async operations overlap: no addition/subtraction for wait decomposition. No event timestamps compared to host clock.','warm_admission_ms':[x['admission_ns']/1e6 for x in out if x['phase']=='sample' and x['arm']=='quad'],'quad_recurrence_ms':[x['recurrence_device_ns']/1e6 for x in out if x['phase']=='sample' and x['arm']=='quad'],'legacy_recurrence_ms':[x['recurrence_device_ns']/1e6 for x in out if x['phase']=='sample' and x['arm']=='legacy'],'conclusion':'Host admission alone cannot explain this observed diagnostic slowdown; recurrence device duration is larger in both orders. Exact reason for order effect, spill cost, cache/residency/power remains unproved. Diagnostic compile-on is not clean baseline timing. Earlier ten-pair quiet rejection remains unchanged.'}
 (A/'root-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 (A/'REPORT.md').write_text('''# GDN host/event diagnostic on Arc B570

The current quad candidate remains rejected by the earlier clean 32768-token ten-pair comparison. This separate compile-on fixture diagnoses where time is spent; it does not qualify model performance or replace those measurements.

Source cd353f30a47dfd42818aace6f464bd227084e779, IntelLLVM2026.1 precise/SPIR64, existing in-order profiling-enabled queue, pinned V2/DirectSubmission0/copy-offload disabled. Fresh build passed with158 common translation-unit flag sets identical after removing only the private diagnostic macro. Root then ran traced host contracts/pre-queue refusal and short24-case/72-positive/one-forced-denial differential. Both fresh32768/chunk2048 processes (opposite first order, two warmup pairs, one recorded pair each) passed. All192 chunk interval rows and full prefix digest/counter records are retained; complete carried state and guarded FP32/FP16 output bits matched. Normal closure59.7245s, no owned cleanup/survivors, no new GPU entries in user-visible journal. No actual model/context lifecycle was tested.

Recorded warmed quad admission sums were0.592/0.641ms per32768 prefix. Quad recurrence command sums were118.195/147.329ms, versus legacy67.832/45.466ms in the corresponding orders. Norm sums also differ although the named normalization source is shared. Thus admission-only overhead cannot explain the observed slowdown. First-use warmup admission/JIT work is much larger and is retained separately. These are two diagnostic samples, not a variance-qualified clean speed estimate. Exact order-effect cause and spill/residency/cache/power remain unresolved; no causal claim is assigned to them.

Host timer includes public-wrapper preparation, admission, asynchronous submits and completion. Device intervals use separate backend timestamps on completed recurrence/norm events. They overlap with host submission; do not sum or subtract host and device clocks to manufacture a wait decomposition. Queries happen after wait and outside fail-stop unknown-completion handling. Successful final buffer drain/free precedes CSV emission. Existing mathematical/default/decoder paths remain unadopted; compile-OFF emission requalification is still separate.

Complete successful API/strace/compiler logs have compact bounded excerpts and original hashes; no raw verbose timeline is required by this closed question. Original source-only reviews and all prior failed receipts retain their statuses. Actual physical262144 model lifecycle, clean whole-model32K timing and candidate adoption remain unqualified.
''')
 shutil.copyfile(__file__,A/Path(__file__).name)
 identities={str(p.relative_to(A)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(A.rglob('*')) if p.is_file()}
 (A/'archive-file-identities.json').write_text(json.dumps(identities,indent=2)+'\n');print(json.dumps({'archive':str(A),'files':len(identities),'bytes':sum(x['bytes'] for x in identities.values())}))
