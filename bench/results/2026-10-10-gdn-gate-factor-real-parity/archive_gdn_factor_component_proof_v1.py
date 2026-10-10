"""Root retains compact closed GDN build/component proofs, original startup failures."""
from pathlib import Path
from collections import Counter
import fcntl,hashlib,json,re,shutil,subprocess
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-prefill-gate-factor-20261010')
A=W/'bench/results/2026-10-10-gdn-gate-factor-real-parity'
def ident(p):
 p=Path(p)
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def copy(src,dst):
 assert not dst.exists(),dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert ident(src)==ident(dst)
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def trace_summary(path):
 text=path.read_text();lines=text.splitlines();handles={};launches=Counter();unknown=[];shapes=Counter()
 for line in lines:
  name=re.search(r'SUCCESS .* in zeKernelGetName\(hKernel=(0x[0-9a-f]+).*pName="([^"]+)"',line)
  if name:handles[name[1]]=name[2]
  launch=re.search(r'SUCCESS .* in zeCommandListAppendLaunchKernel(?:WithArguments)?\(.*hKernel=(0x[0-9a-f]+)',line)
  if launch:
   name=handles.get(launch[1])
   if name:launches[name]+=1
   else:unknown.append(launch[1])
   shape=re.search(r'groupCounts=({[^}]+}).*groupSizes=({[^}]+})',line)
   if shape:shapes[str(name)+' '+shape[1]+' '+shape[2]]+=1
 failures=[line for line in lines if re.search(r'-> UR_RESULT_(?!SUCCESS)|ERROR \(ZE_RESULT_',line)]
 return dict(original_path=str(path),**ident(path),lines=len(lines),actual_successful_kernel_launch_counts=dict(launches),launch_shapes=dict(shapes),unresolved_handles=unknown,LevelZero_success_counts=dict(Counter(re.findall(r'SUCCESS \(ZE_RESULT_SUCCESS\) in (ze\w+)\(',text))),non_success_lines=len(failures),distinct_non_success_counts=dict(Counter(re.sub(r'0x[0-9a-f]+','ADDRESS',line) for line in failures)),first_lines=lines[:8],last_lines=lines[-14:],policy='Closed successful trace compacted. No current complete-history consumer; original hash retained. Failures preserved independently.')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
 builds=B/'gdn-factor-real-cpu-build-v2';d=json.loads((builds/'record.json').read_text());assert d['passed'] and d['complete'] and not d['active']
 copy(builds/'record.json',A/'cpu-build-v2/record.json');copy(B/'build_gdn_factor_real_parity_v2.py',A/'cpu-build-v2/controller.py')
 logs={}
 for c in d['commands']:
  for name,pin in c['logs'].items():
   p=builds/name;assert ident(p)==pin;text=p.read_bytes().decode('utf-8','replace');lines=text.splitlines();logs[name]=dict(**pin,lines=len(lines),warnings=dict(Counter(line for line in lines if 'warning:' in line)),first=lines[:8],last=lines[-12:])
 write(A/'cpu-build-v2/log-summary.json',logs)
 for version in [1,2]:
  run=B/f'gdn-factor-real-runtime-v{version}';d=json.loads((run/'record.json').read_text());assert not d['active'] and not d['passed'] and not d['GPU_runtime_launched']
  for p in sorted(run.iterdir()):
   if p.is_file():copy(p,A/f'startup-failure-v{version}'/p.name)
  src=B/f'run_gdn_factor_real_parity_v{version}.py';dst=A/f'startup-failure-v{version}'/src.name
  if not dst.exists():copy(src,dst)
 names=['gdn-factor-real-runtime-v3']+['gdn-factor-real-'+route+'-runtime-v4' for route in ['cols','serial','quad','keyhead','keyhead-tuned']]
 results=[]
 for name in names:
  run=B/name;d=json.loads((run/'record.json').read_text());assert d['passed'] and d['complete'] and not d['active'] and not d['devcoredump_after'] and not d['visible_kernel_GPU_entries']
  target=A/'runtime'/name
  for p in sorted(run.iterdir()):
   if p.is_file() and p.name!='real-gdn-probe.stderr':copy(p,target/p.name)
  trace=run/'real-gdn-probe.stderr';c=next(c for c in d['commands'] if c['label']=='real-gdn-probe');assert ident(trace)==c['logs'][trace.name]
  compact=trace_summary(trace);assert not compact['unresolved_handles'];write(target/'trace-summary.json',compact)
  route='pipeline' if name.endswith('runtime-v3') else d['component_result']['selected_route']
  recurrence={kernel:count for kernel,count in compact['actual_successful_kernel_launch_counts'].items() if 'gdn_rec_' in kernel or 'GdnRec' in kernel}
  assert sum(count for kernel,count in recurrence.items() if 'factor' in kernel.lower())==22
  assert sum(count for kernel,count in recurrence.items() if 'factor' not in kernel.lower())==22
  if route=='quad':assert recurrence=={'_ZTSN6strata7prefill22GdnRecQuadPipelineSG32E':22,'_ZTSN6strata7prefill28GdnRecQuadFactorPipelineSG32E':22}
  if route=='keyhead-tuned':assert sorted(recurrence.values())==[6,6,16,16]
  results.append(dict(route=route,receipt=ident(run/'record.json'),component=d['component_result'],actual_recurrence_launches=recurrence,all_owned_commands_normal0_emptySID=True,kernel_interval='complete clean',timing_eligible=False))
 for name in ['run_gdn_factor_real_parity_v3.py','run_gdn_factor_real_parity_routes_v4.py']:copy(B/name,A/'runtime'/name)
 write(A/'runtime/aggregate.json',dict(routes=results,paired_chunk_comparisons=132,model_qualified=False,actual_physical_262144=False,performance_claim=False,adopted=False))
 report=A/'REPORT.md';report.write_text(report.read_text()+"""

## Closed actual component qualification

Corrected CPU build v2 passed normal0 with all owned sessions empty. Source622a53eb compiled and linked both production prefill and the real fixture. Root controller setup initially compiled the shell wrapper as Python; corrected extraction loads only its pure diagnostic environment function. Next startup refused because the runtime search path lacked oneMKL. Both failures occurred before any GPU submission, are preserved unchanged, and were corrected by extracting the embedded Python and using the recorded oneAPI build library path with a dependency preflight. Neither was a GPU hang.

The B570 root0000:05:00.0 then passed paired real GDN producer, convolution, recurrence and norm comparison on six selector configurations: pipeline, nonpipeline columns, serial, quadSG32, keyhead, and tuned keyhead. Each compared22 chunks from zero/nonzero initialization with state/history carry, all FP32 and FP16 bytes, beta, materialized-exp factors, immutable inputs and guards.132 paired chunk comparisons passed; each process normal0, all owned sessions empty, complete new kernel journal interval with no GPU entries/dump. Actual native kernel launch identities were resolved from Level Zero handles. Tuned selection used6 tuned chunks per arm and16 pipeline fallbacks; it is not a claim of22 tuned executions. Synthetic total1133 tokens per initialization is a correctness fixture, not a32K performance comparison. No model/physical262144 or speed/adoption result follows.

Successful verbose API traces are compacted with original bytes/hashes, complete launch-name/count/shape observations and error counts. They have no current full-history consumer and may be retired after this archive commit. Future performance runs will use quiet diagnostics and>=32768 input, repeated paired samples and unchanged decode decisions. Real model and full physical262144 lifecycle still need qualification before adoption.
""")
 copy(Path(__file__),A/Path(__file__).name)
 index=A/'archive-file-identities.json';old=json.loads(index.read_text());current={str(p.relative_to(A)):ident(p) for p in sorted(A.rglob('*')) if p.is_file() and p!=index}
 for name,pin in old.items():
  if name!='REPORT.md':assert current[name]==pin
 write(index,current)
 for args in [('diff','--check'),('add','bench/results/2026-10-10-gdn-gate-factor-real-parity'),('diff','--cached','--check'),('commit','-m','test(sycl): retain real GDN factor parity across dispatch families'),('push',),('rev-parse','HEAD')]:subprocess.run(['git',*args],cwd=W,check=True)
 print(json.dumps(dict(paired_chunks=132,dispatch_configurations=6,model_qualified=False,adopted=False)))
