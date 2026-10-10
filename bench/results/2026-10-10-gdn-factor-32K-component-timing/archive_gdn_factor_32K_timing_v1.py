"""Archive closed root timing/build evidence; preserve parser failure."""
from pathlib import Path
from collections import Counter
import fcntl,hashlib,json,re,shutil,subprocess
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-factor-component-timing-20261010')
A=W/'bench/results/2026-10-10-gdn-factor-32K-component-timing'
def ident(p):
 p=Path(p)
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def copy(src,dst):
 assert not dst.exists(),dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert ident(src)==ident(dst)
def write(p,data):p.write_text(json.dumps(data,indent=2)+'\n')
def closed(record):
 assert not record['active']
 for c in record['commands']:
  assert c['observation_complete'] and c['session_empty'] and c['direct_child_reaped'] and not c['cleanup'] and not c['errors'] and not c['survivors']
  assert c['normal_exit']
def trace_summary(path):
 raw=path.read_text();lines=raw.splitlines();handles={};launches=Counter();unknown=[];shapes=Counter()
 for line in lines:
  name=re.search(r'SUCCESS .* in zeKernelGetName\(hKernel=(0x[0-9a-f]+).*pName="([^"]+)"',line)
  if name:handles[name[1]]=name[2]
  launch=re.search(r'SUCCESS .* in zeCommandListAppendLaunchKernel(?:WithArguments)?\(.*hKernel=(0x[0-9a-f]+)',line)
  if launch:
   name=handles.get(launch[1]);launches[name]+=1
   if name is None:unknown.append(launch[1])
   shape=re.search(r'groupCounts=({[^}]+}).*groupSizes=({[^}]+})',line)
   if shape:shapes[str(name)+' '+shape[1]+' '+shape[2]]+=1
 failures=[line for line in lines if re.search(r'-> UR_RESULT_(?!SUCCESS)|ERROR \(ZE_RESULT_',line)]
 assert not unknown
 return dict(original_path=str(path),**ident(path),lines=len(lines),actual_successful_kernel_launch_counts=dict(launches),launch_shapes=dict(shapes),unresolved_handles=unknown,LevelZero_success_counts=dict(Counter(re.findall(r'SUCCESS \(ZE_RESULT_SUCCESS\) in (ze\w+)\(',raw))),non_success_lines=len(failures),distinct_non_success_counts=dict(Counter(re.sub(r'0x[0-9a-f]+','ADDRESS',line) for line in failures)),first_lines=lines[:8],last_lines=lines[-14:],policy='Closed diagnostic, launch identities/counts preserved; no full-history consumer. This does not observe the quiet benchmark.')
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='5ab8ab0a45b98a987b7ec77cb8b19007a35ea5d9'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
 assert not A.exists();A.mkdir(parents=True)
 build=B/'gdn-factor-component-timing-cpu-build-v2';r=json.loads((build/'record.json').read_text());closed(r);assert r['passed'] and r['complete'] and not r['GPU_executed']
 copy(build/'record.json',A/'cpu-build/record.json');copy(B/'build_gdn_factor_component_timing_v2.py',A/'cpu-build/controller.py')
 summary={}
 for c in r['commands']:
  for name,pin in c['logs'].items():
   p=build/name;assert ident(p)==pin;lines=p.read_text(errors='replace').splitlines()
   warnings=Counter(m[1] for line in lines if (m:=re.search(r'warning: (.*)',line)))
   summary[name]=dict(**pin,lines=len(lines),warning_counts=dict(warnings),first=lines[:5],last=lines[-7:])
   if name in ['configure.stdout','configure.stderr','direct-imports.stdout']:copy(p,A/'cpu-build'/name)
 write(A/'cpu-build/log-summary.json',summary)
 for version in [5,6]:
  run=B/f'gdn-factor-component-timing-quad-runtime-v{version}';r=json.loads((run/'record.json').read_text());closed(r)
  assert r['journal_status']=='complete' and not r['visible_kernel_GPU_entries'] and not r['devcoredump_after']
  target=A/f'runtime-v{version}';copy(run/'record.json',target/'record.json')
  copy(B/f'run_gdn_factor_component_timing_quad_v{version}.py',target/'controller.py')
  diag=next(c for c in r['commands'] if c['label']=='real-gdn-probe');trace=run/'real-gdn-probe.stderr';assert ident(trace)==diag['logs'][trace.name]
  compact=trace_summary(trace);assert {n:c for n,c in compact['actual_successful_kernel_launch_counts'].items() if 'GdnRec' in n}=={'_ZTSN6strata7prefill22GdnRecQuadPipelineSG32E':22,'_ZTSN6strata7prefill28GdnRecQuadFactorPipelineSG32E':22}
  write(target/'diagnostic-trace-summary.json',compact)
  if version==5:
   assert not r['passed'] and not r['complete'] and 'timing_result' not in r
   assert (run/'zero-chunk.stderr').read_text()=='FAIL,bench admission\n'
   for name in ['zero-chunk.stdout','zero-chunk.stderr','fault-observation-after-failure.stdout','fault-observation-after-failure.stderr']:copy(run/name,target/name)
  else:
   assert r['passed'] and r['complete'] and r['host_admission_rejections']==6
   for name in ['host-contract.stdout','real-gdn-probe.stdout','quiet-32K-bench.stdout','quiet-32K-bench.stderr','kernel-interval.stdout','kernel-interval.stderr','dynamic-dependencies.stdout']:copy(run/name,target/name)
   for c in r['commands']:
    if c['label'].startswith(('zero-','nondivisor-','too-few-','dirty-')):
     assert c['exit_code']==1
     for name in c['logs']:copy(run/name,target/name)
   write(A/'samples.json',r['timing_result'])
 assert ident(B/'gdn-factor-component-timing-quad-runtime-v5/real-gdn-probe.stdout')==ident(B/'gdn-factor-component-timing-quad-runtime-v6/real-gdn-probe.stdout')
 samples=json.loads((A/'samples.json').read_text());med=samples['service_seconds_medians'];delta=(med['factor']/med['log']-1)*100
 (A/'REPORT.md').write_text(f'''# GDN factor 32K component service comparison

Arc B570 10GiB root0000:05:00.0, oneAPI2026.1, source5ab8ab0a, binary8b7dc40a45e0b1044bcb0201a613cde6b1d509a4cfcc2405796aa2b2705ed6ee. CPU build completed119 steps in327.134599348s, normal0 with empty owned sessions. Source, binary, library and boot identities are pinned in original receipts.

The quiet paired synthetic prefix was32768 inputs in four8192 chunks, three reset samples per arm, complete untimed full-prefix warmup and balanced within-sample order. The clock spans gates+conv+recurrence+norm submissions through queue drain and async-error check. Uploads, readbacks, hashing, checks and progress flushes are excluded. Checks between pairs affect cache conditions. This is combined component service time, not model wall time or throughput.

| Sample | Legacy log gate (ms) | Producer factor (ms) |
| --- | ---: | ---: |
| 0 | 148.836091 | 149.337857 |
| 1 | 148.310772 | 149.213512 |
| 2 | 146.069502 | 148.959475 |
| Median | {med['log']*1000:.6f} | {med['factor']*1000:.6f} |

Factor median is{delta:.6f}% longer and no sample sum improves. No speed benefit was demonstrated; do not adopt from this result. The option remains OFF by default. No model/full-physical262144 qualification or decode performance claim follows.

All12 measured chunk pairs and all warmup chunks passed full bitwise state, convolution history/output, beta, materialized-exp factor, FP32/FP16 output, guards and immutable-input comparisons. Repeated chunk hashes agree. The requested quiet quad configuration is not a quiet native launch trace. The separate default diagnostic identifies22 legacy and22 factor quad kernels and matches earlier default fingerprints. Host-only contract and six before-queue input/logging refusals passed.

Final benchmark normal0, empty owned session, no cleanup/errors/survivors, sampled peakRSS2463166464B, total process wall32.720775828s (includes excluded checks/setup). Complete new journal interval has no GPU entries and no coredump; runtime/binary/source/boot pins remained stable. No recovery or service change occurred.

Root controllerv5 failed while parsing the expected host rejection: it required 'FAIL ' although the pinned fixture correctly wrote 'FAIL,bench admission'. The host exited1 normally without queue construction; earlier diagnostic passed. Original failure receipt and21-byte rejection are retained, with the clean failure-time journal interval. No quiet benchmark launched in v5. New controllerv6 corrects exact expected CSV messages, rejects unknown/empty/reordered rows, checks exact stage order and interval scope (R159 suggestion), and completed qualification. Original v5 remains failed and unchanged.

Successful verbose diagnostics and full build chatter are represented by original bytes/hashes, command/status/flags, warnings and complete launch-name/count/shape summaries. They have no current full-history consumer and may be retired after this archive commit. Keep original rejection evidence and active fixtures. The next independent source experiment is private prefill-only IQ4_NL direct dispatch based on609a270d; its measurements are pending.
''')
 copy(Path(__file__),A/Path(__file__).name)
 write(A/'archive-file-identities.json',{str(p.relative_to(A)):ident(p) for p in sorted(A.rglob('*')) if p.is_file()})
 subprocess.run(['git','diff','--check'],cwd=W,check=True)
 print(json.dumps(dict(archive=str(A),model=False,adopted=False,median_delta_percent=delta)))
