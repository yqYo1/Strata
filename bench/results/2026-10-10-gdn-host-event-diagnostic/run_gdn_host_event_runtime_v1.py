"""Root-owned diagnostic host/event attribution; instrumented synthetic component, no speed/adoption claim."""
from pathlib import Path
import csv,datetime,fcntl,hashlib,json,math,os,resource,shlex,signal,statistics,subprocess,time
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-quad-pipeline-20261010')
G=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
I=Path('/opt/intel/oneapi/compiler/2026.1/bin')
HEAD='cd353f30a47dfd42818aace6f464bd227084e779'
BUILD=W/'build-sycl-gdn-host-event-v1'
OUT=B/'gdn-host-event-runtime-v1'
PINS={'sycl/src/prefill/kernels.dp.cpp': 'cc5254f688b9138fc71800884c853211771067c0f07a0a44a237fd45b59b4bfa', 'sycl/include/strata/prefill/gdn_variant.hpp': 'd9b0c169486205c636658c92db90351f5a9f091bb8e108c07ddce2ec56cd3ef0', 'sycl/CMakeLists.txt': '5911b38ff78b7d185ffd92ee34d0360f272e003b558ae7a3fa0707a1e684f9b6', 'sycl/src/prefill/gdn_quad_parity.cpp': '13b19226ec042d82dd51c7ae8fcde7fe2710a28c79cd6c6324958c82f205ddf6', 'sycl/tools/gdn-quad-pipeline/README.md': '5135966f4b527c2b6673d1d0a58189f6129dee06b73a3d094f994c80d630b231', 'sycl/include/dpct/device.hpp': 'adbcbd0ac45995348e37ac3f5ca42d9f3d4b0555031aeb1781a1e2166a74ed52', 'sycl/include/strata/sycl_error.hpp': '4b64f82b0805e9a113077ea7abc57d18044bff7240d8fe956abe651eb4f90ce4'}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(root,*args):return subprocess.check_output(['/usr/bin/git','-C',str(root),*args],text=True,timeout=20).strip()
def identity(pid):
 try:
  w=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,ppid=int(w[1]),start_ticks=int(w[19]),state=w[0],rss_bytes=int(w[21])*os.sysconf('SC_PAGE_SIZE'))
 except (FileNotFoundError,PermissionError,ProcessLookupError):return None

def main():
 assert __debug__
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
  assert git(G,'rev-parse','HEAD')=='3cf03257f219afbe7334045ff7c6a06ac68c627d' and not git(G,'status','--porcelain')
  for p,h in PINS.items():assert sha(W/p)==h,p
  assert BUILD.is_dir() and not OUT.exists()
  qualified_path=B/'gdn-host-event-cpu-build-v1/record.json';qualified=json.loads(qualified_path.read_text())
  assert qualified['passed'] and qualified['complete'] and not qualified['active'] and not qualified['cleanup'] and not qualified['survivors']
  assert qualified['binary_sha256']==sha(BUILD/'gdn_quad_parity')
  previous=B/'cache-route-pairs-v0141-private-build-v1/record.json';baseline=json.loads(previous.read_text())
  assert baseline['passed'] and baseline['complete'] and not baseline['active'] and not baseline['cleanup'] and not baseline['survivors']
  basebuild=Path(baseline['arms'][0]['build']);baseroot=Path(baseline['root']);basecc=basebuild/'compile_commands.json'
  assert sha(basecc)==baseline['compile_commands_sha256']
  for p in Path('/proc').iterdir():
   if p.name.isdecimal():
    try:comm=(p/'comm').read_text().strip()
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert comm not in ('strata','gdn_quad_parity','strata-xe-health','ninja','icpx','icx','vtune','unitrace'),(p.name,comm)
  OUT.mkdir(mode=0o700);started=time.monotonic();proc=None;fds={}
  r=dict(active=True,complete=False,passed=False,scope='Short traced correctness followed by two fresh counterbalanced 32768 event-capture diagnostic processes; host/device clocks nonadditive; instrumented synthetic component only, no clean timing/model/full-lifecycle/adoption',source_head=HEAD,root=str(W),build=str(BUILD),source_pins=PINS,controller_sha256=sha(__file__),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_seconds=1800,text_budget_bytes=96<<20,limits=dict(AS_each_bytes=64<<30,RSS_owned_poll_bytes=12<<30,CPU_each_seconds=[900,901],FSIZE_each_bytes=64<<20,NOFILE=512,CORE_bytes=0),commands=[],owners={},cleanup=[],survivors=[],gpu_executed=False,model_opened=False,adopted=False,performance_eligible=False,full_lifecycle_passed=False,baseline_receipt_sha256=sha(previous),baseline_compile_commands_sha256=sha(basecc),prior_build_failure_receipt_sha256='a7ddabec6b7a5be0981f06d543f9f4c232c6597d08dd561730ee1a9370115992',qualification_receipt_sha256=sha(qualified_path),binary_sha256=qualified['binary_sha256'],independent_lifetime_review_sha256=sha(B/'research-20261009/gdn-host-event-final-source-review-round107.txt'))
  def save():
   r['elapsed_seconds']=time.monotonic()-started
   temp=OUT/'record.json.tmp';temp.write_text(json.dumps(r,indent=2)+'\n');temp.replace(OUT/'record.json')
  def register(v):
   k=str(v['pid'])
   if k in r['owners']:return
   fd=os.pidfd_open(v['pid']);after=identity(v['pid'])
   if not after or after['start_ticks']!=v['start_ticks']:os.close(fd);return
   r['owners'][k]=dict(pid=v['pid'],ppid=v['ppid'],start_ticks=v['start_ticks']);fds[k]=fd
  def discover():
   rows=[v for p in Path('/proc').iterdir() if p.name.isdecimal() and (v:=identity(int(p.name)))]
   changed=True
   while changed:
    changed=False
    for v in rows:
     if str(v['pid']) not in r['owners'] and str(v['ppid']) in r['owners']:
      try:register(v)
      except ProcessLookupError:continue
      changed=changed or str(v['pid']) in r['owners']
  def live():return [now for v in r['owners'].values() if (now:=identity(v['pid'])) and now['start_ticks']==v['start_ticks'] and now['state']!='Z']
  def stop():
   discover()
   for sig,grace in ((signal.SIGTERM,3),(signal.SIGKILL,3)):
    for now in reversed(live()):
     try:signal.pidfd_send_signal(fds[str(now['pid'])],sig);r['cleanup'].append(dict(pid=now['pid'],start_ticks=now['start_ticks'],signal=sig.name))
     except ProcessLookupError:pass
    if proc:
     try:proc.wait(timeout=grace)
     except subprocess.TimeoutExpired:pass
  env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
  def run(label,argv,cwd=W,wall=240,expected=0):
   nonlocal proc
   so=OUT/(label+'.stdout');se=OUT/(label+'.stderr');begin=time.monotonic()
   e=dict(label=label,argv=argv,cwd=str(cwd),wall_seconds=wall,expected_exit_code=expected);r['commands'].append(e);save()
   def limits():
    for k,p in ((resource.RLIMIT_AS,(64<<30,64<<30)),(resource.RLIMIT_CPU,(900,901)),(resource.RLIMIT_FSIZE,(64<<20,64<<20)),(resource.RLIMIT_NOFILE,(512,512)),(resource.RLIMIT_CORE,(0,0))):resource.setrlimit(k,p)
   with so.open('wb') as out,se.open('wb') as err:
    proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=out,stderr=err,start_new_session=True,preexec_fn=limits)
    v=identity(proc.pid);assert v;register(v);e['owner']=r['owners'][str(proc.pid)];r['active_stage']=label;save()
    while proc.poll() is None:
     discover();rss=sum(x['rss_bytes'] for x in live());r['peak_owned_rss_bytes']=max(r.get('peak_owned_rss_bytes',0),rss)
     assert rss<12<<30,'owned RSS budget';assert time.monotonic()-begin<wall,'stage deadline';assert time.monotonic()-started<1790,'total deadline'
     assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<96<<20,'text budget';save();time.sleep(.3)
   discover();e.update(exit_code=proc.returncode,elapsed_seconds=time.monotonic()-begin,stdout_sha256=sha(so),stderr_sha256=sha(se));save()
   assert proc.returncode==expected,label+' unexpected exit status';assert not live(),label+' descendants'
   return so
  save()
  try:
   env.update({k:v for k,v in qualified['environment'].items() if k in ('PATH','LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','ONEAPI_ROOT','CMAKE_PREFIX_PATH','PKG_CONFIG_PATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','MKLROOT')})
   env.update(UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',EnableDirectSubmission='0',SYCL_CACHE_PERSISTENT='0',STRATA_GDN_QUAD='0',STRATA_GDN_KEYHEAD='0',STRATA_GDN_KEYHEAD_TUNED='0',STRATA_GDN_PIPELINE='1')
   # Reuse only the helper's pure environment function, exactly as debug-run does.
   import types
   recovery=types.ModuleType('xe_diagnostic_environment');helper=W/'sycl/tools/recover-xe.sh'
   source=helper.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
   exec(compile(source,str(helper),'exec'),recovery.__dict__)
   r['environment']=recovery.diagnostic_environment(env);r['debug_runner_sha256']=sha(W/'sycl/tools/debug-run.py');r['recover_environment_helper_sha256']=sha(helper)
   runtime=[Path('/opt/intel/oneapi/compiler/2026.1/lib')/n for n in ('libsycl.so.9','libur_loader.so.0','libur_adapter_level_zero_v2.so.0')]
   runtime += [Path('/usr/lib/x86_64-linux-gnu')/n for n in ('libze_loader.so.1','libze_intel_gpu.so.1')]
   r['runtime_pins']={str(p):dict(realpath=str(p.resolve()),sha256=sha(p)) for p in runtime};save()
   binary=str(BUILD/'gdn_quad_parity');debug=['/usr/bin/python3',str(W/'sycl/tools/debug-run.py'),binary]
   for label,mode,status in [('host-contract','--host-only',0),('host-fail-stop','--host-fail-stop-probe',2)]:
    trace=OUT/(label+'.trace')
    log=run(label,['/usr/bin/strace','--kill-on-exit','-f','-yy','-e','trace=open,openat,openat2,ioctl','-o',str(trace),*debug,mode],wall=60,expected=status)
    text=log.read_text();error=(OUT/(label+'.stderr')).read_text();raw=trace.read_text()
    assert '/dev/dri/' not in raw and '/dev/accel/' not in raw,'host mode opened GPU device'
    assert 'host fail-stop probe unwound' not in error and 'host fail-stop probe unwound' not in text
    if status==0:assert 'PASS host-only' in text and 'queue_lookup=false GPU_submission=false' in text
    else:assert 'FAIL-STOP stage=host injected exception' in error and 'host_unwind=false' in error and 'explicit_USM_release=false' in error
    r.setdefault('host_probes',[]).append(dict(mode=mode,observed_exit_code=status,trace_sha256=sha(trace),trace_bytes=trace.stat().st_size,no_GPU_device_open_observed=True,no_unwind_marker=True,source_queue_API_not_called=True));save()
   # New argument and dirty-environment refusal under an observable device-open trace.
   for label,total in [('diagnostic-invalid-length','32767'),('diagnostic-dirty-environment','32768')]:
    trace=OUT/(label+'.trace')
    args=['--diagnostic-prefix',total,'--chunk','2048','--order','legacy-first','--samples','1']
    run(label,['/usr/bin/strace','--kill-on-exit','-f','-yy','-e','trace=open,openat,openat2,ioctl','-o',str(trace),*debug,*args],wall=60,expected=1)
    raw=trace.read_text();assert '/dev/dri/' not in raw and '/dev/accel/' not in raw
    err=(OUT/(label+'.stderr')).read_text();assert 'FAIL ' in err and 'TIMING_SUMMARY' not in (OUT/(label+'.stdout')).read_text()
    if total=='32768':assert 'quiet timing refused:' in err
    r.setdefault('timing_negative_probes',[]).append(dict(label=label,exit_code=1,no_device_open_observed=True,trace_sha256=sha(trace)));save()
   r['gpu_executed']=True;r['active_gpu_stage']='initial short synthetic differential';save()
   log=run('gpu-short',debug,wall=240)
   lines=log.read_text().splitlines();calls=[x for x in lines if x.startswith('PASS call ')];cases=[x for x in lines if x.startswith('CASE ')]
   assert len(cases)==24 and len(calls)==73,'incomplete short fixture'
   positives=[x for x in calls if 'candidate=1 fallback=0' in x];denied=[x for x in calls if 'candidate=0 fallback=1' in x]
   assert len(positives)==72 and len(denied)==1 and all('compiledSG=32' in x for x in positives)
   assert any(x.startswith('PASS empty_device ') for x in lines)
   assert lines[-1]=='PASS complete candidate-specific synthetic GPU differential; no model/full-lifecycle/performance qualification'
   error=(OUT/'gpu-short.stderr').read_text()
   assert 'FAIL-STOP' not in error and 'SYCL async error:' not in error and 'asynchronous SYCL error:' not in error and 'PROFILE_QUERY_FAILED' not in error
   r.update(cases=24,positive_candidate_calls=72,forced_denial_explicit_fallback_calls=1,full_state_and_active_FP32_FP16_bitwise=True,outer_and_unused_capacity_guards=True,compiled_SG32_observed=True,synthetic_only=True,active_gpu_stage=None)
   r['diagnostic_environment_without_API_layers']=dict(env);r['diagnostic_component_samples']=[];save()
   assert not any(k in env for k in ('STRATA_TRACE','UR_ENABLE_LAYERS','ZE_ENABLE_VALIDATION_LAYER','ZEL_ENABLE_LOADER_LOGGING','LD_PRELOAD'))
   for order in ('legacy-first','quad-first'):
    r['active_gpu_stage']='diagnostic32768 '+order;save()
    log=run('diagnostic-'+order,[binary,'--diagnostic-prefix','32768','--chunk','2048','--order',order,'--samples','1'],wall=600)
    rows=list(csv.reader(log.read_text().splitlines()));events=[x for x in rows if x[0]=='DIAGNOSTIC'];prefixes=[x for x in rows if x[0]=='TIMING']
    assert len(rows)==106 and len(events)==96 and len(prefixes)==6 and len(rows[1])==21
    assert rows[0][0]=='DIAGNOSTIC_META' and 'capture_compiled=true' in rows[0] and 'host_device_not_additive=true' in rows[0]
    assert rows[-1]==['DIAGNOSTIC_SUMMARY','rows=96','recorded_pairs=1','exact_paired_chunks=true','profile_available=true','normal_buffers_released=true','host_device_not_additive=true','model=false','full_lifecycle=false','adopted=false','pass']
    identities={tuple(x[14:18]) for x in prefixes};assert len(identities)==1
    aggregated=[]
    for phase,pair,reverse in [('warmup',1,False),('warmup',2,True),('sample',1,False)]:
     expected_order=('quad-first' if order=='legacy-first' else 'legacy-first') if reverse else order
     for arm in ('legacy','quad'):
      ev=[x for x in events if x[1]==phase and x[2]==str(pair) and x[5]==arm]
      pr=[x for x in prefixes if x[1]=='diagnostic_'+phase and x[2]==str(pair) and x[5]==arm]
      assert len(ev)==16 and len(pr)==1;pr=pr[0]
      assert [int(x[7]) for x in ev]==list(range(0,32768,2048))
      expected_position='1' if (arm=='quad')==(expected_order=='quad-first') else '2'
      for x in ev:
       assert len(x)==21 and x[3:5]==[expected_order,expected_position] and x[8]=='2048'
       assert x[6]==('GdnRecQuadPipelineSG32' if arm=='quad' else 'GdnLegacyPipelineReference')
       assert math.isfinite(float(x[9])) and float(x[9])>0
       ns=[int(v) for v in x[10:]];assert all(v>=0 for v in ns)
       assert ns[7]==ns[6]-ns[5] and ns[10]==ns[9]-ns[8]
       assert ns[6]>=ns[5] and ns[9]>=ns[8]
       if arm=='legacy':assert ns[2]==0
      assert len(pr)==24 and pr[3:6]==[expected_order,expected_position,arm]
      assert pr[6:9]==['32768','2048','16'] and pr[10:12]==['32768','16'] and pr[13]=='16' and pr[-1]=='pass'
      assert pr[12]==('16' if arm=='quad' else '0') and pr[18]==('32' if arm=='quad' else '0')
      assert math.isfinite(float(pr[9])) and float(pr[9])>0
      assert abs(sum(float(x[9]) for x in ev)-float(pr[9]))<2e-8,'prefix service sum rounding'
      aggregated.append(dict(phase=phase,pair=pair,order=expected_order,arm=arm,position=int(expected_position),service_seconds=float(pr[9]),prevalidation_ns=sum(int(x[10]) for x in ev),queue_lookup_ns=sum(int(x[11]) for x in ev),admission_ns=sum(int(x[12]) for x in ev),recurrence_submit_host_ns=sum(int(x[13]) for x in ev),norm_submit_host_ns=sum(int(x[14]) for x in ev),recurrence_device_ns=sum(int(x[17]) for x in ev),norm_device_ns=sum(int(x[20]) for x in ev)))
    err=(OUT/('diagnostic-'+order+'.stderr')).read_text()
    assert all(x not in err for x in ('FAIL-STOP','SYCL async error:','asynchronous SYCL error:','PROFILE_QUERY_FAILED','DIAGNOSTIC_DENIED','DIAGNOSTIC_UNAVAILABLE'))
    r['diagnostic_component_samples'].append(dict(starting_order=order,aggregate_rows=aggregated,repeat_digests=list(identities)[0],per_chunk_rows=96,normal_buffers_released=True));save()
   assert r['diagnostic_component_samples'][0]['repeat_digests']==r['diagnostic_component_samples'][1]['repeat_digests']
   journal=run('kernel-journal',['/usr/bin/journalctl','-k','--since',r['started_utc'],'--no-pager'],wall=30)
   import re
   faults=[x for x in journal.read_text().splitlines() if re.search(r'\b(?:xe|i915|drm)\b|gpu.*(?:hang|fault|reset)|devcoredump',x,re.I)]
   r['visible_kernel_gpu_entries']=faults;assert not faults,'new visible GPU driver entry requires review'
   r.update(diagnostic_event_capture_validated=True,clean_component_timing_validated=False,component_definition='Instrumented host service plus separate backend recurrence/norm timestamps; per-chunk admission observed. Asynchronous intervals overlap: do not add/subtract host and device values; transfers/checks excluded.',active_gpu_stage=None,complete=True,passed=True)
  except BaseException as e:r['error']=type(e).__name__+': '+str(e)
  finally:
   discover()
   if live():stop()
   r['survivors']=live();r['passed']=r['passed'] and not r['cleanup'] and not r['survivors']
   try:
    assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
    assert git(G,'rev-parse','HEAD')=='3cf03257f219afbe7334045ff7c6a06ac68c627d' and not git(G,'status','--porcelain')
    for p,h in PINS.items():assert sha(W/p)==h,p
    assert sha(__file__)==r['controller_sha256'];assert sha(BUILD/'gdn_quad_parity')==qualified['binary_sha256'];r['exit_pin_gate_passed']=True
   except BaseException as e:r['passed']=False;r['exit_pin_error']=repr(e)
   r.update(active=False,active_stage=None,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
   for fd in fds.values():os.close(fd)
  print(json.dumps({k:r.get(k) for k in ('passed','error','elapsed_seconds','cleanup','survivors','cases','positive_candidate_calls','host_probes','gpu_executed','binary_sha256')}));return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
