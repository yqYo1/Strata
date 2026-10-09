"""Root-owned supervised host probes and initial diagnostic GPU differential; no model/performance."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,resource,shlex,signal,subprocess,time
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-quad-pipeline-20261010')
G=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
I=Path('/opt/intel/oneapi/compiler/2026.1/bin')
HEAD='9edf4a5941f029357b2f2d0b19c19bdff4638c06'
BUILD=W/'build-sycl-gdn-quad-v2'
OUT=B/'gdn-quad-prefix-carry-gates-v1'
PINS={'sycl/src/prefill/kernels.dp.cpp':'af16f3d89dc0ef154b89800c278a78959ca0f7de0d59e90c7bc16e998f9968be','sycl/include/strata/prefill/gdn_variant.hpp':'044dfc92139d77e4cb3a1ddbf644339cad64a8155b052f691575cc1ccd818bf7','sycl/CMakeLists.txt':'ce0a6264b8afb008abab3840e4f22fd652cb371483f2273e4e73e01a5df81ee7','sycl/src/prefill/gdn_quad_parity.cpp':'79cf2e0772279a28b624ed9086d5412b71cffd11e2f57ef9f426fb6dd8d87ae1','sycl/tools/gdn-quad-pipeline/README.md':'a5ebc495ee8189ca7c9ba58838a4bca2f66e2fe387e421886a32d53b6d30ccf1','sycl/include/dpct/device.hpp':'adbcbd0ac45995348e37ac3f5ca42d9f3d4b0555031aeb1781a1e2166a74ed52','sycl/include/strata/sycl_error.hpp':'4b64f82b0805e9a113077ea7abc57d18044bff7240d8fe956abe651eb4f90ce4'}
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
  qualified_path=B/'gdn-quad-cpu-build-v2/record.json';qualified=json.loads(qualified_path.read_text())
  assert qualified['passed'] and qualified['complete'] and not qualified['active'] and not qualified['cleanup'] and not qualified['survivors']
  assert qualified['binary_sha256']==sha(BUILD/'gdn_quad_parity')
  short_path=B/'gdn-quad-initial-gates-v1/record.json';short=json.loads(short_path.read_text())
  assert sha(short_path)=='79fd4e18384486e5ba9150013e5b39d70d5cd43fc6483e31ec61d0879f2d4e90'
  assert short['passed'] and short['complete'] and not short['active'] and not short['cleanup'] and not short['survivors']
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
  r=dict(active=True,complete=False,passed=False,scope='Supervised host early contract/fail-stop probes then bounded 32768/262144 synthetic recurrence-and-norm prefix carry; no model/lifecycle/performance qualification',source_head=HEAD,root=str(W),build=str(BUILD),source_pins=PINS,controller_sha256=sha(__file__),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_seconds=1800,text_budget_bytes=96<<20,limits=dict(AS_each_bytes=64<<30,RSS_owned_poll_bytes=12<<30,CPU_each_seconds=[900,901],FSIZE_each_bytes=64<<20,NOFILE=512,CORE_bytes=0),commands=[],owners={},cleanup=[],survivors=[],gpu_executed=False,model_opened=False,adopted=False,performance_eligible=False,full_lifecycle_passed=False,baseline_receipt_sha256=sha(previous),baseline_compile_commands_sha256=sha(basecc),prior_build_failure_receipt_sha256='a7ddabec6b7a5be0981f06d543f9f4c232c6597d08dd561730ee1a9370115992',qualification_receipt_sha256=sha(qualified_path),short_gate_receipt_sha256=sha(short_path),binary_sha256=qualified['binary_sha256'],independent_lifetime_review_sha256=sha(B/'research-20261009/gdn-quad-fail-stop-wrapper-independent-review-round96.txt'))
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
   r['prefixes']=[];save()
   for total in (32768,262144):
    r['gpu_executed']=True;r['active_gpu_stage']='synthetic prefix '+str(total);save()
    log=run('prefix-'+str(total),[*debug,'--prefix',str(total),'--chunk','2048'],wall=800)
    lines=log.read_text().splitlines();calls=[x for x in lines if x.startswith('PASS call ')]
    assert len(calls)==total//2048,'incomplete synthetic carry'
    for ordinal,line in enumerate(calls):
     assert ('PASS call T=2048 prefix='+str(ordinal*2048)+' ') in line
     assert 'candidate=1 fallback=0' in line and 'compiledSG=32' in line
    assert lines[-1]=='PASS complete candidate-specific synthetic GPU differential; no model/full-lifecycle/performance qualification'
    error=(OUT/('prefix-'+str(total)+'.stderr')).read_text();assert 'FAIL-STOP' not in error and 'SYCL async error:' not in error
    r['prefixes'].append(dict(tokens=total,chunk_capacity=2048,candidate_calls=len(calls),full_state_and_active_FP32_FP16_bitwise=True,outer_and_unused_capacity_guards=True,compiled_SG32_observed=True,synthetic_recurrence_and_norm_only=True,model_full_context_lifecycle=False));save()
   r.update(active_gpu_stage=None,complete=True,passed=True)
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
  print(json.dumps({k:r.get(k) for k in ('passed','error','elapsed_seconds','cleanup','survivors','prefixes','gpu_executed','binary_sha256')}));return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
