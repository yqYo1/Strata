"""Root-owned finite production-flag build of GDN quad fixture; no executable/GPU/model run."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,resource,shlex,signal,subprocess,time
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-quad-pipeline-20261010')
G=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
I=Path('/opt/intel/oneapi/compiler/2026.1/bin')
HEAD='9edf4a5941f029357b2f2d0b19c19bdff4638c06'
BUILD=W/'build-sycl-gdn-quad-v2'
OUT=B/'gdn-quad-cpu-build-v2'
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
  assert not BUILD.exists() and not OUT.exists()
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
  r=dict(active=True,complete=False,passed=False,scope='Same production precise-FP/SPIR64 configuration, full dependencies and new correctness executable build; no executable/GPU/model invocation',source_head=HEAD,root=str(W),build=str(BUILD),source_pins=PINS,controller_sha256=sha(__file__),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_seconds=2400,text_budget_bytes=32<<20,limits=dict(AS_each_bytes=32<<30,RSS_owned_poll_bytes=12<<30,CPU_each_seconds=[600,601],FSIZE_each_bytes=1<<30,NOFILE=512,CORE_bytes=0),commands=[],owners={},cleanup=[],survivors=[],gpu_executed=False,model_opened=False,adopted=False,performance_eligible=False,full_lifecycle_passed=False,baseline_receipt_sha256=sha(previous),baseline_compile_commands_sha256=sha(basecc),previous_failed_receipt_sha256='a7ddabec6b7a5be0981f06d543f9f4c232c6597d08dd561730ee1a9370115992',controller_change='Restore setvars CPLUS_INCLUDE_PATH/C_INCLUDE_PATH/MKLROOT; fresh build avoids mixed preprocessor header availability; source/math flags unchanged')
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
  def run(label,argv,cwd=W,wall=1800):
   nonlocal proc
   so=OUT/(label+'.stdout');se=OUT/(label+'.stderr');begin=time.monotonic()
   e=dict(label=label,argv=argv,cwd=str(cwd),wall_seconds=wall);r['commands'].append(e);save()
   def limits():
    for k,p in ((resource.RLIMIT_AS,(32<<30,32<<30)),(resource.RLIMIT_CPU,(600,601)),(resource.RLIMIT_FSIZE,(1<<30,1<<30)),(resource.RLIMIT_NOFILE,(512,512)),(resource.RLIMIT_CORE,(0,0))):resource.setrlimit(k,p)
   with so.open('wb') as out,se.open('wb') as err:
    proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=out,stderr=err,start_new_session=True,preexec_fn=limits)
    v=identity(proc.pid);assert v;register(v);e['owner']=r['owners'][str(proc.pid)];r['active_stage']=label;save()
    while proc.poll() is None:
     discover();rss=sum(x['rss_bytes'] for x in live());r['peak_owned_rss_bytes']=max(r.get('peak_owned_rss_bytes',0),rss)
     assert rss<12<<30,'owned RSS budget';assert time.monotonic()-begin<wall,'stage deadline';assert time.monotonic()-started<2390,'total deadline'
     assert sum(p.stat().st_size for p in OUT.glob('*.std*'))<32<<20,'text budget';save();time.sleep(.3)
   discover();e.update(exit_code=proc.returncode,elapsed_seconds=time.monotonic()-begin,stdout_sha256=sha(so),stderr_sha256=sha(se));save()
   assert proc.returncode==0,label+' failed';assert not live(),label+' descendants'
   return so
  save()
  try:
   # Minimal inherited environment: no credentials or GPU/runtime selectors.
   tool=run('toolchain-env',['/bin/bash','-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'],wall=30)
   raw=dict(x.decode().split('=',1) for x in tool.read_bytes().split(b'\0') if b'=' in x)
   env.update({k:raw[k] for k in ('PATH','LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','ONEAPI_ROOT','CMAKE_PREFIX_PATH','PKG_CONFIG_PATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','MKLROOT') if k in raw})
   r['environment']=env;r['compiler_sha256']={str(I/k):sha(I/k) for k in ('icx','icpx')};save()
   run('compiler',[str(I/'icpx'),'--version'],wall=30)
   options=['-DCMAKE_CXX_COMPILER='+str(I/'icpx'),'-DCMAKE_C_COMPILER='+str(I/'icx'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DSTRATA_GGML_DIR='+str(G),'-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_SYCL_PARITY=ON','-DSTRATA_IQ2S_GCC=OFF','-DSTRATA_SYCL_AOT=','-DSTRATA_SYCL_SPIN_MAX=','-DSTRATA_NATIVE_POOL_TASK_FACTOR=0','-DSTRATA_GDN_QUAD_PARITY=ON']
   run('configure',['/usr/bin/cmake','-S',str(W/'sycl'),'-B',str(BUILD),'-G','Ninja',*options],wall=120)
   run('build',['/usr/bin/cmake','--build',str(BUILD),'--target','gdn_quad_parity','--parallel','4'],wall=1800)
   commands=json.loads((BUILD/'compile_commands.json').read_text());prior=json.loads(basecc.read_text())
   def rows(entries,root,build):
    result={}
    for x in entries:
     n=str(Path(x['file']).resolve()).replace(str(build),'{BUILD}').replace(str(root),'{ROOT}')
     args=shlex.split(x['command']);args=[v.replace(str(build),'{BUILD}').replace(str(root),'{ROOT}') for v in args]
     result[n]=args
    return result
   new=rows(commands,W,BUILD);old=rows(prior,baseroot,basebuild);common=sorted(set(new)&set(old));mismatch=[n for n in common if new[n]!=old[n]]
   r['flag_comparison']=dict(common_translation_units=len(common),mismatches=mismatch,new_files=sorted(set(new)-set(old)),old_files_missing=sorted(set(old)-set(new)))
   assert not mismatch and not set(old)-set(new),'baseline compile flags changed'
   assert set(new)-set(old)=={'{ROOT}/sycl/src/prefill/gdn_quad_parity.cpp'},'unexpected new TU'
   test=new['{ROOT}/sycl/src/prefill/gdn_quad_parity.cpp'];kernel=new['{ROOT}/sycl/src/prefill/kernels.dp.cpp']
   for args in (test,kernel):
    assert '-fp-model=precise' in args and '-fsycl' in args and '-fsycl-default-sub-group-size=32' in args
    assert '-fsycl-device-code-split=per_kernel' in args
    assert not set(args)&{'-ffast-math','-Ofast','-ffinite-math-only','-fassociative-math'}
   imports=run('direct-imports',['/usr/bin/readelf','-d',str(BUILD/'gdn_quad_parity')],wall=30)
   targetcommands=run('target-commands',['/usr/bin/ninja','-C',str(BUILD),'-t','commands','gdn_quad_parity'],wall=30)
   r.update(binary=str(BUILD/'gdn_quad_parity'),binary_sha256=sha(BUILD/'gdn_quad_parity'),binary_bytes=(BUILD/'gdn_quad_parity').stat().st_size,files={n:dict(sha256=sha(BUILD/n),bytes=(BUILD/n).stat().st_size) for n in ('compile_commands.json','CMakeCache.txt','build.ninja','.ninja_log')},target_commands_sha256=sha(targetcommands),complete=True,passed=True)
  except BaseException as e:r['error']=type(e).__name__+': '+str(e)
  finally:
   discover()
   if live():stop()
   r['survivors']=live();r['passed']=r['passed'] and not r['cleanup'] and not r['survivors']
   try:
    assert git(W,'rev-parse','HEAD')==HEAD and not git(W,'status','--porcelain')
    assert git(G,'rev-parse','HEAD')=='3cf03257f219afbe7334045ff7c6a06ac68c627d' and not git(G,'status','--porcelain')
    for p,h in PINS.items():assert sha(W/p)==h,p
    assert sha(__file__)==r['controller_sha256'];r['exit_pin_gate_passed']=True
   except BaseException as e:r['passed']=False;r['exit_pin_error']=repr(e)
   r.update(active=False,active_stage=None,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
   for fd in fds.values():os.close(fd)
  print(json.dumps({k:r.get(k) for k in ('passed','error','elapsed_seconds','cleanup','survivors','binary','binary_sha256','flag_comparison')}));return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
