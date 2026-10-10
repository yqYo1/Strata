"""Root-owned CPU compilation of the private necessary numerical screen; no GPU execution."""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,os,resource,shlex,signal,subprocess,time
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-gdn-gate-factor-probe-20261010')
SNAP=B/'research-source-snapshots/gdn-gate-factor-probe-root-v4/manifest.json'
HELP=B/'build_iq2s_timing_v2.py'
I=Path('/opt/intel/oneapi/compiler/2026.1/bin')
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
with (B/'owned-v0141-measurement.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(SNAP)=='4b488ae17f57bf791594af87dcd93485f00704d10d0cd3d8c2459125a0585f6f'
 assert sha(HELP)=='cbd98b28266a2b04b67adb7a1fb7127174f916f0999fedee813ff1466507f73d'
 spec=importlib.util.spec_from_file_location('root_session_owner',HELP);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 pins=json.loads(SNAP.read_text());compiler_sha=sha(I/'icpx')
 dependency_paths=[Path('/usr/include/level_zero/ze_api.h'),Path('/usr/lib/x86_64-linux-gnu/libze_loader.so'),Path('/opt/intel/oneapi/compiler/2026.1/include/sycl/ext/oneapi/backend/level_zero.hpp'),Path('/opt/intel/oneapi/compiler/2026.1/include/sycl/detail/backend_traits_level_zero.hpp')]
 dependencies={str(p):dict(realpath=str(p.resolve()),sha256=sha(p))for p in dependency_paths}
 def check_pins():
  assert sha(SNAP)=='4b488ae17f57bf791594af87dcd93485f00704d10d0cd3d8c2459125a0585f6f'
  assert sha(HELP)=='cbd98b28266a2b04b67adb7a1fb7127174f916f0999fedee813ff1466507f73d'
  assert sha(I/'icpx')==compiler_sha
  for p,x in dependencies.items():assert sha(p)==x['sha256'] and str(Path(p).resolve())==x['realpath'],p
  for p,x in pins.items():assert sha(W/p)==x['sha256'],p
 check_pins();out=B/'gdn-gate-factor-probe-cpu-build-v3';out.mkdir(mode=0o700);build=W/'build-gdn-gate-factor-probe-v3';assert not build.exists()
 env=dict(PATH=str(I)+':/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
 r=dict(active=True,complete=False,passed=False,scope='Private copied-expression necessary screen compile only, no device/model qualification',commands=[],cleanup=[],survivors=[],gpu_work_submitted=False,inference_run=False,model_opened=False,adopted=False,performance_eligible=False,full_lifecycle_passed=False,source_snapshot_sha256=sha(SNAP),source_pins=pins,controller_sha256=sha(__file__),helper_sha256=sha(HELP),compiler_sha256=compiler_sha,identity_dependency_pins=dependencies,environment=env,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),limits=dict(wall_seconds=600,AS_each_bytes=12<<30,RSS_session_bytes=6<<30,CPU_each_seconds=[300,301],FSIZE_each_bytes=128<<20,text_bytes=16<<20,NOFILE=256,CORE=0),owner_scope='Whole inherited session, including Ninja compiler process groups')
 start=time.monotonic();proc=None
 def save():(out/'record.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,argv):
  global proc
  e=dict(label=label,argv=argv,cwd=str(W));r['commands'].append(e);save();begin=time.monotonic()
  def limits():
   for k,v in [(resource.RLIMIT_AS,(12<<30,12<<30)),(resource.RLIMIT_CPU,(300,301)),(resource.RLIMIT_FSIZE,(128<<20,128<<20)),(resource.RLIMIT_NOFILE,(256,256)),(resource.RLIMIT_CORE,(0,0))]:resource.setrlimit(k,v)
  with (out/(label+'.stdout')).open('wb')as so,(out/(label+'.stderr')).open('wb')as se:
   proc=subprocess.Popen(argv,cwd=W,env=env,stdout=so,stderr=se,start_new_session=True,preexec_fn=limits)
   e['owner']=dict(pid=proc.pid,sid=proc.pid,start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]));save()
   while proc.poll()is None:
    assert time.monotonic()-start<600,'wall bound'
    rss=sum(x['rss_bytes']for x in h.session_members(proc.pid));r['peak_session_rss_bytes']=max(r.get('peak_session_rss_bytes',0),rss);assert rss<=6<<30,'RSS bound'
    assert sum(p.stat().st_size for p in out.glob('*.std*'))<=16<<20,'text bound';time.sleep(.1)
  e.update(exit_code=proc.returncode,elapsed_seconds=time.monotonic()-begin);save();assert proc.returncode==0,label;assert not h.session_members(proc.pid),label+' survivors'
  return(out/(label+'.stdout')).read_text()
 save()
 try:
  run('compiler-version',[str(I/'icpx'),'--version'])
  run('configure',['/usr/bin/cmake','-S',str(W/'sycl/tools/gdn-gate-factor-probe'),'-B',str(build),'-G','Ninja','-DCMAKE_CXX_COMPILER='+str(I/'icpx'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'])
  run('build',['/usr/bin/cmake','--build',str(build),'--target','gdn_gate_factor_probe','--parallel','2'])
  commands=json.loads((build/'compile_commands.json').read_text());assert len(commands)==1
  args=shlex.split(commands[0]['command']);required=['-O3','-DNDEBUG','-std=c++20','-fsycl','-fp-model=precise','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel'];assert all(x in args for x in required)
  assert not any(x in args for x in ['-ffast-math','-Ofast','-ffinite-math-only','-funsafe-math-optimizations','-fno-signed-zeros','-fassociative-math'])
  target=run('target-commands',['/usr/bin/ninja','-C',str(build),'-t','commands','gdn_gate_factor_probe'])
  assert '-Xsycl-target-backend=spir64' in target and '-cl-fp32-correctly-rounded-divide-sqrt' in target
  binary=build/'gdn_gate_factor_probe';imports=run('direct-imports',['/usr/bin/readelf','-d',str(binary)]);assert 'libze_loader.so.1' in imports
  check_pins();assert sha(__file__)==r['controller_sha256']
  r.update(complete=True,passed=True,binary=dict(path=str(binary),bytes=binary.stat().st_size,sha256=sha(binary)),compile_commands=dict(path=str(build/'compile_commands.json'),sha256=sha(build/'compile_commands.json')),required_device_math_flags=required,link_backend_option_checked=True,exit_pin_gate_passed=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  if proc is not None:
   if proc.poll()is None or h.session_members(proc.pid):
    r['cleanup'].append(dict(sid=proc.pid,signal='TERM'))
    for pg in {x['pgid']for x in h.session_members(proc.pid)}:
     try:os.killpg(pg,signal.SIGTERM)
     except ProcessLookupError:pass
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    if proc.poll()is None or h.session_members(proc.pid):
     r['cleanup'].append(dict(sid=proc.pid,signal='KILL'))
     for pg in {x['pgid']for x in h.session_members(proc.pid)}:
      try:os.killpg(pg,signal.SIGKILL)
      except ProcessLookupError:pass
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:r['stranded']=True
   r['survivors']=h.session_members(proc.pid)
   if r['cleanup']or r['survivors']:r['passed']=False
  r.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),logs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.glob('*.std*')});save()
 print(json.dumps(dict(record=str(out/'record.json'),sha256=sha(out/'record.json'),passed=r['passed'],error=r.get('error'),elapsed_seconds=r['elapsed_seconds'])))
 raise SystemExit(0 if r['passed']else 1)
