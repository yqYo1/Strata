"""Root-owned serialized CPU-only role-plan sanitizer build/fixtures."""
import datetime, fcntl, hashlib, json, os, resource, subprocess, time
from pathlib import Path
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-role-plan-v0141-20261010')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 with (B/'owned-v0141-measurement.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  out=B/'native-role-plan-host-cpu-validation-v1';out.mkdir(mode=0o700)
  build=W/'build-native-role-plan-host-v1'
  assert not build.exists()
  relative=['include/strata/artifact/native_role_plan.hpp','src/artifact/native_role_plan.cpp','sycl/tools/native-role-plan-host/CMakeLists.txt','sycl/tools/native-role-plan-host/native_role_plan_fixture.cpp','include/strata/artifact/gguf_reader.hpp','include/strata/artifact/gguf_split.hpp','include/strata/kernels/cpu/expert_layout.hpp','include/strata/kernels/cpu/native_expert.hpp','include/strata/kernels/cpu/expert.hpp']
  pins={str(W/p):sha(W/p) for p in relative}
  env=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8',ASAN_OPTIONS='detect_leaks=1:abort_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
  record=dict(active=True,complete=False,passed=False,scope='Thin standalone shared GGUF reader/role owner with synthetic bytes; ASan+UBSan; no actual model, native kernels, pool or production engine',controller_sha256=sha(Path(__file__)),pins=pins,environment=env,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),commands=[],cleanup=[],survivors=[],peak_rss_bytes=0,wall_deadline_seconds=180,text_budget_bytes=8388608,gpu_work_submitted=False,model_opened=False,inference_run=False,actual_weight_service_measured=False,production_checker_integrated=False,performance_eligible=False,full_lifecycle_passed=False,adopted=False,limits=dict(CPU_soft_seconds=120,CPU_hard_seconds=121,AS_build_bytes=4294967296,AS_fixture_bytes=None,AS_fixture_reason='ASan reserves a large shadow address space; RSS bounded separately',RSS_poll_bytes=1073741824,FSIZE_bytes=8388608,CORE_bytes=0,NOFILE=128))
  rp=out/'record.json';start=time.monotonic();proc=None
  def save():rp.write_text(json.dumps(record,indent=2)+'\n')
  def run(label,argv,asan=False):
   nonlocal proc
   stdout=out/(label+'.stdout');stderr=out/(label+'.stderr')
   def limits():
    for kind,values in ((resource.RLIMIT_CPU,(120,121)),(resource.RLIMIT_FSIZE,(8388608,8388608)),(resource.RLIMIT_NOFILE,(128,128)),(resource.RLIMIT_CORE,(0,0))):resource.setrlimit(kind,values)
    if not asan:resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296))
   c=dict(label=label,argv=argv,cwd=str(W),AS_limit_enabled=not asan,active=True);record['commands'].append(c);save();begin=time.monotonic()
   with stdout.open('wb') as so,stderr.open('wb') as se:
    proc=subprocess.Popen(argv,cwd=W,env=env,stdout=so,stderr=se,preexec_fn=limits,start_new_session=True)
    c['owner']=dict(pid=proc.pid,start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]));save()
    while proc.poll() is None:
     assert time.monotonic()-start<180,'total wall deadline'
     assert sum(p.stat().st_size for p in out.glob('*.std*'))<8388608,'aggregate text budget'
     try:status=Path(f'/proc/{proc.pid}/status').read_text()
     except FileNotFoundError:status=''
     for line in status.splitlines():
      if line.startswith('VmRSS:'):
       rss=int(line.split()[1])*1024;record['peak_rss_bytes']=max(record['peak_rss_bytes'],rss);assert rss<=1073741824,'RSS cap'
     time.sleep(.05)
   c.update(active=False,exit_code=proc.returncode,elapsed_seconds=time.monotonic()-begin)
   assert proc.returncode==0,label+' failed'
   save();return stdout.read_text()
  save()
  try:
   run('compiler',['/usr/bin/g++','--version'])
   run('configure',['/usr/bin/cmake','-S',str(W/'sycl/tools/native-role-plan-host'),'-B',str(build),'-G','Ninja','-DCMAKE_CXX_COMPILER=/usr/bin/g++','-DCMAKE_BUILD_TYPE=Debug','-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer','-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined'])
   run('build',['/usr/bin/cmake','--build',str(build),'--parallel','1'])
   binary=build/'native_role_plan_fixture'
   imports=run('dynamic-imports',['/usr/bin/readelf','-d',str(binary)])
   assert not any(x in imports.lower() for x in ('libsycl','libur_','libze_loader','libmkl')),'unexpected device runtime import'
   result=run('fixtures',[str(binary),str(out/'synthetic-gguf')],asan=True)
   assert result.strip()=='PASS 22 independent role-plan fixture groups','fixture completion/count'
   for p,h in pins.items():assert sha(Path(p))==h,'source changed'
   commands=json.loads((build/'compile_commands.json').read_text())
   assert len(commands)==2 and all('-fsycl' not in c['command'] for c in commands),'thin compile closure'
   record.update(complete=True,passed=True,fixture_groups=22,binary=dict(path=str(binary),bytes=binary.stat().st_size,sha256=sha(binary)),compile_commands=dict(path=str(build/'compile_commands.json'),sha256=sha(build/'compile_commands.json')),static_import_scope='readelf direct NEEDED only; no runtime device-API tripwire claim')
  except BaseException as e:record['error']=type(e).__name__+': '+str(e)
  finally:
   if proc is not None and proc.poll() is None:
    proc.terminate();record['cleanup'].append('TERM owned process')
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();record['cleanup'].append('KILL owned process');proc.wait(timeout=5)
   if proc is not None and proc.poll() is None:record['survivors'].append(record['commands'][-1].get('owner'))
   record['logs']={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.glob('*.std*')}
   record.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
  print(json.dumps(dict(record=str(rp),sha256=sha(rp),passed=record['passed'],elapsed_seconds=record['elapsed_seconds'],error=record.get('error'))))
  return 0 if record['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
