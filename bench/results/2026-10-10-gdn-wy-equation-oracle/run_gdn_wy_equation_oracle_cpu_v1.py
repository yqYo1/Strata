"""Root-only standalone synthetic equation/sanitizer qualification, no GPU/model."""
from pathlib import Path
import csv,datetime,fcntl,hashlib,json,math,os,resource,signal,subprocess,time

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-gdn-wy-equation-oracle-20261010')
S=W/'sycl/tools/gdn-wy-equation-oracle'
PINS={'CMakeLists.txt':'e0d8b41e23da215248e517258bb91f7046ce7dc32e29b324c82f1caa96186627',
      'gdn_wy_equation_oracle.cpp':'335bcdd934fa548385ddd9b20b2c5e4f200574da8a9d104332efd70cbd64cfd4',
      'README.md':'352280fec19875502e13bed20f9034f2a45cda6af7892673a3385dba5ebc4114'}
ENV=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def members(pgid):
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdecimal():continue
  try:
   s=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(s[2])==pgid and s[0]!='Z':result.append(dict(pid=int(p.name),start_ticks=int(s[19]),rss_bytes=int(s[21])*os.sysconf('SC_PAGE_SIZE')))
  except (FileNotFoundError,ProcessLookupError):pass
 return result

def main():
 with (B/'owned-v0141-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for rel,h in PINS.items():assert sha(S/rel)==h
  out=B/'gdn-wy-equation-oracle-cpu-validation-v1';out.mkdir(mode=0o700)
  rp=out/'record.json';start=time.monotonic()
  record=dict(active=True,complete=False,passed=False,controller_sha256=sha(__file__),source_pins=PINS,
              compiler_path='/usr/bin/g++',compiler_sha256=sha('/usr/bin/g++'),commands=[],cleanup=[],survivors=[],
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              scope='Double synthetic GDN/WY identity and finite FP32 characterization; no production emulation/model parity/performance',
              gpu_work_submitted=False,model_opened=False,inference_run=False,performance_eligible=False,adopted=False,full_lifecycle_passed=False)
  def save():rp.write_text(json.dumps(record,indent=2)+'\n')
  def run(label,argv,env=ENV,address_bytes=8<<30,wall=120,rss_cap=1536<<20):
   so=out/(label+'.stdout');se=out/(label+'.stderr');entry=dict(label=label,argv=argv,environment=env,AS_bytes=address_bytes,RSS_group_bytes=rss_cap,wall_seconds=wall,CPU_each_seconds=[120,121],FSIZE_each_bytes=2<<20,NOFILE=256,CORE_bytes=0);record['commands'].append(entry);save();proc=None;t=time.monotonic()
   try:
    with so.open('wb') as f,se.open('wb') as e:
     def limits():
      for kind,pair in ((resource.RLIMIT_AS,(address_bytes,address_bytes) if address_bytes is not None else (resource.RLIM_INFINITY,resource.RLIM_INFINITY)),(resource.RLIMIT_CPU,(120,121)),(resource.RLIMIT_FSIZE,(2<<20,2<<20)),(resource.RLIMIT_NOFILE,(256,256)),(resource.RLIMIT_CORE,(0,0))):resource.setrlimit(kind,pair)
     proc=subprocess.Popen(argv,cwd=W,env=env,stdout=f,stderr=e,preexec_fn=limits,start_new_session=True)
     entry['owner']=dict(pid=proc.pid,pgid=proc.pid,start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]));save()
     while proc.poll() is None:
      assert time.monotonic()-t<wall,'stage deadline'
      rss=sum(x['rss_bytes'] for x in members(proc.pid));entry['peak_group_rss_bytes']=max(entry.get('peak_group_rss_bytes',0),rss);assert rss<rss_cap,'RSS limit'
      assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<8<<20,'text limit'
      time.sleep(.05)
    entry['exit_code']=proc.returncode;assert proc.returncode==0 and not members(proc.pid),'stage failure/descendants'
   finally:
    if proc is not None:
     if proc.poll() is None or members(proc.pid):
      record['cleanup'].append(dict(label=label,pgid=proc.pid,signal='TERM'))
      try:os.killpg(proc.pid,signal.SIGTERM)
      except ProcessLookupError:pass
      try:proc.wait(timeout=5)
      except subprocess.TimeoutExpired:pass
      if proc.poll() is None or members(proc.pid):
       record['cleanup'].append(dict(label=label,pgid=proc.pid,signal='KILL'))
       try:os.killpg(proc.pid,signal.SIGKILL)
       except ProcessLookupError:pass
       proc.wait(timeout=5)
     record['survivors'].extend(members(proc.pid))
    entry.update(elapsed_seconds=time.monotonic()-t,stdout_sha256=sha(so),stderr_sha256=sha(se));save()
   return so
  save()
  try:
   run('compiler',['/usr/bin/g++','--version'])
   variants={}
   for variant in ('release','asan-ubsan'):
    build=W/('build-gdn-wy-oracle-'+variant+'-v1');assert not build.exists()
    config=['/usr/bin/cmake','-S',str(S),'-B',str(build),'-G','Ninja','-DCMAKE_CXX_COMPILER=/usr/bin/g++','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
    if variant=='release':config+=['-DCMAKE_BUILD_TYPE=Release']
    else:config+=['-DCMAKE_BUILD_TYPE=Debug','-DCMAKE_CXX_FLAGS=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer','-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined']
    run(variant+'-configure',config);run(variant+'-build',['/usr/bin/cmake','--build',str(build),'--target','gdn_wy_equation_oracle','--parallel','2'])
    compile_file=build/'compile_commands.json';rules=json.loads(compile_file.read_text());assert len(rules)==1 and Path(rules[0]['file']).resolve()==S/'gdn_wy_equation_oracle.cpp'
    flags=rules[0]['command'].split();assert '-fno-fast-math' in flags and '-ffp-contract=off' in flags
    assert not set(flags)&{'-Ofast','-ffast-math','-fsycl','-fassociative-math','-ffinite-math-only','-funsafe-math-optimizations','-freciprocal-math'}
    (out/(variant+'-compile_commands.json')).write_text(compile_file.read_text())
    binary=build/'gdn_wy_equation_oracle';imports=run(variant+'-imports',['/usr/bin/readelf','-d',str(binary)])
    assert not any(x in imports.read_text().lower() for x in ('libsycl','libur_','libze_loader','libmkl','libigc'))
    env=ENV if variant=='release' else {**ENV,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1:print_stacktrace=1'}
    log=run(variant+'-fixtures',[str(binary)],env,address_bytes=512<<20 if variant=='release' else None,rss_cap=512<<20)
    rows=list(csv.reader(log.read_text().splitlines()));assert rows[-1]==['RESULT','synthetic_math_pass','cases','100','model_parity','false','adopted','false','performance','false','full_lifecycle','false']
    cases=[x for x in rows if x[:1]==['CASE'] and x[1]!='ordinal'];assert len(cases)==100 and all(x[-1]=='pass' for x in cases)
    assert [int(x[1]) for x in cases]==list(range(1,101)) and all(len(x)==18 for x in cases)
    assert all(all(math.isfinite(float(v)) and float(v)>=0 for v in x[10:17]) for x in cases)
    negative=[x for x in rows if x[:1]==['NEGATIVE']];assert len(negative)==1 and negative[0][1]=='contiguous_head_div3' and float(negative[0][2])>1e-6 and negative[0][-1]=='rejected'
    counts=[x for x in rows if x[:1]==['PHYSICAL_COUNTS']];assert len(counts)==5
    variants[variant]=dict(binary_sha256=sha(binary),binary_bytes=binary.stat().st_size,compile_commands_sha256=sha(compile_file),fixture_stdout_sha256=sha(log),positive_cases=100,negative_wrong_head_map_discriminated=True,max_double_output_abs=max(float(x[10]) for x in cases),max_double_state_abs=max(float(x[11]) for x in cases),max_float_output_abs=max(float(x[12]) for x in cases),max_float_state_abs=max(float(x[13]) for x in cases),AS_note='ASan requires large sparse shadow virtual mapping; AS unlimited, fixed fixture bounds plus512MiB polled RSS and CPU/wall/output caps' if variant!='release' else '512MiB AS and RSS',physical_counts_only=counts)
   record.update(variants=variants,complete=True,passed=True)
  except BaseException as e:record['error']=type(e).__name__+': '+str(e)
  finally:
   try:
    assert all(sha(S/rel)==h for rel,h in PINS.items());assert sha('/usr/bin/g++')==record['compiler_sha256'];assert sha(__file__)==record['controller_sha256'];record['exit_pin_gate_passed']=True
   except BaseException as e:record['passed']=False;record['exit_pin_error']=type(e).__name__+': '+str(e)
   record.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
  print(json.dumps(dict(record=str(rp),sha256=sha(rp),passed=record['passed'],error=record.get('error'),elapsed_seconds=record['elapsed_seconds'])))
  return 0 if record['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
