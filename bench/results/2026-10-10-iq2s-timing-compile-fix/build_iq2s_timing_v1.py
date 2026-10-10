"""Root-owned fresh CPU timing-harness build admission."""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, importlib.util, json, os, resource, shlex, signal, subprocess, time

B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-actual-cohort-timing-20261010')
G=Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
I=Path('/opt/intel/oneapi/compiler/2026.1/bin')
BASE=B/'iq2s-actual-cohort-cpu-build-v1/record.json'
HELP=B/'build_iq2s_actual_cohort_cpu_v1.py'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def session_members(sid):
 result=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdecimal():continue
  try:
   s=(p/'stat').read_text().rsplit(')',1)[1].split()
   if int(s[3])==sid and s[0]!='Z':result.append(dict(pid=int(p.name),start_ticks=int(s[19]),pgid=int(s[2]),sid=int(s[3]),rss_bytes=int(s[21])*os.sysconf('SC_PAGE_SIZE')))
  except(FileNotFoundError,ProcessLookupError):pass
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('variant',choices=('release','asan','default'));variant=ap.parse_args().variant
 with (B/'owned-v0141-measurement.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert sha(BASE)=='292849fedd2f0b9460304f3c3f317d3b9fcab483734e3d95103d23e4c1131de1'
  assert sha(HELP)=='1ef9245586c14264a374d0a6e702e70c1d5fb30f19f67948709148fbb9b2cb2f'
  base=json.loads(BASE.read_text());assert base['passed'] and base['complete'] and not base['active'] and not base['cleanup'] and not base['survivors']
  spec=importlib.util.spec_from_file_location('root_build',HELP);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
  def pins():
   for p,v in base['pins'].items():assert sha(p)==v,p
   for p,v in base['production_source_pins'].items():assert sha(W/p)==v,p
   for p,v in json.loads((B/'research-source-snapshots/iq2s-actual-timing-source-v1/manifest.json').read_text()).items():assert sha(W/p)==v['sha256'],p
   for p,v in base['ggml_source_pins'].items():assert sha(G/p)==v,p
   for p,v in base['compiler_pins'].items():assert sha(p)==v,p
   assert sha(base['compile_commands']['path'])==base['compile_commands']['sha256']
  pins();out=B/('iq2s-timing-'+variant+'-build-v1');out.mkdir(mode=0o700)
  build=W/('build-iq2s-timing-'+variant+'-v1');assert not build.exists()
  env=dict(PATH=str(I)+':/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
  rec=dict(active=True,complete=False,passed=False,variant=variant,controller_sha256=sha(__file__),base_build_receipt_sha256=sha(BASE),helper_sha256=sha(HELP),commands=[],cleanup=[],survivors=[],environment=env,gpu_work_submitted=False,inference_run=False,model_opened=False,performance_eligible=False,adopted=False,owner_scope='Entire inherited session, including Ninja compiler process groups',source_commit=subprocess.check_output(['/usr/bin/git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),limits=dict(AS_each_bytes=12<<30,RSS_group_bytes=8<<30,CPU_each_soft_seconds=300,CPU_each_hard_seconds=301,FSIZE_each_bytes=128<<20,NOFILE=256,CORE=0,wall_total_seconds=600,text_budget_bytes=16<<20))
  start=time.monotonic();proc=None
  def save():(out/'record.json').write_text(json.dumps(rec,indent=2)+'\n')
  def run(label,cmd):
   nonlocal proc
   ent=dict(label=label,argv=cmd,cwd=str(W));rec['commands'].append(ent);save();begin=time.monotonic()
   def limits():
    for k,v in [(resource.RLIMIT_AS,(12<<30,12<<30)),(resource.RLIMIT_CPU,(300,301)),(resource.RLIMIT_FSIZE,(128<<20,128<<20)),(resource.RLIMIT_NOFILE,(256,256)),(resource.RLIMIT_CORE,(0,0))]:resource.setrlimit(k,v)
   with (out/(label+'.stdout')).open('wb') as so,(out/(label+'.stderr')).open('wb') as se:
    proc=subprocess.Popen(cmd,cwd=W,env=env,stdout=so,stderr=se,preexec_fn=limits,start_new_session=True)
    ent['owner']=dict(pid=proc.pid,pgid=proc.pid,start_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rsplit(')',1)[1].split()[19]));save()
    while proc.poll() is None:
     assert time.monotonic()-start<600
     rss=sum(x['rss_bytes'] for x in session_members(proc.pid));rec['peak_group_rss_bytes']=max(rec.get('peak_group_rss_bytes',0),rss);assert rss<=8<<30
     assert sum(p.stat().st_size for p in out.glob('*.std*'))<16<<20
     time.sleep(.1)
   ent.update(exit_code=proc.returncode,elapsed_seconds=time.monotonic()-begin);assert proc.returncode==0,label
   assert not session_members(proc.pid);save();return (out/(label+'.stdout')).read_text()
  save()
  try:
   config=['/usr/bin/cmake','-S',str(W/'sycl/tools/native-service-calibration'),'-B',str(build),'-G','Ninja','-DCMAKE_C_COMPILER='+str(I/'icx'),'-DCMAKE_CXX_COMPILER='+str(I/'icpx'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DSTRATA_ROOT='+str(W),'-DGGML_SOURCE_DIR='+str(G/'ggml')]
   if variant=='asan':config+=['-DSTRATA_CALIBRATION_IQ2S_INDEX_CHECK=ON','-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer','-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer','-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined']
   if variant=='release':config+=['-DSTRATA_CALIBRATION_IQ2S_INDEX_CHECK=ON']
   run('configure',config);run('build',['/usr/bin/cmake','--build',str(build),'--target','native_service_calibration','--parallel','4'])
   binary=build/'native_service_calibration';imports=run('direct-imports',['/usr/bin/readelf','-d',str(binary)])
   assert not any(x in imports.lower() for x in ('libsycl','libur_','libze_loader','libmkl','libiomp','libgomp'))
   cc=build/'compile_commands.json';commands=json.loads(cc.read_text());assert all('-fsycl' not in x['command'] for x in commands)
   def canon(c,directory):
    excluded={'-fsanitize=address,undefined','-fno-omit-frame-pointer','-DSTRATA_CALIBRATION_IQ2S_INDEX_CHECK=1'}
    return [x.replace(str(directory),'{BUILD}').replace(str(W),'{ROOT}').replace('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010','{ROOT}') for x in shlex.split(c['command']) if x not in excluded]
   before={x['file'].replace('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-iq2s-nt1-index-spread-20261010','{ROOT}'):canon(x,Path(base['compile_commands']['path']).parent) for x in json.loads(Path(base['compile_commands']['path']).read_text())}
   after={x['file'].replace(str(W),'{ROOT}'):canon(x,build) for x in commands};common=set(before)&set(after);diff=[x for x in sorted(common) if before[x]!=after[x]]
   added=sorted(set(after)-set(before));missing=sorted(set(before)-set(after));expected=[] if variant in ('asan','release') else ['{ROOT}/sycl/tools/native-service-calibration/iq2s_index_dot.cpp']
   rec['compile_comparison']=dict(common_units=len(common),mismatches=diff,added_units=added,missing_units=missing,excluded_flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-DSTRATA_CALIBRATION_IQ2S_INDEX_CHECK=1'])
   assert not diff and not added and missing==expected
   if variant=='asan':assert all('-fsanitize=address,undefined' in x['command'] for x in commands)
   elif variant=='default':
    assert 'STRATA_CALIBRATION_IQ2S_INDEX_CHECK:BOOL=OFF' in (build/'CMakeCache.txt').read_text()
    assert all('STRATA_CALIBRATION_IQ2S_INDEX_CHECK' not in x['command'] for x in commands)
    symbols=run('defined-symbols',['/usr/bin/nm','--defined-only',str(binary)])
    assert 'isolated_iq2s' not in symbols and 'index_correctness' not in symbols
   pins();assert sha(__file__)==rec['controller_sha256']
   rec.update(complete=True,passed=True,binary=dict(path=str(binary),bytes=binary.stat().st_size,sha256=sha(binary)),compile_commands=dict(path=str(cc),bytes=cc.stat().st_size,sha256=sha(cc)),default_option_omitted=(variant=='default'),production_source_pins_stable=True)
  except BaseException as e:rec['error']=type(e).__name__+': '+str(e)
  finally:
   if proc is not None:
    members=session_members(proc.pid)
    if proc.poll() is None or members:
     rec['cleanup'].append(dict(pgid=proc.pid,signal='TERM',members=members))
     try:[os.killpg(g,signal.SIGTERM) for g in sorted({x['pgid'] for x in session_members(proc.pid)})]
     except ProcessLookupError:pass
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:pass
     if proc.poll() is None or session_members(proc.pid):
      rec['cleanup'].append(dict(pgid=proc.pid,signal='KILL'))
      try:[os.killpg(g,signal.SIGKILL) for g in sorted({x['pgid'] for x in session_members(proc.pid)})]
      except ProcessLookupError:pass
      proc.wait(timeout=5)
    rec['survivors']=session_members(proc.pid)
   rec.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),logs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.glob('*.std*')});save()
  print(json.dumps(dict(record=str(out/'record.json'),sha256=sha(out/'record.json'),passed=rec['passed'],error=rec.get('error'),elapsed_seconds=rec['elapsed_seconds'])))
  return 0 if rec['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
