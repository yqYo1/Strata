"""Serial root-owned hardware characterization. No engine changes/adoption."""
from pathlib import Path
import datetime,fcntl,hashlib,json,math,os,re,statistics,subprocess,sys,types
B=Path(__file__).parent
STAGE=sys.argv[1]
assert STAGE in ['cpu','measure-gpu']
assert STAGE in ['build-cpu','cpu','build-gpu','qualify-gpu','measure-gpu']
PARENT=B/'run_gdn_gate_factor_probe_v2.py'
PARENT_SHA='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
def sha(p):
 with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def ident(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
def closed(stage):
 p=B/f'hardware-ceilings-v1-{stage}/record.json';x=json.loads(p.read_text())
 assert x['passed'] and x['complete'] and not x['active']
 return p,x
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(PARENT)==PARENT_SHA
 for mode in ['baseline','pool0']:
  p=B/f'owned-decode-pool-phase-32769-v1-{mode}/record.json';x=json.loads(p.read_text())
  assert x['completed'] and x['healthy'] and x['math_gate_passed'] and not x['active']
 assert not (Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump')).exists()
 module=types.ModuleType('hardware_characterization_owner')
 exec(compile(PARENT.read_text().split('\ndef parse_probe(',1)[0],str(PARENT),'exec'),module.__dict__)
 module.W=B
 out=B/f'hardware-ceilings-repeat-v1-{STAGE}-r3';assert not out.exists();out.mkdir(mode=0o700)
 owner=module.Owner(out)
 source=B/('hardware_ceiling_cpu_v1.cpp' if STAGE in ['build-cpu','cpu'] else 'hardware_ceiling_gpu_v1.cpp')
 binary=B/('hardware-ceiling-cpu-v1' if STAGE in ['build-cpu','cpu'] else 'hardware-ceiling-gpu-v1')
 clean=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
 r=dict(active=True,complete=False,passed=False,stage=STAGE,started_utc=module.utc(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source=dict(path=str(source),**ident(source)),controller=ident(__file__),parent_owner=ident(PARENT),commands=owner.commands,adopted=False,model_executed=False,scope='Hardware streaming and production-shape dense GEMM characterization; logical traffic is not hardware-counter DRAM traffic. Host-inclusive monotonic time, no GPU profiling timestamps. Not model throughput.',limits=dict(AS_each_bytes=16<<30,RSS_session_sampled_bytes=2<<30,CPU_each_seconds=[120,121],wall_seconds=240))
 def save():
  tmp=out/'record.json.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(out/'record.json')
 owner.persist=save
 def run(label,args,env=clean,wall=120):
  e,so,se=owner.run(label,args,env,wall=wall,text_cap=16<<20,file_cap=16<<20);save()
  assert module.completed(e) and e['exit_code']==0,(label,e)
  return so,se
 try:
  if STAGE.startswith('build'):
   assert not binary.exists()
   if STAGE=='build-cpu':
    env=clean
    cmd=['/usr/bin/g++','-std=c++20','-O3','-mavx2','-pthread','-Wall','-Wextra',str(source),'-o',str(binary)]
   else:
    so,se=run('toolchain-environment',['/bin/bash','-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'],wall=30)
    discovered=dict(x.decode().split('=',1) for x in so.read_bytes().split(b'\0') if b'=' in x)
    env=dict(clean)
    for key in ['PATH','LD_LIBRARY_PATH','LIBRARY_PATH','CPATH','ONEAPI_ROOT','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','MKLROOT']:
     if key in discovered:env[key]=discovered[key]
    cmd=['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-std=c++20','-O3','-fsycl','-fp-model=precise','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-qmkl=sequential','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt',str(source),'-lze_loader','-o',str(binary)]
   r['environment']=env;save();run('compile',cmd,env,wall=240)
   r['binary']=dict(path=str(binary),**ident(binary))
  else:
   buildpath,build=closed('build-cpu' if STAGE=='cpu' else 'build-gpu')
   assert build['source']==r['source'] and ident(binary)=={k:build['binary'][k] for k in ['bytes','sha256']}
   r.update(binary=build['binary'],build_receipt=ident(buildpath))
   env=dict(build['environment'])
   gpu=STAGE.endswith('gpu')
   if gpu:
    if STAGE=='measure-gpu':
     qp,q=closed('qualify-gpu');assert q['binary']==build['binary'];r['qualification_receipt']=ident(qp)
    env.update(UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',SYCL_CACHE_PERSISTENT='0',EnableDirectSubmission='0',NEOReadDebugKeys='1')
    if STAGE=='qualify-gpu':env.update(ZEL_ENABLE_LOADER_LOGGING='1',ZEL_LOADER_LOG_CONSOLE='1',ZEL_LOADER_LOGGING_LEVEL='warn',ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT='0',ZE_ENABLE_VALIDATION_LAYER='1',ZE_ENABLE_PARAMETER_VALIDATION='1',UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
    so,se=run('kernel-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager'],wall=10)
    m=re.search(r'^-- cursor: (.+)$',so.read_text(),re.M);assert m;r['kernel_cursor']=m[1]
   r['environment']=env
   r['host_configuration']={str(p):p.read_text().strip() for p in [Path('/proc/cpuinfo'),Path('/proc/meminfo'),Path('/proc/sys/kernel/perf_event_paranoid')]}
   r['cpu_affinity']=sorted(os.sched_getaffinity(0))
   r['cpu_topology']={p.parent.parent.name:p.read_text().strip() for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/topology/thread_siblings_list')}
   r['cpu_frequency_policy']={str(p):p.read_text().strip() for name in ['scaling_governor','scaling_min_freq','scaling_max_freq','energy_performance_preference'] for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/'+name)}
   save()
   args=[str(binary)]+(['--qualify' if STAGE=='qualify-gpu' else '--measure'] if gpu else [])
   so,se=run('benchmark',args,env,wall=240)
   rows=[json.loads(line) for line in so.read_text().splitlines()];assert rows and rows[0]['kind']=='configuration'
   r['result_rows']=rows;r['benchmark_stderr']=ident(se)
   for row in rows:
    if row['kind'] in ['sample','gemm_sample']:assert math.isfinite(row['seconds']) and row['seconds']>0
    if row['kind'] in ['validation','gemm_validation']:assert row['passed']
   if gpu:
    assert rows[0]['device']=='B570' and rows[0]['pci']=='0000:05:00.0'
    assert sum(x['kind']=='validation' for x in rows)==20 and sum(x['kind']=='gemm_validation' for x in rows)==40
    assert sum(x['kind']=='sample' for x in rows)==20*(1 if STAGE=='qualify-gpu' else 7)
    assert sum(x['kind']=='gemm_sample' for x in rows)==40*(1 if STAGE=='qualify-gpu' else 7)
    so2,se2=run('kernel-interval',['/usr/bin/journalctl','-k','--after-cursor='+r['kernel_cursor'],'--no-pager','-o','short-monotonic'],wall=10)
    r['visible_kernel_GPU_entries']=[s for s in so2.read_text().splitlines() if re.search(r'\bxe\b|i915|GPU HANG|devcoredump',s,re.I)]
    r['devcoredump_after']=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
    assert not r['visible_kernel_GPU_entries'] and not r['devcoredump_after']
   else:
    assert sum(x['kind']=='validation' for x in rows)==20
    assert sum(x['kind']=='sample' for x in rows)==160
    assert not se.read_text()
  assert r['source']==dict(path=str(source),**ident(source))
  r.update(passed=True,complete=True)
 except BaseException as e:r['error']=type(e).__name__+': '+str(e)
 finally:
  r.update(active=owner.active is not None,finished_utc=module.utc())
  r['passed']=r['passed'] and not r['active'] and all(module.completed(e) for e in owner.commands)
  save();print(json.dumps({k:r.get(k) for k in ['stage','active','complete','passed','error']}),flush=True)
 if not r['passed']:sys.exit(1)
