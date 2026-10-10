"""Root-only, finite first returned-event check. No inference or speed claim."""
from pathlib import Path
import fcntl,hashlib,json,re,subprocess,sys,types
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-prefill-service-qualification-20261010')
OUT=B/'prefill-service-small-device-qualification-v1'
BUILD=B/'prefill-service-qualification-cpu-build-v1/record.json'
HEAD='7c5ad75a45c7e188ecef7a947b49b829dc5c5710'
def ident(p):
 with Path(p).open('rb') as f:return dict(bytes=Path(p).stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
with (B/'owned-v0141-measurement.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 build=json.loads(BUILD.read_text());assert build['passed'] and build['complete'] and not build['active']
 assert build['commit']==HEAD and not build['gpu_tested']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==HEAD
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
 assert not Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
 for rel,pin in build['sources'].items():assert ident(W/rel)==pin
 binary=BUILD.parent/'qualify_prefill_f16_event';assert ident(binary)==build['binaries'][binary.name]
 parent=B/'run_gdn_gate_factor_probe_v2.py';assert ident(parent)['sha256']=='7df012ee4d9bd047a6094ccedb37fddb3d56d4b054d9fdf468534b0fb6e94e1f'
 source=parent.read_text().split('\ndef parse_probe(',1)[0]
 module=types.ModuleType('service_small_owner');exec(compile(source,str(parent),'exec'),module.__dict__)
 assert not OUT.exists();OUT.mkdir(mode=0o700);module.W=OUT;owner=module.Owner(OUT)
 helper=W/'sycl/tools/recover-xe.sh';helper_source=helper.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
 recovery=types.ModuleType('xe_diag_env');exec(compile(helper_source,str(helper),'exec'),recovery.__dict__)
 env=dict(build['environment']);env.update(UR_ADAPTERS_FORCE_LOAD='/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero_v2.so.0',UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',ONEAPI_DEVICE_SELECTOR='level_zero:gpu',SYCL_CACHE_PERSISTENT='0',EnableDirectSubmission='0',NEOReadDebugKeys='1')
 env=recovery.diagnostic_environment(env)
 r=dict(active=True,complete=False,passed=False,source_commit=HEAD,build_receipt=dict(path=str(BUILD),**ident(BUILD)),
  binary=dict(path=str(binary),**ident(binary)),controller=ident(__file__),parent_owner=ident(parent),
  owner_source_sha256=hashlib.sha256(source.encode()).hexdigest(),diagnostic_helpers={str(p):ident(p) for p in [helper,W/'sycl/tools/debug-run.py']},
  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),environment=env,commands=owner.commands,
  model_executed=False,adopted=False,scope='Real Gemm::f16_event and unchanged f16, exact full119float output/padding check, completion and3profiling fields. No internal-kernel span or production service, clean performance or full-context qualification.',started_utc=module.utc())
 def save():
  p=OUT/'record.json.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(OUT/'record.json')
 owner.persist=save
 def run(label,args,e,wall):
  cmd,so,se=owner.run(label,args,e,wall=wall,text_cap=8<<20,file_cap=32<<20);save()
  assert module.completed(cmd) and cmd['exit_code']==0,(label,cmd);return so,se
 try:
  clean=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',LC_ALL='C.UTF-8')
  before,_=run('kernel-cursor',['/usr/bin/journalctl','-k','-n','0','--show-cursor','--no-pager'],clean,10)
  cursor=re.search(r'^-- cursor: (.+)$',before.read_text(),re.M);assert cursor;r['kernel_cursor']=cursor[1];save()
  stdout,stderr=run('qualification',['/usr/bin/python3',str(W/'sycl/tools/debug-run.py'),str(binary)],env,120)
  text=stdout.read_text();match=re.fullmatch(r'f16 returned-event qualification PASS: T=7 N=11 K=13 ldy=17 half/half/float trans/nontrans; full 119 float outputs and padding; submit=(\d+) start=(\d+) end=(\d+); not internal kernel coverage\n',text)
  assert match,repr(text)
  stamps=list(map(int,match.groups()));assert stamps[0]<=stamps[1]<=stamps[2]
  r['result']=dict(exact_outputs_and_padding=119,profiling_fields=dict(zip(['submit','start','end'],stamps)),profiling_order_valid=True,
     queue_device_admission='Exact B570 name in pinned fixture; selected level_zero:gpu')
  r['stdout']=dict(path=str(stdout),**ident(stdout));r['stderr']=dict(path=str(stderr),**ident(stderr));save()
  after,_=run('kernel-interval',['/usr/bin/journalctl','-k','--after-cursor='+cursor[1],'--no-pager','-o','short-monotonic'],clean,10)
  r['visible_kernel_GPU_entries']=[line for line in after.read_text().splitlines() if re.search(r'\bxe\b|i915|GPU HANG|devcoredump',line,re.I)]
  r['devcoredump_after']=Path('/sys/bus/pci/devices/0000:05:00.0/devcoredump').exists()
  assert not r['visible_kernel_GPU_entries'] and not r['devcoredump_after']
  assert ident(binary)==build['binaries'][binary.name]
  for rel,pin in build['sources'].items():assert ident(W/rel)==pin
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==HEAD
  assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)
  r.update(passed=True,complete=True)
 except BaseException as error:r['error']=type(error).__name__+': '+str(error)
 finally:
  r.update(active=owner.active is not None,finished_utc=module.utc());r['passed']=r['passed'] and not r['active'] and all(module.completed(cmd) for cmd in owner.commands)
  save();print(json.dumps({k:r.get(k) for k in ['passed','complete','active','error','result']}),flush=True)
 if not r['passed']:sys.exit(1)
