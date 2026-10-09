"""Root-only bounded CPU fixture; no engine build or GPU/model access."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,signal,subprocess,time
B=Path(__file__).parent
W=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-cache-route-pairs-v0141-20261010')
O=B/'cache-route-pairs-cpu-validation-v2'
assert not O.exists();O.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=(B/'owned-v0141-measurement.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pins={'sycl/include/strata/core/cache_route_pairs.hpp':'c06b174f8059a3fec6638bfdde59820126f00c1db809a30ce89a6087a888ad42','sycl/tools/cache-route-pairs-test.cpp':'fc9a5a3c036b385a3c1b26321dabf63d5416810509280a6988d446d24dbc326a','sycl/src/program/generate.cpp':'eb591e263398abd9b178ab00c61e4f6af53a53bad37f837b18cd323af8dbab05'}
for p,h in pins.items():assert sha(W/p)==h
closed=json.loads((B/'owned-repeat-capture-v0141-two-full-diagnostic-r3/record.json').read_text())
assert not closed['active'] and closed['exit_code']==0 and closed['exit_signal'] is None and not any(closed['cleanup'].values())
for who in ['inferior','debugger']:assert not Path('/proc',str(closed[who]['pid'])).exists()
start=time.monotonic();record={'active':True,'complete':False,'passed':False,'gpu_executed':False,'model_opened':False,'scope':'Portable pair-count CPU fixture only; actual SYCL engine build/runtime remains unqualified','boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_sha256':sha(__file__),'pins':pins,'steps':[],'cleanup':[],'deadline_seconds':120,'text_budget_bytes':67108864}
def save(): (O/'record.json').write_text(json.dumps(record,indent=2)+'\n')
def identity(pid):
 try:
  raw=Path('/proc',str(pid),'stat').read_text();a=raw[raw.rfind(')')+2:].split();return {'pid':pid,'start_ticks':int(a[19]),'pgrp':int(a[2])}
 except FileNotFoundError:return None
record['controller_identity']=identity(os.getpid());save()
env=os.environ.copy();env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1');record['sanitizer_environment']={k:env[k] for k in ['ASAN_OPTIONS','UBSAN_OPTIONS']}
def run(label,argv,seconds):
 step={'label':label,'argv':argv,'timeout_seconds':seconds};record['steps'].append(step)
 stdout=O/(label+'.stdout');stderr=O/(label+'.stderr');s=time.monotonic()
 with stdout.open('wb') as out,stderr.open('wb') as err:
  child=subprocess.Popen(argv,cwd=W,env=env,stdout=out,stderr=err,start_new_session=True);owner=identity(child.pid);step['owner']=owner;save()
  try:child.wait(timeout=seconds)
  except subprocess.TimeoutExpired:
   now=identity(child.pid)
   if now and now==owner:os.killpg(child.pid,signal.SIGKILL);record['cleanup'].append({'label':label,'killed_owned_group':owner})
   child.wait(timeout=5);raise
 step.update(exit_code=child.returncode,elapsed_seconds=time.monotonic()-s,stdout_sha256=sha(stdout),stderr_sha256=sha(stderr),stdout_bytes=stdout.stat().st_size,stderr_bytes=stderr.stat().st_size)
 assert step['stdout_bytes']+step['stderr_bytes']<=67108864 and child.returncode==0
 assert identity(child.pid) is None;save()
try:
 compiler=Path('/usr/bin/c++');record['compiler']=str(compiler.resolve());record['compiler_sha256']=sha(compiler)
 run('compile',[str(compiler),'-std=c++17','-pthread','-O1','-g','-UNDEBUG','-fno-omit-frame-pointer','-fsanitize=address,undefined','-I',str(W/'sycl/include'),str(W/'sycl/tools/cache-route-pairs-test.cpp'),'-o',str(O/'cache-route-pairs-test')],90)
 record['binary_sha256']=sha(O/'cache-route-pairs-test')
 run('fixture',[str(O/'cache-route-pairs-test')],20)
 for p,h in pins.items():assert sha(W/p)==h
 record.update(complete=True,passed=True)
except BaseException as e:record['error']=repr(e)
finally:
 record.update(active=False,elapsed_seconds=time.monotonic()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
print(json.dumps({k:record.get(k) for k in ['complete','passed','gpu_executed','error','cleanup','elapsed_seconds']}))
if not record['passed']:raise SystemExit(1)
