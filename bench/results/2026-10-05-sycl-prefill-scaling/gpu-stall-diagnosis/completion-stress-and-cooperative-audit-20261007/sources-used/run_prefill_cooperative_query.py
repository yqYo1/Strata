"""Bounded diagnostic query of actual kernel resource limits; no kernel launch."""
from pathlib import Path
import json,types,datetime,hashlib,os,sys,re
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(root/'sycl/tools'));from owned_gdb import process_identity
failed=json.loads((base/'owned-upstream-refresh-2048-recheck/record.json').read_text())
health=json.loads((base/'post-upstream-refresh-2048-recheck-health/record.json').read_text());assert health['healthy'] and health['started_utc']>failed['finished_utc']
for key in ['inferior','debugger']:
 old=failed[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
for case in ['dequant-completion-stress-usm','dequant-completion-stress-retire','dequant-gemm-completion-stress-usm','dequant-gemm-completion-stress-retire']:
 old=json.loads((base/case/'record.json').read_text());assert old['passed'] and not old['active'] and not old['new_fault_messages']
 for key in ['inferior','debugger']:
  owner=old[key];now=process_identity(owner['pid']);assert not now or now['start_ticks']!=owner['start_ticks']
out=base/'prefill-cooperative-query';build=json.loads((out/'build-record.json').read_text());assert build['passed']
source=root/'sycl/tools/recover-xe.sh';m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
runner=m.Runner(out);cursor=m.journal_cursor(runner,'kernel-before-query')
r=dict(active=True,passed=False,scope='Actual unchanged prefill/IQ object resource query at observed cooperative local sizes, dynamiclocalbytes0; no kernel launch or reset; reported limits are optimistic if original launch used additional dynamic local memory',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
try:
 env=m.health_environment();env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0');env=m.diagnostic_environment(env)
 r['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['NEOReadDebugKeys','EnableDirectSubmission']}
 binary=out/'query';assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binary_sha256']
 stdout=runner.run('query',[str(binary),str(out/'targets.txt')],seconds=60,env=env)
 results=[json.loads(s) for s in stdout.splitlines()];r['results']=results
 assert results[-1]['stage']=='PASS' and results[-1]['queried']==build['target_count'] and results[-1]['kernels_submitted']==0
 assert len(results)==build['target_count']+1
 logs=(out/'query.stderr').read_text(errors='replace')
 assert not re.search(r'---> urEnqueueKernelLaunch|SUCCESS .*zeCommandListAppendLaunch',logs)
 r['passed']=True;r['driver_kernel_submissions_observed']=0
 print(json.dumps(results[-1]),flush=True)
except BaseException as e:r['error']=repr(e);raise
finally:
 kernel=runner.run('kernel-after-query',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'],seconds=5)
 rows=[json.loads(s) for s in kernel.splitlines() if s.startswith('{')]
 r['new_fault_messages']=[x['MESSAGE'] for x in rows if ('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and m.FAULT.search(x.get('MESSAGE',''))]
 r['passed']=r['passed'] and not r['new_fault_messages'];r['steps']=runner.calls;r['active']=False;r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (out/'query-record.json').write_text(json.dumps(r,indent=2)+'\n')
if not r['passed']:raise SystemExit(1)
