"""Bounded diagnostic query of actual kernel resource limits; no kernel launch."""
from pathlib import Path
import json,types,datetime,hashlib,os,sys,re
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(root/'sycl/tools'));from owned_gdb import process_identity
failed=json.loads((base/'owned-expanded-fp16-refresh-128-2048/record.json').read_text())
health=json.loads((base/'post-expanded-fp16-128-health/record.json').read_text());assert health['healthy'] and health['started_utc']>failed['finished_utc']
for key in ['inferior','debugger']:
 old=failed[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
out=base/'dequant-cooperative-limit-query';build=json.loads((out/'build-record.json').read_text());assert build['passed']
source=root/'sycl/tools/recover-xe.sh';m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
runner=m.Runner(out);cursor=m.journal_cursor(runner,'kernel-before-query')
r=dict(active=True,passed=False,scope='Actual updated IQ object kernel metadata query,installedmax_num_work_groups API; no kernel submitted,no device reset; does not establish root_group-specific semantics',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
try:
 env=m.health_environment();env.update(NEOReadDebugKeys='1',EnableDirectSubmission='0');env=m.diagnostic_environment(env)
 r['environment']={k:v for k,v in env.items() if k.startswith(('STRATA_','SYCL_','UR_','ZE_','ZEL_','ONEAPI_')) or k in ['NEOReadDebugKeys','EnableDirectSubmission']}
 binary=out/'query';assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binary_sha256']
 stdout=runner.run('query',[str(binary)],seconds=60,env=env)
 print(stdout,flush=True);r['stdout']=stdout;r['passed']='queried 2 kernels; no kernel submitted' in stdout
except BaseException as e:r['error']=repr(e);raise
finally:
 kernel=runner.run('kernel-after-query',['/usr/bin/journalctl','-k','--after-cursor',cursor,'--no-pager','-o','json'],seconds=5)
 rows=[json.loads(s) for s in kernel.splitlines() if s.startswith('{')]
 r['new_fault_messages']=[x['MESSAGE'] for x in rows if ('0000:05:00.0' in x.get('MESSAGE','') or re.search(r'\bxe\b',x.get('MESSAGE',''))) and m.FAULT.search(x.get('MESSAGE',''))]
 r['passed']=r['passed'] and not r['new_fault_messages'];r['steps']=runner.calls;r['active']=False;r['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (out/'query-record.json').write_text(json.dumps(r,indent=2)+'\n')
if not r['passed']:raise SystemExit(1)
