from pathlib import Path
import datetime, hashlib, json, os, sys, types
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0,str(root/'sycl/tools'))
from owned_gdb import process_identity
out=base/'post-chunk12k-v01402-health';out.mkdir(mode=0o700)
job=json.loads((base/'owned-main-vmm-full-ram-chunk12k-v01402-code32k-diagnostic-r1/record.json').read_text())
assert job['completed'] and not job['active'] and job['exit_code']==0 and job['exit_signal'] is None
assert not job['new_fault_messages'] and not job['math_gate_passed']
assert not any(job['cleanup'][k] for k in ['inferior_survived','gdb_survived','forced'])
for key in ['inferior','debugger']:
 old=job[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks']
record={'active':True,'healthy':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'scope':'Read-only diagnostic exact-word GPU health after a normally exited32K mathematical rejection. No reset or recovery action.','controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
source=root/'sycl/tools/recover-xe.sh';m=types.ModuleType('health_only');exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__);r=m.Runner(out)
try:
 os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
 m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'))
 record['healthy']=True
except BaseException as error:
 record['error']=repr(error);raise
finally:
 record.update(active=False,steps=r.calls,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'healthy':record['healthy'],'scope':record['scope']},indent=2))
