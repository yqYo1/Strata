from pathlib import Path
import types,json,datetime,hashlib,os,sys
b=Path(__file__).parent;r=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05');sys.path.insert(0,str(r/'sycl/tools'))
from owned_gdb import process_identity
out=b/'post-event-ack-v01402-stall-health';out.mkdir(mode=0o700);probes=out/'probes';probes.mkdir()
m=types.ModuleType('read_only_health');source=r/'sycl/tools/recover-xe.sh';exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
runner=m.Runner(probes)
record={'active':True,'healthy':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'scope':'Logged read-only H2D/kernel/D2H health after the private actual-DMA event candidate32K first-chunk watchdog, with no new xe faults in that job. No reset/rebind/reboot/service changes.','controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
def save():record['steps']=runner.calls;(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
save()
try:
 previous=json.loads((b/'owned-event-ack-v01402-code32k-diagnostic-r1/record.json').read_text());assert not previous['active'];assert not any(previous['cleanup'][k] for k in ['inferior_survived','gdb_survived'])
 sequence=json.loads((b/'event-ack-v01402-state-sequence/record.json').read_text());assert not sequence['active'] and not sequence['passed']
 for key in ['inferior','debugger']:
  old=previous[key];now=process_identity(old['pid']);assert not now or now['start_ticks']!=old['start_ticks'] or now['state']=='Z'
 os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
 m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),runner,Path('/home/yayoi/.local/bin/strata-xe-health'))
 record['healthy']=True
except BaseException as e:record['error']=repr(e)
finally:record['active']=False;record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps({k:v for k,v in record.items() if k!='steps'},indent=2))
if not record['healthy']:raise SystemExit(1)
