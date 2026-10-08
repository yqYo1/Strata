"""Read-only health after a controller metadata error before any GEN request."""
from pathlib import Path
import datetime,hashlib,json,os,types
base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=base/'post-controller-assert-v01402-health-v2';out.mkdir(mode=0o700)
source=root/'sycl/tools/recover-xe.sh'
m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
r=m.Runner(out)
record={'active':True,'healthy':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Read-only exact-word device/driver/runtime health; no recovery/reset/rebind/service action.'}
try:
    old_path=base/'owned-full-kv-access-v01402-full256k-diagnostic-r3/record.json'
    old=json.loads(old_path.read_text())
    assert not old['active'] and old['error']=="KeyError('mtp_counts')" and not old['requests']
    assert not old['new_fault_messages'] and not old['cleanup']['inferior_survived'] and not old['cleanup']['gdb_survived']
    record['terminal_controller_receipt_sha256']=hashlib.sha256(old_path.read_bytes()).hexdigest()
    os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'))
    record['healthy']=True
except BaseException as e:record['error']=repr(e);raise
finally:
    record.update(active=False,steps=r.calls,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'healthy':record['healthy'],'record_sha256':hashlib.sha256((out/'record.json').read_bytes()).hexdigest()},indent=2))
