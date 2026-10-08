"""Read-only health after the owned 32K timestamp experiment stopped."""
from pathlib import Path
import datetime, hashlib, json, os, types

base=Path(__file__).parent
root=Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out=base/'post-device-profile-abort-v01402-health-v1'
out.mkdir(mode=0o700)
source=root/'sycl/tools/recover-xe.sh'
m=types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],str(source),'exec'),m.__dict__)
r=m.Runner(out)
record={'active':True,'healthy':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Read-only exact-word device/driver/runtime check after owned model and debugger exit; no recovery/reset/rebind/service action.'}
try:
    old_path=base/'owned-device-profile-v01402-code32k-diagnostic-r1/record.json'
    old=json.loads(old_path.read_text())
    assert not old['active'] and not old['completed'] and not old['healthy']
    assert 'SIGABRT' in old['error'] and not old['new_fault_messages']
    assert not any(old['cleanup'][k] for k in ['inferior_survived','gdb_survived'])
    assert old['boot_id']==record['boot_id'] and old['binary_sha256']=='dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
    assert all(not Path('/proc',str(old[k]['pid'])).exists() for k in ['inferior','debugger'])
    record['terminal_controller_receipt_sha256']=hashlib.sha256(old_path.read_bytes()).hexdigest()
    record['owned_debugger_model_absent']=True
    os.environ.update(NEOReadDebugKeys='1',EnableDirectSubmission='0')
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'),r,Path('/home/yayoi/.local/bin/strata-xe-health'))
    record['healthy']=True
except BaseException as error:
    record['error']=repr(error)
    raise
finally:
    record.update(active=False,steps=r.calls,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'healthy':record['healthy'],'record_sha256':hashlib.sha256((out/'record.json').read_bytes()).hexdigest()},indent=2))
