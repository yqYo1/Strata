"""Read-only device health after the owned r5 process stopped on SIGABRT."""
from pathlib import Path
import datetime, hashlib, json, os, types

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = base / 'post-full256k-abort-v01402-health-v1'
out.mkdir(mode=0o700)
source = root / 'sycl/tools/recover-xe.sh'
m = types.ModuleType('read_only_health')
exec(compile(source.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(source), 'exec'), m.__dict__)
r = m.Runner(out)
record = {'active': True, 'healthy': False,
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'Read-only exact-word device/driver/runtime health; no recovery/reset/rebind/service action.'}
try:
    old_path = base / 'owned-full-kv-access-v01402-full256k-diagnostic-r5/record.json'
    old = json.loads(old_path.read_text())
    digest = hashlib.sha256(old_path.read_bytes()).hexdigest()
    assert digest == 'e7e541f91ef053acebaa6b9ed2bc3139010e269f0b671d23f423869e2c7a882f'
    assert not old['active'] and not old['completed'] and not old['healthy']
    assert 'SIGABRT' in old['error'] and not old['new_fault_messages']
    assert not old['cleanup']['inferior_survived'] and not old['cleanup']['gdb_survived']
    assert record['boot_id'] == old['boot_id']
    assert all(not Path('/proc', str(pid)).exists() for pid in (3396819, 3396909, 3396925))
    record['terminal_controller_receipt_sha256'] = digest
    record['owned_controller_debugger_model_absent'] = True
    os.environ.update(NEOReadDebugKeys='1', EnableDirectSubmission='0')
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'), r,
                   Path('/home/yayoi/.local/bin/strata-xe-health'))
    record['healthy'] = True
except BaseException as e:
    record['error'] = repr(e)
    raise
finally:
    record.update(active=False, steps=r.calls,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'healthy': record['healthy'],
                  'record_sha256': hashlib.sha256((out/'record.json').read_bytes()).hexdigest()}, indent=2))
