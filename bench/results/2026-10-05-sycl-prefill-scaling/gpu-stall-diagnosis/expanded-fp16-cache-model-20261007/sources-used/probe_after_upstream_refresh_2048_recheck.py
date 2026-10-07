"""One bounded logged GPU probe after verified owned full-context cleanup."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys
import types

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0, str(root / 'sycl/tools'))
from owned_gdb import process_identity

previous = json.loads((base / 'owned-upstream-refresh-2048-recheck/record.json').read_text())
assert not previous['active'] and not previous['healthy']
assert not previous['new_fault_messages']
assert not previous['cleanup']['inferior_survived'] and not previous['cleanup']['gdb_survived']
for item in [previous['inferior'], previous['debugger']]:
    current = process_identity(item['pid'])
    assert not current or current['start_ticks'] != item['start_ticks']
for path in base.rglob('stalled-writer.json'):
    item = json.loads(path.read_text())
    current = process_identity(item['pid'])
    assert not current or current['start_ticks'] != item['start_ticks']
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot == previous['boot_id']
out = base / 'post-upstream-refresh-2048-recheck-health'
out.mkdir(mode=0o700)
source = root / 'sycl/tools/recover-xe.sh'
module = types.ModuleType('health_only')
exec(compile(source.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(source), 'exec'), module.__dict__)
runner = module.Runner(out)
record = {'scope': 'One logged B570 H2D/kernel/D2H probe after unchanged updated-upstream control recheck first2K stalls at layer17/token256, watchdog SIGABRT, no new xe fault; both owned processes gone; controller-owned cleanup completed, no reset or service/driver change',
          'healthy': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': boot, 'helper_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
try:
    os.environ['NEOReadDebugKeys'] = '1'
    os.environ['EnableDirectSubmission'] = '0'
    record['NEO_environment'] = {k: os.environ[k] for k in ['NEOReadDebugKeys', 'EnableDirectSubmission']}
    binary = Path('/home/yayoi/.local/bin/strata-xe-health')
    record['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    module.check_health(types.SimpleNamespace(bdf='0000:05:00.0'), runner, binary)
    record['healthy'] = True
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    record['steps'] = runner.calls
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
