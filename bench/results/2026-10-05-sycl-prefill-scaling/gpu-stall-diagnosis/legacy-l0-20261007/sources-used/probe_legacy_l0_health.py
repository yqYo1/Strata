"""One bounded GPU health probe after verified owned diagnostic cleanup."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import types

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = base / 'legacy-l0-health'
out.mkdir(mode=0o700)
src = root / 'sycl/tools/recover-xe.sh'
m = types.ModuleType('health_only')
exec(compile(src.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(src), 'exec'), m.__dict__)
r = m.Runner(out)
record = {'scope': 'One bounded logged B570 H2D/kernel/D2H probe with the installed legacy Level Zero adapter and its documented copy-engine-off setting; no reset or service/driver changes',
          'healthy': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'helper_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
          'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
try:
    previous = json.loads((base/'full-context-expert-wait-1-serve/record.json').read_text())
    assert not previous['active'] and not previous['completed']
    assert previous['boot_id'] == record['boot_id'] and not previous['new_fault_messages']
    assert not previous['cleanup']['inferior_survived'] and not previous['cleanup']['gdb_survived']
    for key in ['inferior', 'debugger']:
        assert not Path('/proc', str(previous[key]['pid'])).exists()
    for p in base.rglob('stalled-writer.json'):
        old = json.loads(p.read_text())
        q = Path('/proc', str(old['pid']), 'stat')
        if q.exists():
            text = q.read_text(); fields = text[text.rfind(')')+2:].split()
            assert int(fields[19]) != old['start_ticks'] or fields[0] == 'Z'
    os.environ['NEOReadDebugKeys'] = '1'
    os.environ['EnableDirectSubmission'] = '0'
    record['NEO_environment'] = {k: os.environ[k] for k in ['NEOReadDebugKeys', 'EnableDirectSubmission']}
    binary = Path('/home/yayoi/.local/bin/strata-xe-health')
    record['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    latest = json.loads((base / 'post-expert-wait-one-stall-health/record.json').read_text())
    assert latest['healthy'] and latest['boot_id'] == record['boot_id']
    adapter = Path('/opt/intel/oneapi/compiler/2026.1/lib/libur_adapter_level_zero.so.0')
    assert adapter.is_file()
    record['adapter_path'] = str(adapter)
    record['adapter_sha256'] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    def legacy_environment(runner, executable):
        assert executable.is_file() and os.access(executable, os.X_OK)
        env = m.health_environment()
        env['UR_ADAPTERS_FORCE_LOAD'] = str(adapter)
        env.pop('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', None)
        env['UR_L0_USE_COPY_ENGINE'] = '0'
        runner.run('health-runtime', ['/usr/bin/python3', '-c', 'import ctypes,sys; ctypes.CDLL(sys.argv[1])', str(adapter)], seconds=5, env=env)
        return env
    m.validate_probe_runtime = legacy_environment
    m.check_health(types.SimpleNamespace(bdf='0000:05:00.0'), r, binary)
    logged = (out / 'health.stderr').read_text(errors='replace')
    assert 'libur_adapter_level_zero.so' in logged and 'libur_adapter_level_zero_v2.so' not in logged
    record['adapter_loaded_log_verified'] = True
    record['healthy'] = True
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    record['steps'] = r.calls
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'record.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
