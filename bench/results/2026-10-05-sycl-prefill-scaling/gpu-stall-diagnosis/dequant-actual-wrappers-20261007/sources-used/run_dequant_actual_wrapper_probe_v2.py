"""Deferred first diagnostic comparison of the two actual dequant wrappers.

Run only after the prior full-context controller has a terminal result and
its owned processes are gone. After failure, require the subsequent logged
same-boot GPU health pass with no surviving probe process. This
controller never resets the GPU or stops unrelated processes/services.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import selectors
import sys
import time
import tty
import types

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
sys.path.insert(0, str(root / 'sycl/tools'))
from owned_gdb import OwnedGdb, process_identity


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


# Refuse before creating output or loading any GPU runtime.
previous = json.loads((base / 'full-context-verifier-lazy-restore-serve/record.json').read_text())
continuation = json.loads((base / 'profiler-continuation.json').read_text())
assert not previous['active']
if not previous['healthy']:
    health = json.loads((base / 'post-full-verifier-lazy-restore-health/record.json').read_text())
    assert health['healthy'] and health['boot_id'] == previous['boot_id']
    assert health['started_utc'] > previous['finished_utc']
    assert all(step['exit_code'] == 0 and not step['timed_out'] and not step['still_alive'] for step in health['steps'])
    for step in health['steps']:
        assert not Path('/proc', str(step['pid'])).exists()
else:
    assert previous['completed']
assert not continuation['live_gpu_jobs'], 'Poll and record the previous PTY terminal result first'
assert previous['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for item in [previous['inferior'], previous['debugger']]:
    identity = process_identity(item['pid'])
    assert not identity or identity['start_ticks'] != item['start_ticks'], 'Previous owned process remains'
for receipt in base.rglob('stalled-writer.json'):
    item = json.loads(receipt.read_text())
    identity = process_identity(item['pid'])
    assert not identity or identity['start_ticks'] != item['start_ticks'], 'A previous owned writer remains'

fixture = json.loads((base / 'dequant-no-root-probe-build/record.json').read_text())
assert fixture['passed'] and fixture['inputs_unchanged']
for item in fixture['binaries'].values():
    assert digest(Path(item['path'])) == item['sha256']
out = base / 'dequant-actual-wrapper-diagnostic-v2'
out.mkdir(mode=0o700)
probes = out / 'probes'
probes.mkdir()
source = root / 'sycl/tools/recover-xe.sh'
module = types.ModuleType('diagnostic_only')
exec(compile(source.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0], str(source), 'exec'), module.__dict__)
runner = module.Runner(probes)
record = {
    'scope': 'First logged actual-wrapper original/candidate GPU comparison: 144 complete guarded FP16 outputs per binary, same host fixture object, only two dequant launch properties changed. No throughput or full-model/capacity/hang-prevention proof.',
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'boot_id': previous['boot_id'], 'controller_sha256': digest(Path(__file__)),
    'fixture_build_record_sha256': digest(base / 'dequant-no-root-probe-build/record.json'),
    'helper_sha256': digest(source), 'active': True, 'passed': False, 'runs': [],
    'deadline_per_binary_seconds': 180, 'log_limit_per_binary_bytes': 512 * 1024**2,
}
cursor = None
started = time.monotonic()


def save():
    record.update(elapsed_seconds=time.monotonic() - started, steps=runner.calls)
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')


def execute(label, binary):
    directory = out / label
    directory.mkdir()
    master, slave = os.openpty()
    tty.setraw(slave)
    os.set_blocking(master, False)
    child = None
    item = {'label': label, 'binary': binary, 'cases': [], 'gpu_pdevs': []}
    record['runs'].append(item)
    before = time.monotonic()
    try:
        child = OwnedGdb([binary['path'], str(directory / 'heads')], directory / 'debugger',
                         env, inferior_tty_fd=slave)
        child.run()
        pending = bytearray()
        next_save = before
        with (directory / 'stdout.raw').open('wb') as raw, selectors.DefaultSelector() as ready:
            ready.register(master, selectors.EVENT_READ)

            def drain():
                if not ready.select(.01):
                    return False
                try:
                    data = os.read(master, 65536)
                except (BlockingIOError, OSError):
                    return False
                if not data:
                    return False
                raw.write(data)
                raw.flush()
                pending.extend(data)
                while b'\n' in pending:
                    line, _, tail = pending.partition(b'\n')
                    pending[:] = tail
                    item['cases'].append(json.loads(line.decode()))
                return True

            while child.exit_code is None and child.exit_signal is None:
                child.poll(.01)
                if child.stops and child.stops[-1] != 'resumed' and child.exit_code is None and child.exit_signal is None:
                    raise RuntimeError('inferior stopped: ' + child.stops[-1])
                if time.monotonic() - before > record['deadline_per_binary_seconds']:
                    raise TimeoutError('bounded actual-wrapper deadline')
                logs = list((directory / 'debugger').glob('*.stderr'))
                if sum(p.stat().st_size for p in logs) > record['log_limit_per_binary_bytes']:
                    raise RuntimeError('bounded API log limit')
                if child.inferior:
                    item.update(inferior=child.inferior, debugger=child.debugger_identity)
                    if not item['gpu_pdevs']:
                        try:
                            for info in Path('/proc', str(child.inferior['pid']), 'fdinfo').iterdir():
                                item['gpu_pdevs'].extend(re.findall(r'^drm-pdev:\s*(\S+)', info.read_text(), re.M))
                        except (FileNotFoundError, ProcessLookupError):
                            pass
                drain()
                if time.monotonic() >= next_save:
                    save()
                    next_save = time.monotonic() + 2
            while drain():
                pass
            assert not pending
        assert child.exit_code == 0 and child.exit_signal is None
        assert len(item['cases']) == 144
        assert '0000:05:00.0' in item['gpu_pdevs']
        assert all(c['finite'] and c['guards_intact'] for c in item['cases'])
    except BaseException:
        if child:
            try:
                item['failure_snapshot'] = child.snapshot('failure', resume=False)
            except BaseException as error:
                item['snapshot_error'] = repr(error)
        raise
    finally:
        if child:
            item.update(exit_code=child.exit_code, exit_signal=child.exit_signal, cleanup=child.close())
        os.close(master)
        os.close(slave)
        item['diagnostic_seconds'] = time.monotonic() - before
        save()
    assert not any(item['cleanup'].values())
    for identity in [item['inferior'], item['debugger']]:
        remaining = process_identity(identity['pid'])
        assert not remaining or remaining['start_ticks'] != identity['start_ticks']
    return item


try:
    env = module.diagnostic_environment()
    env.update(NEOReadDebugKeys='1', EnableDirectSubmission='0')
    record['environment'] = {k: v for k, v in env.items()
                             if k.startswith(('SYCL_', 'UR_', 'ZE_', 'ZEL_', 'ONEAPI_', 'STRATA_', 'NEO'))
                             or k in ('EnableDirectSubmission', 'LD_LIBRARY_PATH')}
    cursor = module.journal_cursor(runner, 'kernel-before')
    save()
    original = execute('original', fixture['binaries']['original'])
    candidate = execute('candidate', fixture['binaries']['candidate'])
    assert original['cases'] == candidate['cases']
    matches = []
    names = {c['file'] for c in original['cases']}
    assert len(names) == 144
    assert {p.name for p in (out / 'original/heads').iterdir()} == names
    assert {p.name for p in (out / 'candidate/heads').iterdir()} == names
    for name in sorted(names):
        a = out / 'original/heads' / name
        b = out / 'candidate/heads' / name
        assert a.read_bytes() == b.read_bytes(), name + ' whole FP16 bytes differ'
        matches.append({'file': name, 'bytes': a.stat().st_size, 'sha256': digest(a), 'all_bytes_equal': True})
    record['whole_guarded_outputs'] = matches
    record['comparisons_complete'] = True
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    if cursor:
        try:
            text = runner.run('kernel-after', ['/usr/bin/journalctl', '-k', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
            rows = [json.loads(s) for s in text.splitlines() if s.startswith('{')]
            record['new_fault_messages'] = [r.get('MESSAGE', '') for r in rows
                if ('0000:05:00.0' in r.get('MESSAGE', '') or re.search(r'\bxe\b', r.get('MESSAGE', '')))
                and module.FAULT.search(r.get('MESSAGE', ''))]
        except BaseException as error:
            record['kernel_error'] = repr(error)
    record['active'] = False
    record['passed'] = bool(record.get('comparisons_complete') and not record.get('error')
                            and not record.get('kernel_error') and not record.get('new_fault_messages'))
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({k: v for k, v in record.items() if k not in ('environment', 'steps', 'whole_guarded_outputs')}, indent=2))
raise SystemExit(0 if record['passed'] else 1)
