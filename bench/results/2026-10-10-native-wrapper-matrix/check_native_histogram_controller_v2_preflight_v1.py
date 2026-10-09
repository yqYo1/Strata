"""Root-owned serial CPU reader validation; no GPU, model or profiler."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

B = Path(__file__).parent
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-native-dispatch-histogram-v0141-20261009')
OUT = B / 'native-histogram-controller-v2-cpu-preflight-v1'
CONTROLLER = B / 'run_owned_native_dispatch_histogram_v0141_code32k_v2.py'
CONTROLLER_PIN = 'ff6d8ad537b9e9fb81511725dd5cf936a3a0d460bae271d79707777f38d0a599'
BUILD_PIN = '7f59396de97d54760539d88e97dac453fd4b1ccfa3ff1e923abcb4b1b72928b6'
FLAGS_PIN = '42be22930d202f9b04761bde8eb33c62bd500d9d856725a473f2f4ae2d1ce31d'
PINS = {CONTROLLER: CONTROLLER_PIN,
        B / 'native-dispatch-histogram-v0141-private-build-v1/record.json': BUILD_PIN,
        B / 'native-dispatch-histogram-v0141-uniform-build-flags-v1.json': FLAGS_PIN}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Pinned child owns the same exclusive lock; parent only supervises its lifetime.
assert not OUT.exists()
head = subprocess.check_output(['git', '-C', str(W), 'rev-parse', 'HEAD'], text=True).strip()
assert head == '96bd5bb4e054f6ddcf677fd96e161b499a013fd7'
assert not subprocess.check_output(['git', '-C', str(W), 'status', '--porcelain'], text=True)
for path, expected in PINS.items():
    assert sha(path) == expected, path
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        comm = (proc / 'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    assert comm not in ['strata', 'strata-xe-health', 'gdb', 'ninja', 'icpx', 'vtune',
                        'gprofng', 'gp-collect-app'], (proc.name, comm)
OUT.mkdir(mode=0o700)
start = time.monotonic()
record = {
    'active': True, 'complete': False, 'passed': False, 'gpu_executed': False,
    'model_opened': False, 'profiler_executed': False, 'source_commit': head,
    'scope': 'CPU-only receipt/source/history/raw-reference admission preflight of root Hcontroller v2. Child exits before GPU helpers/model launch. Runtime histogram and model math remain unexecuted.',
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'controller_sha256': sha(__file__), 'python': sys.executable,
    'python_sha256': sha(sys.executable), 'pins': {str(p): h for p, h in PINS.items()},
    'deadline_seconds': 180, 'text_budget_bytes': 64 * 1024**2,
    'binary_budget_bytes': 140 * 1024**2, 'owners': {}, 'cleanup': [], 'survivors': [],
    'adopted': False, 'performance_eligible': False, 'full_lifecycle_passed': False,
}
pidfds = {}
child = None


def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': int(fields[19])}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def register(row):
    key = str(row['pid'])
    if key in record['owners']:
        return
    fd = os.pidfd_open(row['pid'])
    after = identity(row['pid'])
    if after is None or after['start_ticks'] != row['start_ticks']:
        os.close(fd)
        return
    record['owners'][key] = row
    pidfds[key] = fd


def discover():
    rows = [row for p in Path('/proc').iterdir() if p.name.isdigit()
            and (row := identity(int(p.name)))]
    while True:
        added = False
        for row in rows:
            if str(row['pid']) not in record['owners'] and str(row['ppid']) in record['owners']:
                try:
                    register(row)
                except ProcessLookupError:
                    continue
                added = str(row['pid']) in record['owners'] or added
        if not added:
            break


def save():
    record['elapsed_seconds'] = time.monotonic() - start
    temp = OUT / 'record.json.tmp'
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temp.replace(OUT / 'record.json')


def sizes():
    totals = {'text': 0, 'binary': 0}
    for root in [OUT]:
        if root.exists():
            for path in root.rglob('*'):
                if path.is_file() and not path.is_symlink():
                    totals['binary' if path.suffix == '.bin' else 'text'] += path.stat().st_size
    return totals


def cleanup_owned():
    discover()
    for sig, grace in [(signal.SIGTERM, 2), (signal.SIGKILL, 2)]:
        for key, row in reversed(list(record['owners'].items())):
            now = identity(row['pid'])
            if now and now['start_ticks'] == row['start_ticks']:
                try:
                    signal.pidfd_send_signal(pidfds[key], sig)
                    record['cleanup'].append({'identity': row, 'signal': sig.name})
                except ProcessLookupError:
                    pass
        if child is not None:
            try:
                child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass


save()
try:
    argv = [sys.executable, str(CONTROLLER), 'tasks6', 'diagnostic', '1',
            '--cpu-preflight', '--self-sha256', CONTROLLER_PIN,
            '--build-sha256', BUILD_PIN, '--flags-sha256', FLAGS_PIN]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    record['argv'] = argv
    record['effective_env_changes'] = {'PYTHONDONTWRITEBYTECODE': '1'}
    with (OUT / 'test.stdout').open('wb') as out, (OUT / 'test.stderr').open('wb') as err:
        child = subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True,
                                 cwd=str(W))
        row = identity(child.pid)
        assert row is not None
        try:
            register(row)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=2)
            raise
        save()
        while child.poll() is None:
            discover()
            assert time.monotonic() - start < 175, 'working deadline exceeded'
            current = sizes()
            assert current['text'] <= record['text_budget_bytes'], 'text budget exceeded'
            assert current['binary'] <= record['binary_budget_bytes'], 'binary budget exceeded'
            time.sleep(.1)
        record['exit_code'] = child.returncode
    result = json.loads((OUT / 'test.stdout').read_text())
    record['preflight_result'] = result
    assert child.returncode == 0 and result['cpu_preflight_passed'] and not result['gpu_executed']
    assert result['controller_sha256'] == CONTROLLER_PIN
    assert result['build_receipt_sha256'] == BUILD_PIN and result['flags_receipt_sha256'] == FLAGS_PIN
    assert result['runtime_qualification_claimed'] is False
    for path, expected in PINS.items():
        assert sha(path) == expected, path
    record['complete'] = True
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    discover()
    live = [r for r in record['owners'].values()
            if (now := identity(r['pid'])) and now['start_ticks'] == r['start_ticks']]
    if live:
        cleanup_owned()
    if child is not None:
        record['exit_code'] = child.poll()
    record['survivors'] = [r for r in record['owners'].values()
                          if (now := identity(r['pid'])) and now['start_ticks'] == r['start_ticks']]
    record['passed'] = record['passed'] and not record['survivors'] and not record['cleanup']
    record['sizes'] = sizes()
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    for fd in pidfds.values():
        os.close(fd)
print(json.dumps({k: record.get(k) for k in
                 ['passed', 'complete', 'case_count', 'failed_cases', 'exit_code',
                  'error', 'elapsed_seconds', 'cleanup', 'survivors', 'sizes']}))
if not record['passed']:
    raise SystemExit(1)
