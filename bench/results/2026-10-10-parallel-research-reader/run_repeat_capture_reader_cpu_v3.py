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
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/fix-repeat-capture-reader-20261010')
SRC = W / 'bench/results/2026-10-10-repeat-capture-reader'
OUT = B / 'repeat-capture-reader-cpu-supervisor-v3'
FIX = B / 'repeat-capture-reader-cpu-validation-v3'
PRODUCER = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/debug-sycl-layer-major-repeat-v0141-20261009/sycl/src/prefill/prefill.cpp')
HEADER = PRODUCER.parent / 'repeat_capture.hpp'
PINS = {
    HEADER: '1db25fde8987a4f778d9cba2a5dfdc811dc4f0e36778b9e6b34a779fdc7bd509',
    PRODUCER: 'acd062d1a4f4ab29230630e066fbc29080f92c270cb1fc3e900be12502a17b56',
    SRC / 'capture_reader.py': '7232413521856141c6b1f919e6b19bffc90a0fa06311fc684b9c48aa78512506',
    SRC / 'test_capture_reader.py': 'bcde3bf9c1a0b7d349de2c4f6ac28dfdaea29ed83bab42aa7c3736bc2136de7e',
    B / 'repeat-capture-cpu-validation-v2/normal.bin': '8faa1215acf1c2ef256824eb6c5b12946567edadea65cf75ab79f0af69429395',
    B / 'repeat-capture-cpu-validation-v2/partial-write.bin': '76ba9f7888fc22fdbf2d118e2e718ec326f3471b6f062a41e86dabe57b1da368',
}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUT.exists() and not FIX.exists()
head = subprocess.check_output(['git', '-C', str(W), 'rev-parse', 'HEAD'], text=True).strip()
assert head == '4a565c0302dc3e4df9c4b6798da6c8bc21504d3d'
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
    'scope': 'CPU parser/framing/pair fixtures, Root schema/metadata-contract correction4a565c; independently encoded pinned producer grammar and whole-file verification. No real device proof.',
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'controller_sha256': sha(__file__), 'python': sys.executable,
    'python_sha256': sha(sys.executable), 'pins': {str(p): h for p, h in PINS.items()},
    'deadline_seconds': 240, 'text_budget_bytes': 64 * 1024**2,
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
    for root in [OUT, FIX]:
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
    argv = [sys.executable, str(SRC / 'test_capture_reader.py'), '--out', str(FIX),
            '--producer-source', str(PRODUCER), '--producer-header', str(HEADER), '--real-normal', str(B / 'repeat-capture-cpu-validation-v2/normal.bin'),
            '--real-partial', str(B / 'repeat-capture-cpu-validation-v2/partial-write.bin')]
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
            assert time.monotonic() - start < 235, 'working deadline exceeded'
            current = sizes()
            assert current['text'] <= record['text_budget_bytes'], 'text budget exceeded'
            assert current['binary'] <= record['binary_budget_bytes'], 'binary budget exceeded'
            time.sleep(.1)
        record['exit_code'] = child.returncode
    result = json.loads((FIX / 'record.json').read_text())
    record['test_record_sha256'] = sha(FIX / 'record.json')
    record['case_count'] = len(result['cases'])
    record['failed_cases'] = [r for r in result['cases'] if not r['passed']]
    assert child.returncode == 0 and result['passed'] and not result['active']
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
