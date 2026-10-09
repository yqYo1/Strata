"""Root-owned ordinary-user CPU timer controls, no profiler/model/GPU."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time

B = Path(__file__).parent
OUT = B / 'cpu-timer-control-matrix-v1'
source = B / 'timer_control_matrix_v1.c'
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUT.exists()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: comm = (proc / 'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError): continue
    assert comm not in ['strata', 'strata-xe-health', 'gdb', 'ninja', 'icpx', 'vtune', 'gprofng', 'gp-collect-app'], (proc.name, comm)
OUT.mkdir(mode=0o700)
start = time.monotonic()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
record = {'active': True, 'complete': False, 'passed': False, 'gpu_executed': False, 'model_opened': False,
          'profiler_executed': False, 'scope': 'Five separate disposable CPU processes distinguish clock/notification/legacy timer capability, recording immediate returns and errno. No host settings or limits changed.',
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'controller_sha256': sha(Path(__file__)), 'source_sha256': sha(source),
          'compiler': '/usr/bin/cc', 'compiler_sha256': sha(Path('/usr/bin/cc')),
          'text_budget_bytes': 64 * 1024**2, 'deadline_seconds': 60, 'steps': [], 'cases': [],
          'owners': {}, 'cleanup': [], 'survivors': [], 'errno_note': 'errno is interpreted only for failing calls; raw immediate errno is retained for every call.'}


def identity(pid):
    try:
        f = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(f[1]), 'start_ticks': int(f[19])}
    except (FileNotFoundError, PermissionError, ProcessLookupError): return None


def discover():
    rows = [r for p in Path('/proc').iterdir() if p.name.isdigit() and (r := identity(int(p.name)))]
    while True:
        added = False
        for row in rows:
            key = str(row['pid'])
            if key not in record['owners'] and str(row['ppid']) in record['owners']:
                record['owners'][key] = row; added = True
        if not added: break


def save():
    record['elapsed_seconds'] = time.monotonic() - start
    temp = OUT / 'record.json.tmp'
    temp.write_text(json.dumps(record, indent=2) + '\n'); temp.replace(OUT / 'record.json')


def run(label, argv, deadline):
    step = {'label': label, 'argv': argv, 'deadline_seconds': deadline}
    record['steps'].append(step)
    with (OUT / (label + '.stdout')).open('wb') as out, (OUT / (label + '.stderr')).open('wb') as err:
        child = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        row = identity(child.pid); assert row
        step['identity'] = row; record['owners'][str(child.pid)] = row
        begin = time.monotonic(); save()
        try:
            while child.poll() is None:
                discover()
                assert time.monotonic() - start < 60 and time.monotonic() - begin < deadline
                assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < record['text_budget_bytes']
                time.sleep(.05)
        finally:
            if child.poll() is None:
                discover()
                for sig, grace in [(signal.SIGTERM, 3), (signal.SIGKILL, 2)]:
                    for row in reversed(list(record['owners'].values())):
                        now = identity(row['pid'])
                        if now and now['start_ticks'] == row['start_ticks']:
                            try: os.kill(row['pid'], sig); record['cleanup'].append({'identity': row, 'signal': sig.name})
                            except ProcessLookupError: pass
                    try: child.wait(timeout=grace)
                    except subprocess.TimeoutExpired: pass
            step['exit_code'] = child.returncode; step['elapsed_seconds'] = time.monotonic() - begin; save()
        assert child.returncode == 0
    return (OUT / (label + '.stdout')).read_text()


save()
try:
    binary = OUT / 'timer-control'
    run('build', ['/usr/bin/cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', str(source), '-o', str(binary), '-lrt'], 20)
    record['binary_sha256'] = sha(binary)
    for case in ['thread-directed', 'thread-signal', 'monotonic-directed', 'monotonic-signal', 'legacy-prof']:
        data = run(case, [str(binary), case], 5)
        assert len(data) < 4096
        result = json.loads(data); assert result['case'] == case
        record['cases'].append(result); save()
    record['complete'] = True
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    discover()
    record['survivors'] = [r for r in record['owners'].values() if (now := identity(r['pid'])) and now['start_ticks'] == r['start_ticks']]
    record['passed'] = record['passed'] and not record['survivors'] and not record['cleanup']
    record['active'] = False; record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
print(json.dumps({k: record.get(k) for k in ['complete', 'passed', 'cases', 'cleanup', 'survivors', 'error', 'elapsed_seconds']}))
if not record['passed']: raise SystemExit(1)
