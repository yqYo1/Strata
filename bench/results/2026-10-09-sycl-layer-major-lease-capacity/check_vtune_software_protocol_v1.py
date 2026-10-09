"""Root-owned CPU-only VTune/software collector and isolated protocol smoke test."""
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
OUT = B / 'vtune-software-protocol-smoke-v1'
VTUNE = Path('/opt/intel/oneapi/vtune/2026.4/bin64/vtune')
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUT.exists()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: name = (proc / 'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError): continue
    assert name not in ['strata', 'strata-xe-health', 'gdb', 'ninja', 'vtune'], (proc.name, name)
OUT.mkdir(mode=0o700)
for name in ['request', 'reply']: os.mkfifo(OUT / name, 0o600)
request = os.open(OUT / 'request', os.O_RDWR | os.O_NONBLOCK)
reply = os.open(OUT / 'reply', os.O_RDWR | os.O_NONBLOCK)
target = OUT / 'target.py'
target.write_text('''import json, os, sys, time
from pathlib import Path
base=Path(sys.argv[1])
with (base/'request').open() as inp, (base/'reply').open('w', buffering=1) as out:
 print('READY '+str(os.getpid()),file=out,flush=True)
 assert inp.readline()=='RUN\\n'
 start=time.monotonic(); x=1
 while time.monotonic()-start<4: x=(x*1103515245+12345)&0x7fffffff
 print('DONE '+str(x),file=out,flush=True)
 assert inp.readline()=='QUIT\\n'
 print('BYE',file=out,flush=True)
''')
start = time.monotonic()
record = {'active': True, 'passed': False, 'gpu_executed': False, 'model_opened': False,
          'scope': 'Software Hotspots on a harmless Python CPU target with dedicated FIFO protocol; no model or GPU',
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'vtune': str(VTUNE), 'vtune_sha256': hashlib.sha256(VTUNE.read_bytes()).hexdigest(),
          'target_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
          'perf_event_paranoid': Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip(),
          'ptrace_scope': Path('/proc/sys/kernel/yama/ptrace_scope').read_text().strip(),
          'deadline_seconds': 60, 'text_budget_bytes': 64 * 1024**2,
          'result_budget_bytes': 128 * 1024**2, 'data_limit_mib': 64,
          'owners': {}, 'steps': [], 'protocol': [], 'cleanup': []}

def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(fields[19]), 'ppid': int(fields[1])}
    except (FileNotFoundError, PermissionError, ProcessLookupError): return None

def discover():
    known = set(record['owners'])
    rows = [x for p in Path('/proc').iterdir() if p.name.isdigit() and (x := identity(int(p.name)))]
    changed = True
    while changed:
        changed = False
        for row in rows:
            key = str(row['pid'])
            if key not in known and str(row['ppid']) in known:
                record['owners'][key] = row; known.add(key); changed = True

def save():
    record['elapsed_seconds'] = time.monotonic() - start
    tmp = OUT / 'record.json.tmp'; tmp.write_text(json.dumps(record, indent=2) + '\n'); tmp.replace(OUT / 'record.json')

def poll():
    discover(); save()
    assert time.monotonic() - start < 60, 'CPU profiler total deadline'
    sizes = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert sizes < record['result_budget_bytes'], 'bounded CPU profiler output limit'
    text_sizes = sum(p.stat().st_size for p in OUT.rglob('*')
                     if p.is_file() and p.suffix in ['.stdout', '.stderr', '.json', '.tmp'])
    assert text_sizes < record['text_budget_bytes'], 'bounded CPU profiler text limit'

pending = bytearray()
def line():
    end = time.monotonic() + 20
    while time.monotonic() < end:
        if b'\n' in pending:
            value, _, tail = pending.partition(b'\n'); pending[:] = tail
            text = value.decode(); record['protocol'].append(text); save(); return text
        poll()
        if child.poll() is not None: raise RuntimeError('VTune exited before target protocol: '+str(child.returncode))
        try: data = os.read(reply, 65536)
        except BlockingIOError: data = b''
        pending.extend(data); time.sleep(.05)
    raise TimeoutError('isolated target protocol deadline')

def control(action):
    argv = [str(VTUNE), '-command', action, '-r', str(OUT / 'result')]
    with (OUT / (action+'.stdout')).open('wb') as stdout, (OUT / (action+'.stderr')).open('wb') as stderr:
        p = subprocess.Popen(argv, stdout=stdout, stderr=stderr)
        row = identity(p.pid)
        if row: record['owners'][str(p.pid)] = row
        discover()
        try: code = p.wait(timeout=10)
        except BaseException:
            p.terminate()
            try: p.wait(timeout=3)
            except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=3)
            raise
    record['steps'].append({'action': action, 'argv': argv, 'exit_code': code}); save()
    assert code == 0, action

child = None
save()
try:
    argv = [str(VTUNE), '-collect', 'hotspots', '-knob', 'sampling-mode=sw', '-start-paused',
            '-data-limit=64', '-return-app-exitcode', '-result-dir', str(OUT/'result'),
            '--', '/usr/bin/python3', str(target), str(OUT)]
    record['argv'] = argv
    with (OUT/'collector.stdout').open('wb') as stdout, (OUT/'collector.stderr').open('wb') as stderr:
        child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        row = identity(child.pid); assert row; record['owners'][str(child.pid)] = row; save()
        ready = line(); assert ready.startswith('READY ')
        target_identity = identity(int(ready.split()[1])); assert target_identity
        record['owners'][str(target_identity['pid'])] = target_identity
        record['target_executable'] = os.readlink('/proc/'+str(target_identity['pid'])+'/exe')
        control('resume'); os.write(request, b'RUN\n')
        done = line(); assert done.startswith('DONE ')
        control('pause'); os.write(request, b'QUIT\n')
        assert line() == 'BYE'
        child.wait(timeout=15); record['collector_exit_code'] = child.returncode
        assert child.returncode == 0
        record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    discover()
    # Own only exact discovered identities; never signal by executable name.
    for sig, grace in [(signal.SIGTERM, 3), (signal.SIGKILL, 2)]:
        live = [x for x in record['owners'].values() if (now := identity(x['pid'])) and now['start_ticks'] == x['start_ticks']]
        for x in reversed(live):
            if x['pid'] == os.getpid(): continue
            try: os.kill(x['pid'], sig); record['cleanup'].append({'identity':x,'signal':sig.name})
            except ProcessLookupError: pass
        if child:
            try: child.wait(timeout=grace)
            except subprocess.TimeoutExpired: pass
    record['survivors'] = [x for x in record['owners'].values() if (now := identity(x['pid'])) and now['start_ticks'] == x['start_ticks']]
    record['passed'] = record['passed'] and not record['survivors'] and not record['cleanup']
    if child: record['collector_exit_code'] = child.returncode
    for fd in [request, reply]: os.close(fd)
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({k:record.get(k) for k in ['passed','gpu_executed','collector_exit_code','error','cleanup','survivors','elapsed_seconds']}))
if not record['passed']: raise SystemExit(1)
