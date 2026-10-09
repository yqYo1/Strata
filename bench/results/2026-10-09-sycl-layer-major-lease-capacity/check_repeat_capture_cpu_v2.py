"""Root-owned ASan/UBSan tests of the real producer header with a CPU queue double."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import signal
import struct
import subprocess
import time

B = Path(__file__).parent
FIXTURE = B / 'repeat-capture-cpu-fixture-v2'
ROOT = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/debug-sycl-layer-major-repeat-v0141-20261009')
OUT = B / 'repeat-capture-cpu-validation-v2'
HEADER = ROOT / 'sycl/src/prefill/repeat_capture.hpp'
sha = lambda p: hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()
assert sha(HEADER) == '1db25fde8987a4f778d9cba2a5dfdc811dc4f0e36778b9e6b34a779fdc7bd509'
assert subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == '9c2ebde89e5157c81a9c9ae135719452a0244ca3'
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUT.exists()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        comm = (proc / 'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    assert comm not in ['strata', 'strata-xe-health', 'gdb', 'ninja', 'icpx', 'vtune', 'gprofng', 'gp-collect-app'], (proc.name, comm)
OUT.mkdir(mode=0o700)
start = time.monotonic()
record = {
    'active': True, 'complete': False, 'passed': False,
    'gpu_executed': False, 'model_opened': False, 'profiler_executed': False, 'adopted': False,
    'scope': 'Exact production RepeatCapture header, fake SYCL namespace queue/event delaying a host memcpy. CPU ASan/UBSan tests producer framing, budget and error ownership. Does not validate real SYCL retirement, USM/kernel behavior or model math.',
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'controller_sha256': sha(Path(__file__)), 'production_header_sha256': sha(HEADER),
    'fake_queue_sha256': sha(FIXTURE / 'sycl/sycl.hpp'),
    'harness_sha256': sha(FIXTURE / 'producer_test.cpp'),
    'compiler': '/usr/bin/g++', 'compiler_sha256': sha('/usr/bin/g++'),
    'text_budget_bytes': 64 * 1024**2, 'binary_budget_bytes': 144 * 1024**2,
    'deadline_seconds': 60, 'steps': [], 'cases': [], 'owners': {},
    'cleanup': [], 'survivors': [], 'retirement': [],
    'prior_failed_mock': 'v1 controller rejected 133495680 bytes vs133495872 because mock used floor(pos/4), not floor((pos+1)/4), for completed blocks. Original failed receipt/source unchanged. No production correction from this test.',
    'leak_oracle': 'LSan enabled except the two unknown-retirement cases; those intentionally retain one process-lifetime vector. ASan remains enabled and late memcpy runs after capture destruction. No fake reclamation of a possibly live allocation.',
}


def identity(pid):
    try:
        f = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(f[1]), 'start_ticks': int(f[19])}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def discover():
    rows = [r for p in Path('/proc').iterdir() if p.name.isdigit() and (r := identity(int(p.name)))]
    while True:
        added = False
        for row in rows:
            key = str(row['pid'])
            parent = record['owners'].get(str(row['ppid']))
            now = identity(row['ppid']) if parent else None
            if key not in record['owners'] and parent and now and now['start_ticks'] == parent['start_ticks']:
                record['owners'][key] = row
                added = True
        if not added:
            break


def save():
    record['elapsed_seconds'] = time.monotonic() - start
    temp = OUT / 'record.json.tmp'
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(OUT / 'record.json')


def run(label, argv, deadline, env):
    step = {'label': label, 'argv': argv, 'deadline_seconds': deadline,
            'environment': {key: env.get(key) for key in ['STRATA_PREFILL_REPEAT_CAPTURE', 'ASAN_OPTIONS', 'UBSAN_OPTIONS', 'LD_PRELOAD']}}
    record['steps'].append(step)
    with (OUT / (label + '.stdout')).open('wb') as out, (OUT / (label + '.stderr')).open('wb') as err:
        child = subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True)
        row = identity(child.pid)
        assert row
        step['identity'] = row
        record['owners'][str(child.pid)] = row
        begin = time.monotonic()
        save()
        try:
            while child.poll() is None:
                discover()
                assert time.monotonic() - start < 60 and time.monotonic() - begin < deadline
                assert sum(p.stat().st_size for p in OUT.glob('*.stdout')) + sum(p.stat().st_size for p in OUT.glob('*.stderr')) < record['text_budget_bytes']
                assert sum(p.stat().st_size for p in OUT.glob('*.bin')) < record['binary_budget_bytes']
                time.sleep(.02)
        finally:
            if child.poll() is None:
                discover()
                for sig, grace in [(signal.SIGTERM, 3), (signal.SIGKILL, 2)]:
                    for row in reversed(list(record['owners'].values())):
                        now = identity(row['pid'])
                        if now and now['start_ticks'] == row['start_ticks']:
                            try:
                                os.kill(row['pid'], sig)
                                record['cleanup'].append({'identity': row, 'signal': sig.name})
                            except ProcessLookupError:
                                pass
                    try:
                        child.wait(timeout=grace)
                    except subprocess.TimeoutExpired:
                        pass
            step['exit_code'] = child.returncode
            step['elapsed_seconds'] = time.monotonic() - begin
            save()
        assert child.returncode == 0, (label, child.returncode)
    return (OUT / (label + '.stdout')).read_text()


def parse_capture(path):
    records = []
    with path.open('rb') as f:
        while raw := f.read(128):
            if len(raw) != 128:
                return {'valid': False, 'error': 'truncated-header', 'records': records}
            words = struct.unpack('=12Q', raw[:96])
            magic, version, layer, p0, total, first, rows, elements, kind, width, ordinal, header_size = words
            assert magic == 0x5354524152505431 and version == 1 and header_size == 128
            assert ordinal == len(records) and kind in [1, 2] and rows > 0 and width > 0
            assert elements == rows * width and elements <= 128 * 1024**2 // 4
            phase_raw = raw[96:]
            assert b'\0' in phase_raw
            phase = phase_raw.split(b'\0', 1)[0].decode('ascii')
            payload = f.read(elements * 4)
            if len(payload) != elements * 4:
                return {'valid': False, 'error': 'truncated-payload', 'records': records}
            records.append({'ordinal': ordinal, 'layer': layer, 'p0': p0, 'T': total, 'first': first,
                            'rows': rows, 'width': width, 'type': kind, 'phase': phase,
                            'payload_sha256': hashlib.sha256(payload).hexdigest()})
    return {'valid': True, 'records': records}


save()
try:
    binary = OUT / 'producer-test'
    base_env = dict(os.environ)
    base_env.pop('STRATA_PREFILL_REPEAT_CAPTURE', None)
    base_env['UBSAN_OPTIONS'] = 'halt_on_error=1:print_stacktrace=1'
    base_env['ASAN_OPTIONS'] = 'detect_leaks=1:abort_on_error=1'
    run('build', ['/usr/bin/g++', '-std=c++17', '-O1', '-g', '-Wall', '-Wextra', '-Werror',
                  '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-fno-pie', '-no-pie',
                  '-I' + str(FIXTURE), '-I' + str(HEADER.parent), str(FIXTURE / 'producer_test.cpp'),
                  '-o', str(binary), '-pthread'], 20, base_env)
    record['binary_sha256'] = sha(binary)
    cases = ['disabled', 'normal', 'bad-shape', 'existing', 'symlink', 'initial-error',
             'event-drained', 'submit-drained', 'event-unknown', 'submit-unknown', 'partial-write', 'two-full-budget']
    for name in cases:
        capture = OUT / (name + '.bin')
        env = dict(base_env)
        if name != 'disabled':
            env['STRATA_PREFILL_REPEAT_CAPTURE'] = str(capture)
        if name in ['event-unknown', 'submit-unknown']:
            env['ASAN_OPTIONS'] = 'detect_leaks=0:abort_on_error=1'
        sentinel = b'unchanged-test-sentinel\n'
        if name == 'existing':
            capture.write_bytes(sentinel)
        if name == 'symlink':
            target = OUT / 'symlink-target'
            target.write_bytes(sentinel)
            capture.symlink_to(target)
        data = run(name, [str(binary), name], 20 if name == 'two-full-budget' else 5, env)
        result = json.loads(data)
        assert result['case'] == name and not result['pending']
        result['leak_detection'] = name not in ['event-unknown', 'submit-unknown']
        stderr = (OUT / (name + '.stderr')).read_text()
        assert 'ERROR: AddressSanitizer' not in stderr and 'runtime error:' not in stderr
        result['stderr_sha256'] = sha(OUT / (name + '.stderr'))
        if name in ['disabled', 'bad-shape']:
            assert not capture.exists()
        elif name in ['existing', 'symlink']:
            assert capture.read_bytes() == sentinel
        else:
            stat = capture.stat()
            assert stat.st_uid == os.geteuid() and stat.st_mode & 0o777 == 0o600 and stat.st_nlink == 1
            result['capture'] = {'path': str(capture), 'size': stat.st_size, 'sha256': sha(capture)}
            parsed = parse_capture(capture)
            result['parse_valid'] = parsed['valid']
            result['parsed_records'] = len(parsed['records'])
            if name == 'normal':
                assert parsed['valid'] and len(parsed['records']) == 2
                assert parsed['records'][0]['payload_sha256'] == hashlib.sha256(struct.pack('=8f', 1.25, -2.5, 3, 4, 5, 6, 7, 8)).hexdigest()
                assert parsed['records'][1]['payload_sha256'] == hashlib.sha256(struct.pack('=4i', -1, 0, 17, 262143)).hexdigest()
                result['frames'] = parsed['records']
            elif name == 'partial-write':
                assert not parsed['valid'] and parsed['error'] == 'truncated-payload' and stat.st_size == 150
                result['parse_error'] = parsed['error']
            elif name == 'two-full-budget':
                assert parsed['valid'] and len(parsed['records']) == 690 and stat.st_size == 133495872, ('geometry', len(parsed['records']), stat.st_size)
                result['margin_bytes'] = 134217728 - stat.st_size
                result['parsed_geometry_sha256'] = hashlib.sha256(json.dumps(parsed['records'], sort_keys=True).encode()).hexdigest()
                result['numerical_data'] = 'Synthetic repeated 0x3f800000; framing/budget only, not real IDs or model values.'
            else:
                assert parsed['valid'] and not parsed['records'] and stat.st_size == 0
        record['cases'].append(result)
        save()
        if name == 'two-full-budget':
            plan = {'path': str(capture), 'size': stat.st_size, 'sha256': result['capture']['sha256'],
                    'reason': 'Disposable mock geometry parsed and hashed; regenerated by recorded harness; no real GPU/model evidence or current raw consumer.', 'status': 'verified-before-unlink'}
            record['retirement'].append(plan)
            save()
            assert sha(capture) == plan['sha256'] and capture.stat().st_ino == stat.st_ino
            capture.unlink()
            plan['status'] = 'removed'
            save()
    assert sha(HEADER) == record['production_header_sha256']
    record['complete'] = True
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    discover()
    record['survivors'] = [r for r in record['owners'].values() if (now := identity(r['pid'])) and now['start_ticks'] == r['start_ticks']]
    record['passed'] = record['passed'] and not record['survivors'] and not record['cleanup']
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({k: record.get(k) for k in ['complete', 'passed', 'elapsed_seconds', 'error', 'cleanup', 'survivors']}))
if not record['passed']:
    raise SystemExit(1)
