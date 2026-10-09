"""Root-owned serial, fresh-process production-wrapper CPU correctness matrix.

Synthetic dimensions only. No model, queue, profiler, performance or adoption.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import math
import os
import signal
import stat
import struct
import subprocess
import time

B = Path(__file__).parent
OUT = B / 'native-wrapper-matched-cpu-matrix-v1'
BUILD = B / 'native-wrapper-matched-private-build-v1/record.json'
FLAGS = B / 'native-wrapper-matched-build-flags-v1.json'
PINS = {BUILD: '735686998e2c69c82c8741dbef90215d660007ae12273ba18a9c4c00985685c8',
        FLAGS: 'c9d50ddb45490c56909f57f98e1aa3a6d7dd2911c12551a2f2776375e21e558e'}
CONTROLS = ['STRATA_IQ_MT_MIN', 'STRATA_NO_IQ256', 'STRATA_NO_IQ512',
            'STRATA_NO_IQ4NL', 'STRATA_IQ256_GATHER', 'STRATA_IQ3S_MT1',
            'STRATA_NO_AVXVNNI', 'STRATA_KQ256', 'STRATA_FORCE_ISA',
            'STRATA_Q2_BITPLANE', 'STRATA_NO_Q8K_AVX2',
            'STRATA_NATIVE_DISPATCH_HISTOGRAM', 'STRATA_FORCE_AVX2',
            'STRATA_CPU_YMM', 'STRATA_IQ_GATHER', 'STRATA_IQ_PREFETCH',
            'STRATA_IQ_PARITY_DISPATCH_ONLY']
PROFILES = [('default', {}), ('mt-min1', {'STRATA_IQ_MT_MIN': '1'}),
            ('mt-min2', {'STRATA_IQ_MT_MIN': '2'}),
            ('mt-min3', {'STRATA_IQ_MT_MIN': '3'}),
            ('no-iq256', {'STRATA_NO_IQ256': '1'}),
            ('no-iq4nl', {'STRATA_NO_IQ4NL': '1'}),
            ('gather0', {'STRATA_IQ256_GATHER': '0'}),
            ('gather1', {'STRATA_IQ256_GATHER': '1'}),
            ('iq3s-mt1', {'STRATA_IQ3S_MT1': '1'})]
EXPECTED = [(phase, qt, nt, rows) for phase, types, rows in
            [('gu', [18, 21, 22, 23], 640), ('down', [20, 42], 2560)]
            for qt in types for nt in range(1, 9)]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, timeout=15).strip()


def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': int(fields[19])}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUT.exists()
for path, digest in PINS.items():
    assert sha(path) == digest
build, flags = json.loads(BUILD.read_text()), json.loads(FLAGS.read_text())
assert build['passed'] and build['complete'] and not build['active']
assert not build['cleanup'] and not build['survivors'] and not build['gpu_executed']
assert flags['passed'] and flags['build_receipt_sha256'] == PINS[BUILD]
assert flags['matched_target_object_count'] == 34 and flags['baseline_matched_library_object_count'] == 33
assert not flags['missing_objects'] and not flags['different_flags']
assert [a['label'] for a in build['arms']] == ['T', 'H']
for arm in build['arms']:
    root = Path(arm['root'])
    assert git(root, 'rev-parse', 'HEAD') == arm['source_commit']
    assert not git(root, 'status', '--porcelain')
    assert sha(arm['binary']) == arm['binary_sha256']
    for rel, digest in build['source_pins'].items():
        assert sha(root / rel) == digest
    for rel, digest in arm['files'].items():
        assert sha(Path(arm['build']) / rel) == digest
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        comm = (proc / 'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    assert comm not in ['strata', 'strata-xe-health', 'gdb', 'ninja', 'icpx', 'icx',
                        'vtune', 'gprofng', 'gp-collect-app', 'native_wrapper_'], (proc.name, comm)
OUT.mkdir(mode=0o700)
started = time.monotonic()
record = {'active': True, 'complete': False, 'passed': False, 'gpu_executed': False,
          'model_opened': False, 'full_model_shape_qualified': False,
          'performance_eligible': False, 'adopted': False,
          'original_C_math_gate_passed': False,
          'controller_sha256': sha(__file__),
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'source_arms': build['arms'], 'source_pins': build['source_pins'],
          'build_receipt_sha256': PINS[BUILD], 'flag_proof_sha256': PINS[FLAGS],
          'profiles': [{'name': name, 'overrides': overrides} for name, overrides in PROFILES],
          'deadline_seconds': 1800, 'per_process_deadline_seconds': 180,
          'text_budget_bytes': 64 * 1024**2, 'binary_budget_bytes': 40 * 1024**2,
          'expected_binary_bytes_per_process': 1107104,
          'steps': [], 'pairs': [], 'owners': {}, 'cleanup': [], 'survivors': []}
pidfds = {}
child = None


def save():
    record['elapsed_seconds'] = time.monotonic() - started
    temporary = OUT / 'record.json.tmp'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(OUT / 'record.json')


def register(row):
    key = str(row['pid'])
    if key in record['owners']:
        return
    fd = os.pidfd_open(row['pid'])
    after = identity(row['pid'])
    if not after or after['start_ticks'] != row['start_ticks']:
        os.close(fd)
        return
    record['owners'][key] = row
    pidfds[key] = fd


def discover():
    rows = [r for p in Path('/proc').iterdir() if p.name.isdigit() and (r := identity(int(p.name)))]
    while True:
        added = False
        for row in rows:
            if str(row['pid']) not in record['owners'] and str(row['ppid']) in record['owners']:
                parent = record['owners'][str(row['ppid'])]
                live_parent = identity(parent['pid'])
                if not live_parent or live_parent['start_ticks'] != parent['start_ticks']:
                    continue
                try:
                    register(row)
                except ProcessLookupError:
                    continue
                added = added or str(row['pid']) in record['owners']
        if not added:
            break


def live_owners():
    return [r for r in record['owners'].values()
            if (now := identity(r['pid'])) and now['start_ticks'] == r['start_ticks']]


def cleanup_owned():
    discover()
    for sig, grace in [(signal.SIGTERM, 3), (signal.SIGKILL, 2)]:
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


def parse_stdout(path, overrides):
    lines = path.read_text().splitlines()
    cases = []
    env_seen = {}
    for line in lines:
        if line.startswith('env '):
            key, value = line[4:].split('=', 1)
            assert key not in env_seen
            env_seen[key] = value
        if line.startswith('case '):
            fields = dict(item.split('=', 1) for item in line[5:].split())
            cases.append(fields)
    assert len(cases) == 48
    for case, expected in zip(cases, EXPECTED):
        assert (case['phase'], int(case['type']), int(case['nt']), int(case['rows'])) == expected
    for key, value in env_seen.items():
        assert value == overrides.get(key, '<unset>'), (key, value)
    assert set(env_seen) == set(CONTROLS[:12])
    summary = [line for line in lines if line.startswith('summary ')]
    assert len(summary) == 1
    fields = dict(item.split('=', 1) for item in summary[0][8:].split())
    assert fields['cases'] == '48' and fields['bytes'] == '1107104'
    failed = []
    for case in cases:
        local = all(case[key] == '1' for key in
                    ['finite', 'legal_actq', 'padding', 'reference_gate', 'empty_unchanged', 'passed'])
        local = local and all(case[key] == '0' for key in ['partition_differ', 'range_differ'])
        local = local and all(math.isfinite(float(case[key])) and float(case[key]) <= 1e-5
                             for key in ['reference_rel', 'direct_rel'])
        if not local:
            failed.append(case)
    assert int(fields['failures']) == len(failed)
    assert fields['local_gates_passed'] == ('0' if failed else '1')
    return {'case_results': cases, 'summary': fields, 'failed_cases': failed,
            'cpu': lines[0], 'effective': [line for line in lines if line.startswith('effective ')],
            'reported_dispatch_environment': env_seen}


def parse_binary(path):
    st = path.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_uid == os.geteuid()
    assert stat.S_IMODE(st.st_mode) == 0o600 and st.st_nlink == 1 and st.st_size == 1107104
    raw = path.read_bytes()
    assert struct.unpack_from('<8I', raw) == (0x53575031, 1, 48, 2560, 640, 8, 0, 0)
    pos, cases = 32, []
    for phase, qt, nt, rows in EXPECTED:
        header = struct.unpack_from('<6I', raw, pos)
        assert header == (1 if phase == 'gu' else 2, qt, nt, rows, rows * nt, 0)
        pos += 24
        payload = raw[pos:pos + rows * nt * 4]
        assert all(math.isfinite(v) for v, in struct.iter_unpack('<f', payload))
        cases.append({'phase': phase, 'type': qt, 'nt': nt, 'rows': rows,
                      'payload_offset': pos, 'payload_bytes': len(payload),
                      'payload_sha256': hashlib.sha256(payload).hexdigest()})
        pos += len(payload)
    assert pos == len(raw)
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'cases': cases, 'all_payload_finite': True}, raw


def run(profile, overrides, arm, base_env):
    global child
    label = profile + '-' + arm['label']
    target = OUT / (label + '.bin')
    outpath, errpath = OUT / (label + '.stdout'), OUT / (label + '.stderr')
    env = {key: value for key, value in base_env.items() if key not in CONTROLS}
    env.update(overrides)
    argv = [arm['binary'], '--out', str(target)]
    step = {'label': label, 'profile': profile, 'arm': arm['label'], 'argv': argv,
            'cwd': arm['root'], 'effective_dispatch_environment': {k: env.get(k) for k in CONTROLS},
            'deadline_seconds': 180}
    record['steps'].append(step)
    begin = time.monotonic()
    with outpath.open('wb') as out, errpath.open('wb') as err:
        child = subprocess.Popen(argv, cwd=arm['root'], env=env, stdout=out, stderr=err, start_new_session=True)
        row = identity(child.pid)
        assert row
        try:
            register(row)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=2)
            raise
        step['identity'] = row
        record['active_stage'] = label
        save()
        while child.poll() is None:
            discover()
            assert time.monotonic() - started < 1790, 'whole matrix deadline exceeded'
            assert time.monotonic() - begin < 180, label + ' deadline exceeded'
            assert sum(p.stat().st_size for p in OUT.glob('*.stdout')) + sum(p.stat().st_size for p in OUT.glob('*.stderr')) <= record['text_budget_bytes']
            assert sum(p.stat().st_size for p in OUT.glob('*.bin')) <= record['binary_budget_bytes']
            assert not target.exists() or target.stat().st_size <= 2 * 1024**2
            save(); time.sleep(.1)
        step['exit_code'] = child.returncode
        step['elapsed_seconds'] = time.monotonic() - begin
        step['stdout_sha256'], step['stderr_sha256'] = sha(outpath), sha(errpath)
        step['stdout_bytes'], step['stderr_bytes'] = outpath.stat().st_size, errpath.stat().st_size
        save()
        step['stdout_findings'] = parse_stdout(outpath, overrides)
        step['binary_findings'], raw = parse_binary(target)
        step['local_passed'] = not step['stdout_findings']['failed_cases'] and child.returncode == 0
        save()
        assert step['local_passed'], label + ' local correctness gates rejected'
        assert not live_owners(), label + ' owned process survived normal exit'
    print(json.dumps({'label': label, 'local_passed': True, 'elapsed_seconds': step['elapsed_seconds']}), flush=True)
    return raw


save()
try:
    raw_env = subprocess.check_output(['/bin/bash', '-c', 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'], timeout=30)
    env = dict(item.decode().split('=', 1) for item in raw_env.split(b'\0') if b'=' in item)
    record['toolchain_environment'] = {k: env[k] for k in ['PATH', 'LD_LIBRARY_PATH', 'ONEAPI_ROOT'] if k in env}
    cpuinfo = Path('/proc/cpuinfo').read_text().split('\n\n')[0]
    record['cpuinfo'] = [line for line in cpuinfo.splitlines() if line.startswith(('vendor_id', 'model name', 'flags'))]
    for name, overrides in PROFILES:
        raws = [run(name, overrides, arm, env) for arm in build['arms']]
        pair = {'profile': name, 'T_H_bitwise_equal': raws[0] == raws[1],
                'bytes_each': len(raws[0]), 'first_different_byte': next((i for i, (a, b) in enumerate(zip(*raws)) if a != b), None)}
        record['pairs'].append(pair)
        save()
        assert pair['T_H_bitwise_equal'], name + ' T/H complete outputs differ'
    for arm in build['arms']:
        assert sha(arm['binary']) == arm['binary_sha256']
        assert git(arm['root'], 'rev-parse', 'HEAD') == arm['source_commit']
        assert not git(arm['root'], 'status', '--porcelain')
        for rel, digest in build['source_pins'].items():
            assert sha(Path(arm['root']) / rel) == digest
    record['complete'] = True
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    discover()
    if live_owners():
        cleanup_owned()
    record['survivors'] = live_owners()
    record['passed'] = record['passed'] and not record['cleanup'] and not record['survivors']
    record['active'] = False
    record['active_stage'] = None
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    for fd in pidfds.values():
        os.close(fd)
print(json.dumps({k: record.get(k) for k in ['complete', 'passed', 'error', 'elapsed_seconds', 'cleanup', 'survivors']}), flush=True)
if not record['passed']:
    raise SystemExit(1)
