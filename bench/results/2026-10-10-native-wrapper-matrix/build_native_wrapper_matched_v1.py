"""Root serial matched CPU target builds, no model/GPU target invocation."""
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
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree')
OUT = B / 'native-wrapper-matched-private-build-v1'
GGML = Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
ARMS = [('T', W / 'diag-native-wrapper-parity-baseline-20261010',
         '61a70650a004d27ec622046b3d4d6499e7012ca0'),
        ('H', W / 'diag-native-wrapper-parity-20261010',
         '87b50497bc4bac7b40113f181ad51b47e64ba7ae')]
PINS = {'sycl/CMakeLists.txt': 'a6bd64de6ec4dd9c4298608e777551ae53b00321745087752200ed4ccc2848dc',
        'sycl/tools/native_wrapper_parity.cpp': '339658d5c8d13baf59ffe246aac7ac3a0911d60ce01619773c5ce9f587f76fd6',
        'src/kernels/cpu/iq_avx2_parity.cpp': 'fb4e0b0629e749b816c4ec71da836f383330cf93aed49a11d03607937e1232d9'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def identity(pid):
    try:
        f = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(f[1]), 'start_ticks': int(f[19])}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUT.exists()
assert git(GGML, 'rev-parse', 'HEAD') == '3cf03257f219afbe7334045ff7c6a06ac68c627d'
assert not git(GGML, 'status', '--porcelain')
for label, root, head in ARMS:
    assert git(root, 'rev-parse', 'HEAD') == head and not git(root, 'status', '--porcelain')
    assert not (root / 'build-sycl-native-wrapper-v1').exists()
    for rel, expected in PINS.items():
        assert sha(root / rel) == expected
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        comm = (proc / 'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    assert comm not in ['strata', 'strata-xe-health', 'gdb', 'ninja', 'icpx', 'icx',
                        'vtune', 'gprofng', 'gp-collect-app'], (proc.name, comm)
OUT.mkdir(mode=0o700)
started = time.monotonic()
record = {'active': True, 'complete': False, 'passed': False, 'gpu_executed': False,
          'model_opened': False, 'adopted': False, 'performance_eligible': False,
          'original_C_math_gate_passed': False,
          'scope': 'Only native_wrapper_parity target linked actual SYCL CPU library; same two-file test patch in frozenT and candidateH. Global SYCL link flags preserved; no target executable invoked by this build.',
          'controller_sha256': sha(__file__), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'baseline_build_receipt_sha256': sha(B / 'decode-pool-phase-timing-v0141-private-build-v2/record.json'),
          'ggml_commit': git(GGML, 'rev-parse', 'HEAD'), 'source_pins': PINS,
          'deadline_seconds': 1200, 'text_budget_bytes': 64 * 1024**2,
          'steps': [], 'arms': [], 'owners': {}, 'cleanup': [], 'survivors': []}
assert record['baseline_build_receipt_sha256'] == '3632b1514544ef2dd89046c112b823c75a27d0215aca2fce58c4b8df9e43d7f8'
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
                try:
                    register(row)
                except ProcessLookupError:
                    continue
                added = added or str(row['pid']) in record['owners']
        if not added:
            break


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


def command(label, argv, root, env, timeout):
    global child
    step = {'label': label, 'argv': argv, 'cwd': str(root), 'deadline_seconds': timeout}
    record['steps'].append(step)
    begin = time.monotonic()
    with (OUT / (label + '.stdout')).open('wb') as out, (OUT / (label + '.stderr')).open('wb') as err:
        child = subprocess.Popen(argv, cwd=root, env=env, stdout=out, stderr=err, start_new_session=True)
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
            assert time.monotonic() - started < 1190, 'total working deadline exceeded'
            assert time.monotonic() - begin < timeout, label + ' deadline exceeded'
            assert sum(p.stat().st_size for p in OUT.glob('*.stdout')) + \
                   sum(p.stat().st_size for p in OUT.glob('*.stderr')) <= record['text_budget_bytes']
            save(); time.sleep(.5)
        step['exit_code'] = child.returncode
        step['elapsed_seconds'] = time.monotonic() - begin
        save()
        assert child.returncode == 0, label


save()
try:
    # Env is used internally; only toolchain paths/variables are retained, not
    # arbitrary inherited values or credentials.
    raw_env = subprocess.check_output(['/bin/bash', '-c', 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'])
    env = dict(item.decode().split('=', 1) for item in raw_env.split(b'\0') if b'=' in item)
    record['toolchain_env'] = {k: env[k] for k in ['PATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH', 'ONEAPI_ROOT'] if k in env}
    for compiler in ['/opt/intel/oneapi/compiler/2026.1/bin/icpx', '/opt/intel/oneapi/compiler/2026.1/bin/icx']:
        record.setdefault('compiler_sha256', {})[compiler] = sha(compiler)
    for label, root, head in ARMS:
        build = root / 'build-sycl-native-wrapper-v1'
        argv = ['/usr/bin/cmake', '-S', str(root / 'sycl'), '-B', str(build), '-G', 'Ninja',
                '-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx',
                '-DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx',
                '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DSTRATA_GGML_DIR=' + str(GGML), '-DSTRATA_NATIVE_EXPERTS=ON',
                '-DSTRATA_SYCL_PARITY=OFF', '-DSTRATA_SYCL_NATIVE_WRAPPER_PARITY=ON',
                '-DSTRATA_IQ2S_GCC=OFF', '-DSTRATA_SYCL_AOT=', '-DSTRATA_SYCL_SPIN_MAX=',
                '-DSTRATA_NATIVE_POOL_TASK_FACTOR=0']
        command(label + '-configure', argv, root, env, 120)
        command(label + '-build', ['/usr/bin/cmake', '--build', str(build), '--target',
                                 'native_wrapper_parity', '--parallel', '4'], root, env, 600)
        assert git(root, 'rev-parse', 'HEAD') == head and not git(root, 'status', '--porcelain')
        for rel, expected in PINS.items():
            assert sha(root / rel) == expected
        binary = build / 'native_wrapper_parity'
        arm = {'label': label, 'source_commit': head, 'root': str(root), 'build': str(build),
               'binary': str(binary), 'binary_sha256': sha(binary),
               'files': {rel: sha(build / rel) for rel in ['compile_commands.json', 'CMakeCache.txt', 'build.ninja', '.ninja_log']},
               'runtime_tested': False}
        record['arms'].append(arm)
        save()
    assert not git(GGML, 'status', '--porcelain') and git(GGML, 'rev-parse', 'HEAD') == record['ggml_commit']
    record['complete'] = True
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    discover()
    live = [r for r in record['owners'].values() if (now := identity(r['pid'])) and now['start_ticks'] == r['start_ticks']]
    if live:
        cleanup_owned()
    if child is not None:
        record['last_child_exit_code'] = child.poll()
    record['survivors'] = [r for r in record['owners'].values() if (now := identity(r['pid'])) and now['start_ticks'] == r['start_ticks']]
    record['passed'] = record['passed'] and not record['survivors'] and not record['cleanup']
    record['active'] = False
    record['active_stage'] = None
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    for fd in pidfds.values():
        os.close(fd)
print(json.dumps({k: record.get(k) for k in ['complete', 'passed', 'error', 'elapsed_seconds', 'cleanup', 'survivors']}))
if not record['passed']:
    raise SystemExit(1)
