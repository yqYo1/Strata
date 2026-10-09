"""Root serial full cache-route-pairs engine build, no model/GPU target invocation."""
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
OUT = B / 'cache-route-pairs-v0141-private-build-v1'
GGML = Path('/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned')
ARMS = [('P', W / 'diag-sycl-cache-route-pairs-v0141-20261010',
         '66aaec3a66ad9ed12e6d042cffe743c47e364432')]
PINS = {'sycl/CMakeLists.txt': 'd8fc6c99428f96021c0d986bcad77680dbe66f4fa6a64490166ff58f24e274b1', 'sycl/src/prefill/prefill.cpp': 'f3c9bb38b46cdc111e9e663cc606d655f60f07c05873aff5499d51895e886c77', 'sycl/src/program/generate.cpp': 'eb591e263398abd9b178ab00c61e4f6af53a53bad37f837b18cd323af8dbab05', 'sycl/include/dpct/device.hpp': 'adbcbd0ac45995348e37ac3f5ca42d9f3d4b0555031aeb1781a1e2166a74ed52', 'include/strata/kernels/cpu/native_expert.hpp': '87ff3c6d2aedf0d1985b002480086113426d79d284d1e7ac9cf9f01e55f7163e', 'include/strata/kernels/cpu/pool.hpp': '5c200f068a27d43414cbca2159459944a2541476d0d99cf55f8b962b4b62299b', 'src/kernels/cpu/expert_layout.cpp': '9826d86cc83330d447d62a62189dc782f549ffb815c7bdccd2ad4f4582672f4d', 'src/kernels/cpu/native_expert.cpp': '767f02f33aca4da9fabbe113c50ce1d7cb2aa9f25b4234df59224ec46a5985cd', 'src/kernels/cpu/pool.cpp': '5ef33f7d2020a17ecc36ad15796c6f021b4b5963998aa964dccedbe733637368'}
PINS.update({'sycl/include/strata/core/cache_route_pairs.hpp': 'c06b174f8059a3fec6638bfdde59820126f00c1db809a30ce89a6087a888ad42', 'sycl/include/strata/core/expert_source.hpp': 'f94d0f2ed4d14f5aa4ad402fcda041b2a46b2690976d08ff1a0b740555043247', 'sycl/src/core/expert_source.cpp': '41b98031b865338f0429b195ef4aa10ba94cbaf2c0f2224ab1c3a4de7c37f4aa', 'sycl/tools/cache-route-pairs-test.cpp': 'fc9a5a3c036b385a3c1b26321dabf63d5416810509280a6988d446d24dbc326a'})


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
cpu_fixture = B / 'cache-route-pairs-cpu-validation-v2/record.json'
assert sha(cpu_fixture) == '2803c78cf7621b8f5ebcda50d67f1ba769a28e07d32568508115e91168e35994'
cpu_status = json.loads(cpu_fixture.read_text())
assert cpu_status['complete'] and cpu_status['passed'] and not cpu_status['active'] and not cpu_status['cleanup']
assert sha(B / 'native-wrapper-matched-cpu-matrix-v1/record.json') == '60616fc79ce250e574383e83943cbcaa9564cb1cdab6e81e6a737eefb030dfdc'
qualification = json.loads((B / 'native-wrapper-matched-cpu-matrix-v1/record.json').read_text())
assert qualification['complete'] and qualification['passed'] and not qualification['active']
assert len(qualification['steps']) == 18 and len(qualification['pairs']) == 9
assert all(p['T_H_bitwise_equal'] for p in qualification['pairs'])
assert not qualification['cleanup'] and not qualification['survivors']
assert git(GGML, 'rev-parse', 'HEAD') == '3cf03257f219afbe7334045ff7c6a06ac68c627d'
assert not git(GGML, 'status', '--porcelain')
for label, root, head in ARMS:
    assert git(root, 'rev-parse', 'HEAD') == head and not git(root, 'status', '--porcelain')
    assert not (root / 'build-sycl-cache-route-pairs-v1').exists()
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
          'scope': 'Uniform full cache-route-pairs engine build with qualified T/H production flags. Portable ASan/UBSan fixture passed separately. No executable/GPU/model/profiler invoked; cache counter and original C remain unadopted.',
          'controller_sha256': sha(__file__), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'baseline_build_receipt_sha256': sha(B / 'decode-pool-phase-timing-v0141-private-build-v2/record.json'),
          'ggml_commit': git(GGML, 'rev-parse', 'HEAD'), 'source_pins': PINS, 'portable_cpu_receipt_sha256': sha(cpu_fixture),
          'deadline_seconds': 2400, 'text_budget_bytes': 64 * 1024**2,
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
            assert time.monotonic() - started < 2390, 'total working deadline exceeded'
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
        build = root / 'build-sycl-cache-route-pairs-v1'
        argv = ['/usr/bin/cmake', '-S', str(root / 'sycl'), '-B', str(build), '-G', 'Ninja',
                '-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx',
                '-DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx',
                '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DSTRATA_GGML_DIR=' + str(GGML), '-DSTRATA_NATIVE_EXPERTS=ON',
                '-DSTRATA_SYCL_PARITY=ON',
                '-DSTRATA_IQ2S_GCC=OFF', '-DSTRATA_SYCL_AOT=', '-DSTRATA_SYCL_SPIN_MAX=',
                '-DSTRATA_NATIVE_POOL_TASK_FACTOR=0']
        command(label + '-configure', argv, root, env, 120)
        command(label + '-build', ['/usr/bin/cmake', '--build', str(build), '--target',
                                 'strata', '--parallel', '4'], root, env, 1800)
        assert git(root, 'rev-parse', 'HEAD') == head and not git(root, 'status', '--porcelain')
        for rel, expected in PINS.items():
            assert sha(root / rel) == expected
        binary = build / 'strata'
        arm = {'label': label, 'source_commit': head, 'root': str(root), 'build': str(build),
               'binary': str(binary), 'binary_sha256': sha(binary),
               'files': {rel: sha(build / rel) for rel in ['compile_commands.json', 'CMakeCache.txt', 'build.ninja', '.ninja_log']},
               'runtime_tested': False}
        record['arms'].append(arm)
        save()
    assert not git(GGML, 'status', '--porcelain') and git(GGML, 'rev-parse', 'HEAD') == record['ggml_commit']
    record['source_head'] = ARMS[0][2]
    record['root'] = str(ARMS[0][1])
    record['binary'] = record['arms'][0]['binary']
    record['binary_sha256'] = record['arms'][0]['binary_sha256']
    record['compile_commands_sha256'] = record['arms'][0]['files']['compile_commands.json']
    record['build_ninja_sha256'] = record['arms'][0]['files']['build.ninja']
    record['source_sha256'] = PINS
    record['compiled_engine'] = True
    record['gpu_tested'] = False
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
