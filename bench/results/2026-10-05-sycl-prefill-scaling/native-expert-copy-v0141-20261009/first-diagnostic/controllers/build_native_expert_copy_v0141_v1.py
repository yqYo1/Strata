"""CPU-only uniform build, admitted after the live quiet sequence finishes."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

B = Path(__file__).parent
ROOT = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-prefill-native-copy-v0141-20261009')
OUT = B / 'native-expert-copy-v0141-private-build-v1'
BUILD = ROOT / 'build-sycl-native-expert-copy-v0141-v1'
QUIET = B / 'host-prefill-accounting-v0141-quiet32k-comparison-sequence-v1/record.json'
SOURCE_HEAD = os.environ['STRATA_NATIVE_COPY_SOURCE_HEAD']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def alive(ident):
    try:
        text = (Path('/proc') / str(ident['pid']) / 'stat').read_text()
    except FileNotFoundError:
        return False
    return int(text.rsplit(')', 1)[1].split()[19]) == ident['start_ticks']


assert len(SOURCE_HEAD) == 40 and git('rev-parse', 'HEAD') == SOURCE_HEAD
assert not git('status', '--porcelain')
assert not OUT.exists() and not BUILD.exists()
OUT.mkdir(mode=0o700)
started = time.monotonic()
record = {'active': True, 'passed': False, 'gpu_tested': False, 'adopted': False,
          'compiled_engine': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'CPU-only configure and full rebuild of all engine translation units against one native-copy candidate header; same production compiler flags/settings. No GPU command or adoption.',
          'source_head': SOURCE_HEAD, 'root': str(ROOT), 'controller_sha256': digest(__file__),
          'source_sha256': {p: digest(ROOT / p) for p in ['sycl/include/dpct/device.hpp', 'sycl/src/prefill/prefill.cpp', 'sycl/CMakeLists.txt']},
          'admission_deadline_seconds': 2400, 'steps': []}


def save():
    record['elapsed_seconds'] = time.monotonic() - started
    p = OUT / 'record.json.tmp'
    p.write_text(json.dumps(record, indent=2) + '\n')
    p.replace(OUT / 'record.json')


def command(label, argv, env, deadline):
    step = {'label': label, 'argv': argv, 'timeout_seconds': deadline}
    record['steps'].append(step)
    with (OUT / (label + '.stdout')).open('w') as stdout, (OUT / (label + '.stderr')).open('w') as stderr:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        step['pid'] = child.pid
        begin = time.monotonic()
        save()
        try:
            while child.poll() is None:
                if time.monotonic() - begin >= deadline:
                    raise TimeoutError(label + ' CPU build deadline')
                save()
                time.sleep(2)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
            raise
        step['exit_code'] = child.returncode
        step['elapsed_seconds'] = time.monotonic() - begin
        save()
        assert child.returncode == 0, label


save()
try:
    while True:
        q = json.loads(QUIET.read_text())
        record['active_stage'] = 'wait-for-terminal-quiet-comparison'
        save()
        if not q['active']:
            break
        if time.monotonic() - started > record['admission_deadline_seconds']:
            raise TimeoutError('quiet comparison has not terminated; no CPU build started')
        time.sleep(5)
    assert q['passed'] and len(q['steps']) == 6
    for step in q['steps']:
        p = Path(step['receipt'])
        assert digest(p) == step['receipt_sha256']
        d = json.loads(p.read_text())
        assert not d['active'] and d['completed'] and d['healthy'] and d['math_gate_passed'] and d['host_accounting_gate_passed']
        assert d['exit_code'] == 0 and not d['exit_signal'] and not d['new_fault_messages'] and not any(d['cleanup'].values())
        assert not any(alive(d[k]) for k in ['inferior', 'debugger']) and not alive(step['controller_identity'])
    assert git('rev-parse', 'HEAD') == SOURCE_HEAD and not git('status', '--porcelain')
    record['quiet_sequence_sha256'] = digest(QUIET)
    raw = subprocess.check_output(['/bin/bash', '-c', 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'])
    env = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if b'=' in item)
    record['active_stage'] = 'configure-and-uniform-CPU-build'
    cmake = ['/usr/bin/cmake', '-S', str(ROOT/'sycl'), '-B', str(BUILD), '-G', 'Ninja',
             '-DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx',
             '-DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx', '-DCMAKE_BUILD_TYPE=Release',
             '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
             '-DSTRATA_GGML_DIR=/home/yayoi/orca/workspaces/Strata/ggml-sycl-pinned',
             '-DSTRATA_NATIVE_EXPERTS=ON', '-DSTRATA_SYCL_PARITY=ON', '-DSTRATA_IQ2S_GCC=OFF',
             '-DSTRATA_SYCL_AOT=', '-DSTRATA_SYCL_SPIN_MAX=', '-DSTRATA_NATIVE_POOL_TASK_FACTOR=0']
    command('configure', cmake, env, 180)
    command('build-strata', ['/usr/bin/cmake', '--build', str(BUILD), '--target', 'strata', '--parallel', '4'], env, 1800)
    for relative, expected in record['source_sha256'].items():
        assert digest(ROOT/relative) == expected
    assert git('rev-parse', 'HEAD') == SOURCE_HEAD and not git('status', '--porcelain')
    binary = BUILD / 'strata'
    record['binary'] = str(binary)
    record['binary_sha256'] = digest(binary)
    record['compile_commands_sha256'] = digest(BUILD/'compile_commands.json')
    record['CMakeCache_sha256'] = digest(BUILD/'CMakeCache.txt')
    record['build_ninja_sha256'] = digest(BUILD/'build.ninja')
    record['compiled_engine'] = True
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    record['active'] = False
    record['active_stage'] = None
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({k:v for k,v in record.items() if k not in ['steps','source_sha256']}, indent=2))
if not record['passed']:
    raise SystemExit(1)
