"""CPU-only uniform build of isolated existing pool task-count CLI candidate; no GPU invocation."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

B = Path(__file__).parent
ROOT = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/perf-sycl-decode-pool-phase-timing-v0141-20261009')
OUT = B / 'decode-pool-phase-timing-v0141-private-build-v1'
BUILD = ROOT / 'build-sycl-decode-pool-phase-timing-v0141-v1'
FULL = B / 'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json'
SOURCE_HEAD = '7d0105f2a942ac72ca20fef69d4c2e7960653de3'


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


# Preparation only: terminal current-C admission is not yet filled in.
raise RuntimeError('Root must add current C terminal receipt and owned lock before this draft can build')
# Every previous owned GPU execution must be terminal before this CPU build.
full=json.loads(FULL.read_text())
assert digest(FULL)=='168d7896cbe3a60dfc604a08b6fe310f16d823a17d13d8a013260bfc7facb0a4'
assert not full['active'] and full['completed'] and full['healthy'] and full['math_gate_passed']
assert full['physical256k_sequence_completed'] and full['capacity_sequence_completed']
assert full['exit_code']==0 and not full['exit_signal'] and not full['new_fault_messages'] and not any(full['cleanup'].values())
assert full['binary_sha256']=='86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8'
assert full['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not any(alive(full[k]) for k in ['inferior','debugger'])
proof_path=B/'decode-pool-phase-timing-v0141-source-review-v1.json'
assert digest(proof_path)=='d58482d3fa6d8dbe55d3f672a2142862e07c32b465bcc7a9ffcc1bd2efb9325f'
proof=json.loads(proof_path.read_text())
assert proof['passed'] and proof['removing_diagnostic_reproduces_entire_original_generate_source_byte_identical'] and not proof['gpu_executed'] and not proof['compiled']
assert digest(ROOT/'sycl/src/program/generate.cpp')==proof['source_sha256']=='8bb78a21e3cbb9f03e40db3699d056225181e9ebb638d7eea04fadb41d852ced'
assert digest(ROOT/'sycl/src/prefill/prefill.cpp')=='f3c9bb38b46cdc111e9e663cc606d655f60f07c05873aff5499d51895e886c77'
assert digest(ROOT/'sycl/include/dpct/device.hpp')=='adbcbd0ac45995348e37ac3f5ca42d9f3d4b0555031aeb1781a1e2166a74ed52'
assert len(SOURCE_HEAD)==40 and git('rev-parse','HEAD')==SOURCE_HEAD and not git('status','--porcelain')
seq_path=B/'native-prefill-copy-phase-v0141-decode-repeat-comparison-sequence-v1/record.json'
seq=json.loads(seq_path.read_text());assert not seq['active'] and not seq['passed'] and len(seq['steps'])==17
admission=[]
owned_paths=[]
for step in seq['steps']:
    path=Path(step['receipt'])
    if 'receipt_sha256' in step:
        assert digest(path)==step['receipt_sha256']
    else:
        assert step is seq['steps'][-1] and step['mode']=='phaseon' and step['block']==6 and step['exit_code']==1
        assert digest(path)=='aa5c7c7b262ade3e1f9ddd287abbe4db16459c3146da36733207e5cf0628fbfc'
    assert not step['active'] and not alive(step['controller_identity'])
    owned_paths.append(path)
owned_paths += [B/relative for relative in [
    'owned-native-prefill-copy-phase-v0141-failure-localization-phaseon-diagnostic-r1/record.json',
    'owned-native-prefill-copy-phase-v0141-minimal-diagnostic-phaseon-r1/record.json',
    'owned-native-prefill-copy-phase-v0141-minimal-diagnostic-phaseoff-r1/record.json',
    'owned-native-prefill-copy-phase-v0141-minimal-diagnostic-phaseon-r2/record.json',
    'owned-stager-affinity-v0141-code32k-affinity-diagnostic-r1/record.json']]
for path in owned_paths:
    owned=json.loads(path.read_text())
    assert not owned['active'] and owned['completed'] and owned['healthy'] and owned['exit_code']==0 and not owned['exit_signal'] and not owned['new_fault_messages'] and not any(owned['cleanup'].values())
    assert owned['boot_id']==full['boot_id'] and not any(alive(owned[k]) for k in ['inferior','debugger'])
    failed=digest(path)=='aa5c7c7b262ade3e1f9ddd287abbe4db16459c3146da36733207e5cf0628fbfc'
    assert owned['math_gate_passed'] is (not failed)
    admission.append({'path':str(path),'sha256':digest(path),'terminal_owned_processes_absent':True,'math_gate_passed':owned['math_gate_passed'],'held_prior_candidate_failure_preserved':failed})
assert sum(x['held_prior_candidate_failure_preserved'] for x in admission)==1
assert not OUT.exists() and not BUILD.exists()
OUT.mkdir(mode=0o700)
started = time.monotonic()
record = {'active': True, 'passed': False, 'gpu_tested': False, 'adopted': False,
          'compiled_engine': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'CPU-only configure and full rebuild of every engine translation unit. Existing pool phase counters sampled under the existing diagnostic flag only; inherited pool tasks CLI unchanged. No native copy-only factory, prefill/stager, queue, kernel or common CPU-source change. Same qualified compiler flags/settings. No GPU command or adoption.',
          'source_head': SOURCE_HEAD, 'root': str(ROOT), 'controller_sha256': digest(__file__),
          'source_sha256': {p: digest(ROOT / p) for p in ['sycl/include/dpct/device.hpp', 'sycl/src/prefill/prefill.cpp', 'sycl/src/program/generate.cpp', 'sycl/CMakeLists.txt']},
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
    assert git('rev-parse', 'HEAD') == SOURCE_HEAD and not git('status', '--porcelain')
    record['qualified_baseline_full_receipt_sha256'] = digest(FULL)
    record['source_bound_default_and_host_review_sha256'] = digest(proof_path)
    record['prior_owned_gpu_admission'] = admission
    record['prior_numerical_failure_remains_held'] = True
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
