"""Root-owned uniform build of bounded repeated-prefill capture; no GPU invocation.

Requires the terminal C full-context receipt SHA256 as its only positional
argument. C is a separate candidate; its known numerical rejection is preserved.
Only the independently qualified integrated baseline admits this CPU-only build. Optional --cpu-preflight verifies admission without building.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import fcntl
import sys
import signal
import subprocess
import time

B = Path(__file__).parent
ROOT = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/debug-sycl-layer-major-repeat-v0141-20261009')
OUT = B / 'repeat-capture-v0141-private-build-v1'
BUILD = ROOT / 'build-sycl-repeat-capture-v0141-v1'
FULL = B / 'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json'
SOURCE_HEAD = '9c2ebde89e5157c81a9c9ae135719452a0244ca3'


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


assert len(sys.argv) in [2,3] and (len(sys.argv)==2 or sys.argv[2]=='--cpu-preflight')
assert len(sys.argv[1])==64 and all(c in '0123456789abcdef' for c in sys.argv[1])
lock=(B/'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
last_C_full=B/'owned-layer-major-streamed-kv-v0141-full256k-streamed-lease-diagnostic-r1/record.json'
assert digest(last_C_full)==sys.argv[1]
last=json.loads(last_C_full.read_text())
assert not last['active'] and last['completed'] and last['healthy']
# This build changes D's existing host counter reporting, not C's prefill path.
# Preserve the exact terminal rejection; this never admits C timing or adoption.
assert digest(last_C_full)=='a3413ac4c3f20ae593d8507a7db0ac405cd7fab2de286547460e6ba82b0d9404'
assert last['math_gate_passed'] is False and last['full_lifecycle_passed'] is False
assert last['physical256k_sequence_completed'] is False
assert last['mathematical_rejection']=='mathematical gate rejected; no timing/adoption'
assert [q['name'] for q in last['requests']]==['control32k-before','resume32k-reference','full256k-first','control32k-between-full-reads','full256k-repeat']
assert all(q['math_gate_passed'] for q in last['requests'][:-1])
assert not last['requests'][-1]['math_gate_passed']
assert last['requests'][-1]['comparison']['ids_equal'] and not last['requests'][-1]['comparison']['all_live_state_equal']
assert last['sessions'][1]['all_saved_state_and_kv_bytes_equal_same_version_baseline']
assert last['sessions'][-1]['name']=='save-numerical-failure' and last['sessions'][-1]['passed']
localization=B/'streamed-kv-repeat-localization-v1.json'
assert digest(localization)=='96f908d51cb3fe535c892e38550fb55c6f41a35a2aa54e24f1e8c52f4dffcb23'
local=json.loads(localization.read_text())
assert local['complete'] and not local['gpu_executed'] and local['original_math_gate_passed'] is False
assert last['exit_code']==0 and not last['exit_signal'] and not last['new_fault_messages'] and not any(last['cleanup'].values())
assert last['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not any(alive(last[k]) for k in ['inferior','debugger'])
C32=B/'owned-layer-major-streamed-kv-v0141-code32k-streamed-lease-diagnostic-r1/record.json'
assert digest(C32)=='f54eb757d9553403b59a3f72eb07b036c5d90068f57466b1c97f629b7003b7b2'
assert last['first_four_fresh_C_lease_receipt_sha256']==digest(C32)
assert last['binary_sha256']=='66d5bee60f4bbdb6f710b395ae11a1e6e14a61e04db417bcd6092d4f7956377e'
for item in subprocess.check_output(['ps','-eo','pid=,comm='],text=True).splitlines():
    pid,name=item.split(maxsplit=1)
    assert name not in ['strata','gdb','icpx','icx','ninja','strata-xe-health'],item
# Every previous owned GPU execution must be terminal before this CPU build.
full=json.loads(FULL.read_text())
assert digest(FULL)=='168d7896cbe3a60dfc604a08b6fe310f16d823a17d13d8a013260bfc7facb0a4'
assert not full['active'] and full['completed'] and full['healthy'] and full['math_gate_passed']
assert full['physical256k_sequence_completed'] and full['capacity_sequence_completed']
assert full['exit_code']==0 and not full['exit_signal'] and not full['new_fault_messages'] and not any(full['cleanup'].values())
assert full['binary_sha256']=='86972697ecb1903750d202f8be29a7a1da351eb3415678c3e3b6af790804d1c8'
assert full['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not any(alive(full[k]) for k in ['inferior','debugger'])
proof_path=B/'repeat-capture-v0141-source-review-v2.json'
assert digest(proof_path)=='a5ed2740c973830ed67275a89f8f911910abaa6d497a2f3424f5b45a595838b7'
proof=json.loads(proof_path.read_text())
assert proof['passed'] and proof['generate_identical_to_C'] and proof['unknown_copy_destination_retained_until_process_exit']
assert proof['original_C_math_gate_passed'] is False and not proof['gpu_executed'] and not proof['compiled']
assert proof['source_head']==SOURCE_HEAD
for relative,expected in proof['source_sha256'].items():assert digest(ROOT/relative)==expected
T_path=B/'owned-decode-pool-phase-timing-v0141-code32k-tasks6-diagnostic-r1/record.json'
assert digest(T_path)=='e770766de708a5ab8842b4aa3ea80a6d1453765059e8b7816e36aed09331148c'
T=json.loads(T_path.read_text());assert not T['active'] and T['completed'] and T['healthy'] and T['math_gate_passed']
assert T['exit_code']==0 and not T['exit_signal'] and not T['new_fault_messages'] and not any(T['cleanup'].values())
assert not any(alive(T[k]) for k in ['inferior','debugger'])
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
if sys.argv[2:]==['--cpu-preflight']:
    print(json.dumps({'passed':True,'gpu_executed':False,'compiled':False,'source_head':SOURCE_HEAD,'latest_C_full_receipt_sha256':digest(last_C_full),'original_C_math_gate_passed':False,'C_qualified':False,'source_review_sha256':digest(proof_path)}))
    sys.exit(0)
OUT.mkdir(mode=0o700)
started = time.monotonic()
record = {'active': True, 'passed': False, 'gpu_tested': False, 'adopted': False,
          'compiled_engine': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'CPU-only uniform rebuild of opt-in bounded repeated-prefill row capture. Based on numerically rejected C; capture destination quarantine fixes an unproven error lifetime, not the C numerical failure. Same qualified compiler flags/settings. No GPU command, mathematical qualification or adoption.',
          'source_head': SOURCE_HEAD, 'root': str(ROOT), 'controller_sha256': digest(__file__),
          'source_sha256': {p: digest(ROOT / p) for p in ['sycl/include/dpct/device.hpp', 'sycl/src/prefill/prefill.cpp', 'sycl/src/program/generate.cpp', 'sycl/src/prefill/repeat_capture.hpp', 'sycl/CMakeLists.txt']},
          'admission_deadline_seconds': 2400, 'log_limit_bytes':64*1024**2, 'steps': []}


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
                if sum(p.stat().st_size for p in OUT.glob('*.stdout'))+sum(p.stat().st_size for p in OUT.glob('*.stderr'))>=record['log_limit_bytes']:
                    raise RuntimeError(label+' build text budget exceeded')
                if time.monotonic()-started>=record['admission_deadline_seconds']:
                    raise TimeoutError('total build deadline')
                if time.monotonic() - begin >= deadline:
                    raise TimeoutError(label + ' CPU build deadline')
                save()
                time.sleep(2)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait(timeout=10)
            step['terminated_on_error']=True
            step['exit_code']=child.returncode
            save()
            raise
        step['exit_code'] = child.returncode
        step['elapsed_seconds'] = time.monotonic() - begin
        save()
        assert child.returncode == 0, label


save()
try:
    assert git('rev-parse', 'HEAD') == SOURCE_HEAD and not git('status', '--porcelain')
    record['qualified_baseline_full_receipt_sha256'] = digest(FULL)
    record['last_owned_C_full_receipt_sha256']=digest(last_C_full)
    record['prior_C32_lease_receipt_sha256']=digest(C32)
    record['latest_C_full_processes_absent']=True
    record['original_C_math_gate_passed']=False
    record['C_candidate_qualified']=False
    record['C_failure_localization_sha256']=digest(localization)
    record['build_independent_of_C']='Diagnostic build only; original C full-context failure remains held. No performance/adoption admission.'
    record['exclusive_measurement_lock']=True
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
