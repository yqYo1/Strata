"""Root-owned cache-pair controller-v3 pure CPU fixture only; no inference."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, signal, subprocess, time

B = Path(__file__).parent
OUT = B / 'cache-route-pairs-protocol-v3-cpu-validation-v1'
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/diag-sycl-cache-route-pairs-v0141-20261010')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
pins = {
    'run_owned_cache_route_pairs_v0141_code32k_v3.py': 'cbce9a3cd0fedac9daadae66071bf6b7e5baccf355eb99538869f90b91808518',
    'test_cache_route_pairs_protocol_integration_v3.py': 'd303e8845a3f0e874cf42dced0852348167897e277f2bc29697212584802b868',
    'bounded_engine_protocol_v1.py': 'aa72bd7f1a9babf7c6e325b10634f1a046fe78fc2b2738e691d90624197131c1',
    'test_bounded_engine_protocol_v1.py': 'd40df828556798d66900fa54ed8fcc5159cbd53d736b56794da33823d9e30e44',
    'test_cache_route_pairs_parser_v2.py': 'e80c9ffbaeb0a772998977dde210d5b75fe6fec445d3e7282776b915d86f7266',
    'cache_route_pairs_parser_v1.py': '2ff44099f7385054dc5663f04a8e226d364d4d349a84b2493bc2bb1d6f3585cd',
    'run_owned_cache_route_pairs_v0141_code32k_v2.py': '312bb2d93b28033623a2997e47742fdbb1df33e555d116747496c0c29d213673',
    'cache-route-pairs-v0141-private-build-v1/record.json': 'b812f9df5186681f31d3a82c1fa7c07b7f73e5c16cb4ea7232c0e0c62edcd833',
}
for name, expected in pins.items():
    assert sha(B / name) == expected
build = json.loads((B / 'cache-route-pairs-v0141-private-build-v1/record.json').read_text())
assert build['passed'] and build['complete'] and not build['active']
assert not build['cleanup'] and not build['survivors']
binary = Path(build['binary'])
assert sha(binary) == build['binary_sha256'] == '194968fa045e1afe5805d2cc87ebb4c3c06acf23c86fd9e780655e2ca428b972'
flags_path = B / 'cache-route-pairs-v0141-uniform-build-flags-v1.json'
flags_hash = sha(flags_path)
flags = json.loads(flags_path.read_text())
assert flags['passed'] and flags['build_receipt_sha256'] == pins['cache-route-pairs-v0141-private-build-v1/record.json']
assert flags['candidate_compile_count'] == 115 and not flags['different_flags'] and not flags['missing_sources']
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not OUT.exists()
OUT.mkdir(mode=0o700)
start = time.monotonic()
record = {'active': True, 'complete': False, 'passed': False,
          'scope': 'AST-extracted v3 protocol/budget functions with independent fake source; not controller/PTY/model integration; no controller integration/inference/GPU helper/model payload read',
          'gpu_work_submitted': False, 'model_opened': False, 'adopted': False,
          'performance_eligible': False, 'full_lifecycle_passed': False,
          'original_C_D_math_failures_preserved': True,
          'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'controller_sha256': sha(__file__), 'pins': pins,
          'flags_receipt_sha256': flags_hash, 'binary_sha256': sha(binary),
          'deadline_seconds': 120, 'text_budget_bytes': 2*1024**2,
          'steps': [], 'cleanup': [], 'survivors': []}
env = os.environ.copy()
env.update(build['toolchain_env'])
env['PYTHONDONTWRITEBYTECODE'] = '1'
env.pop('STRATA_CACHE_ROUTE_PAIRS', None)
env.update(ZE_ENABLE_VALIDATION_LAYER='1', ZE_ENABLE_PARAMETER_VALIDATION='1',
           ZEL_ENABLE_LOADER_LOGGING='1', ZEL_LOADER_LOGGING_LEVEL='warn')
record['test_environment'] = {k: env[k] for k in ('LD_LIBRARY_PATH', 'ZE_ENABLE_VALIDATION_LAYER', 'ZE_ENABLE_PARAMETER_VALIDATION', 'ZEL_ENABLE_LOADER_LOGGING', 'ZEL_LOADER_LOGGING_LEVEL', 'PYTHONDONTWRITEBYTECODE') if k in env}
def save():
    record['elapsed_seconds'] = time.monotonic()-start
    temporary = OUT / 'record.json.tmp'
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(OUT / 'record.json')
def identity(pid):
    try:
        values = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(values[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None
def run(label, argv, expected=0, extra=None, child_owns_lock=False):
    # The preflight itself acquires the shared lock. Transfer the lock rather
    # than causing a false recursive-lock rejection; all root steps are serial.
    if child_owns_lock:
        fcntl.flock(lock, fcntl.LOCK_UN)
    step = {'label': label, 'argv': argv, 'expected_exit_code': expected,
            'deadline_seconds': 30, 'lock_owner': 'child-controller' if child_owns_lock else 'root-wrapper'}
    record['steps'].append(step)
    test_env = env.copy()
    if extra is not None:
        test_env.update(extra)
        step['case_environment'] = extra
    stdout = OUT / (label+'.stdout')
    stderr = OUT / (label+'.stderr')
    begin = time.monotonic()
    child = None
    try:
        with stdout.open('wb') as output, stderr.open('wb') as error:
            child = subprocess.Popen(argv, cwd=W, env=test_env, stdout=output, stderr=error, start_new_session=True)
            owner = identity(child.pid)
            assert owner is not None
            step['owner'] = owner
            save()
            while child.poll() is None:
                assert time.monotonic()-begin < 30 and time.monotonic()-start < 115, 'CPU qualification deadline'
                assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) <= record['text_budget_bytes'], 'CPU output budget'
                time.sleep(.05)
            step['exit_code'] = child.returncode
            step['elapsed_seconds'] = time.monotonic()-begin
            assert child.returncode == expected, label+' unexpected exit'
            assert identity(child.pid) is None
    finally:
        if child is not None and child.poll() is None:
            if identity(child.pid) == step.get('owner'):
                os.killpg(child.pid, signal.SIGKILL)
                record['cleanup'].append({'label': label, 'killed_owned_group': step['owner']})
            child.wait(timeout=5)
        if child_owns_lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for kind, path in [('stdout', stdout), ('stderr', stderr)]:
            if path.exists():
                step[kind+'_bytes'] = path.stat().st_size
                step[kind+'_sha256'] = sha(path)
        save()
    return stdout, stderr

save()
try:
    _, errors = run('protocol-fixture', ['/usr/bin/python3', str(B / 'test_cache_route_pairs_protocol_integration_v3.py')])
    assert 'Ran 9 tests' in errors.read_text() and errors.read_text().rstrip().endswith('OK')
    for name, expected in pins.items():
        assert sha(B / name) == expected
    record['controller_integration_executed'] = False
    record.update(complete=True, passed=True)
except BaseException as error:
    record['error'] = repr(error)
finally:
    record['active'] = False
    record['passed'] = record['passed'] and not record['cleanup']
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({key: record.get(key) for key in ('complete', 'passed', 'error', 'elapsed_seconds', 'cleanup', 'gpu_work_submitted')}))
if not record['passed']:
    raise SystemExit(1)
