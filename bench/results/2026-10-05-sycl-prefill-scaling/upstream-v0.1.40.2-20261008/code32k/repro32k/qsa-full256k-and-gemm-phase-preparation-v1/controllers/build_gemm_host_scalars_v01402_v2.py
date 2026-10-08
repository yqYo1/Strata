"""Build exactly one private GEMM object on the fully qualified DD5 baseline.

Fail before any mutation or large reads while the full256K controller is active.
This builds a candidate; it does not execute an engine or adopt an optimization.
"""
from pathlib import Path
import datetime, hashlib, json, os, shlex, shutil, signal, subprocess, time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
build = root / 'build-sycl-event-ack-registered-copy-v3-20261008/build'
current_path = base/'owned-qsa-reduce12-sg32-v01402-full256k-diagnostic-r1/record.json'
current = json.loads(current_path.read_text())
assert not current['active'], 'active QSA full256K job: no compile or large input reads'
assert current['completed'] and current['healthy'] and current['math_gate_passed']
assert current['physical256k_sequence_completed'] and current['capacity_sequence_completed']
assert current['exit_code'] == 0 and not current['exit_signal'] and not current.get('error')
assert not current['new_fault_messages'] and not any(current['cleanup'].values())
assert current['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for role in ['inferior', 'debugger']:
    row = current[role]
    try:
        stat = (Path('/proc')/str(row['pid'])/'stat').read_text()
    except FileNotFoundError:
        continue
    assert int(stat[stat.rindex(')')+2:].split()[19]) != row['start_ticks'], 'owned QSA process remains live'

full_path = base / 'owned-profile-definition-v01402-full256k-diagnostic-r7/record.json'
full = json.loads(full_path.read_text())
assert not full['active'], 'active full256K job: do not build concurrently'
assert full['completed'] and full['healthy'] and full['math_gate_passed']
assert full['physical256k_sequence_completed'] and full['capacity_sequence_completed']
assert full['exit_code'] == 0 and not full['exit_signal']
assert not full['new_fault_messages'] and not any(full['cleanup'].values())
assert full['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert full['actual_executable_identity']['sha256'] == 'dd5efb9002167481d180f370b86fb6978d724f37745f7e4a923432296725ad34'
for role in ['inferior', 'debugger']:
    row = full[role]
    try:
        stat = (Path('/proc') / str(row['pid']) / 'stat').read_text()
    except FileNotFoundError:
        continue
    assert int(stat[stat.rindex(')') + 2:].split()[19]) != row['start_ticks'], 'owned process remains live'

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

source_dir = base / 'gemm-host-scalars-v01402-source-v1'
source_record_path = source_dir / 'record.json'
prepared = json.loads(source_record_path.read_text())
source = Path(prepared['candidate_source'])
assert prepared['prepared'] and not prepared['compiled'] and not prepared['gpu_tested'] and not prepared['adopted']
assert prepared['changed_default_call_sites'] == 2 and all(prepared['source_contract_review'].values())
assert digest(source) == prepared['candidate_source_sha256'] == '0225c809c1608a59719d84809075b3349bddcf09e5a9bb7aada4bc611aada955'
original_source = root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/prefill/gemm.dp.cpp'
assert digest(original_source) == prepared['source_inputs_sha256'][str(original_source)] == '11a2c943ccc323c328b58d99119d77b3b4ed95c92a0f680480bf5ede96bcb705'
assert all(digest(path) == expected for path,expected in prepared['source_inputs_sha256'].items())
uniform_path = base / 'dpct-profile-definition-v01402-build-v1/record.json'
assert digest(uniform_path) == 'b4e79456dc8222ef6187c1667ce798dc4b3f67d18f9da28aee9ca06adb56d2c0'
uniform = json.loads(uniform_path.read_text())
assert uniform['passed'] and not uniform['active'] and uniform['baseline_inputs_unchanged']
baseline = Path(uniform['candidate_binary'])
assert digest(baseline) == uniform['candidate_binary_sha256'] == full['actual_executable_identity']['sha256']
shadow = Path(uniform['shadow_header'])
assert digest(shadow) == uniform['shadow_header_sha256']
candidate = root / 'build-sycl-gemm-host-scalars-v2-20261008'
out = base / 'gemm-host-scalars-v01402-build-v2'
candidate.mkdir()
out.mkdir(mode=0o700)
started = time.monotonic()
record = {
    'active': True, 'passed': False, 'gpu_tested': False, 'adopted': False, 'steps': [],
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'controller_sha256': digest(__file__),
    'scope': 'Private compile/link only: replace one gemm.dp.cpp.o member on physical256K-qualified DD5. Preserve all other actual link inputs and uniform DPCT header definitions. No GPU workload or speed claim.',
    'preceding_qsa_full256k_receipt_sha256': digest(current_path),
    'baseline_full256k_receipt_sha256': digest(full_path),
    'baseline_binary_sha256': digest(baseline),
    'baseline_build_receipt_sha256': digest(uniform_path),
    'source_review_sha256': digest(source_record_path),
    'candidate_source_sha256': digest(source), 'candidate_source': str(source),
    'shadow_header': str(shadow), 'shadow_header_sha256': digest(shadow),
    'minimum_performance_input_tokens': 32768}
def save():
    record['elapsed_seconds'] = time.monotonic() - started
    (out / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
env = dict(os.environ, PATH='/usr/bin:/bin:/opt/intel/oneapi/compiler/2026.1/bin',
           MKLROOT='/opt/intel/oneapi/mkl/2026.1',
           LIBRARY_PATH='/opt/intel/oneapi/mkl/2026.1/lib:/opt/intel/oneapi/compiler/2026.1/lib:/usr/lib/x86_64-linux-gnu')
env.pop('LD_LIBRARY_PATH', None)
env.pop('LD_PRELOAD', None)
def run(label, argv):
    step = {'label': label, 'argv': argv}
    record['steps'].append(step)
    save()
    then = time.monotonic()
    with (out / (label + '.stdout')).open('wb') as a, (out / (label + '.stderr')).open('wb') as b:
        proc = subprocess.Popen(argv, cwd=build, env=env, stdout=a, stderr=b, start_new_session=True)
        step['pid'] = proc.pid
        stat = (Path('/proc') / str(proc.pid) / 'stat').read_text()
        step['start_ticks'] = int(stat[stat.rindex(')') + 2:].split()[19])
        save()
        try:
            rc = proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
    step.update(exit_code=rc, elapsed_seconds=time.monotonic() - then)
    save()
    assert rc == 0, (label, rc)

save()
try:
    link = list(uniform['steps'][-1]['argv'])
    assert uniform['steps'][-1]['label'] == 'link'
    inputs = {str((build / arg).resolve()): digest((build / arg).resolve())
              for arg in link if arg.endswith(('.a', '.o')) and (build / arg).is_file()}
    assert inputs == full['uniform_candidate_link_input_sha256']
    record['baseline_link_input_sha256'] = inputs
    prefill_paths = [Path(path) for path in inputs if Path(path).name == 'libstrata_prefill.a']
    assert len(prefill_paths) == 1
    original_archive = prefill_paths[0]
    assert digest(original_archive) == '1c4a8d644ce2d5d74d0466e59e109ea902b337bee084151f862b215026b8db1f'
    private_archive = candidate / 'libstrata_prefill.a'
    object_path = candidate / 'gemm.dp.cpp.o'
    dep = candidate / 'gemm.dp.cpp.o.d'
    target = 'CMakeFiles/strata_prefill.dir/src/prefill/gemm.dp.cpp.o'
    command = subprocess.check_output(['/usr/bin/ninja', '-t', 'commands', '-s', target], cwd=build, text=True, timeout=30)
    assert len(command.splitlines()) == 1
    (out / 'original-compile-command.txt').write_text(command)
    argv = shlex.split(command)
    assert argv[0] == '/opt/intel/oneapi/compiler/2026.1/bin/icpx'
    assert Path(argv[argv.index('-c') + 1]) == original_source
    assert '-fp-model=precise' in argv and '-fsycl-default-sub-group-size=32' in argv
    for flag, value in [('-c', str(source)), ('-o', str(object_path)), ('-MT', str(object_path)), ('-MF', str(dep))]:
        argv[argv.index(flag) + 1] = value
    argv.insert(1, '-I' + str(shadow.parent.parent))
    run('compile', argv)
    dependencies = shlex.split(dep.read_text().replace('\\\n', ' '))[1:]
    resolved_deps = [str((build / name).resolve()) for name in dependencies]
    assert str(source.resolve()) in resolved_deps and str(shadow.resolve()) in resolved_deps
    actual_dpct_device_headers = {path for path in resolved_deps if path.endswith('/dpct/device.hpp')}
    assert actual_dpct_device_headers == {str(shadow.resolve())}
    dep_rows = [{'path': path, 'sha256': digest(path)} for path in sorted(set(resolved_deps))]
    (out / 'actual-dependency-files.json').write_text(json.dumps(dep_rows, indent=2) + '\n')
    record.update(actual_dependency_file_sha256=digest(dep), actual_dependency_inventory_sha256=digest(out / 'actual-dependency-files.json'),
                  candidate_object_sha256=digest(object_path), original_profiled_dpct_define_retained=True)
    shutil.copy2(original_archive, private_archive)
    members = subprocess.check_output(['/usr/bin/ar', 't', original_archive], text=True).splitlines()
    assert members.count(object_path.name) == 1
    before_members = {member: hashlib.sha256(subprocess.check_output(['/usr/bin/ar', 'p', original_archive, member])).hexdigest()
                      for member in members}
    original_object = build/target
    assert original_object.is_file()
    assert before_members[object_path.name] == digest(original_object)
    record['original_gemm_object_sha256'] = digest(original_object)
    run('archive-replace', ['/usr/bin/ar', 'r', str(private_archive), str(object_path)])
    assert subprocess.check_output(['/usr/bin/ar', 't', private_archive], text=True).splitlines() == members
    rows = []
    for member in members:
        after = hashlib.sha256(subprocess.check_output(['/usr/bin/ar', 'p', private_archive, member])).hexdigest()
        changed = after != before_members[member]
        assert changed == (member == object_path.name)
        if changed:
            assert after == digest(object_path)
        rows.append({'member': member, 'before_sha256': before_members[member], 'after_sha256': after, 'replaced': changed})
    record['archive_members'] = rows
    assert link.count(str(original_archive)) == 1
    link = [str(private_archive) if arg == str(original_archive) else arg for arg in link]
    binary = candidate / 'strata'
    link[link.index('-o') + 1] = str(binary)
    run('link', link)
    assert digest(baseline) == record['baseline_binary_sha256']
    assert all(digest(path) == sha for path, sha in inputs.items())
    assert all(digest(row['path']) == row['sha256'] for row in dep_rows)
    record.update(passed=True, baseline_inputs_unchanged=True, only_gemm_archive_member_replaced=True,
                  candidate_binary=str(binary), candidate_binary_sha256=digest(binary), candidate_archive_sha256=digest(private_archive),
                  pending=['Initial logged>=32768 full numerical/session gate',
                           'Quiet>=32768 matched first/later full-read comparison',
                           'Repeated physical256K, actual disk restoration, clipped tail, capacity refusals and later fresh32K before adoption'])
except BaseException as error:
    record['error'] = repr(error)
    raise
finally:
    record.update(active=False, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
print(json.dumps({k: record.get(k) for k in ['passed', 'error', 'elapsed_seconds', 'candidate_binary_sha256', 'only_gemm_archive_member_replaced']}))
