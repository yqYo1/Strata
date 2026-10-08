"""Finite private CPU build, admitted after the updated baseline finishes."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-v0.1.41-20261009')
plan_path = base / 'host-prefill-accounting-v0141-private-build-plan-v1/record.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


assert digest(plan_path) == 'abb2df8e7b332ff0254a108dadad9a14e9ce42e44bb2b8c01cebd578477b761e'
plan = json.loads(plan_path.read_text())
assert plan['prepared'] and not plan['compiled_engine'] and plan['compiler_flags_preserved']
assert not git('status', '--porcelain')
source_head = git('rev-parse', 'HEAD')
subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_commit'], source_head], cwd=root, check=True)
# Later documentation and the uncompiled launcher fix may be committed first.
# Engine or header changes invalidate the exact compiled input plan.
changes = git('diff', '--name-only', plan['source_commit'], source_head).splitlines()
assert all(p == 'sycl/serve/strata-sycl.sh' or p.startswith(('docs/', 'bench/')) for p in changes), changes
for rel, expected in plan['original_link_input_sha256'].items():
    assert digest(Path(rel) if rel.startswith('/') else Path(plan['cwd']) / rel) == expected, rel
assert digest(base / 'host-prefill-accounting-v0141-source-v1/prefill.cpp') == plan['candidate_source_sha256']
cpu_path = base / 'host-prefill-accounting-v0141-cpu-check-v1/record.json'
assert digest(cpu_path) == plan['cpu_struct_check_receipt_sha256']
cpu = json.loads(cpu_path.read_text())
assert cpu['passed'] and not cpu['active']
full_path = base / 'owned-upstream-v0141-integrated-full256k-diagnostic-r2/record.json'
tuned_path = base / 'upstream-v0141-tuned-matched32k-comparison-sequence-v1/record.json'
full = json.loads(full_path.read_text())
tuned = json.loads(tuned_path.read_text())
assert not full['active'] and full['healthy'] and full['math_gate_passed']
assert full['physical256k_sequence_completed'] and full['capacity_sequence_completed']
assert full['exit_code'] == 0 and not full['exit_signal'] and not full['new_fault_messages']
assert not any(full['cleanup'].values())
assert not tuned['active'] and tuned['passed'] and len(tuned['steps']) == 4
assert full['binary_sha256'] == plan['original_binary_sha256']
assert digest(root / 'build-sycl-integrated-20261009/strata') == plan['original_binary_sha256']
out = Path(plan['output'])
out.mkdir(mode=0o700)
env_raw = subprocess.check_output(['/bin/bash', '-c', 'source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 && env -0'])
env = dict(x.decode().split('=', 1) for x in env_raw.split(b'\0') if b'=' in x)
record = {
    'active': True, 'passed': False, 'compiled_engine': False,
    'gpu_tested': False, 'adopted': False,
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'CPU-only private host-accounting build; exact ninja compile flags, one private prefill object/archive and unchanged original other link inputs. No tracked source mutation, GPU execution or speed/adoption claim.',
    'plan_receipt_sha256': digest(plan_path),
    'physical_baseline_receipt_sha256': digest(full_path),
    'quiet_baseline_receipt_sha256': digest(tuned_path),
    'source_head_observed': source_head, 'compiled_engine_source_commit': plan['source_commit'],
    'candidate_source_sha256': plan['candidate_source_sha256'],
    'source_sha256': digest(Path(__file__)), 'steps': [],
}
started = time.monotonic()


def save():
    record['elapsed_seconds'] = time.monotonic() - started
    p = out / 'record.json.tmp'
    p.write_text(json.dumps(record, indent=2) + '\n')
    p.replace(out / 'record.json')


def run(label, argv, timeout):
    step = {'label': label, 'argv': argv, 'cwd': plan['cwd'], 'timeout_seconds': timeout}
    record['steps'].append(step)
    with (out / (label + '.stdout')).open('w') as stdout, (out / (label + '.stderr')).open('w') as stderr:
        child = subprocess.Popen(argv, cwd=plan['cwd'], env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        step['pid'] = child.pid
        begin = time.monotonic()
        save()
        try:
            while child.poll() is None:
                if time.monotonic() - begin >= timeout:
                    raise TimeoutError(label + ' private build deadline')
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
    run('compile-prefill', plan['compile_argv'], 900)
    setup = plan['archive_setup']
    shutil.copy2(setup['copy_from'], setup['copy_to'])
    run('replace-private-archive-member', setup['replace_member_argv'], 20)
    run('index-private-archive', setup['ranlib_argv'], 20)
    members = subprocess.check_output(['/usr/bin/ar', 't', setup['copy_to']], text=True).splitlines()
    assert members == setup['expected_members']
    original_members = subprocess.check_output(['/usr/bin/ar', 't', setup['copy_from']], text=True).splitlines()
    assert members == original_members
    for member in members:
        old = subprocess.check_output(['/usr/bin/ar', 'p', setup['copy_from'], member])
        new = subprocess.check_output(['/usr/bin/ar', 'p', setup['copy_to'], member])
        if member != 'prefill.cpp.o':
            assert old == new, member
    run('link-private-binary', plan['link_argv'], 300)
    for rel, expected in plan['original_link_input_sha256'].items():
        assert digest(Path(rel) if rel.startswith('/') else Path(plan['cwd']) / rel) == expected, rel
    assert git('rev-parse', 'HEAD') == source_head and not git('status', '--porcelain')
    binary = out / 'strata'
    record['binary'] = str(binary)
    record['binary_sha256'] = digest(binary)
    record['private_object_sha256'] = digest(out / 'prefill.cpp.o')
    record['private_archive_sha256'] = digest(setup['copy_to'])
    record['all_other_archive_members_identical'] = True
    record['original_link_inputs_unchanged'] = True
    record['compiled_engine'] = True
    record['passed'] = True
except BaseException as error:
    record['error'] = repr(error)
finally:
    record['active'] = False
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps(record, ensure_ascii=False))
if not record['passed']:
    raise SystemExit(1)
