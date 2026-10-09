"""Root-only closed CPU fixture retirement; original result status is immutable."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import time

A = Path(__file__).parent
W = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
PLAN = A / 'reader-synthetic-retirement-plan.json'
RESULT = A / 'reader-synthetic-retirement-result.json'
PLAN_SHA = 'a616745a7f6a6f0323fbffd361e8e4dd2214621974aa5d9af6f53dd5f52d80ef'
start = time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(fields[19])}
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not RESULT.exists()
assert sha(PLAN) == PLAN_SHA
git = lambda *args: subprocess.check_output(['git', '-C', str(W), *args], text=True).strip()
head = git('rev-parse', 'HEAD')
assert head == git('rev-parse', 'origin/docs/sycl-storage-retention-2026-10-09')
committed = subprocess.check_output(['git', '-C', str(W), 'show',
                                   'HEAD:' + str(PLAN.relative_to(W))])
assert hashlib.sha256(committed).hexdigest() == PLAN_SHA
plan = json.loads(PLAN.read_text())
assert len(plan['delete_files']) == 164 and not plan['current_raw_consumers']
for row in plan['retained_evidence']:
    assert sha(row['path']) == row['sha256'], row['path']
archive = json.loads((A / 'archive-identity.json').read_text())
for row in archive['byte_identical_files']:
    assert sha(row['original']) == sha(row['archived']) == row['sha256']
receipts = []
roots = []
for version in [1, 2, 3]:
    for kind in ['validation', 'supervisor']:
        root = B / ('repeat-capture-reader-cpu-%s-v%d' % (kind, version))
        roots.append(root)
        path = root / 'record.json'
        data = json.loads(path.read_text())
        assert not data['active'] and not data['gpu_executed'] and not data['model_opened']
        assert data['passed'] == (version == 3)
        assert data['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        for row in data.get('owners', {}).values():
            now = identity(row['pid'])
            assert not now or now['start_ticks'] != row['start_ticks']
        receipts.append({'path': str(path), 'sha256': sha(path), 'passed': data['passed']})

# Verify all planned files before any unlink, including both names of a test
# hardlink. Link counts can then decrease because another planned name is gone.
seen = set()
for row in plan['delete_files']:
    assert time.monotonic() - start < 60
    path = Path(row['path'])
    assert path not in seen
    seen.add(path)
    assert any(path.is_relative_to(root) for root in roots)
    assert path.name != 'record.json'
    for parent in path.parents:
        assert not parent.is_symlink()
        if parent == B:
            break
    st = path.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid()
    assert (st.st_dev, st.st_ino, st.st_size, st.st_nlink) == \
        (row['dev'], row['ino'], row['bytes'], row['nlink'])
    assert sha(path) == row['sha256'], path

result = {'complete': False, 'passed': False, 'plan_sha256': PLAN_SHA,
          'committed_pushed_evidence_head': head, 'original_receipts': receipts,
          'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'removed': [], 'available_space_delta_measured': False,
          'gpu_executed': False, 'source_models_sessions_services_untouched': True}


def save():
    result['elapsed_seconds'] = time.monotonic() - start
    temporary = RESULT.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(RESULT)


save()
try:
    for row in plan['delete_files']:
        assert time.monotonic() - start < 60
        path = Path(row['path'])
        st = path.lstat()
        assert stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid()
        assert (st.st_dev, st.st_ino, st.st_size) == (row['dev'], row['ino'], row['bytes'])
        path.unlink()
        result['removed'].append({'path': str(path), 'bytes': row['bytes'],
                                  'sha256': row['sha256']})
        save()
    assert all(not Path(row['path']).exists() for row in plan['delete_files'])
    for row in receipts:
        assert sha(row['path']) == row['sha256']
        assert json.loads(Path(row['path']).read_text())['passed'] == row['passed']
    result['removed_logical_bytes_sum'] = plan['logical_bytes_sum']
    result['removed_unique_inode_allocation_bytes_sum'] = plan['unique_inode_allocation_bytes_sum']
    result['original_receipts_unchanged'] = True
    result['complete'] = True
    result['passed'] = True
except BaseException as error:
    result['error'] = repr(error)
finally:
    result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
print(json.dumps({k: result.get(k) for k in ['complete', 'passed', 'error',
                 'removed_logical_bytes_sum', 'removed_unique_inode_allocation_bytes_sum',
                 'original_receipts_unchanged', 'elapsed_seconds']}))
if not result['passed']:
    raise SystemExit(1)
