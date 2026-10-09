"""Retire only the resolved disposable mock; immutable receipts stay byte-identical."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import time

A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sha = lambda p: hashlib.file_digest(Path(p).open('rb'), 'sha256').hexdigest()
plan_path = A / 'mock-v1-artifact-retirement-plan.json'
assert sha(plan_path) == 'd6c9e00650fb8896bfc6bf05540d68b7a1f3ad763245d9fcca96064b7ef0a625'
plan = json.loads(plan_path.read_text())
lock = (B / 'owned-v0141-measurement.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
start = time.monotonic()

def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[19])
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None

for name, expected, archive in [
    ('repeat-capture-cpu-validation-v1/record.json', plan['original_failed_receipt_sha256'], 'repeat-capture-CPU-mock-rejected-v1-record.json'),
    ('repeat-capture-cpu-validation-v2/record.json', plan['corrected_passed_receipt_sha256'], 'repeat-capture-CPU-mock-passed-v2-record.json')]:
    path = B / name
    assert sha(path) == sha(A / archive) == expected
    receipt = json.loads(path.read_text())
    assert not receipt['active'] and not receipt['cleanup'] and not receipt['survivors']
    for row in receipt['owners'].values():
        assert identity(row['pid']) != row['start_ticks']
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: name = (proc / 'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError): continue
    assert name not in ['strata', 'strata-xe-health', 'gdb', 'ninja', 'icpx', 'vtune', 'gprofng', 'gp-collect-app'], (proc.name, name)
assert not (A / 'mock-v1-artifact-retirement-result.json').exists()
path = Path(plan['path'])
assert path.parent == B / 'repeat-capture-cpu-validation-v1' and not path.is_symlink()
st = path.stat()
assert st.st_dev == plan['device'] and st.st_ino == plan['inode'] and st.st_mtime_ns == plan['mtime_ns']
assert st.st_size == plan['size'] and sha(path) == plan['sha256']
result = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope': 'Resolved synthetic mock only; no models/source/services/GPU/receipt changes.',
          'script_sha256': sha(Path(__file__)), 'plan_sha256': sha(plan_path),
          'removed_path': str(path), 'original_size': st.st_size, 'original_sha256': plan['sha256'],
          'logical_removed_bytes': st.st_size, 'file_allocation_removed_bytes': st.st_blocks * 512,
          'available_space_delta_measured': False, 'original_failed_status_preserved': True,
          'complete': False}
path.unlink()
assert not path.exists()
for name, expected in [('repeat-capture-cpu-validation-v1/record.json', plan['original_failed_receipt_sha256']),
                       ('repeat-capture-cpu-validation-v2/record.json', plan['corrected_passed_receipt_sha256'])]:
    assert sha(B / name) == expected
result['complete'] = True
result['elapsed_seconds'] = time.monotonic() - start
with (A / 'mock-v1-artifact-retirement-result.json').open('x') as stream:
    json.dump(result, stream, indent=2); stream.write('\n')
print(json.dumps(result))
