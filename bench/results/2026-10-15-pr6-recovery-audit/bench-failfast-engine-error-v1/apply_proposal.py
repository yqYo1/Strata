"""Apply reviewed proposal to a specified checkout; no tests or native build."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
HERE = Path(__file__).resolve().parent
root = Path(sys.argv[1]).resolve()
manifest = json.loads((HERE/'source-manifest.json').read_text())
for name, expected in manifest['original_files'].items():
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'original hash mismatch: {name}: {actual}')
target = root/'tools/test_strata_bench_failfast.py'
if target.exists():
    raise SystemExit(f'refusing to overwrite {target}')
subprocess.run(['git', 'apply', '--check', str(HERE/'source.diff')], cwd=root, check=True)
subprocess.run(['git', 'apply', str(HERE/'source.diff')], cwd=root, check=True)
shutil.copyfile(HERE/'test_strata_bench_failfast.py', target)
for name, expected in manifest['proposed_files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'proposed hash mismatch: {name}')
