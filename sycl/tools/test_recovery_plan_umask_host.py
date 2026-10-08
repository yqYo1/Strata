"""Reproduce the saved-plan regression with actual old/new definitions only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
import time
import types


def check(source):
    module = types.ModuleType('cpu_plan_reproduction')
    embedded = source.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    exec(compile(embedded, '<recovery-plan>', 'exec'), module.__dict__)
    previous = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory() as directory:
            module.BASE = Path(directory)
            output = module.BASE / 'display-test'; output.mkdir(mode=0o700)
            path = output / 'plan.json'; user = pwd.getpwuid(os.getuid())
            plan = dict(approved=True, service=module.SERVICE, bdf=module.BDF,
                        user=user.pw_name, uid=user.pw_uid, clients=[(1, 1)],
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                        approved_monotonic=time.monotonic(), worker_started=False,
                        display_stop_attempted=False, display_restore_requested=False)
            module.save_plan(path, plan)
            initial = path.stat().st_mode & 0o777
            os.umask(0o022)
            plan['display_stop_attempted'] = True
            module.save_plan(path, plan)
            error = None
            try:
                module.read_plan(path)
            except RuntimeError as failure:
                error = str(failure)
            return dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        launcher_umask='0077', worker_umask='0022',
                        initial_mode=oct(initial), replacement_mode=oct(path.stat().st_mode & 0o777),
                        read_error=error)
    finally:
        os.umask(previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    control = '6d782060ae2b0633589fa0304bd3bc1c4255e950'
    old = subprocess.check_output(['git', 'show', control + ':sycl/tools/recover-xe-display.sh'],
                                  cwd=root, text=True)
    candidate = Path(__file__).with_name('recover-xe-display.sh').read_text()
    rows = [dict(version=name, **check(source)) for name, source in [('old', old), ('candidate', candidate)]]
    assert rows[0]['replacement_mode'] == '0o644'
    assert rows[0]['read_error'] == 'Unsafe recovery plan ownership/permissions'
    assert rows[1]['replacement_mode'] == '0o600' and rows[1]['read_error'] is None
    record = dict(control_commit=control, rows=rows, passed=True,
                  scope='Actual save/read definitions; CPU temporary files only; no sudo, GUI, reset or GPU operation')
    text = json.dumps(record, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
