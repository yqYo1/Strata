#!/usr/bin/env python3
"""One native launch per owned benchmark root; no retries or environment mutation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def identity(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), sid=int(fields[3]))


def main():
    root = Path(os.environ['STRATA_MEASUREMENT_ROOT'])
    plan = json.loads((root / 'plan.json').read_text())
    binary = plan['binary']
    with Path(binary['path']).open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != binary['sha256']:
            raise RuntimeError('engine binary changed')
    claim = dict(wrapper=identity(os.getpid()), binary=binary, args=sys.argv[1:])
    # A server restart must fail before another native process/GPU launch.
    with (root / 'launch.json').open('x') as stream:
        json.dump(claim, stream)
    child = subprocess.Popen([binary['path'], *sys.argv[1:]], env=dict(os.environ))
    claim['native'] = identity(child.pid)
    (root / 'engine-start.json').write_text(json.dumps(claim) + '\n')
    code = child.wait()
    (root / 'engine-exit.json').write_text(json.dumps(dict(claim, exit_code=code)) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
