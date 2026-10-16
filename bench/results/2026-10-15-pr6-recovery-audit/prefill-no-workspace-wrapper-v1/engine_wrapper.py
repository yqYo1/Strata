#!/home/yayoi/.local/share/strata-sycl/venv/bin/python
"""Keep native streams intact and record its identity before and after execution."""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ENGINE = '/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-prefill-no-workspace-build-root-v1/strata'
ENGINE_SHA256 = '5de4d2695a3226f037ab8eb9330da0b33b5e9a4761598837c51bb5e93c0fbe99'


def identity(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ppid=int(fields[1]), sid=int(fields[3]), start_ticks=int(fields[19]))


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    env = dict(os.environ)
    observer = None
    assert 'LD_PRELOAD' not in env and 'STRATA_MEASUREMENT_THROW_OBSERVER' not in env
    assert env['STRATA_SYCL_PARALLEL_IQ_DEQUANT'] in ('0', '1')
    assert env['STRATA_SYCL_PROMPT_DIRECT'] == '0'
    with open(ENGINE, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == ENGINE_SHA256, 'selective native hash mismatch'
    child = subprocess.Popen([ENGINE, *sys.argv[1:]], env=env)
    entry = dict(engine=ENGINE, binary_sha256=digest, exception_observer=observer,
                 argv=[ENGINE, *sys.argv[1:]], wrapper=identity(os.getpid()),
                 native=identity(child.pid), started_utc=started)
    pipe = os.stat('/proc/' + str(child.pid) + '/fd/0')
    entry['stdin'] = dict(device=pipe.st_dev, inode=pipe.st_ino)
    Path(os.environ['STRATA_MEASUREMENT_ENGINE_START']).write_text(json.dumps(entry, indent=2) + '\n')
    code = child.wait()
    entry.update(pid=child.pid, exit_code=code,
                 finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    Path(os.environ['STRATA_MEASUREMENT_ENGINE_EXIT']).write_text(json.dumps(entry, indent=2) + '\n')
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    main()
