#!/home/yayoi/.local/share/strata-sycl/venv/bin/python
"""Keep native streams intact and record its identity before and after execution."""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from owned_thread_sampler import Sampler
from bounded_stderr_relay import Relay
from pathlib import Path

ENGINE = '/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-native-build-root-v1/build/strata'


def identity(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ppid=int(fields[1]), sid=int(fields[3]), start_ticks=int(fields[19]))


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    env = dict(os.environ)
    observer = env.pop('STRATA_MEASUREMENT_THROW_OBSERVER', None)
    if observer:
        assert Path(observer).is_file() and 'LD_PRELOAD' not in env
        env['LD_PRELOAD'] = observer
    with open(ENGINE, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    child = subprocess.Popen([ENGINE, *sys.argv[1:]], env=env, stderr=subprocess.PIPE)
    entry = dict(engine=ENGINE, binary_sha256=digest, exception_observer=observer,
                 argv=[ENGINE, *sys.argv[1:]], wrapper=identity(os.getpid()),
                 native=identity(child.pid), started_utc=started)
    pipe = os.stat('/proc/' + str(child.pid) + '/fd/0')
    entry['stdin'] = dict(device=pipe.st_dev, inode=pipe.st_ino)
    Path(os.environ['STRATA_MEASUREMENT_ENGINE_START']).write_text(json.dumps(entry, indent=2) + '\n')
    sampler = Sampler(entry['native'], Path(os.environ['STRATA_MEASUREMENT_ENGINE_START']).parent)
    sampler.start()
    relay = Relay(child.stderr, Path(os.environ['STRATA_MEASUREMENT_ENGINE_START']).parent)
    relay.start()
    try:
        code = child.wait()
    finally:
        sampler.close()
        relay.close()
    entry.update(pid=child.pid, exit_code=code,
                 finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    Path(os.environ['STRATA_MEASUREMENT_ENGINE_EXIT']).write_text(json.dumps(entry, indent=2) + '\n')
    sys.exit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    main()
