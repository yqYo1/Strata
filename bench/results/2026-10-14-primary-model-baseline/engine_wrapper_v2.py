#!/home/yayoi/.local/share/strata-sycl/venv/bin/python
"""Record the real engine's exit status without changing its argv or streams."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ENGINE = '/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-primary-native-build-root-v1/build/strata'

if __name__ == '__main__':
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    env = dict(os.environ)
    observer = env.pop('STRATA_MEASUREMENT_THROW_OBSERVER', None)
    if observer:
        assert Path(observer).is_file() and 'LD_PRELOAD' not in env
        env['LD_PRELOAD'] = observer
    child = subprocess.Popen([ENGINE, *sys.argv[1:]], env=env)
    code = child.wait()
    Path(os.environ['STRATA_MEASUREMENT_ENGINE_EXIT']).write_text(json.dumps(dict(
        engine=ENGINE, exception_observer=observer, argv=[ENGINE, *sys.argv[1:]], pid=child.pid, exit_code=code,
        started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2) + '\n')
    sys.exit(code if code >= 0 else 128 - code)
