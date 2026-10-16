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
    child = subprocess.Popen([ENGINE, *sys.argv[1:]])
    code = child.wait()
    Path(os.environ['STRATA_MEASUREMENT_ENGINE_EXIT']).write_text(json.dumps(dict(
        engine=ENGINE, argv=[ENGINE, *sys.argv[1:]], pid=child.pid, exit_code=code,
        started_utc=started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2) + '\n')
    sys.exit(code if code >= 0 else 128 - code)
