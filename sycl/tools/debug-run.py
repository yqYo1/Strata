#!/usr/bin/python3
"""Run a first SYCL correctness check with flushed UR/Level Zero API logs."""
import os
from pathlib import Path
import sys
import types


if len(sys.argv) < 2:
    raise SystemExit('Usage: python3 sycl/tools/debug-run.py EXECUTABLE [arguments...]')

# Reuse the recovery probe's diagnostic settings without calling recovery or
# changing the caller's device, library paths, submission or copy settings.
helper = Path(__file__).with_name('recover-xe.sh')
source = helper.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
recovery = types.ModuleType('xe_diagnostic_environment')
exec(compile(source, str(helper), 'exec'), recovery.__dict__)
env = recovery.diagnostic_environment(os.environ)
os.execvpe(sys.argv[1], sys.argv[1:], env)
