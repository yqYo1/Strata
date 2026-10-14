#!/usr/bin/env python3
"""Preserve profiler injection and record the actual runtime child's exit."""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def identity(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, ppid=int(fields[1]), sid=int(fields[3]), start_ticks=int(fields[19]))


root, executable = map(Path, sys.argv[1:])
entry = dict(binary=str(executable), binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
    wrapper=identity(os.getpid()), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    instrumentation_environment={k:v for k,v in os.environ.items() if k.startswith(('AMPLXE_', 'INTEL_LIBITTNOTIFY')) or k == 'LD_PRELOAD'})
child = subprocess.Popen([str(executable)], env=dict(os.environ))
entry['native'] = identity(child.pid)
(root/'runtime-start.json').write_text(json.dumps(entry, indent=2)+'\n')
code = child.wait()
entry.update(exit_code=code, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(root/'runtime-exit.json').write_text(json.dumps(entry, indent=2)+'\n')
sys.exit(code if code >= 0 else 128-code)
