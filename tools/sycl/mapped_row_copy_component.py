#!/usr/bin/env python3
"""Shared input/provenance helpers retained for existing diagnostic consumers.

The closed row-copy comparison and its unused native fixture are recoverable
from source85f33092fb8aa308e3a5eec1fbca331dc0540753. This legacy module name
remains for device_fp64_probe and measure_q6_columns; it offers no experiment
CLI and does not qualify changed or historical measurement inputs.
Main-authored, repository LICENSE.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import benchmark_runtime as python_runtime
import measurement_guards as guards
import measurement_owner as owner
import ur_validation as validation

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
HELPERS = ('tools/sycl/mapped_row_copy_component.py', 'tools/sycl/benchmark_runtime.py',
    'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py',
    'tools/sycl/ur_validation.py', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
PACKAGES = ('libc6', 'libc6-dev', 'linux-libc-dev', 'libstdc++6',
            'libstdc++-13-dev', 'libgcc-s1', 'libgcc-13-dev', 'binutils')
require, sha = guards.require, owner.sha


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)


def signature(pins):
    return dict(files=len(pins), aggregate_sha256=hashlib.sha256(
        json.dumps(pins, sort_keys=True).encode()).hexdigest())


def packages():
    return subprocess.check_output(['/usr/bin/dpkg-query', '-W',
        '-f=${binary:Package} ${Version}\n', *PACKAGES], text=True, timeout=15).splitlines()
