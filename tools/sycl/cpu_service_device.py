#!/usr/bin/env python3
"""Reuse the owned small phase/device checks for the CPU observer generation.

Main-authored; repository LICENSE. The maintained runner is loaded into a private
module, with only build/root/control selection changed before execution. Neither
its shared module nor the process environment is mutated. Real-pool parity is a
prerequisite. These payload checks do not qualify CPU service or model speed.
"""
import fcntl
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_cpu_service_model as build
import check_verify_window_phase as original

SELF = Path(__file__).resolve()
ROOT = build.B / 'cpu-service-phase-device-v3'
KEY = 'verify_window_cpu_service_device_v3'
PREVIOUS = 'verify_window_cpu_service_device_v2'


def context():
    spec = importlib.util.spec_from_file_location('strata_cpu_observer_phase_context', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.build, ctx.ROOT, ctx.KEY, ctx.SELF = build, ROOT, KEY, SELF
    built = ctx.built

    def qualified_build():
        native = built()
        host = native.get('host_result', {})
        cases = host.get('cases', [])
        build.require(host.get('passed') and host.get('complete') and host.get('ownership_closed')
            and not host.get('active') and not host.get('GPU_executed') and len(cases) == 2
            and [(r['mode'], r['exit_code'], r['cases']) for r in cases] == [(0, 0, 28), (1, 0, 28)]
            and cases[0]['checksum'] == cases[1]['checksum'],
            'closed same-build observer off/on real-pool parity required')
        return native

    ctx.built = qualified_build
    return ctx


def prepare(ctx):
    prior = build.base.committed(build.RESULT)[PREVIOUS]
    build.require(prior['stage'] == 'failed-preparation' and prior['ownership_closed']
        and not prior['GPU_executed'] and not prior['active']
        and prior['preparation_failure']['exact_prefix_verified'],
        'preserve the closed V2 metadata-size failure before the corrected plan')
    ctx.prepare()


if __name__ == '__main__':
    build.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'),
                  'usage: cpu_service_device.py prepare|run')
    ctx = context()
    with (build.B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.base.guards.no_model_owners([])
        {'prepare': lambda: prepare(ctx), 'run': ctx.run}[sys.argv[1]]()
