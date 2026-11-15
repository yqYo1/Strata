#!/usr/bin/env python3
"""Small traced GPU admission after the unchanged-library host-v2 gate.

Repository LICENSE. Reuse the maintained submission/phase fixture runner in a
private context. No model, Q6 arithmetic, throughput or ordinary teardown claim.
"""
import fcntl
import importlib.util
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
import iq3_host_v2 as host
import check_verify_window_phase as original

P, B, RESULT, base = host.P, host.B, host.RESULT, host.base
ROOT = B / 'decode-iq3-phase-device-v1'
KEY = 'decode_iq3_phase_device_v1'
SELF = Path(__file__).resolve()
require = host.require


def context():
    spec = importlib.util.spec_from_file_location('strata_iq3_phase_admission', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.build = host.build.context()
    ctx.ROOT, ctx.KEY, ctx.SELF = ROOT, KEY, SELF
    built = ctx.built

    def qualified_build():
        native = built()
        checked = json.loads((host.ROOT / 'record.json').read_text())
        require(base.committed(RESULT)[host.KEY] == host.compact(checked) and
                checked['stage'] == 'closed' and checked['passed'] and checked['complete'] and
                checked['ownership_closed'] and not checked['active'] and not checked['GPU_executed'] and
                not checked['native_rebuilt'] and checked['binaries'] == native['binaries'] and
                checked['native_source_commit'] == native['source_commit'] and
                [(c['kind'], c['arm'], c['passed'], c['exit_code']) for c in checked['results']] ==
                    [('exact', 0, True, 0), ('pool', 0, True, 0),
                     ('exact', 1, True, 0), ('pool', 1, True, 0)] and
                checked['pool_checksum'] == host.build.component()['correctness']['pool_checksum'],
                'closed same-build host-v2 arithmetic/route/pool qualification required')
        host.unchanged(checked)
        return native

    ctx.built = qualified_build
    return ctx


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'run'),
            'usage: iq3_device.py prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        ctx = context()
        {'prepare': ctx.prepare, 'run': ctx.run}[sys.argv[1]]()
