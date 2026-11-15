#!/usr/bin/env python3
"""Fresh owned native checks for the compiled bulk-mirror engine.

Main-authored from existing source/recipe/fixture admission; repository LICENSE.
The prior numerical producer result does not qualify attention or model speed.
"""
import fcntl
import importlib.util
import sys

sys.dont_write_bytecode = True
import build_qsa_bulk_model as build
import check_qsa16_stream_model as original

P, B = build.P, build.B
ROOT, KEY = B / 'qsa16-bulk-mirror-model-device-v1', 'qsa16_bulk_mirror_model_device_v1'
THIS = 'tools/sycl/check_qsa_bulk_model.py'
STAGES = original.STAGES


def context():
    spec = importlib.util.spec_from_file_location('qsa_bulk_native_checks', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.NATIVE, ctx.NATIVE_KEY = build.ROOT, build.KEY
    return ctx


if __name__ == '__main__':
    build.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', *STAGES),
        'check_qsa_bulk_model.py prepare|' + '|'.join(STAGES))
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original.guards.no_model_owners([])
        ctx = context()
        if sys.argv[1] == 'prepare':
            ctx.prepare()
        else:
            ctx.execute(sys.argv[1])
