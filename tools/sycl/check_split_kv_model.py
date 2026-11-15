#!/usr/bin/env python3
"""Run the unchanged owned model handoff protocol on the B1 generation.

Main-authored adapter; repository LICENSE. No copied external implementation,
new native build, inference setting, or inherited model result.
"""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import check_named_kv_model as original
import check_split_kv_oracle as oracle

P, B = oracle.P, oracle.B
ROOT, KEY = B / 'phase-kv-split-model-v1', 'phase_kv_split_model_v1'
THIS = 'tools/sycl/check_split_kv_model.py'


def controller():
    spec = importlib.util.spec_from_file_location('split_kv_model_driver', original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.KEY, module.THIS = ROOT, KEY, THIS
    module.oracle = oracle.controller()
    module.native_device = oracle.native
    return module


def prepare():
    module = controller()
    module.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    oracle.native.require(row['stage'] == 'prepared' and not row['results']
                          and not row['commands'], 'fresh B1 model plan only')
    row['plan'] = (
        'B1 only: cooperative decode TU extraction and required linking, compiled '
        'coherently at0569e6924. One fresh2f83ee16engine process with unchanged '
        'thinking-on32769/82/32769inputs and32outputs each. Preserve placement, '
        '8192chunk,MTPspec4,head2/dense1,large-GU and immutable safety settings. '
        'Require host INFO/stage and literal KV controls, two actual scalar-prefill/'
        'cooperative-decode4 handoffs, typed QSA/GU/Down/job conservation, MTP '
        'restore/cache/recapture,normal0,quiet validation,no fresh fault,closed '
        'ownership and15full asset hashes before/after. Stop first failure; '
        'no retry or changed pins. No old model result is inherited.')
    row['limits'] = (
        'Production route/state handoff only. Clean matched PP/TTFT and sustained '
        'thinking decode remain separate comparisons; no naming-only static-image '
        'compatibility exception applies to B1. Task/agent,physical262144model '
        'fill/tails/RESTORE,ordinary teardown and default adoption remain separate.')
    row['retention'] = (
        'No new native generation. Preserve both independent phase references '
        'through the imminent B1 per-unit comparison; retire successful raw model '
        'logs/cache after compact routes,work totals,provenance and closure are '
        'verified. Do not keep full row histograms as permanent evidence.')
    module.controller().context(row['native_observer_admission']).save(row)
    print('Prepared B1 production handoff;commit before host/model.', flush=True)


if __name__ == '__main__':
    module = controller()
    module.require(Path(sys.prefix) == Path(module.coherent.PYTHON).parent.parent,
                   'use qualified model venv')
    module.require(len(sys.argv) == 2 and sys.argv[1] in module.coherent.ACTIONS,
                   'prepare|host|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        module.common.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        elif sys.argv[1] == 'host':
            controls = module.previous.controls()
            module.controller().execute('host')
            row = json.loads((ROOT / 'record.json').read_text())
            row['KV_host_controls'] = controls
            module.controller().context(row['native_observer_admission']).save(row)
        else:
            row = json.loads((ROOT / 'record.json').read_text())
            module.require(row['KV_host_controls']['passed'], 'literal KV controls required')
            module.controller().execute('run')
