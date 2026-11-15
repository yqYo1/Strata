#!/usr/bin/env python3
"""Apply the owned KV oracle to B1 without another driver copy.

Main-authored adapter over check_named_kv_oracle; repository LICENSE.
All numerical, graph, ownership and fault gates remain in that shared driver.
"""
import fcntl
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
import check_named_kv_oracle as original
import check_split_kv as native

P, B, RESULT = native.P, native.B, native.RESULT
ROOT, KEY = B / 'phase-kv-split-oracle-v1', 'phase_kv_split_oracle_v1'
THIS = 'tools/sycl/check_split_kv_oracle.py'


def controller():
    spec = importlib.util.spec_from_file_location('split_kv_oracle_driver', original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.KEY, module.THIS = ROOT, KEY, THIS
    module.native = native
    module.OBSERVERS = {native.THIS, THIS}
    return module


def prepare():
    module = controller()
    module.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    native.require(row['stage'] == 'built' and not row['cases'] and not row['commands'],
                   'fresh B1 oracle plan only')
    row['plan']['source'] = (
        'B1 only: independent cooperative decode TU and required production/fixture '
        'linking from97ed15507,compiled coherently at0569e6924. Keep scalar body, '
        'geometry,placement,settings and every original oracle criterion unchanged. '
        'Use this generation\'s actual14th fixture after its11fresh native stages; '
        'no old GPU/model/performance result is inherited.')
    row['plan']['limits'] = (
        'Component correctness only;model handoff,matched prefill/TTFT and sustained '
        'thinking decode comparisons remain required. No naming-only static-image '
        'compatibility exception applies to this different B1 image. Physical262144 '
        'model fill,RESTORE,task/agent and ordinary teardown remain separate.')
    row['retention'] = (
        'Use the existing B1fixture;no new binary. Retain fast14508and validated A '
        'references for the immediate independent phase comparisons. Retire successful '
        'oracle trace/cache after useful correctness/closure/provenance evidence is '
        'preserved and its actual consumers close. No fourth engine generation.')
    module.save(row)
    print('Prepared B1 same-build KV oracle;commit before smoke/full.', flush=True)


if __name__ == '__main__':
    native.require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'smoke', 'full'),
                   'prepare|smoke|full')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        native.common.guards.no_model_owners([])
        if sys.argv[1] == 'prepare':
            prepare()
        else:
            controller().runner().run(sys.argv[1])
