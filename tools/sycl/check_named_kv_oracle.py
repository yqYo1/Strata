#!/usr/bin/env python3
"""Fresh same-build KV oracle after the naming unit's11native stages.

Main-owned use of the unchanged Strata oracle/graph/ownership driver;
repository LICENSE. No old fixture execution or performance gate is inherited.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import check_named_kv as native
import fixture_build as common
import kv_planner_experiment as driver

P, B, RESULT = native.P, native.B, native.RESULT
ROOT, KEY = B / 'phase-kv-naming-oracle-v1', 'phase_kv_naming_oracle_v1'
THIS = 'tools/sycl/check_named_kv_oracle.py'
OBSERVERS = {native.THIS, THIS}
require, sha, signature = native.require, native.sha, native.signature


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('controller', 'runtime_files', 'commands')}
    out['controller'] = {k: v for k, v in row['controller'].items() if k != 'source_sha256'}
    out['runtime_files_sha256'] = signature(row['runtime_files'])
    return out


def save(row):
    common.write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); common.write(RESULT, data)


def basis(expected=None):
    n = native.native(expected, observers=OBSERVERS)
    prior = common.validation.committed(RESULT, (native.KEY,))
    require(prior['stage'] == 'closed' and not prior['active']
            and all(prior[k] for k in ('passed', 'complete', 'ownership_closed'))
            and len(prior['results']) == len(native.STAGES) == 11
            and [v['mode'] for v in prior['results']] == list(native.STAGES)
            and all(v['passed'] and v['ownership_closed'] and v['native_exit_code'] == 0
                    and not v.get('fresh_faults', []) for v in prior['results'])
            and prior['native_source_commit'] == n['native_source_commit']
            and prior['binaries'] == n['binaries'], 'all11fresh same-build native stages required')
    proof = copy.deepcopy(n['observer_admission'])
    current = common.source.pin(P, common.PREFIXES)['source_sha256']
    require(THIS in proof['added_observers'] and THIS in current, 'new oracle observer must be explicit')
    del proof['added_observers'][THIS]; del current[THIS]
    proof['observer_source_files'] -= 1
    proof['observer_source_sha256'] = signature(current)
    require(proof == prior['native_observer_admission'],
            'only this uncompiled observer may extend the original native admission')
    return n, prior


def check(row, *, committed=True):
    if committed:
        require(common.validation.committed(RESULT, (KEY,)) == compact(row),
                'commit the exact oracle plan/result before execution')
    n, gate = basis(row['native_observer_admission'])
    require(signature(gate) == row['native_gate_sha256']
            and row['binary'] == n['binaries']['sycl_kv_copy_groups_bench']
            and sha(row['native_record']['path']) == row['native_record']['sha256'],
            'original native gate or same-build KV image changed')
    controller = row['controller']
    recovered = {name: hashlib.sha256(raw).hexdigest() for name, raw in
                 common.source.blobs(P, controller['revision'], controller['source_sha256'])}
    require(recovered == controller['source_sha256']
            and common.source.pin(P, ['tools/sycl'])['source_sha256'] == recovered,
            'frozen oracle/controller source changed')
    manifest = row['runtime_manifest']
    require(sha(manifest['path']) == manifest['sha256']
            and json.loads(Path(manifest['path']).read_text()) == row['runtime_files']
            and all(sha(p) == h for p, h in row['runtime_files'].items())
            and common.guards.driver_guard() == row['driver']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'],
            'original runtime/driver/boot changed')


def context():
    spec = importlib.util.spec_from_file_location('named_kv_oracle_environment', common.__file__)
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    env.ROOT = ROOT
    return SimpleNamespace(check=check, environment=env.environment, save=save)


def runner():
    spec = importlib.util.spec_from_file_location('named_kv_oracle_driver', driver.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT, module.KEY, module.context = ROOT, KEY, context
    return module


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'one fresh oracle plan only')
    n, gate = basis()
    manifest = gate['runtime_manifest']; require(sha(manifest['path']) == manifest['sha256'], 'runtime map changed')
    row = dict(stage='built', controller=common.source.pin(P, ['tools/sycl']),
        native_source_commit=n['native_source_commit'], native_observer_admission=n['observer_admission'],
        native_record=dict(path=str(native.build.ROOT / 'record.json'), sha256=sha(native.build.ROOT / 'record.json')),
        native_gate_sha256=signature(gate), binary=n['binaries']['sycl_kv_copy_groups_bench'],
        recipe=dict(environment=copy.deepcopy(n['recipe']['environment'])),
        native_validation_probe=n['native_validation_probe'], runtime_manifest=manifest,
        runtime_files=json.loads(Path(manifest['path']).read_text()),
        driver_library=gate['environment']['ZE_ENABLE_ALT_DRIVERS'], boot=gate['boot'], driver=gate['driver'],
        cases=[], commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False, performance_qualified=False,
        plan=dict(order=['smoke', 'full'],
            source='Same-build14th KVfixture from the naming-only coherent generation. '
                   'All scalar/cooperative body,geometry,policy and oracle definitions unchanged;no TU split.',
            smoke='Separate immutable child with full URtrace/parameter validation:14direct semantics,'
                  'refusals/null-stream and20graph replays. Preserve qualified copy-offload/direct-submission/'
                  'queue settings and pre-main provider probe.',
            full='Separate quiet child:795exact oracle records,60graph replays,15physical-tail graph replays '
                 'and12prospective timing-shape checks with two exact replays each. Guards,map/control/'
                 'payload/source/input invariants and final cell262143 remain required. No timing gain claim.',
            stopping='Commit exact source/recipe/plan before smoke;commit its closure before full. '
                     'Stop first failed identity,budget,oracle,ownership,validation or fresh-fault gate;'
                     'no retries,changed pins or reclassified results.',
            limits='Component correctness only. Do not claim model262144fill,RESTORE,quality,ordinary '
                   'teardown or speed. Actual production phase/state handoff and original comparison '
                   'obligations remain separate;static ELF equivalence alone changes none of them.'),
        retention='Use the existing14-target generation;no new binaries. Keep compact counts,'
                  'exactness,provider,fault/closure and source identities;retire successful trace/cache/'
                  'raw records after this oracle decision. Three named phase-comparison generations '
                  'remain bounded by their existing retirement point.')
    check(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    save(row)
    print('Prepared fresh same-build KVoracle;commit before smoke/full.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'smoke', 'full'), 'prepare|smoke|full')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        common.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: runner().run(sys.argv[1])
