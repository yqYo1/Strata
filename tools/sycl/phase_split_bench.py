#!/usr/bin/env python3
"""Compare B1 alone with A using the owned bench.sh phase drivers.

Main-authored; repository LICENSE. This adapter reuses launch, routing and
sample readers, but never the naming-only image-identity exception.
"""
import copy
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import phase_naming_bench as original
import check_split_kv_model as model

P, B = model.P, model.B
THIS, TEST = 'tools/sycl/phase_split_bench.py', 'tools/test_phase_split_bench.py'
require, sha, signature = original.require, original.sha, original.signature


def basis(module):
    device = model.oracle.native
    observers = {device.THIS, model.oracle.THIS, model.THIS, THIS}
    n = device.native(observers=observers)
    live = device.common.source.pin(P, device.common.PREFIXES)
    gates = {}
    for key, source in [
        ('phase_kv_naming_device_v1', module.BUILDS['reference'][3]),
        ('phase_kv_naming_oracle_v1', module.BUILDS['reference'][3]),
        ('phase_kv_naming_model_v1', module.BUILDS['reference'][3]),
        (device.KEY, n['native_source_commit']),
        (model.oracle.KEY, n['native_source_commit']),
        (model.KEY, n['native_source_commit'])]:
        gate = device.common.validation.committed(device.RESULT, (key,))
        require(gate['stage'] == 'closed' and not gate['active']
                and all(gate[x] for x in ('passed', 'complete', 'ownership_closed'))
                and gate['native_source_commit'] == source,
                'closed same-build native/oracle/model gates required')
        gates[key] = signature(gate)
    gate = device.common.validation.committed(device.RESULT, (model.KEY,))
    proof = copy.deepcopy(n['observer_admission'])
    before = dict(live['source_sha256'])
    require(THIS in proof['added_observers'] and THIS in before,
            'declare this exact uncompiled benchmark observer')
    del proof['added_observers'][THIS]; del before[THIS]
    proof['observer_source_files'] -= 1
    proof['observer_source_sha256'] = signature(before)
    admission = gate['native_observer_admission']
    require(proof == {k: admission[k] for k in proof}
            and admission['device_evidence_sha256'] == gates[device.KEY]
            and admission['oracle_evidence_sha256'] == gates[model.oracle.KEY],
            'only this benchmark may extend the qualified B1 source')
    for key in ('phase_kv_naming_model_v1', model.KEY):
        handoff = device.common.validation.committed(device.RESULT, (key,))
        require(handoff['results'][0]['routes']['decode_KV']['actual_decode4_handoffs'] == 2
                and handoff['results'][0]['routes']['decode_KV']['prefill_cooperative_calls'] == 0,
                'actual independent phase handoffs required')
        retirement = device.common.validation.committed(device.RESULT,
            (handoff['public_retention']['private_raw_retirement_key'],))
        require(retirement['complete'] and retirement['state'] == 'retired',
                'consume curated gates after successful raw retirement')
    seed = json.loads(device.build.SEED.read_text())
    require(device.build.split_prerequisite() == n['build_prerequisite'],
            'recover original A source,recipe,census,class definitions and retained images')
    native = {arm: dict(binary=row['binaries']['strata'],
        native_source_commit=row['native_source_commit'],
        native_validation_probe=row['native_validation_probe'])
        for arm, row in [('reference', seed), ('candidate', n)]}
    prior = json.loads((B / 'coherent-generations-prefill-v1/record.json').read_text())
    require(device.common.validation.committed(device.RESULT, ('coherent_generations_prefill_v1',))
            == module.kv.compact(prior) and prior['stage'] == 'closed'
            and all(prior[x] for x in ('passed', 'complete', 'ownership_closed'))
            and not prior['active'], 'exact closed common frontend/configuration required')
    require(native['reference']['native_validation_probe'] == native['candidate']['native_validation_probe']
            == prior['native_validation_probe'], 'same qualified provider in both arms')
    files = json.loads(Path(prior['runtime_manifest']['path']).read_text())
    require(sha(prior['runtime_manifest']['path']) == prior['runtime_manifest']['sha256']
            and all(sha(p) == h for p, h in files.items())
            and device.common.guards.driver_guard() == prior['driver']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == prior['boot'],
            'qualified providers,boot and driver changed')
    for arm, item in native.items():
        item['boot'] = prior['boot']
        require(item['native_source_commit'] == module.BUILDS[arm][3]
                and item['binary']['sha256'] == module.BUILDS[arm][5],
                'exact original native generations required')
    return native, dict(candidate=n['observer_admission'],
        reference=n['build_prerequisite']['reference_build'], gates=gates), prior, files, live


def decision(module, processes):
    require(len(processes) == len(module.cases()) and all(p['index'] == i
            and p['kernel'] == module.cases()[i][1]
            and p['native_source_commit'] == module.BUILDS[p['kernel']][3]
            and p['binary']['sha256'] == module.BUILDS[p['kernel']][5]
            for i, p in enumerate(processes)), 'complete phase/source-bound cohort required')
    if module.MODE == 'prefill':
        result = module.prefill.decision(processes)
        result['phase_preservation_eligible'] = result['median_preservation_screen_passed']
    else:
        require(all(p['passed'] and p['complete'] and p['ownership_closed']
                and p['native_exit_code'] == 0 and not p['fresh_faults']
                and len(p['samples']) == 2 and all(type(s.get(f)) in (int, float)
                    and math.isfinite(s[f]) and s[f] > 0 for s in p['samples']
                    for f in ('exact_native_decode_tps', 'prompt_ms', 'ttft_s'))
                for p in processes), 'complete positive finite thinking decode samples required')
        result = module.kv.decision(processes)
        for name, value in result['workloads'].items():
            value['TG_median_preserved'] = statistics.median(p['TG'] for p in value['paired_ratios']) >= 1
            accepted = [p['samples'][module.kv.WORKLOADS.index(name)]['acceptance_rate']
                        for p in processes if p['kernel'] == 'candidate']
            value['candidate_acceptance'] = accepted
            value['candidate_MTP_class_confirmed'] = all(v >= .60 for v in accepted) \
                if name == 'regular-code' else all(v <= .40 for v in accepted)
        result['phase_preservation_eligible'] = all(v['TG_median_preserved']
            and v['reference_MTP_class_confirmed'] and v['candidate_MTP_class_confirmed']
            for v in result['workloads'].values())
    result.update(unit='B1:cooperative decode TU extraction and required linking only',
        inference_adopted=False, other_phase_may_not_offset_failure=True,
        naming_identity_exception_applied=False)
    return result


def controller(mode):
    spec = importlib.util.spec_from_file_location('split_kv_phase_driver', original.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.configure(mode)
    module.ROOT = B / ('phase-kv-split-' + mode + '-v1')
    module.KEY = 'phase_kv_split_' + mode + '_v1'
    module.THIS, module.TEST = THIS, TEST
    module.BUILDS = dict(reference=original.BUILDS['candidate'], candidate=(
        'phase-kv-split-native-v1', 'phase_kv_split_native_v1',
        '9477e2160da27470c2a7cf876946bcc238088253',
        '0569e6924b0277ecc7cbd44033bd8751e0c6e666',
        '9807e41d5d73074e03c8e553f4d2f0a977a57cfddc3b0eba7e7745aa180266e1',
        '2f83ee16d7286f83c3278ec23e342121a71a0e20b0eccd49fb4a0bfd301a67b8',
        14, 142, 137, 1474))
    module.UNIT_TESTS = ['tools.test_phase_split_bench', *module.prefill.UNIT_TESTS]
    module.basis = lambda: basis(module)
    module.decision = lambda processes: decision(module, processes)
    return module


def prepare(module):
    module.prepare()
    row = json.loads((module.ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not row['processes'], 'fresh B1 comparison only')
    row['plan'].update(
        unit='B1:97ed15507 cooperative decode TU extraction,wrapper removal and required CMake linking. '
             'These three files are inseparable:one definition and both production/fixture links must remain valid. '
             'No B1s,scalar-body restoration,kernel geometry,placement or setting change.',
        decision='Record all samples and three paired ratios per series. Prefill independently requires '
                 'candidate first/warmed32K/64K prompt-time and clientTTFT medians no slower in all eight '
                 'series. Decode independently requires median paired TG no lower for both observed '
                 'MTP classes;both arms need strong>=.60 and weak<=.40 acceptance. Record all-pair '
                 'consistency and gain separately. No naming identity exception applies. A failed '
                 'phase keeps that separation defect unresolved;the other phase cannot offset it. '
                 'No posthoc exclusion,threshold change or extra samples within this plan.',
        retention='Keep compact individual samples,configuration,original source/build and route/closure '
                  'evidence. Retire successful raw/cache at verified closure. Retain fast14508,A and '
                  'B1only until this independent phase comparison selects and retires a superseded '
                  'generation;no fourth engine build.',
        limits='This smallest-unit comparison does not qualify B1s/B2,agent/task,physical262144model '
               'fill/tails/RESTORE,ordinary teardown,hardware saturation or default inference adoption.')
    module.save(row)
    print('Prepared B1-only ' + module.MODE + ';commit before host/native.', flush=True)


if __name__ == '__main__':
    require(sys.executable == original.PYTHON and len(sys.argv) == 3,
            'qualified Python;prefill|decode prepare|host|index')
    module = controller(sys.argv[1]); action = sys.argv[2]
    require(action in ('prepare', 'host', *map(str, range(len(module.cases())))), 'one declared stage')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        module.kv.build.guards.no_model_owners([])
        if action == 'prepare': prepare(module)
        else:
            with (module.ROOT / ('host-attempt.json' if action == 'host' else action + '/attempt.json')).open('x') as f:
                json.dump(dict(action=action, result_key=module.KEY), f)
            try:
                if action == 'host': module.host()
                elif module.MODE == 'prefill': module.kv.runner.comparison.run(int(action), module.context())
                else: module.context().run(int(action))
            except BaseException as error:
                row = json.loads((module.ROOT / 'record.json').read_text())
                if row['stage'] != 'failed':
                    row.update(stage='failed', passed=False, complete=False, error=repr(error), failed_action=action)
                    module.save(row)
                raise
