#!/usr/bin/env python3
"""Fresh device qualification for the independently built KV naming unit.

Main-owned application of this fork's coherent numerical/ownership gates;
repository LICENSE. No old GPU outcome, speed, or model route is inherited.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_named_kv as build
import check_decode_kv as previous

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B / 'phase-kv-naming-device-v1', 'phase_kv_naming_device_v1'
THIS = 'tools/sycl/check_named_kv.py'
STAGES = previous.STAGES
common = build.previous.fixture
require, sha, signature = build.require, build.sha, build.signature


def native(expected=None, *, observers=None):
    observers = {THIS} if observers is None else set(observers)
    require(THIS in observers and all(p.startswith('tools/sycl/') and p.endswith('.py')
            for p in observers), 'declare exact uncompiled qualification observers')
    row = json.loads((build.ROOT / 'record.json').read_text())
    require(common.validation.committed(RESULT, (build.KEY,)) == build.context().compact(row)
            and row['stage'] == 'built' and not row['active'] and not row['GPU_executed']
            and all(row[k] for k in ('passed', 'complete', 'ownership_closed')),
            'committed complete naming-only full build required')
    before = row['project']['source_sha256']
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                 common.source.blobs(P, row['project']['revision'], before)}
    after = common.source.pin(P, common.PREFIXES)['source_sha256']
    require(recovered == before and set(after) - set(before) == observers
            and not set(before) - set(after) and all(after[n] == h for n, h in before.items()),
            'native build changed beyond named committed Python observers')
    actual = row['actual_inputs']
    require(actual['source_commit'] == row['native_source_commit'] == row['project']['revision']
            and actual['translation_units'] == 136 and actual['uncovered'] == 0
            and len(row['planned_compilation']) == 140
            and set(row['binaries']) == set(build.TARGETS), 'exact14-target/140-object build required')
    coherence = actual['class_definition_coherence']
    require(coherence['fresh_complete_build'] and not coherence['partial_archive_reuse']
            and set(coherence['headers']) == set(build.previous.HEADERS)
            and all(v['header_sha256'] == sha(P / n) and v['all_objects_from_fresh_build']
                    for n, v in coherence['headers'].items()), 'actual class-definition closure changed')
    require(build.prerequisite() == row['build_prerequisite'],
            'original six-file naming unit,retired legacy holds,or phase references changed')
    for pin in row['dependencies']:
        recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                     common.source.blobs(Path(pin['root']), pin['revision'], pin['source_sha256'])}
        require(recovered == pin['source_sha256'] and all(sha(Path(pin['root']) / n) == h
                for n, h in recovered.items()), 'original dependency inputs changed')
    require(sha(build.SEED) == row['recipe_seed']['sha256']
            and sha(row['external_inputs']['path']) == row['external_inputs']['sha256']
            and all(sha(p) == h for p, h in row['toolchain'].items())
            and all(sha(p) == h for p, h in json.loads(Path(row['external_inputs']['path']).read_text()).items()),
            'original compiler/recipe/SDK input changed')
    seed = json.loads(build.SEED.read_text())
    recipe = copy.deepcopy(seed['recipe']); old = str(build.SEED.parent)
    recipe['configure'] = [str(build.ROOT) + v[len(old):] if v.startswith(old) else v
                           for v in recipe['configure']]
    recipe['configure'][recipe['configure'].index('-S') + 1] = str(P)
    recipe['environment'] = {k: v.replace(old, str(build.ROOT))
                             for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = build.TARGETS
    require(recipe == row['recipe'], 'the naming build recipe changed')
    require(build.previous.base.probe_qualification() ==
            (row['native_validation_probe'], row['validation_qualification']), 'runtime probe changed')
    require(all(sha(p) == h for p, h in (row['configured_files'] | row['generated_inputs']).items())
            and sha(actual['path']) == actual['sha256'], 'configured/generated/compiler input changed')
    census = json.loads(Path(actual['path']).read_text())
    require(all(sha(p) == h for family in ('compiled', 'generated', 'links')
                for p, h in census[family].items()), 'actual compiler/link dependency changed')
    require(not any(str(P / n) in census[family] for n in observers
                    for family in ('compiled', 'generated', 'links')), 'an observer is compiled/linked input')
    for image in [*row['binaries'].values(), *row['preserved_binaries']]:
        require(Path(image['path']).stat().st_size == image['bytes']
                and sha(image['path']) == image['sha256'], 'fresh fixture or retained phase image changed')
    proof = dict(native_source_commit=row['native_source_commit'], original_source_files=len(before),
        observer_source_files=len(after), observer_source_sha256=signature(after),
        added_observers={n: after[n] for n in sorted(observers)},
        actual_census_sha256=actual['sha256'], class_coherence=coherence,
        exact_fourteen_ELFs_census_recipe=True, exact_seed_recipe_sha256=signature(recipe),
        native_recompiled=False, GPU_result_inherited=False,
        naming_unit_origin=build.UNIT, TU_split_included=False)
    require(expected is None or proof == expected, 'frozen native qualification admission changed')
    return dict(row, observer_admission=proof)


def controller():
    spec = importlib.util.spec_from_file_location('named_kv_device_controls', previous.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.KEY, module.THIS = ROOT, KEY, THIS
    module.build, module.native = build, native
    return module


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'one fresh native qualification')
    module = controller()
    module.prepare()
    # Replace the old campaign's bookkeeping before its new plan is committed.
    # Its tested numerical/trace commands,immutable safety settings and actual
    # ownership/fault checks are reused;none of its execution results are copied.
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not row['results'] and not row['commands']
            and row['native_source_commit'] == json.loads((build.ROOT / 'record.json').read_text())['native_source_commit'],
            'fresh source-bound native stage plan required')
    row['plan'].update(
        next='After all11fresh stages pass,commit separate traced KVsmoke/full795-oracle checks '
             'using this same-build14th fixture. Then qualify actual production phase/handoff '
             'and compare the naming unit alone before the independent TU split. '
             'Do not inherit a failed component/model performance gate.',
        source_unit='Only ae74403d1 mechanical decode names and required six-file closure;'
                    'no kernel-body,geometry,placement,setting or TU-split change.')
    row['limits'] = ('Fresh same-build CPU/GPU numerical/safety gates for14coherent ELFs. '
        'No model throughput/task/agent,physical262144/RESTORE,ordinary teardown or default adoption. '
        'Small fixtures may use scalar fallback;actual production cooperative dispatch needs model evidence.')
    row['retention'] = ('Three named engine generations only for this immediate naming comparison: '
        '14508fast prefill reference,1eecunchanged comparison and the new naming candidate. '
        'All three legacy holds are retired;no old standalone component is retained or invoked. '
        'Retire successful raw native traces/cache after compact evidence/source recovery. '
        'Select/retire a superseded generation before building the independently evaluated TU split.')
    head = module.controller().context().context().context(row['native_observer_admission'])
    head.save(row)
    print('Prepared fresh11-stage naming qualification;commit before CPU/GPU tests.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', *STAGES),
            'prepare|' + '|'.join(STAGES))
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        common.guards.no_model_owners([])
        action = sys.argv[1]
        if action == 'prepare': prepare()
        elif action.startswith('gu-'): controller().controller().gu_stage(action)
        elif action in previous.previous.original.STAGES:
            controller().controller().context().context().execute(action)
        else: controller().controller().context().extra(action)
