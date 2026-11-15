#!/usr/bin/env python3
"""Fresh qualification of B1 alone, authored from this fork's owned A controls.

Repository LICENSE. Integrate this uncompiled observer only AFTER the B1 build;
its independent source proposal does not qualify a build or any GPU result.
B1s scalar isolation/restoration and all settings changes remain excluded.
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

build.select_unit('split')
P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B / 'phase-kv-split-device-v1', 'phase_kv_split_device_v1'
THIS = 'tools/sycl/check_split_kv.py'
STAGES = previous.STAGES
common = build.previous.fixture
require, sha, signature = build.require, build.sha, build.signature
HOST_CONTROLS = dict(command=[
    '/home/yayoi/.local/share/strata-sycl/venv/bin/python3', '-B', '-m', 'unittest',
    'tools.test_phase_kv_split_qualification'], wall_seconds=90, CPU_seconds=60,
    RSS_bytes=1 << 30, text_bytes=128 << 10, total_bytes=4 << 20,
    GPU_executed=False, prerequisites='Commit this source,controls and recipe first;'
        'Main runs them only at a serialized no-GPU-job boundary.')


def check_census(row):
    """Refuse a reused A receipt or partial B1 link before any device work."""
    actual = row['actual_inputs']
    require(actual['source_commit'] == row['native_source_commit'] == row['project']['revision']
            and actual['translation_units'] == 137 and actual['uncovered'] == 0
            and len(row['planned_compilation']) == 142
            and set(row['binaries']) == set(build.TARGETS),
            'exact B1 fourteen-target/142-object/137-TU build required')


def native(expected=None, *, observers=None):
    observers = {THIS} if observers is None else set(observers)
    require(THIS in observers and all(p.startswith('tools/sycl/') and p.endswith('.py')
            for p in observers), 'declare exact uncompiled qualification observers')
    row = json.loads((build.ROOT / 'record.json').read_text())
    require(common.validation.committed(RESULT, (build.KEY,)) == build.context().compact(row)
            and row['stage'] == 'built' and not row['active'] and not row['GPU_executed']
            and all(row[k] for k in ('passed', 'complete', 'ownership_closed')),
            'committed complete B1-only full build required')
    before = row['project']['source_sha256']
    recovered = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                 common.source.blobs(P, row['project']['revision'], before)}
    after = common.source.pin(P, common.PREFIXES)['source_sha256']
    require(recovered == before and set(after) - set(before) == observers
            and not set(before) - set(after) and all(after[n] == h for n, h in before.items()),
            'native build changed beyond named committed Python observers')
    actual = row['actual_inputs']
    check_census(row)
    coherence = actual['class_definition_coherence']
    require(coherence['fresh_complete_build'] and not coherence['partial_archive_reuse']
            and set(coherence['headers']) == set(build.previous.HEADERS)
            and all(v['header_sha256'] == sha(P / n) and v['all_objects_from_fresh_build']
                    for n, v in coherence['headers'].items()), 'actual class-definition closure changed')
    require(build.prerequisite() == row['build_prerequisite'],
            'original three-file B1 unit,closed A phase controls,retirement or phase references changed')
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
    require(recipe == row['recipe'], 'the B1 build recipe changed')
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
        source_unit_origin=build.SPLIT_UNIT, TU_split_included=True,
        scalar_isolation_or_restoration_included=False)
    require(expected is None or proof == expected, 'frozen native qualification admission changed')
    return dict(row, observer_admission=proof)


def controller():
    spec = importlib.util.spec_from_file_location('split_kv_device_controls', previous.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT, module.KEY, module.THIS = ROOT, KEY, THIS
    module.build, module.native = build, native
    return module


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()),
            'one fresh B1 native qualification')
    module = controller(); module.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not row['results'] and not row['commands']
            and row['native_source_commit'] ==
                json.loads((build.ROOT / 'record.json').read_text())['native_source_commit'],
            'fresh B1 source-bound stage plan required')
    row['plan'].update(source_unit='B1 only:97ed15507 cooperative TU extraction and required '
        'link/CMake wiring;scalar body SHA34418f3b0 remains unchanged. No B1s or B2.',
        next='After11fresh stages pass,commit same-build KVsmoke/full oracle and actual production '
             'handoff plans,then independently compare both phases. No static-identity or A '
             'timing-control criterion applies to this different B1 image.')
    row['limits'] = ('Fresh component gates only;no old numerical/model/performance outcome inherited. '
        'Require actual per-phase production routing,matched PP/TTFT and long-thinking decode, '
        'agent/task and physical262144/RESTORE before corresponding adoption claims.')
    row['retention'] = ('Fast14508 plus validated A and this temporary B1 generation only,after '
        'exact old1eec retirement. On failure retain both references and retire B1 after compact '
        'source/evidence recovery. Select and retire a superseded generation before another build.')
    module.controller().context().context().context(row['native_observer_admission']).save(row)
    print('Prepared fresh B1 eleven-stage qualification;commit before tests.', flush=True)


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
