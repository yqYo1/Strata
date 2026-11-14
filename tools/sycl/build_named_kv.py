#!/usr/bin/env python3
"""Build an explicitly selected, independently compared KV isolation unit.

Main-owned use of this fork's full-build/census helpers; repository LICENSE.
Naming and cooperative TU extraction are separate selections and generations.
Compilation is not GPU, phase-preservation, or model-route qualification.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_decode_kv as previous
import coherent_native_reuse as recovery

P, B, RESULT = previous.P, previous.B, previous.RESULT
PRIMARY = Path('/home/yayoi/orca/workspaces/Strata/sycl')
ROOT, KEY = B / 'phase-kv-naming-native-v1', 'phase_kv_naming_native_v1'
SEED = previous.ROOT / 'record.json'
UNIT = 'ae74403d163208bec938e0af92c5d10eb3d92263'
POLICY = '6b6098593628b4b707dee7290a85cf050a53812a'
SOURCE_PREFIXES = ['CMakeLists.txt', 'cmake', 'src', 'include', 'third_party',
                   'tests', 'bench/sycl']
RENAMES = {
    b'verify_coalesced_kv': b'decode_verify_coalesced_kv',
    b'kv_stream_resolve_with_coalesced_plan': b'decode_kv_stream_resolve_coalesced',
}
UNIT_FILES = {
    'include/strata/core/verify_window_policy.hpp',
    'include/strata/kernels/kv_stream.hpp',
    'src/core/verify.cpp', 'src/kernels/sycl/kv_stream.cpp',
    'tests/core/verify_window_policy_test.cpp',
    'tools/sycl/kv_copy_groups_bench.cpp',
}
TARGETS = [*previous.TARGETS, 'sycl_kv_copy_groups_bench']
require, sha, signature = previous.require, previous.sha, previous.signature
MODE, SEED_KEY = 'naming', previous.KEY
SPLIT_UNIT = '97ed1550789b90ded5c9cbade96857c24d2ef4fa'
SPLIT_FILES = {'cmake/sycl_backend.cmake', 'src/kernels/sycl/kv_stream.cpp',
               'src/kernels/sycl/decode_kv_stream.cpp'}


def select_unit(mode):
    global MODE, ROOT, KEY, SEED, SEED_KEY
    require(mode in ('naming', 'split'), 'select naming or split explicitly')
    MODE = mode
    if mode == 'split':
        ROOT, KEY = B / 'phase-kv-split-native-v1', 'phase_kv_split_native_v1'
        SEED, SEED_KEY = B / 'phase-kv-naming-native-v1/record.json', 'phase_kv_naming_native_v1'
    else:
        ROOT, KEY = B / 'phase-kv-naming-native-v1', 'phase_kv_naming_native_v1'
        SEED, SEED_KEY = previous.ROOT / 'record.json', previous.KEY


def split_unit_sources():
    """B1 contains the required link unit only; B2 restoration stays excluded."""
    source = previous.base.common.source
    require(set(source.git(P, 'diff', '--name-only', UNIT, SPLIT_UNIT).decode().splitlines())
            == SPLIT_FILES, 'review the exact three-file B1 unit')
    paths = source.git(P, 'ls-tree', '-r', '--name-only', '-z', SPLIT_UNIT,
                       '--', *SOURCE_PREFIXES, 'tools/sycl').decode().strip('\0').split('\0')
    paths = {n for n in paths if not n.startswith('tools/sycl/')
             or Path(n).suffix in ('.cpp', '.c', '.h', '.hpp', '.inc')}
    live = source.git(P, 'ls-files', '-z', '--', *SOURCE_PREFIXES,
                      'tools/sycl').decode().strip('\0').split('\0')
    live = {n for n in live if not n.startswith('tools/sycl/')
            or Path(n).suffix in ('.cpp', '.c', '.h', '.hpp', '.inc')}
    require(paths == live, 'B1 native input was added or removed independently')
    expected = {n: hashlib.sha256(raw).hexdigest()
                for n, raw in source.blobs(P, SPLIT_UNIT, sorted(paths))}
    require(all(sha(P / n) == h for n, h in expected.items()),
            'select B1 alone; scalar restoration and other native edits are excluded')
    require(expected['src/kernels/sycl/kv_stream.cpp']
            == '34418f3b0b5ae8c2b4e0a217d54413197f60246c9e76299a22298619300bde57',
            'keep the original scalar body except its moved decode wrapper')
    return dict(origin_commit=SPLIT_UNIT, naming_base=UNIT, files=len(expected),
        native_source_sha256=signature(expected),
        changed_files={n: expected[n] for n in sorted(SPLIT_FILES)},
        indivisible_link_unit='Remove the old public definition, compile its independent '
            'replacement and link both production and KV fixture targets.',
        scalar_restoration_or_settings_changed=False)


def split_prerequisite():
    """Recover A exactly, without reviving its retired historical fallback ELFs."""
    source = previous.base.common.source
    data = json.loads(source.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    for phase in ('prefill', 'decode'):
        gate = data['phase_kv_naming_' + phase + '_v1']
        require(gate['stage'] == 'closed' and not gate['active']
                and all(gate[k] for k in ('passed', 'complete', 'ownership_closed'))
                and gate['decision']['naming_control_eligible'],
                'both original A phase controls must qualify before building B1')
    retirement = data['phase_kv_naming_reference_retirement_v1']
    require(retirement['complete'] and retirement['ownership_closed']
            and retirement['retired_native_source_commit']
                == '1eec28c77f393d34814dc748502113514973ea8e'
            and not (B / 'decode-kv-native-v1/build').exists(),
            'retire the ended old generation before admitting a third build')
    require(sha(SEED) == '2dc7d4f3ead6068be12afa6ef36c589324f75d31ddd95f6fa96421552da57c9c',
            'original A build receipt changed')
    seed = json.loads(SEED.read_text())
    original = json.loads(source.git(P, 'show', '9ff38147e5d9bbbaec96dc8dcae86ffd969f457c:'
                                  + str(RESULT.relative_to(P))))[SEED_KEY]
    require(data[SEED_KEY] == original == context().compact(seed)
            and seed['stage'] == 'built' and not seed['active']
            and all(seed[k] for k in ('passed', 'complete', 'ownership_closed'))
            and seed['native_source_commit'] == '9849e35803f011aaafb2753c1e0f31411e40308e',
            'exact original A coherent build required')
    recovered = recovery.recover(seed['project'])
    for pin in seed['dependencies']: recovered.update(recovery.recover(pin))
    require(sha(seed['recipe_seed']['path']) == seed['recipe_seed']['sha256']
            and sha(seed['external_inputs']['path']) == seed['external_inputs']['sha256']
            and all(sha(p) == h for p, h in seed['toolchain'].items()), 'A recipe/tool inputs changed')
    external = json.loads(Path(seed['external_inputs']['path']).read_text())
    require(all(sha(p) == h for p, h in external.items()), 'A SDK/link inputs changed')
    for p, h in (seed['configured_files'] | seed['generated_inputs']).items():
        require(recovered.get(p) == h if p in recovered else sha(p) == h,
                'A configured/generated input changed')
    census = seed['actual_inputs']
    require(sha(census['path']) == census['sha256'] and census['uncovered'] == 0
            and census['translation_units'] == 136
            and len(seed['planned_compilation']) == 140, 'A actual compiler census changed')
    actual = json.loads(Path(census['path']).read_text())
    allowed = recovered | external | seed['generated_inputs']
    require(len(actual['compiled']) == 1473 and len(actual['generated']) == 1
            and len(actual['links']) == 29 and all(allowed.get(p) == h
                and (p in recovered or sha(p) == h) for family in ('compiled', 'generated', 'links')
                for p, h in actual[family].items()), 'recover A original compiler/link input bytes')
    coherence = census['class_definition_coherence']
    require(coherence['fresh_complete_build'] and not coherence['partial_archive_reuse']
            and len(coherence['headers']) == 5 and all(v['all_objects_from_fresh_build']
                and v['header_sha256'] == recovered[str(P / n)]
                for n, v in coherence['headers'].items()), 'A class-definition closure changed')
    retained = [v for v in seed['preserved_binaries']
                if Path(v['path']).is_relative_to(B / 'coherent-prefill-native-v1')]
    require(len(retained) == 12 and len(seed['binaries']) == 14,
            'retain exactly the fast prefill and A generations')
    for item in [*retained, *seed['binaries'].values()]:
        require(Path(item['path']).stat().st_size == item['bytes']
                and sha(item['path']) == item['sha256'], 'retained phase fallback changed')
    return dict(unit=split_unit_sources(),
        reference_build=dict(path=str(SEED), sha256=sha(SEED),
            source_commit=seed['native_source_commit'], actual_inputs_sha256=census['sha256']),
        naming_controls={phase: signature(data['phase_kv_naming_' + phase + '_v1'])
                         for phase in ('prefill', 'decode')},
        retirement_sha256=signature(retirement),
        inherited_GPU_model_or_performance_qualification=False)


def unit_sources():
    """Prove the six-file naming unit and exclude the subsequent TU split."""
    if MODE == 'split': return split_unit_sources()
    source = previous.base.common.source
    names = source.git(P, 'diff', '--name-only', POLICY, UNIT).decode().splitlines()
    require(set(names) == UNIT_FILES, 'review the original six-file naming unit')
    old = dict(source.blobs(P, POLICY, names))
    new = dict(source.blobs(P, UNIT, names))
    for name in names:
        expected = old[name]
        for before, after in RENAMES.items():
            expected = expected.replace(before, after)
        require(new[name] == expected, 'naming unit includes another mechanism: ' + name)
    def selected(revision=None):
        args = ['ls-files', '-z'] if revision is None else [
            'ls-tree', '-r', '--name-only', '-z', revision]
        paths = source.git(P, *args, '--', *SOURCE_PREFIXES).decode().strip('\0').split('\0')
        tools = source.git(P, *args, '--', 'tools/sycl').decode().strip('\0').split('\0')
        return set(paths) | {n for n in tools if Path(n).suffix in ('.cpp', '.c', '.h', '.hpp', '.inc')}
    paths = sorted(selected(UNIT))
    require(set(paths) == selected(), 'native input was added or removed outside the naming unit')
    expected = {n: hashlib.sha256(raw).hexdigest()
                for n, raw in source.blobs(P, UNIT, paths)}
    require(all(sha(P / n) == h for n, h in expected.items()),
            'select naming alone before compiling; the TU split or other native edits are present')
    require(not (P / 'src/kernels/sycl/decode_kv_stream.cpp').exists(),
            'the independently compared TU split must remain excluded')
    return dict(origin_commit=UNIT, policy_base=POLICY, files=len(expected),
                native_source_sha256=signature(expected),
                changed_files={n: hashlib.sha256(new[n]).hexdigest() for n in sorted(names)},
                kernel_bodies_geometry_placement_and_settings_changed=False)


def prerequisite():
    if MODE == 'split': return split_prerequisite()
    source = previous.base.common.source
    data = json.loads(source.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    retirement = data['kv_planner_speed_v1']['production_preparation']['build_retirement']
    require(retirement['state'] == 'retired' and retirement['complete']
            and retirement['legacy_remaining'] == {},
            'retire the three reconciled legacy build generations before another engine build')
    seed = json.loads(SEED.read_text())
    require(data[previous.KEY] == previous.context().compact(seed)
            and seed['stage'] == 'built' and not seed['active']
            and all(seed[k] for k in ('passed', 'complete', 'ownership_closed')),
            'preserve the original closed coherent full-engine receipt')
    for pin in (seed['project'], *seed['dependencies']):
        found = {n: hashlib.sha256(raw).hexdigest() for n, raw in
                 source.blobs(Path(pin['root']), pin['revision'], pin['source_sha256'])}
        require(found == pin['source_sha256'], 'original baseline source is not recoverable')
    require(sha(seed['actual_inputs']['path']) == seed['actual_inputs']['sha256'],
            'original baseline compiler/link census changed')
    for item in [*seed['binaries'].values(), *seed['preserved_binaries']]:
        require(Path(item['path']).stat().st_size == item['bytes']
                and sha(item['path']) == item['sha256'], 'preserve both phase reference generations')
    return dict(unit=unit_sources(),
                reference_build=dict(path=str(SEED), sha256=sha(SEED),
                                     source_commit=seed['native_source_commit']),
                inherited_GPU_model_or_performance_qualification=False)


def context():
    spec = importlib.util.spec_from_file_location('named_kv_full_build', previous.base.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.TARGETS, ctx.MODEL_BUILD = ROOT, KEY, TARGETS, True
    ctx.SEED, ctx.SEED_KEY, ctx.component_reference = SEED, SEED_KEY, prerequisite
    ctx.common.PREFIXES = previous.fixture.PREFIXES
    original = ctx.context
    duplicates = dict(previous.coherent.DUPLICATES)
    duplicates[str(P / 'src/kernels/sycl/kv_stream.cpp')] = [
        'strata_kernels', 'sycl_kv_copy_groups_bench']
    if MODE == 'split':
        duplicates[str(P / 'src/kernels/sycl/decode_kv_stream.cpp')] = [
            'strata_kernels', 'sycl_kv_copy_groups_bench']
    def native_context():
        native = original()
        parser, census = native.planned_compilation, native.census
        native.planned_compilation = lambda text: parser(text, allowed_duplicate_targets=duplicates)
        def checked_census(row, dep, link, **kwargs):
            result = census(row, dep, link, **kwargs)
            # Use this generation's actual complete dependency closure.
            old_root = previous.ROOT
            try:
                previous.ROOT = ROOT
                result['class_definition_coherence'] = previous.definition_census(row, dep)
            finally:
                previous.ROOT = old_root
            return result
        native.census = checked_census
        return native
    ctx.context = native_context
    return ctx


def prepare():
    require(P == PRIMARY and not ROOT.exists()
            and KEY not in json.loads(RESULT.read_text()), 'fresh primary-worktree selected-unit build only')
    proof = prerequisite()
    seed = json.loads(SEED.read_text())
    preserved = seed['preserved_binaries']
    if MODE == 'split':
        preserved = [v for v in preserved
                     if Path(v['path']).is_relative_to(B / 'coherent-prefill-native-v1')]
    project = previous.base.common.source.pin(P, previous.fixture.PREFIXES)
    recipe = copy.deepcopy(seed['recipe'])
    old = str(SEED.parent)
    recipe['configure'] = [str(ROOT) + v[len(old):] if v.startswith(old) else v
                           for v in recipe['configure']]
    recipe['configure'][recipe['configure'].index('-S') + 1] = str(P)
    recipe['environment'] = {k: v.replace(old, str(ROOT))
                             for k, v in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = TARGETS
    require(recipe['configure'][recipe['configure'].index('-B') + 1] == str(ROOT / 'build'),
            'fresh full build required; no partial relink')
    probe, validation = previous.base.probe_qualification()
    row = dict(stage='prepared', project=project, dependencies=copy.deepcopy(seed['dependencies']),
        recipe=recipe, recipe_seed=dict(path=str(SEED), sha256=sha(SEED)),
        toolchain=seed['toolchain'], external_inputs=seed['external_inputs'],
        native_source_commit=project['revision'],
        preserved_binaries=[*preserved, *seed['binaries'].values()],
        native_validation_probe=probe, validation_qualification=validation,
        build_prerequisite=proof,
        plan='Unit A only: two mechanical decode names with their six required callers/headers/tests. '
            'No scheduling, placement, environment setting, kernel-body or TU split change. '
            'Commit exact source/recipe/plan; run existing compiler-census host controls; '
            'configure and commit its actual commands/generated inputs; compile every selected TU '
            'fresh, including 14 engine/fixture targets; require complete dependency/link and class '
            'coherence census. Stop on first failure, no retries or refreshed pins. Build success '
            'does not admit Unit B. After build commit, a separate committed plan must exercise '
            'actual phase routing, small traced device checks, complete KV oracle/graph tests, '
            'prefill/decode handoff, repeated matched >=32K prefill/TTFT and >=1K thinking-on '
            'MTP-friendly/weak decode against the unchanged 1eec engine. Preserve the faster14508 '
            'prefill reference and all physical262144/task/RESTORE obligations. No speedup claim '
            'from this naming change or inherited model/phase qualification.',
        retention='Explicit bounded temporary third generation: retain14508 for faster prefill '
            'and1eec for the sole-change naming comparison. All three legacy holds must already '
            'be retired. Do not create the independent TU-split generation before this comparison '
            'selects/retire its superseded generation after exact source/recipe recovery. Preserve '
            'the old two builds if the new build or native qualification fails; retire the failed '
            'generation. Review at naming native and matched-model closure; no unrelated archive.',
        budgets=dict(build=dict(wall=2400, cpu=7200, rss_cap=12 << 30,
            text_cap=32 << 20, file_cap=512 << 20, total_cap=4 << 30)),
        commands=[], outcomes={}, passed=False, complete=False, active=False,
        ownership_closed=True, GPU_executed=False, model_qualified=False,
        physical262144_model_qualified=False, performance_qualified=False)
    if MODE == 'split':
        row['plan'] = ('Unit B1 only: independent cooperative decode TU and the required '
            'production/fixture link wiring. Keep scalar/helper bytes except the moved public '
            'decode wrapper; do not restore14508, change placement, settings or scheduling. '
            'Require both closed A phase controls and old-generation retirement. Commit all '
            'source/recipe/generated inputs before their stage; compile all14 targets fresh '
            'and verify actual dependency/link and five-header coherence, including both '
            'new-TU objects. Stop on the first failure, no refreshed pins. Compilation admits '
            'neither B2 nor model execution: commit fresh small traced/numerical/KV/state '
            'qualification and matched >=32K prefill/TTFT plus >=1K thinking-on strong/weak '
            'MTP decode plans separately. Compare B1 against A before considering B2 or '
            'their combination. Preserve physical262144/RESTORE and task obligations.')
        row['retention'] = ('Temporary third generation B1 only after old1eec retirement; '
            'retain fast14508/12ELFs and A/14ELFs unchanged. On B1 qualification failure '
            'retire B1 after preserving evidence; on completed comparison select and retire '
            'the superseded generation before B2. No fourth generation or old-ELF revival.')
    ctx = context()
    ctx.current(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    ctx.save(row)
    print('Prepared ' + MODE + ' full build; commit before configure. No GPU executed.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        select_unit(sys.argv.pop(1))
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'configure', 'compile'),
            'use [naming|split] prepare|configure|compile')
    require(P == PRIMARY, 'select the unit source in the existing primary worktree')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous.base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: context().build(sys.argv[1])
