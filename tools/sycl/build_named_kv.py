#!/usr/bin/env python3
"""Build only the independently selected KV phase-naming change.

Main-owned use of this fork's full-build/census helpers; repository LICENSE.
The later scalar/cooperative TU split is deliberately excluded from this unit.
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
    'include/strata/kernels/kv_stream_copy_groups.hpp',
    'src/core/verify.cpp', 'src/kernels/sycl/kv_stream.cpp',
    'tests/core/verify_window_policy_test.cpp',
    'tools/sycl/kv_copy_groups_bench.cpp',
}
TARGETS = [*previous.TARGETS, 'sycl_kv_copy_groups_bench']
require, sha, signature = previous.require, previous.sha, previous.signature


def unit_sources():
    """Prove the six-file naming unit and exclude the subsequent TU split."""
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
    ctx.SEED, ctx.SEED_KEY, ctx.component_reference = SEED, previous.KEY, prerequisite
    ctx.common.PREFIXES = previous.fixture.PREFIXES
    original = ctx.context
    duplicates = dict(previous.coherent.DUPLICATES)
    duplicates[str(P / 'src/kernels/sycl/kv_stream.cpp')] = [
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
            and KEY not in json.loads(RESULT.read_text()), 'fresh primary-worktree naming build only')
    proof = prerequisite()
    seed = json.loads(SEED.read_text())
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
        preserved_binaries=[*seed['preserved_binaries'], *seed['binaries'].values()],
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
    ctx = context()
    ctx.current(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    ctx.save(row)
    print('Prepared naming-only full build; commit before configure. No GPU executed.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'configure', 'compile'),
            'use prepare|configure|compile')
    require(P == PRIMARY, 'select the naming-only source in the existing primary worktree')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous.base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: context().build(sys.argv[1])
