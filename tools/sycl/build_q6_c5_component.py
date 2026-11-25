#!/usr/bin/env python3
"""Build the five-column canonical Q6 fixture with the retained SYCL recipe."""

import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_compat_profile as base

P, B = base.P, base.B
ROOT = B / 'q6-canonical-c5-component-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'q6_canonical_c5_component_build_v1'
TARGET = 'sycl_q6_canonical_component'
REFERENCE = 'd6f2b82947018f85e8e6169bd2072fb076c03f19'
NATIVE = B / 'dflash2-target-snapshot-native-v2'
FOUNDATION = B / 'coherent-prefill-native-v1' / 'record.json'
VERIFY_POLICY = 'include/strata/core/verify_window_policy.hpp'
THIS = str(Path(__file__).resolve().relative_to(P))
UNITS = (
    'src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
    'src/kernels/sycl/native_mmvq.cpp', 'src/kernels/sycl/native_q5_head.cpp',
    'src/kernels/sycl/native_q3_esimd.cpp', 'src/kernels/sycl/native_q4_esimd.cpp',
    'src/kernels/sycl/native_q2_esimd.cpp', 'src/kernels/sycl/native_iq4xs_esimd.cpp',
    'src/kernels/sycl/native_iq_esimd.cpp', 'src/kernels/sycl/native_mmq_xmx.cpp',
    'src/kernels/sycl/iq.cpp', 'src/kernels/sycl/q6_dpas_component.cpp', 'bench/sycl/q6_canonical.cpp',
)
require, sha = base.require, base.digest


def q6_dispatch_source(path):
    source = path.read_bytes()
    start = source.index(b'constexpr bool verify_q6_columns(')
    return source[start:source.index(b'\n}', start) + 2]


def reference():
    archived = json.loads(base.source.git(P, 'show', REFERENCE + ':' + str(RESULT.relative_to(P))))
    measured = archived['dflash2_snapshot_regular_code_v1']
    record = NATIVE / 'record.json'
    native = json.loads(record.read_text())
    committed = archived['dflash2_target_snapshot_native_v2']
    for name in ('native_source_commit', 'binaries', 'recipe', 'external_inputs', 'build_tools'):
        require(native[name] == committed[name], 'the recorded native build differs: ' + name)
    source_files = native['source_files']
    require(base.signature(native['foundation']) == committed['foundation_sha256']
            and dict(files=len(source_files), sha256=base.signature(source_files)) == committed['source_files'],
            'the native compiler or source manifest differs from Git')
    selected = measured['snapshot_arm']
    require(measured['stage'] == 'closed' and measured['passed'] and measured['ownership_closed']
            and native['stage'] == 'built' and native['passed'] and native['ownership_closed']
            and selected['native_source_commit'] == native['native_source_commit']
            and selected['native_build_sha256'] == base.signature(committed)
            and selected['binary'] == native['binaries']['strata'], 'retain the checked model and recipe')
    require(sha(FOUNDATION) == native['foundation']['original_record_sha256'], 'compiler foundation changed')
    compiler = json.loads(FOUNDATION.read_text())
    paths = set(UNITS[:-2])
    paths |= {p for p in source_files if p.startswith(('src/sycl/', 'include/strata/sycl/'))}
    sources = {name: source_files[name] for name in sorted(paths)}
    require(all(sha(P / name) == value for name, value in sources.items()), 'reference or runtime source changed')
    # The primary header has a separate KV policy and graph counters. This
    # fixture consumes its Q6 dispatch predicate and pins the entire header.
    native_policy = NATIVE / 'source' / VERIFY_POLICY
    component_policy = P / VERIFY_POLICY
    require(sha(native_policy) == source_files[VERIFY_POLICY]
            and q6_dispatch_source(component_policy) == q6_dispatch_source(native_policy), 'Q6 dispatch policy changed')
    policy = dict(native_sha256=source_files[VERIFY_POLICY], component_sha256=sha(component_policy))
    return native, compiler, dict(record=dict(path=str(record), sha256=sha(record)),
        evidence_commit=REFERENCE, source_commit=native['native_source_commit'],
        engine=selected['binary'], runtime_sources=sources, verify_policy=policy)


def context():
    spec = importlib.util.spec_from_file_location('canonical_c5_fixture_build', base.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.RESULT, ctx.KEY, ctx.TARGET, ctx.UNITS = ROOT, RESULT, KEY, TARGET, UNITS

    def current(row, committed=True):
        if committed:
            require(ctx.feature.validation.committed(RESULT, (KEY,)) == ctx.compact(row), 'commit exact fixture plan')
        _, _, identity = reference()
        require(identity == row['selected_native'], 'native build provenance changed')
        required_sources = {*UNITS, THIS, VERIFY_POLICY, 'tools/sycl/build_compat_profile.py'}
        require(required_sources <= row['project']['source_sha256'].keys(),
                'required fixture and builder sources are missing')
        ctx.build.check_inputs(row)
        require(row['recipe']['targets'] == row['recipe']['build_targets'] == [TARGET]
                and 'LD_PRELOAD' not in row['recipe']['environment'], 'fixture-only build recipe required')

    ctx.current = current
    return ctx


def prepare(ctx):
    require(not ROOT.exists() and KEY not in ctx.feature.validation.committed(RESULT), 'fresh component build required')
    native, compiler, identity = reference()
    project = ctx.source.pin(P, ('CMakeLists.txt', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md', 'cmake',
                                'src', 'include', 'third_party', 'bench/sycl', 'tools/sycl'))
    recipe = copy.deepcopy(native['recipe'])
    recipe['configure'] = [value.replace(str(NATIVE / 'source'), str(P)).replace(str(NATIVE), str(ROOT))
                           for value in recipe['configure']]
    recipe['environment'] = {key: value.replace(str(NATIVE), str(ROOT))
                             for key, value in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = [TARGET]
    require(recipe['configure'][recipe['configure'].index('-S') + 1] == str(P)
            and recipe['configure'][recipe['configure'].index('-B') + 1] == str(ROOT / 'build'), 'isolated build paths')
    row = dict(stage='prepared', project=project, dependencies=compiler['dependencies'],
        toolchain=compiler['toolchain'], external_inputs=native['external_inputs'], recipe=recipe,
        fallbacks=dict(strata=identity['engine']), selected_native=identity,
        budgets=ctx.BUDGET, commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        plan=dict(compile_target=TARGET, initial_native_checks=['--host', '--host-c5', '--check-c5-small'],
            scope='One fixture-only build. Match the retained DPC++ recipe. Validate device loading and five-column numerical results in a separate bounded run.',
            next='After the small guard/oracle checks, measure head and dense shapes independently in three fresh processes.'),
        retention='One standalone fixture for the immediate C5 comparison. Retire the build after the component decision and preserve its exact source and recipe.')
    ctx.current(row, committed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    ctx.save(row)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['build']), 'use prepare or build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.feature.guards.no_model_owners([])
        ctx = context()
        if sys.argv[1] == 'prepare':
            prepare(ctx)
        else:
            ctx.compile_fixture()
