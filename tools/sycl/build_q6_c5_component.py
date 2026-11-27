#!/usr/bin/env python3
"""Build the five-column Q6 fixture with the retained SYCL recipe."""

import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_compat_profile as base

P, B = base.P, base.B
ROOT = B / 'q6-canonical-c5-component-v2'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'q6_canonical_c5_component_build_v2'
PACKED_ROOT = B / 'q6-packed-c5-component-v1'
PACKED_KEY = 'q6_packed_c5_component_build_v1'
TARGET = 'sycl_q6_canonical_component'
REFERENCE = 'd6f2b82947018f85e8e6169bd2072fb076c03f19'
NATIVE = B / 'dflash2-target-snapshot-native-v2'
FOUNDATION = B / 'coherent-prefill-native-v1' / 'record.json'
VERIFY_POLICY = 'include/strata/core/verify_window_policy.hpp'
VERIFY_POLICY_SOURCE = '02d64c4ca533e357b0c408047a749626790b476b'
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


def q6_dispatch_source(source):
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
    native_policy = base.source.git(P, 'show', VERIFY_POLICY_SOURCE + ':' + VERIFY_POLICY)
    component_policy = (P / VERIFY_POLICY).read_bytes()
    require(hashlib.sha256(native_policy).hexdigest() == source_files[VERIFY_POLICY]
            and q6_dispatch_source(component_policy) == q6_dispatch_source(native_policy), 'Q6 dispatch policy changed')
    policy = dict(native_sha256=source_files[VERIFY_POLICY], component_sha256=hashlib.sha256(component_policy).hexdigest())
    return native, compiler, dict(record=dict(path=str(record), sha256=sha(record)),
        evidence_commit=REFERENCE, source_commit=native['native_source_commit'],
        engine=selected['binary'], runtime_sources=sources, verify_policy=policy)


def context(packed=False):
    spec = importlib.util.spec_from_file_location('canonical_c5_fixture_build', base.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY = (PACKED_ROOT, PACKED_KEY) if packed else (ROOT, KEY)
    ctx.RESULT, ctx.TARGET, ctx.UNITS = RESULT, TARGET, UNITS
    ctx.PACKED = packed

    def current(row, committed=True):
        if committed:
            require(ctx.feature.validation.committed(RESULT, (ctx.KEY,)) == ctx.compact(row), 'commit exact fixture plan')
        _, _, identity = reference()
        require(identity == row['selected_native'], 'native build provenance changed')
        required_sources = {*UNITS, THIS, VERIFY_POLICY, 'tools/sycl/build_compat_profile.py'}
        require(required_sources <= row['project']['source_sha256'].keys(),
                'required fixture and builder sources are missing')
        ctx.build.check_inputs(row)
        require(row['recipe']['targets'] == row['recipe']['build_targets'] == [TARGET]
                and 'LD_PRELOAD' not in row['recipe']['environment'], 'fixture-only build recipe required')
        if 'binary' in row:
            image = row['binary']
            require(Path(image['path']).stat().st_size == image['bytes'] and sha(image['path']) == image['sha256'],
                    'fixture executable changed')
            actual = row['actual_inputs']
            require(sha(actual['path']) == actual['sha256'], 'actual compiler inputs changed')
            inputs = json.loads(Path(actual['path']).read_text())
            require(all(sha(path) == value for name in ('compiled', 'generated', 'links')
                        for path, value in inputs[name].items()), 'a compiled or linked input changed')

    ctx.current = current
    return ctx


def prepare(ctx):
    root = ctx.ROOT
    require(not root.exists() and ctx.KEY not in ctx.feature.validation.committed(RESULT), 'fresh component build required')
    native, compiler, identity = reference()
    project = ctx.source.pin(P, ('CMakeLists.txt', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md', 'cmake',
                                'src', 'include', 'third_party', 'bench/sycl', 'tools/sycl'))
    recipe = copy.deepcopy(native['recipe'])
    recipe['configure'] = [value.replace(str(NATIVE / 'source'), str(P)).replace(str(NATIVE), str(root))
                           for value in recipe['configure']
                           if not value.startswith('-DSTRATA_QSA_PREFILL_PV_SOURCE:')]
    recipe['environment'] = {key: value.replace(str(NATIVE), str(root))
                             for key, value in recipe['environment'].items()}
    recipe['targets'] = recipe['build_targets'] = [TARGET]
    require(recipe['configure'][recipe['configure'].index('-S') + 1] == str(P)
            and recipe['configure'][recipe['configure'].index('-B') + 1] == str(root / 'build'), 'isolated build paths')
    row = dict(stage='prepared', project=project, dependencies=compiler['dependencies'],
        toolchain=compiler['toolchain'], external_inputs=native['external_inputs'], recipe=recipe,
        fallbacks={}, selected_native=identity,
        budgets=ctx.BUDGET, commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        host_budget=dict(wall=120, cpu=120, rss_cap=1 << 30, text_cap=1 << 20,
                         file_cap=2 << 20, total_cap=64 << 20),
        plan=dict(compile_target=TARGET, initial_native_checks=['--host', '--host-c5', '--check-c5-small'],
            scope='One fixture-only build. Omit the generated prefill test source. Preserve the retained DPC++ flags. Validate device loading and five-column numerical results in a separate bounded run.',
            next='After the small guard/oracle checks, measure head and dense shapes independently in three fresh processes.'),
        retention='One standalone fixture for the immediate C5 comparison. Retire the build after the component decision and preserve its exact source and recipe.')
    if ctx.PACKED:
        row['plan'].update(initial_native_checks=['--host', '--host-c5', '--host-packed-c5'],
            scope='Build the canonical/packed comparison on one coherent SYCL fixture. Model dispatch stays outside this comparison.',
            next='Require traced small and full-head numerical checks, then three clean matched timing processes.')
    ctx.current(row, committed=False)
    root.mkdir(mode=0o700)
    for name in ('tmp', 'cache'):
        (root / name).mkdir(mode=0o700)
    ctx.save(row)


def host(ctx, extra_controls=()):
    root = ctx.ROOT
    row = json.loads((root / 'record.json').read_text())
    ctx.current(row)
    require(row['stage'] == 'built' and row['passed'] and row['ownership_closed']
            and 'host' not in row, 'a fresh committed fixture is required')
    def persist():
        ctx.write(root / 'host-execution.json', dict(active=bool(owner.active), commands=owner.commands))

    owner = base.ownership.Owner(root / 'owned-host', persist)
    controls = [('--host', 'HOST passed=1 cases=31 GPU=0'),
                ('--host-c5', 'HOST_C5 passed=1 cases=10 GPU=0')]
    if ctx.PACKED:
        controls.append(('--host-packed-c5', 'HOST_PACKED_C5 passed=1 cases=7 GPU=0'))
    controls.extend(extra_controls)
    outcomes = []
    try:
        require(row['plan']['initial_native_checks'][:len(controls)] == [flag for flag, _ in controls],
                'native host controls differ from the committed plan')
        for flag, expected in controls:
            entry, out, _ = owner.run(flag.removeprefix('--'), [row['binary']['path'], flag],
                                      row['recipe']['environment'], P, **row['host_budget'])
            require(base.ownership.closed(entry) and entry['exit_code'] == 0
                    and out.read_text().splitlines() == [expected], 'host controls differ: ' + flag)
            outcomes.append(dict(argument=flag, exit_code=0, result=expected))
    except BaseException as error:
        row['host'] = dict(passed=False, error=repr(error), outcomes=outcomes)
        raise
    else:
        row['host'] = dict(passed=True, outcomes=outcomes)
    finally:
        row['host']['ownership_closed'] = not owner.active and all(base.ownership.closed(c) for c in owner.commands)
        ctx.save(row)


if __name__ == '__main__':
    require(len(sys.argv) in (2, 3) and sys.argv[1] in ('prepare', 'build', 'host')
            and sys.argv[2:] in ([], ['--packed']), 'use prepare, build or host with optional --packed')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.feature.guards.no_model_owners([])
        ctx = context(packed=sys.argv[2:] == ['--packed'])
        if sys.argv[1] == 'prepare':
            prepare(ctx)
        elif sys.argv[1] == 'build':
            ctx.compile_fixture()
        else:
            host(ctx)
