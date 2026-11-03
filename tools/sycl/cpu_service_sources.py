#!/usr/bin/env python3
"""Compose only the owned CPU observer onto the selected worker/map engine.

Repository LICENSE; independently authored. The historical observer comparison
failed its overall gate. This construction reuses source, not that qualification.
No worktree generator/layout changes enter the projected engine. A new build,
same-build pool/device checks and prospective observer comparison remain required.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import cache_map_model_entry as selected

P, B = selected.P, selected.B
require, digest = selected.require, selected.digest
NATIVE = '479a6fdac8cbaa35a9bf3a41d2b6e23be5260e54'
ENGINE = '552b1616d21ad1b9aa6a317cb940d375c003246b896a255a935e09292804c96a'
HEADER = 'include/strata/kernels/cpu/service_profile.hpp'
POOL = ('include/strata/kernels/cpu/pool.hpp', 'src/kernels/cpu/pool.cpp')
PROGRAM = 'src/program/generate.cpp'
FIXTURES = ('tests/core/cpu_service_profile_test.cpp',
            'tests/core/cpu_service_pool_test.cpp')
PATCHES = (
    ('35f3683bc80cfb36790e4b6a8c54e547737862fe',
     ('CMakeLists.txt', *POOL, HEADER, PROGRAM, FIXTURES[0])),
    ('f26979a6505ca55cf62cc5d1a366eee21d615318', (HEADER,)),
    ('44b520e309494b056d9814918a8a33933d14b6e8', ('CMakeLists.txt', FIXTURES[1])),
    ('589ff46aa8d046edbf395aa46161f7d5f8ce29b2', (FIXTURES[1],)),
)
CHANGED = {'CMakeLists.txt', *POOL, HEADER, PROGRAM, *FIXTURES}


def aggregate(values):
    pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in values.items()}
    return pins, hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def projection():
    """Recover committed inputs and prove each observer delta is reversible."""
    _, before = selected.model.projection(worker=True)
    native = json.loads((selected.model.ROOT / 'record.json').read_text())
    require(native['source_commit'] == NATIVE and len(before) == 558 and
            aggregate(before)[0] == native['native_sources'] and
            native['binaries']['strata']['sha256'] == ENGINE and
            digest(native['binaries']['strata']['path']) == ENGINE,
            'selected source/engine changed; make a new prospective construction')
    values = dict(before)
    merge = selected.base.basis.merge_delta
    steps = []
    for commit, names in PATCHES:
        changed = {}
        for name in names:
            original = values.get(name)
            if original is None:
                values[name] = selected.base.basis.build.git(P, 'show', commit + ':' + name)
            else:
                values[name] = merge(original, name, commit + '^', commit)
                require(merge(values[name], name, commit, commit + '^') == original,
                        'observer patch changed unrelated source: ' + name)
            changed[name] = original
        steps.append((commit, changed))

    require(len(values) == 561 and set(values) - set(before) == {HEADER, *FIXTURES} and
            {name for name in values if values[name] != before.get(name)} == CHANGED,
            'only the seven declared observer/build/fixture files may differ')
    # Undo the complete stack as well as checking each local three-way merge.
    restored = dict(values)
    for commit, originals in reversed(steps):
        for name, original in originals.items():
            if original is None:
                del restored[name]
            else:
                restored[name] = merge(restored[name], name, commit, commit + '^')
                require(restored[name] == original, 'observer stack is not reversible: ' + name)
    require(restored == before, 'observer removal must recover every selected source byte')
    # The live pool currently contains exactly this observer. Its generator and
    # CMake files also contain other work and are deliberately not copied.
    require(all(values[name] == (P / name).read_bytes() for name in (*POOL, HEADER)),
            'reviewed pool observer differs from the declared pinned deltas')
    return native, values


def recipe(native, root):
    """Same compiler, optimization, dependencies and safety; one observer flag."""
    root = Path(root)
    require(root.is_absolute() and root.parent == B and root != selected.model.ROOT,
            'use a distinct external comparison root; preserve the validated engine')
    old = str(selected.model.ROOT)
    configure = [value.replace(old, str(root)) for value in native['configure_argv']]
    require(not any(value.startswith('-DSTRATA_CPU_SERVICE_PROFILE') for value in configure),
            'selected recipe already contains an observer setting')
    configure.append('-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON')
    environment = {name: value.replace(old, str(root))
                   for name, value in native['environment'].items()}
    return dict(configure=configure, environment=environment,
                targets=['strata', 'cpu_service_pool_test', 'sycl_submission_test'],
                host_modes=[{'STRATA_CPU_SERVICE': '0'}, {'STRATA_CPU_SERVICE': '1'}],
                retention='One imminent observer comparison against selected552b. '
                    'Retire the extra generation after source/recipe recovery at closure; '
                    'do not overwrite the selected engine or its eight same-build fixtures.')


def main():
    native, values = projection()
    pins, fingerprint = aggregate(values)
    plan = recipe(native, B / 'cpu-service-cache-map-v1')
    print(json.dumps(dict(selected_native=NATIVE, selected_engine_sha256=ENGINE,
        source_files=len(values), source_aggregate_sha256=fingerprint,
        changed_files=sorted(CHANGED), observer_header_sha256=pins[HEADER],
        reversible_to_selected=True, recipe=plan,
        build_executed=False, GPU_executed=False, model_executed=False,
        observer_overhead_qualified=False, inference_adopted=False), sort_keys=True))


if __name__ == '__main__':
    main()
