#!/usr/bin/env python3
"""Compose only the owned CPU observer onto the measured phase/Q6 engine.

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
import verify_window_model as phase

P, B = selected.P, selected.B
require, digest = selected.require, selected.digest
NATIVE = '05a9b45c0a3a266779d4757a855b45a3bb42a62e'
ENGINE = 'a99aa1ac98c397933ca76d59361fec32aba335f6904b63c25d95efc52ce22734'
NATIVE_ROOT = B / 'verify-window-model-v3'
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
    ('49c20a41c360da05111b8203f1d6ef480875f3dd', (HEADER, PROGRAM, FIXTURES[0])),
)
CHANGED = {'CMakeLists.txt', *POOL, HEADER, PROGRAM, *FIXTURES}


def aggregate(values):
    pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in values.items()}
    return pins, hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()


def projection():
    """Recover committed inputs and prove each observer delta is reversible."""
    # Recover the actual composed 560-file snapshot, including the original
    # 6e40 pool bytes. A plain checkout of 05a9 is not that measured source.
    _, before = phase.projection(NATIVE)
    native = json.loads((NATIVE_ROOT / 'record.json').read_text())
    require(native['source_commit'] == NATIVE and len(before) == 560 and
            aggregate(before)[0] == native['native_sources'] and
            native['binaries']['strata']['sha256'] == ENGINE and
            native['passed'] and native['complete'] and native['ownership_closed'] and not native['active'],
            'measured source/build identity changed; make a new prospective construction')
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

    require(len(values) == 563 and set(values) - set(before) == {HEADER, *FIXTURES} and
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
    # Preserve all phase ownership and byte-read safety code, not just the Q6
    # selectors. The generator's observer delta is independently reversed above.
    require(all(values[name] == before[name] for name in
                (*phase.Q6_FILES, *[n for n in phase.PHASE_FILES if n != PROGRAM], phase.POLICY, phase.FIXTURE)),
            'observer changed the measured Q6/phase implementation')
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
    require(root != NATIVE_ROOT, 'preserve the source generation until explicit retirement')
    old = str(NATIVE_ROOT)
    configure = [value.replace(old, str(root)) for value in native['recipe']['configure']]
    require(not any(value.startswith('-DSTRATA_CPU_SERVICE_PROFILE') for value in configure),
            'selected recipe already contains an observer setting')
    configure.append('-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON')
    environment = {name: value.replace(old, str(root))
                   for name, value in native['recipe']['environment'].items()}
    return dict(configure=configure, environment=environment,
                targets=['strata', 'cpu_service_profile_test', 'cpu_service_pool_test',
                         'sycl_submission_test', 'sycl_verifier_retirement_test'],
                host_modes=[{'STRATA_CPU_SERVICE': '0'}, {'STRATA_CPU_SERVICE': '1'}],
                retention='One imminent observer on/off comparison on the measured05a9 source. '
                    'Keep validated479a as fallback;retire the closed a99aa1 comparison outputs '
                    'only after exact source/recipe recovery and consumer updates, before building '
                    'this replacement. No third engine generation and no inference adoption.')


def main():
    native, values = projection()
    pins, fingerprint = aggregate(values)
    plan = recipe(native, B / 'cpu-service-verify-window-v2')
    print(json.dumps(dict(selected_native=NATIVE, selected_engine_sha256=ENGINE,
        source_files=len(values), source_aggregate_sha256=fingerprint,
        changed_files=sorted(CHANGED), observer_header_sha256=pins[HEADER],
        reversible_to_selected=True, recipe=plan,
        build_executed=False, GPU_executed=False, model_executed=False,
        observer_overhead_qualified=False, inference_adopted=False), sort_keys=True))


if __name__ == '__main__':
    main()
