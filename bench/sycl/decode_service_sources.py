#!/usr/bin/env python3
"""Compose disjoint service attribution onto the actual 92832 generation.

Main-authored, repository LICENSE. This recovers source, not qualification.
Do not copy the newer worktree generator or change the frozen model inputs.
The CLI waits for the original centering cohort to close; a new owned build,
retirement decision and host/device/model qualification remain separate work.
"""
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import q6_centering_sources as previous

B, require = previous.B, previous.require
SOURCE_ROOT = B / 'q6-centering-model-v2'
ROOT = B / 'decode-service-model-v1'
NATIVE = '92832da86d9a63f996e6fcebbf4e919e2c8f746a'
ENGINE = 'a1b0b7e336495bde2482fcbe13b070cba303d49fac09a134f610699c9eaf1616'
PATCH = '96b9cc511b81e5fe4c18e53030019a2ad34c6b4d'
EXISTING = ('include/strata/sycl/cpu_wait_profile.hpp',
            'src/core/verify.cpp', 'src/program/generate.cpp')
ADDED = ('include/strata/sycl/decode_service_profile.hpp',
         'tests/core/decode_service_profile_test.cpp')
CHANGED = (*EXISTING, *ADDED)


def cohort():
    data = previous.base.committed(previous.original.RESULT)
    row = data['q6_centering_generation_speed_v1']
    require(row['stage'] == 'closed' and row['complete'] and row['passed']
            and row['ownership_closed'] and not row['active']
            and len(row['processes']) == 6
            and [p['index'] for p in row['processes']] == list(range(6))
            and row['binary']['sha256'] == ENGINE
            and row['native_source_commit'] == NATIVE
            and all(p['complete'] and p['passed'] and p['ownership_closed']
                and p['native_exit_code'] == 0 and not p['fresh_faults']
                and p['boot_unchanged'] and p['centering_route']['passed']
                for p in row['processes'])
            and type(row.get('decision', {}).get('centering_model_TG_gain_qualified')) is bool,
            'close and adjudicate the original six processes before this projection')
    # Execution success and the original performance decision remain distinct.
    return row


def projection():
    selected = cohort()
    _, _, before = previous.projection()
    native = json.loads((SOURCE_ROOT / 'record.json').read_text())
    require(native['source_commit'] == NATIVE and len(before) == 564
            and previous.pins(before) == native['native_sources']
            and native['binaries']['strata']['sha256'] == ENGINE
            and selected['binary'] == native['binaries']['strata'],
            'recover every actual 92832 input before adding the observer')
    values = dict(before)
    for name in EXISTING:
        values[name] = previous.merge(before[name], name, PATCH + '^', PATCH)
    for name in ADDED:
        require(name not in before, 'observer input already exists: ' + name)
        values[name] = previous.blob(PATCH, name)
        require((P / name).read_bytes() == values[name],
                'reviewed observer header/fixture changed: ' + name)
    require(len(values) == 566 and set(values) - set(before) == set(ADDED)
            and {n for n in values if values[n] != before.get(n)} == set(CHANGED),
            'only the five declared observer files may change')
    # A full reverse check protects Q6, IQ3, prefill, cache and runtime changes
    # sharing these files. No newer worktree file substitutes for the delta.
    restored = dict(values)
    for name in EXISTING:
        restored[name] = previous.merge(values[name], name, PATCH, PATCH + '^')
    for name in ADDED:
        del restored[name]
    require(restored == before, 'removing the observer must recover all 564 inputs')
    require(all(values[n] == before[n] for n in before if n not in EXISTING),
            'observer changed another production source')
    return native, values


def recipe(native, root=ROOT):
    root = Path(root)
    require(root == ROOT and root != SOURCE_ROOT
            and root != previous.phase.selected.model.ROOT
            and native['source_commit'] == NATIVE
            and native['binaries']['strata']['sha256'] == ENGINE,
            'use only the named diagnostic root and exact native recipe')
    value = copy.deepcopy(native['recipe'])
    value['configure'] = [s.replace(str(SOURCE_ROOT), str(root)) for s in value['configure']]
    value['environment'] = {k: s.replace(str(SOURCE_ROOT), str(root))
                            for k, s in value['environment'].items()}
    require(sum('--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13' in s
                for s in value['configure']) == 1
            and '-DSTRATA_CPU_SERVICE_PROFILE:BOOL=ON' in value['configure'],
            'preserve the exact GCC13 selection and existing CPU observer build')
    # Follow the original closed gate, rather than selecting a favorable sample.
    selected = int(cohort()['decision']['centering_model_TG_gain_qualified'])
    value['environment'].update(STRATA_SYCL_DECODE_Q6_CENTERING=str(selected),
        STRATA_SYCL_DECODE_SERVICE_PROFILE='0')
    value['retention'] = ('Preserve the validated552b fallback. After cohort and '
        'standalone half-screen consumers close,recover and retire a1b0 products '
        'before compiling this replacement. No third inference generation. '
        'Keep the exact original recipe/toolchain and qualify the new build. '
        'Separate same-binary profile on/off processes keep CPU service/census '
        'enabled in both;clean speed and physical traffic remain separate.')
    return value


if __name__ == '__main__':
    native, values = projection()
    print(json.dumps(dict(native_source=NATIVE, native_engine_sha256=ENGINE,
        next_root=str(ROOT),
        observer_delta=PATCH, files=len(values), changed=list(CHANGED),
        aggregate_sha256=previous.signature(previous.pins(values)),
        reversible_to_native=True, build_executed=False, GPU_executed=False,
        model_executed=False, hardware_bottleneck_qualified=False)))
