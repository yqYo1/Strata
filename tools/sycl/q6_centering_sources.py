#!/usr/bin/env python3
"""Recover the measured b310 engine and add only qualified Q6/phase changes.

Main-authored, repository LICENSE. Historical constructors also check the live
tree, which now contains newer candidates. Recover their declared Git deltas
without changing those old admission gates; require every recovered byte to
match the original 563-file build map before composing the new candidate.
"""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_iq3_model as original
import cpu_service_sources as observer
import verify_window_model as phase

P, B, base = original.P, original.B, original.base
BASE_SOURCE = '9922f8a5aff9631c87cda6b31e9526f0e4e4e266'
COMPONENT_SOURCE = 'f18b8e548391a9c4981f216d30c410aeceae2aad'
SELECTOR_SOURCE = 'dc2299e1fd97bd9e3b04faf9b94067e570161ce6'
COMPONENT_EVIDENCE = '2197e506384137dbb958eada61a042faf5fab858'
POLICY_EVIDENCE = '92d06632a'
CENTER_HEADER = 'include/strata/kernels/q6_packed_center.hpp'
SELECTOR_FILES = ('include/strata/core/verify.hpp', 'src/core/verify.cpp',
                  phase.POLICY, phase.FIXTURE)
CHANGED = (*phase.Q6_FILES, CENTER_HEADER, *SELECTOR_FILES)
require = base.require


def blob(revision, name):
    return base.basis.build.git(P, 'show', revision + ':' + name)


def pins(values):
    return {n: hashlib.sha256(raw).hexdigest() for n, raw in values.items()}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def merge(value, name, before, after):
    result = base.basis.merge_delta(value, name, before, after)
    require(base.basis.merge_delta(result, name, after, before) == value,
            'source delta is not exactly reversible: ' + name)
    return result


def baseline():
    full, values = phase.selected.model.projection(worker=True)
    fallback = json.loads((phase.selected.model.ROOT / 'record.json').read_text())
    require(fallback['source_commit'] == '479a6fdac8cbaa35a9bf3a41d2b6e23be5260e54'
            and pins(values) == fallback['native_sources'],
            'recover every measured prefill fallback byte first')
    for name in phase.Q6_FILES:
        require(values[name] == blob('79e240181', name),
                'unexpected original Q6 reference: ' + name)
        values[name] = blob(phase.Q6, name)
    for name in phase.PHASE_FILES:
        values[name] = merge(values[name], name, phase.BEFORE, observer.NATIVE)
    for name in (phase.POLICY, phase.FIXTURE):
        require(name not in values, 'unexpected pre-existing phase fixture')
        values[name] = blob(observer.NATIVE, name)
    old_phase = json.loads((observer.NATIVE_ROOT / 'record.json').read_text())
    require(len(values) == 560 and pins(values) == old_phase['native_sources'],
            'recover every original Q6/phase engine input')
    for revision, names in observer.PATCHES:
        for name in names:
            values[name] = (merge(values[name], name, revision + '^', revision)
                            if name in values else blob(revision, name))
    for name in original.CPU_FILES:
        values[name] = blob(original.COMPONENT_SOURCE, name)
    values[observer.PROGRAM] = merge(values[observer.PROGRAM], observer.PROGRAM,
                                      original.AUDIT_SOURCE + '^', original.AUDIT_SOURCE)
    recorded = json.loads((original.ROOT / 'record.json').read_text())
    require(recorded['source_commit'] == BASE_SOURCE and len(values) == 563 and
            pins(values) == recorded['native_sources'],
            'Git reconstruction differs from actual b310 build inputs')
    return full, recorded, values


def qualification():
    name = str(original.RESULT.relative_to(P))
    component = json.loads(blob(COMPONENT_EVIDENCE, name))['q6_centering_component_v2']
    host = json.loads(blob(POLICY_EVIDENCE, name))['q6_centering_policy_host_v1']
    require(component['source_commit'] == COMPONENT_SOURCE and
            component['stage'] == 'closed' and component['passed'] and component['complete']
            and component['ownership_closed'] and not component['active']
            and len(component['processes']) == 3
            and component['decision']['performance_eligible_shapes'] == 18
            and sum(s['all_three_faster'] for s in component['decision']['shapes']) == 18,
            'closed repeated 18-shape component gate required')
    require(host['source_commit'] == SELECTOR_SOURCE and host['passed'] and host['complete']
            and host['ownership_closed'] and not host['active']
            and host['controls']['centering_shapes'] == 18
            and host['controls']['UBSan_errors'] == 0,
            'closed real C++ policy/cache gate required')
    return dict(component_commit=COMPONENT_EVIDENCE, policy_commit=POLICY_EVIDENCE,
                component_sha256=signature(component), policy_sha256=signature(host),
                eligible_shapes=[s['shape'] for s in component['decision']['shapes']
                                 if s['all_three_faster']], model_qualified=False)


def projection():
    full, recorded, before = baseline()
    qualification()
    values = dict(before)
    for name in (*phase.Q6_FILES, CENTER_HEADER):
        values[name] = blob(COMPONENT_SOURCE, name)
        require((P / name).read_bytes() == values[name],
                'qualified Q6 implementation differs from active source: ' + name)
    for name in SELECTOR_FILES:
        values[name] = merge(before[name], name, SELECTOR_SOURCE + '^', SELECTOR_SOURCE)
    require(len(values) == 564 and set(values) - set(before) == {CENTER_HEADER}
            and {n for n in values if before.get(n) != values[n]} == set(CHANGED),
            'only seven declared Q6/phase files may change')
    require(all(values[n] == before[n] for n in values
                if n.startswith('src/prefill/') or n.startswith('src/kernels/cpu/')
                or n in ('src/program/generate.cpp', 'src/sycl/runtime.cpp',
                         'src/sycl/compat.cpp', 'CMakeLists.txt')),
            'preserve prefill,CPU,generator,placement and runtime inputs')
    return full, recorded, values


if __name__ == '__main__':
    _, old, values = projection()
    print(json.dumps(dict(baseline_source=BASE_SOURCE, baseline_files=563,
        source_files=len(values), changed=list(CHANGED),
        aggregate_sha256=signature(pins(values)),
        original_engine_sha256=old['binaries']['strata']['sha256'],
        build_executed=False, GPU_executed=False)))
