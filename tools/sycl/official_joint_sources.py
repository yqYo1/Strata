#!/usr/bin/env python3
"""Compose the reviewed joint head change on the exact official candidate.

Main-authored; repository LICENSE. No compilation, device work or adoption.
Preserve the original constructor and all unrelated working-tree experiments.
"""
import json

import official_model_sources as original

P, B = original.P, original.B
require = original.require
JOINT = '29f8657ee454f9f841ee752e3c9c6ff6c515ef4d'
BEFORE = '0391301a65f1a1e61afb95ebd2f7afa7bade0721^'
FILES = ('include/strata/core/prefill_head_scope.hpp',
         'include/strata/core/verify.hpp', 'src/core/verify.cpp')
GENERATE = 'src/program/generate.cpp'


def section(raw):
    start = b'struct NativeDensePhase {'
    end = b"// Windows' WDDM driver model:"
    require(raw.count(start) == raw.count(end) == 1, 'ambiguous shared phase section')
    first, last = raw.index(start), raw.index(end)
    require(first < last, 'reversed shared phase section')
    return first, last, raw[first:last]


def sources():
    full, values = original.sources()
    require(original.aggregate(values) ==
            '5f468f91a381399d120eafa9b65baa274ea6b2137176e490f8676e7c1d1f89d0',
            'original official candidate changed')
    for name in FILES:
        require(values[name] == original.split.blob(BEFORE, name),
                'joint prerequisite differs from selected source: ' + name)
        values[name] = original.split.blob(JOINT, name)
    raw = values[GENERATE]
    first, last, selected = section(raw)
    require(selected == section(original.split.blob(BEFORE, GENERATE))[2],
            'selected phase controller differs from joint prerequisite')
    replacement = section(original.split.blob(JOINT, GENERATE))[2]
    values[GENERATE] = raw[:first] + replacement + raw[last:]
    a, b, _ = section(values[GENERATE])
    require(values[GENERATE][:a] + selected + values[GENERATE][b:] == raw,
            'joint composition changed unrelated controller code')
    require(len(values) == 554, 'unexpected joint source count')
    return full, values


if __name__ == '__main__':
    _, values = sources()
    print(json.dumps(dict(source_files=len(values), aggregate_sha256=original.aggregate(values),
        changed_paths=[*FILES, GENERATE], joint_source=JOINT,
        build_or_GPU_executed=False, inference_adopted=False)))
