#!/usr/bin/env python3
"""Compose two prefill strategies/GDN3 on the measured official source.

Main-authored; repository LICENSE. No XeStrata input, compilation or GPU work.
Only the named deltas enter this projection, not other working-tree experiments.
"""
import official_joint_sources as original

P, B = original.P, original.B
require = original.require
BEFORE = 'd942e775e8511bd96250d21774b689cfdcc26d48'
IMPLEMENTATION = '271333f64af10e088897753e142dd600c1aad3d4'
BASE_AGGREGATE = 'c3b781a2378a0d208c6ac7852bff505ce36f8c4955ae2956877c0db44dfd7dbc'
POLICY = ('include/strata/prefill/prefill.hpp', 'src/prefill/prefill.cpp',
          'src/prefill/layer_major.inc', 'src/program/generate.cpp')


def sources():
    full, values = original.sources()
    aggregate = original.original.aggregate
    require(aggregate(values) == BASE_AGGREGATE, 'official source basis changed')
    merge = original.original.merge_delta
    blob = original.original.split.blob
    for name in POLICY:
        before = values[name]
        values[name] = merge(before, name, BEFORE, IMPLEMENTATION)
        require(merge(values[name], name, IMPLEMENTATION, BEFORE) == before,
                'policy composition changed unrelated source: ' + name)
    name = 'include/strata/prefill/strategy.hpp'
    require(name not in values, 'unexpected existing strategy header')
    values[name] = blob(IMPLEMENTATION, name)

    name = 'src/kernels/sycl/fused_gdn.cpp'
    before, after = values[name], blob(IMPLEMENTATION, name)
    clean = after.replace(b'#include <cstdio>\n#include <cstdlib>\n', b'')
    seam = b'// Main-authored scheduling variant'
    require(before[:before.index(seam)] == clean[:clean.index(seam)],
            'existing decode GDN arithmetic changed')
    first = b'} // namespace\nvoid fused_gdn_conv_l2'
    old_end, new_end = b'void gdn_recurrence_partitioned(', b'namespace {\nvoid gdn_recurrence_separate('
    require(before[before.index(first):before.index(old_end)].rstrip() ==
            after[after.index(first):after.index(new_end)].rstrip(),
            'existing conv/gates/fused prefill/decode wrappers changed')
    require(b'recurrence_subgroups' not in after, 'rejected subgroup prototype entered model')
    values[name] = after
    for name in ('include/strata/prefill/kernels.hpp', 'bench/sycl/gdn_columns.cpp', 'tests/sycl/compat.cpp'):
        values[name] = blob(IMPLEMENTATION, name)
    name = 'cmake/sycl_backend.cmake'
    old = b'''add_executable(sycl_gdn_columns_bench EXCLUDE_FROM_ALL bench/sycl/gdn_columns.cpp)
target_link_libraries(sycl_gdn_columns_bench PRIVATE strata_kernels)
'''
    new = b'''add_executable(sycl_gdn_columns_bench EXCLUDE_FROM_ALL
  bench/sycl/gdn_columns.cpp src/kernels/sycl/fused_gdn.cpp)
target_link_libraries(sycl_gdn_columns_bench PRIVATE strata_sycl_runtime)
'''
    values[name] = original.original.once(values[name], old, new)
    require(len(values) == 555, 'unexpected projected model source count')
    # All unlisted source bytes, including decode kernels/scheduler, stay fixed.
    return full, values


if __name__ == '__main__':
    import json
    _, values = sources()
    print(json.dumps(dict(source_files=len(values), aggregate_sha256=original.original.aggregate(values),
                         implementation=IMPLEMENTATION, build_or_GPU_executed=False)))
