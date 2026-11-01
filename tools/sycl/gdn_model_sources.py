#!/usr/bin/env python3
"""Selected engine plus only the owned GDN storage integration. Repository MIT.

The active working tree contains other work. Project explicit reversible deltas
onto the recoverable selected555-file source instead of building all of HEAD.
"""
import hashlib
from pathlib import Path

import retained_prefill_candidate as retained

P, B = retained.P, retained.B
base, require = retained.base, retained.require
BEFORE = 'ccba29e0a9f7add85807168858d5d7c7ee659a8b'
TYPED_BEFORE = '409e5951faccd5fc146c0d7660a4380e4d8905e4'
TYPED = '3a1294459f8a6d0d8a6276188fed39b66c85bb3c'
DELTA = ('include/strata/prefill/gemm.hpp', 'src/prefill/sycl/gemm.cpp',
         'src/prefill/prefill.cpp')
KERNEL, HEADER = 'src/kernels/sycl/fused_gdn.cpp', 'include/strata/prefill/kernels.hpp'
CAST = 'include/strata/prefill/detail/gdn_storage_casts.hpp'
FIXTURE = 'bench/sycl/gdn_storage_pipeline.cpp'
CMAKE = 'cmake/sycl_backend.cmake'
MARKER = b'# Main-owned production-wrapper GDN storage qualification (repository LICENSE).\n'


def blob(revision, name):
    return base.basis.build.git(P, 'show', revision + ':' + name)


def sources(revision):
    full, values = retained.original.sources()
    require(len(values) == 555, 'selected source count changed')
    original = dict(values)
    merge = base.basis.merge_delta
    for name in DELTA:
        require((P/name).read_bytes() == blob(revision,name), 'commit exact integration: '+name)
        values[name] = merge(values[name], name, BEFORE, revision)
        require(merge(values[name],name,revision,BEFORE) == original[name],
                'integration changed unrelated source: '+name)
    for name in (KERNEL,HEADER):
        require((P/name).read_bytes() == blob(TYPED,name), 'qualified typed input changed: '+name)
        values[name] = merge(values[name],name,TYPED_BEFORE,TYPED)
        require(merge(values[name],name,TYPED,TYPED_BEFORE) == original[name],
                'typed source changed unrelated source: '+name)
    require(CAST not in values and FIXTURE not in values, 'unexpected existing typed files')
    values[CAST] = blob(TYPED,CAST)
    require((P/CAST).read_bytes() == values[CAST], 'qualified conversion helper changed')
    values[FIXTURE] = blob(revision,FIXTURE)
    require((P/FIXTURE).read_bytes() == values[FIXTURE], 'commit exact production fixture')
    cmake = blob(revision,CMAKE)
    require((P/CMAKE).read_bytes() == cmake and cmake.count(MARKER)==1,
            'commit exact unique fixture target')
    appendix = MARKER + cmake.split(MARKER,1)[1]
    require(b'sycl_gdn_storage_pipeline_bench' not in values[CMAKE], 'duplicate fixture target')
    values[CMAKE] += b'\n' + appendix
    touched = {*DELTA,KERNEL,HEADER,CAST,FIXTURE,CMAKE}
    require(len(values)==557 and all(name in touched or raw == original[name]
            for name,raw in values.items()), 'unrelated/decode source entered candidate')
    # Preserve original decode math and existing FP32 fused prefill entrypoint.
    old,new = original[KERNEL],values[KERNEL]
    seam=b'// Main-authored scheduling variant'
    clean=new.replace(b'#include <type_traits>\n',b'').replace(
        b'#include "strata/prefill/detail/gdn_storage_casts.hpp"\n',b'')
    require(old[:old.index(seam)]==clean[:clean.index(seam)], 'decode GDN arithmetic changed')
    start=b'} // namespace\nvoid fused_gdn_conv_l2';end=b'} // namespace strata::kernels'
    require(old[old.index(start):old.index(end)]==new[new.index(start):new.index(end)],
            'decode GDN wrappers changed')
    return full, values


def input_pins(revision):
    names=(*DELTA,KERNEL,HEADER,CAST,FIXTURE,CMAKE)
    return {name:hashlib.sha256(blob(revision,name)).hexdigest() for name in names}
