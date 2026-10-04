#!/usr/bin/env python3
"""Lower pinned CUDA tile loaders mechanically; keep their arithmetic unchanged."""
import argparse
import hashlib
import json
from pathlib import Path

FORMATS = ['q2_0', 'iq2_xxs', 'iq2_xs', 'iq2_s', 'iq3_xxs', 'iq3_s', 'iq4_nl', 'iq4_xs', 'q8_0']


def function(source, name, prefix):
    at = source.index(name + '(')
    start = source.rfind(prefix, 0, at)
    if start < 0:
        raise ValueError('missing function prefix: ' + name)
    end = source.index('\n}\n', at) + 3
    return source[start:end]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ggml', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / 'docs/sycl-audit/source-inventory.json').read_text())
    hashes = {f['path']: f['sha256'] for f in manifest['external_sources']}
    sources = {}
    for path in ['ggml/src/ggml-cuda/mmq-load-tiles.cuh', 'ggml/src/ggml-cuda/vecdotq.cuh',
                 'ggml/src/ggml-quants.c', 'ggml/src/ggml-common.h']:
        blob = (a.ggml / path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != hashes[path]:
            raise SystemExit('Pinned upstream source mismatch: ' + path)
        sources[Path(path).name] = blob.decode()
    common = '''// Generated from pinned GGML (MIT, third_party/ggml/LICENSE).
#pragma once
#include "strata/sycl_upstream/mmq_loader_shim.hpp"
namespace strata::sycl_upstream::original_loaders {
#pragma push_macro("TURING_MMA_AVAILABLE")
#define TURING_MMA_AVAILABLE 1
'''
    out = common
    for name in ['get_int_b2', 'get_int_b4', 'get_int_from_table_16', 'unpack_ksigns']:
        out += function(sources['vecdotq.cuh'], name, 'static __device__').replace('__device__ ', '').replace('__forceinline__', 'inline') + '\n'
    for fmt in FORMATS:
        body = function(sources['mmq-load-tiles.cuh'], 'ggml_cuda_mmq_load_tiles_' + fmt, 'template <')
        body = body.replace('__device__ ', '').replace('__forceinline__', 'inline')
        needle = 'const int stride) {'
        assert body.count(needle) == 1
        body = body.replace(needle, 'const int stride, ThreadIdx threadIdx) {')
        out += body + '\n'
    out += '#pragma pop_macro("TURING_MMA_AVAILABLE")\n}\n'
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'upstream_mmq_loaders.hpp').write_text(out)
    # Independent CPU dequantizers are not the GPU loader or its scale formula.
    cpu = '''// Generated verbatim from pinned GGML ggml-quants.c (MIT).
#pragma once
#include "strata/sycl_upstream/mmq_loader_shim.hpp"
#include <cassert>
#include <cstring>
#pragma push_macro("GGML_FP16_TO_FP32")
#define GGML_FP16_TO_FP32(x) float(x)
namespace upstream_cpu {
'''
    for fmt in FORMATS:
        cpu += function(sources['ggml-quants.c'], 'dequantize_row_' + fmt, 'void ') + '\n'
    cpu += '}\n#pragma pop_macro("GGML_FP16_TO_FP32")\n'
    (a.output / 'upstream_cpu_dequant.hpp').write_text(cpu)


if __name__ == '__main__':
    main()
