#!/usr/bin/env python3
"""Lower original dequant helper syntax; keep the pinned arithmetic and tables."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from extract_mmq_loaders import function

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ggml', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / 'docs/sycl-audit/source-inventory.json').read_text())
    hashes = {r['path']: r['upstream_sha256'] for r in manifest['files']}
    external = {r['path']: r['sha256'] for r in manifest['external_sources']}
    def read(base, path, expected):
        raw = (base / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[path]:
            raise SystemExit('Pinned original source mismatch: ' + path)
        return raw.decode()
    dense = read(root, 'src/kernels/cuda/dequant_bf16.cu', hashes)
    iq = read(root, 'src/kernels/cuda/iq_kernels.cu', hashes)
    read(args.ggml, 'ggml/src/ggml-quants.c', external)
    read(args.ggml, 'ggml/src/ggml-common.h', external)
    preamble = '''// Generated from original Strata CUDA helpers and pinned GGML (MIT).
#pragma once
#include "strata/sycl_upstream/mmq_loader_shim.hpp"
#include <cstring>
namespace strata::sycl_upstream::original_dequant {
inline float __half2float(sycl::half v) { return float(v); }
inline sycl::half __ushort_as_half(uint16_t v) { return sycl::bit_cast<sycl::half>(v); }
inline sycl::half __float2half_rn(float v) { return sycl::half(v); }
inline sycl::half __float2half(float v) { return sycl::half(v); }
inline uint16_t __half_as_ushort(sycl::half v) { return sycl::bit_cast<uint16_t>(v); }
inline uint32_t __float_as_uint(float v) { return sycl::bit_cast<uint32_t>(v); }
inline float __uint_as_float(uint32_t v) { return sycl::bit_cast<float>(v); }
struct float2 { float x, y; };
inline sycl::half __low2half(sycl::half2 v) { return v[0]; }
inline sycl::half __high2half(sycl::half2 v) { return v[1]; }
inline float2 __half22float2(sycl::half2 v) { return {float(v[0]), float(v[1])}; }
'''
    helpers = dense[dense.index('__device__ __forceinline__ float h2f'):dense.index('template <int TYPE, typename T>\n__global__')]
    geometry = function(dense, 'geometry', 'bool ')
    iq_helpers = iq[iq.index('template<typename dst_t> __device__ __forceinline__ dst_t cvt'):iq.index('// flat: superblock')]
    row_bytes = function(iq, 'iq_row_bytes', 'size_t ')
    out = preamble + helpers + geometry + '\n' + iq_helpers + row_bytes + '\n}\n'
    out = out.replace('__device__ ', '').replace('__forceinline__', 'inline')
    out = out.replace('__constant__ ', 'inline constexpr ')
    out = re.sub(r'\b__half\b', 'sycl::half', out)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'upstream_dequant.hpp').write_text(out)

if __name__ == '__main__':
    main()
