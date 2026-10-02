// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/sycl/esimd_half.hpp"
#include "strata/sycl/launch.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>

namespace strata::kernels {
namespace {
namespace e = sycl::ext::intel::esimd;
namespace ex = sycl::ext::intel::experimental::esimd;
using namespace sycl_backend;
constexpr int V = 16;
using U = e::simd<uint32_t, V>;
using I = e::simd<int32_t, V>;
using F = e::simd<float, V>;

F block_dot(const uint8_t *w, const uint8_t *x) SYCL_ESIMD_FUNCTION {
  const U lane(0, 1);
  const U part = lane % 4u, offset = 2u * (lane / 4u);
  const auto *w32 = reinterpret_cast<const uint32_t *>(w);
  const U vl0 = e::gather<uint32_t, V>(w32, 16u + 16u * offset + 4u * part);
  const U vl1 = e::gather<uint32_t, V>(w32, 32u + 16u * offset + 4u * part);
  const U jm = (offset / 2u) & 1u;
  const auto *w16 = reinterpret_cast<const uint16_t *>(w);
  const U s0 = e::gather<uint16_t, V>(w16, 4u + 2u * jm);
  const U s2 = e::gather<uint16_t, V>(w16, 8u + 2u * jm);
  const U s4 = e::gather<uint16_t, V>(w16, 12u + 2u * jm);
  U sc = s0 & 0x3f3fu, mn = s2 & 0x3f3fu;
  sc.merge((s4 & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2), offset >= 4u);
  mn.merge(((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2), offset >= 4u);
  const U dm = e::gather<uint32_t, V>(w32, U(0));
  const F d = esimd_half_decode<V>(dm & 65535u);
  const F m = esimd_half_decode<V>(dm >> 16);
  F sd = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const U base = (offset + uint32_t(i)) * 36u;
    const auto *x32 = reinterpret_cast<const uint32_t *>(x);
    const I u0 = e::gather<int32_t, V>(reinterpret_cast<const int32_t *>(x),
                                       base + 4u + 4u * part);
    const I u1 = e::gather<int32_t, V>(reinterpret_cast<const int32_t *>(x),
                                       base + 20u + 4u * part);
    const F dx =
        esimd_half_decode<V>(e::gather<uint32_t, V>(x32, base) & 65535u);
    const U v0 = (vl0 >> (4 * i)) & 0x0f0f0f0fu;
    const U v1 = (vl1 >> (4 * i)) & 0x0f0f0f0fu;
    const I dot =
        e::dp4a<int32_t>(e::dp4a<int32_t>(I(0), e::convert<int32_t>(v0), u0),
                         e::convert<int32_t>(v1), u1);
    const I sum = e::dp4a<int32_t>(e::dp4a<int32_t>(I(0), I(0x01010101), u0),
                                   I(0x01010101), u1);
    sd += ex::fma(
        dx,
        e::convert<float>(dot * e::convert<int32_t>((sc >> (8 * i)) & 255u)),
        F(0.f));
    sm += ex::fma(
        dx,
        e::convert<float>(sum * e::convert<int32_t>((mn >> (8 * i)) & 255u)),
        F(0.f));
  }
  // Preserve the two products before affine subtraction and lane accumulation.
  return ex::fma(d, sd, F(0.f)) - ex::fma(m, sm, F(0.f));
}
float row_dot(const uint8_t *w, const uint8_t *x,
              int blocks) SYCL_ESIMD_FUNCTION {
  F acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    acc[j] = 0.f;
  for (int first = 0; first < blocks; first += 8) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (first + j < blocks)
        acc[j] += block_dot(w + (first + j) * 144, x + (first + j) * 8 * 36);
  }
  // Original 128 virtual lanes: ascending cross-warp sums, then XOR tree.
  F low = ((acc[0] + acc[2]) + acc[4]) + acc[6];
  F high = ((acc[1] + acc[3]) + acc[5]) + acc[7];
  F s16 = low + high;
  e::simd<float, 8> s8 = s16.select<8, 1>(0) + s16.select<8, 1>(8);
  e::simd<float, 4> s4 = s8.select<4, 1>(0) + s8.select<4, 1>(4);
  e::simd<float, 2> s2 = s4.select<2, 1>(0) + s4.select<2, 1>(2);
  return s2[0] + s2[1];
}
} // namespace

void native_q4_k_esimd(const void *weights, const void *activation, float *out,
                       int width, int rows, int columns, void *stream) {
  const int blocks = width / 256;
  const auto *w = static_cast<const uint8_t *>(weights);
  const auto *x = static_cast<const uint8_t *>(activation);
  const size_t padded = (size_t(rows) + 15) & ~size_t(15);
  for (int column = 0; column < columns; ++column) {
    const auto *input = x + size_t(column) * (width / 32) * 36;
    auto *result = out + size_t(column) * rows;
    queue_for(stream).parallel_for(
        sycl::nd_range<1>(padded, 16),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
          const size_t row = item.get_global_linear_id();
          if (row < size_t(rows))
            result[row] = row_dot(w + row * blocks * 144, input, blocks);
        });
  }
}
} // namespace strata::kernels
