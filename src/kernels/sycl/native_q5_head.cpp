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

#include "strata/sycl/launch.hpp"
#include <sycl/ext/intel/esimd.hpp>

namespace strata::kernels {
namespace {
namespace e = sycl::ext::intel::esimd;
using namespace sycl_backend;
constexpr int V = 16;
using U = e::simd<uint32_t, V>;
using I = e::simd<int32_t, V>;
using F = e::simd<float, V>;

F half_decode(U h) SYCL_ESIMD_FUNCTION {
  const U sign = (h & 0x8000u) << 16;
  const U exponent = (h >> 10) & 31u;
  const U mantissa = h & 1023u;
  U bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  F result = bits.bit_cast_view<float>();
  F sub = e::convert<float>(mantissa) * 0x1p-24f;
  U sub_bits = sub.bit_cast_view<uint32_t>();
  sub_bits |= sign;
  result.merge(sub_bits.bit_cast_view<float>(), exponent == 0);
  U special = sign | 0x7f800000u | (mantissa << 13);
  result.merge(special.bit_cast_view<float>(), exponent == 31);
  return result;
}

F block_dot(const uint8_t *w, const uint8_t *x) SYCL_ESIMD_FUNCTION {
  const U lane(0, 1);
  const U part = lane % 4u, offset = 2u * (lane / 4u);
  const auto *w32 = reinterpret_cast<const uint32_t *>(w);
  const U vl0 = e::gather<uint32_t, V>(w32, 48u + 16u * offset + 4u * part);
  const U vl1 = e::gather<uint32_t, V>(w32, 64u + 16u * offset + 4u * part);
  const U vh0 = e::gather<uint32_t, V>(w32, 16u + 4u * part) >> offset;
  const U vh1 = e::gather<uint32_t, V>(w32, 32u + 4u * part) >> offset;
  const U jm = (offset / 2u) & 1u;
  const auto *w16 = reinterpret_cast<const uint16_t *>(w);
  const U s0 = e::gather<uint16_t, V>(w16, 4u + 2u * jm);
  const U s2 = e::gather<uint16_t, V>(w16, 8u + 2u * jm);
  const U s4 = e::gather<uint16_t, V>(w16, 12u + 2u * jm);
  U sc = s0 & 0x3f3fu, mn = s2 & 0x3f3fu;
  sc.merge((s4 & 0x0f0fu) | ((s0 & 0xc0c0u) >> 2), offset >= 4u);
  mn.merge(((s4 >> 4) & 0x0f0fu) | ((s2 & 0xc0c0u) >> 2), offset >= 4u);
  const U dm = e::gather<uint32_t, V>(w32, U(0));
  const F d = half_decode(dm & 65535u), m = half_decode(dm >> 16);
  F sd = 0.f, sm = 0.f;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const U base = (offset + uint32_t(i)) * 36u;
    const auto *x32 = reinterpret_cast<const uint32_t *>(x);
    const I u0 = e::gather<int32_t, V>(reinterpret_cast<const int32_t *>(x),
                                       base + 4u + 4u * part);
    const I u1 = e::gather<int32_t, V>(reinterpret_cast<const int32_t *>(x),
                                       base + 20u + 4u * part);
    const U qs = e::gather<uint32_t, V>(x32, base);
    const F dx = half_decode(qs & 65535u);
    const U v0 =
        ((vl0 >> (4 * i)) & 0x0f0f0f0fu) | (((vh0 >> i) << 4) & 0x10101010u);
    const U v1 =
        ((vl1 >> (4 * i)) & 0x0f0f0f0fu) | (((vh1 >> i) << 4) & 0x10101010u);
    const I dot =
        e::dp4a<int32_t>(e::dp4a<int32_t>(I(0), e::convert<int32_t>(v1), u1),
                         e::convert<int32_t>(v0), u0);
    const I sum = e::dp4a<int32_t>(e::dp4a<int32_t>(I(0), I(0x01010101), u1),
                                   I(0x01010101), u0);
    sd += dx *
          e::convert<float>(dot * e::convert<int32_t>((sc >> (8 * i)) & 255u));
    sm += dx *
          e::convert<float>(sum * e::convert<int32_t>((mn >> (8 * i)) & 255u));
  }
  return d * sd - m * sm;
}
} // namespace

// Fixed to the native Q5_K head's 2560 input width. Eight 16-lane virtual
// blocks retain the ordinary MMVQ accumulator and reduction order. ESIMD
// floating outputs can still differ in their last bits from SPMD.
void native_q5_head_esimd(const void *weights, const void *activation,
                          float *out, int rows, int cols, void *stream) {
  auto *w = static_cast<const uint8_t *>(weights);
  auto *x = static_cast<const uint8_t *>(activation);
  const size_t count = (size_t(rows) + 15) & ~size_t(15);
  for (int col = 0; col < cols; ++col) {
    const auto *column = x + size_t(col) * 80 * 36;
    auto *result = out + size_t(col) * rows;
    queue_for(stream).parallel_for(
        sycl::nd_range<1>(count, 16),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
          const size_t row = item.get_global_linear_id();
          if (row >= size_t(rows))
            return;
          F acc[8];
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc[i] = 0.f;
#pragma unroll
          for (int b = 0; b < 10; ++b)
            acc[b % 8] +=
                block_dot(w + (row * 10 + b) * 176, column + b * 8 * 36);
          F low = ((acc[0] + acc[2]) + acc[4]) + acc[6];
          F high = ((acc[1] + acc[3]) + acc[5]) + acc[7];
          F s16 = low + high;
          e::simd<float, 8> s8 = s16.select<8, 1>(0) + s16.select<8, 1>(8);
          e::simd<float, 4> s4 = s8.select<4, 1>(0) + s8.select<4, 1>(4);
          e::simd<float, 2> s2 = s4.select<2, 1>(0) + s4.select<2, 1>(2);
          result[row] = s2[0] + s2[1];
        });
  }
}
} // namespace strata::kernels
