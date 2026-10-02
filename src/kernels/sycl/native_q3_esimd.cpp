// Q3_K block arithmetic adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d vecdotq.cuh and ggml-common.h.
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

#include "strata/kernels/iq_kernels.hpp"
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
F half_decode(U h) SYCL_ESIMD_FUNCTION {
  const U sign = (h & 0x8000u) << 16;
  const U exponent = (h >> 10) & 31u, mantissa = h & 1023u;
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
// Q3_K's 110-byte stride aligns alternate blocks to two bytes, not four.
U load32(const uint8_t *p, U offsets) SYCL_ESIMD_FUNCTION {
  const auto *words = reinterpret_cast<const uint16_t *>(p);
  return e::convert<uint32_t>(e::gather<uint16_t, V>(words, offsets)) |
         (e::convert<uint32_t>(e::gather<uint16_t, V>(words, offsets + 2u))
          << 16);
}
F block_dot(const uint8_t *w, const uint8_t *x) SYCL_ESIMD_FUNCTION {
  const U lane(0, 1), part = lane % 8u;
  const U offset = 4u * (lane / 8u);
  const U scale_offset = lane - part + part / 4u;
  const U vl = load32(w, 32u + 4u * lane);
  const U vh = ~load32(w, 4u * part) >> offset;
  const F d = half_decode(
      e::gather<uint16_t, V>(reinterpret_cast<const uint16_t *>(w), U(108u)));
  F sum = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const U isc = scale_offset + 2u * uint32_t(i);
    const U sl = e::gather<uint8_t, V>(w, 96u + isc % 8u);
    const U sh = e::gather<uint8_t, V>(w, 104u + isc % 4u);
    const U scales = ((sl >> (4u * (isc / 8u))) & 15u) |
                     (((sh >> (2u * (isc / 4u))) & 3u) << 4);
    const I sc = e::convert<int32_t>(scales) - 32;
    const U low = (vl >> (2 * i)) & 0x03030303u;
    const U high = ((vh >> i) << 2) & 0x04040404u;
    const U base = (offset + uint32_t(i)) * 36u;
    const I u = e::gather<int32_t, V>(reinterpret_cast<const int32_t *>(x),
                                      base + 4u + 4u * part);
    const F dx = half_decode(
        e::gather<uint16_t, V>(reinterpret_cast<const uint16_t *>(x), base));
    // low/high bytes are 0..3/0..4. Two signed DP4As implement the
    // per-byte subtraction without carries between packed bytes.
    const I dot = e::dp4a<int32_t>(I(0), e::convert<int32_t>(low), u) -
                  e::dp4a<int32_t>(I(0), e::convert<int32_t>(high), u);
    // Explicit product rounding prevents the ESIMD backend from fusing this
    // multiply with the sum, even when source -ffp-contract=off is set.
    sum += ex::fma(dx, e::convert<float>(dot * sc), F(0.f));
  }
  // Also round the block product before adding it to its virtual lane.
  return ex::fma(d, sum, F(0.f));
}
float row_dot(const uint8_t *w, const uint8_t *input,
              int blocks) SYCL_ESIMD_FUNCTION {
  F acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    acc[j] = 0.f;
  for (int first = 0; first < blocks; first += 8) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (first + j < blocks)
        acc[j] +=
            block_dot(w + (first + j) * 110, input + (first + j) * 8 * 36);
  }
  // Match the original 128 virtual lanes: ascending warp sums,
  // then the XOR tree with offsets 16, 8, 4, 2, 1.
  F low = ((acc[0] + acc[2]) + acc[4]) + acc[6];
  F high = ((acc[1] + acc[3]) + acc[5]) + acc[7];
  F s16 = low + high;
  e::simd<float, 8> s8 = s16.select<8, 1>(0) + s16.select<8, 1>(8);
  e::simd<float, 4> s4 = s8.select<4, 1>(0) + s8.select<4, 1>(4);
  e::simd<float, 2> s2 = s4.select<2, 1>(0) + s4.select<2, 1>(2);
  return s2[0] + s2[1];
}
} // namespace
void native_q3_k_esimd(const void *weights, const void *activation, float *out,
                       int width, int rows, int columns, void *stream) {
  const int blocks = width / 256;
  const auto *w = static_cast<const uint8_t *>(weights);
  const auto *x = static_cast<const uint8_t *>(activation);
  const size_t count = (size_t(rows) + 15) & ~size_t(15);
  for (int column = 0; column < columns; ++column) {
    const auto *input = x + size_t(column) * (width / 32) * 36;
    auto *result = out + size_t(column) * rows;
    queue_for(stream).parallel_for(
        sycl::nd_range<1>(count, 16),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
          const size_t row = item.get_global_linear_id();
          if (row >= size_t(rows))
            return;
          result[row] = row_dot(w + row * blocks * 110, input, blocks);
        });
  }
}
} // namespace strata::kernels
