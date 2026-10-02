// IQ4_XS native Q8_1 projections. Block layout and virtual lane mapping are
// adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d.
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
// Map the nonlinear nibble codebook into packed signed bytes for DP4A.
U lookup(U packed, int shift) SYCL_ESIMD_FUNCTION {
  U out = 0u;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    const U index = (packed >> (8 * byte + shift)) & 15u;
    U low = 0xbfad9881u, high = 0x26190d01u;
    low.merge(U(0xf6eaddcfu), (index & 4u) != 0u);
    high.merge(U(0x71594535u), (index & 4u) != 0u);
    low.merge(high, (index & 8u) != 0u);
    out |= ((low >> (8u * (index & 3u))) & 255u) << (8 * byte);
  }
  return out;
}
F block_part(const uint8_t *w, const uint8_t *x, U block, int blocks)
    SYCL_ESIMD_FUNCTION {
  const U lane(0, 1), group = lane % 8u;
  const e::simd_mask<V> valid = block < uint32_t(blocks);
  const U wo = block * 136u, xo = (block * 8u + group) * 36u;
  const F d = half_decode(e::gather<uint16_t, V>(
      reinterpret_cast<const uint16_t *>(w), wo, valid));
  const F dx = half_decode(e::gather<uint16_t, V>(
      reinterpret_cast<const uint16_t *>(x), xo, valid));
  const U sh = e::gather<uint16_t, V>(
      reinterpret_cast<const uint16_t *>(w), wo + 2u, valid);
  const U sl = e::gather<uint8_t, V>(w, wo + 4u + group / 2u, valid);
  const U scale = ((sl >> (4u * (group % 2u))) & 15u) |
                  (((sh >> (2u * group)) & 3u) << 4);
  I dot = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const U codes = e::gather<uint32_t, V>(
        reinterpret_cast<const uint32_t *>(w), wo + 8u + group * 16u + 4u * j,
        valid);
    const I low = e::gather<int32_t, V>(
        reinterpret_cast<const int32_t *>(x), xo + 4u + 4u * j, valid);
    const I high = e::gather<int32_t, V>(
        reinterpret_cast<const int32_t *>(x), xo + 20u + 4u * j, valid);
    dot = e::dp4a<int32_t>(dot, e::convert<int32_t>(lookup(codes, 0)), low);
    dot = e::dp4a<int32_t>(dot, e::convert<int32_t>(lookup(codes, 4)), high);
  }
  dot *= e::convert<int32_t>(scale) - 32;
  F result = ex::fma(ex::fma(d, dx, F(0.f)), e::convert<float>(dot), F(0.f));
  result.merge(F(0.f), !valid);
  return result;
}
float row_dot(const uint8_t *w, const uint8_t *x, int blocks)
    SYCL_ESIMD_FUNCTION {
  const U lane(0, 1);
  F acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    acc[j] = 0.f;
  for (int first = 0; first < blocks; first += 16) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (first + j * 2 < blocks)
        acc[j] += block_part(w, x, U(first + j * 2) + lane / 8u, blocks);
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
void native_iq4_xs_esimd(const void *weights, const void *activation, float *out,
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
            result[row] = row_dot(w + row * blocks * 136, input, blocks);
        });
  }
}
} // namespace strata::kernels
