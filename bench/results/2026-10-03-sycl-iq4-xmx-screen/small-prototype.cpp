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
#include "strata/sycl/esimd_half.hpp"
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
  const F d = esimd_half_decode<V>(e::gather<uint16_t, V>(
      reinterpret_cast<const uint16_t *>(w), wo, valid));
  const F dx = esimd_half_decode<V>(e::gather<uint16_t, V>(
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
// Sixteen weight rows become DPAS columns, retaining each thirty-two-value
// integer partial and the original floating-point virtual-lane additions.
template <int Blocks>
F xmx_tile(const uint8_t *w, const uint8_t *x) SYCL_ESIMD_FUNCTION {
  const U row(0, 1);
  e::simd<float, 256> even = 0.f, odd = 0.f;
#pragma unroll
  for (int bin = 0; bin < 8; ++bin) {
    e::simd<float, 256> values = 0.f;
#pragma unroll 1
    for (int first = 0; first < Blocks; first += 16) {
#pragma unroll 1
      for (int cell = 0; cell < 16; ++cell) {
        const int block = first + bin * 2 + cell / 8, group = cell % 8;
        if (block >= Blocks) continue;
        const U wo = row * uint32_t(Blocks * 136) + uint32_t(block * 136);
        e::simd<int32_t, 128> b;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const U codes = e::gather<uint32_t, 16>(
              reinterpret_cast<const uint32_t *>(w),
              wo + uint32_t(8 + group * 16 + 4 * j));
          U low = lookup(codes, 0), high = lookup(codes, 4);
          b.select<16, 1>(j * 16) = low.bit_cast_view<int32_t>();
          b.select<16, 1>((j + 4) * 16) = high.bit_cast_view<int32_t>();
        }
        const int xo = (block * 8 + group) * 36;
        e::simd<int32_t, 8> a = e::gather<int32_t, 8>(
            reinterpret_cast<const int32_t *>(x),
            e::simd<uint32_t, 8>(4, 4) + uint32_t(xo));
        I dot = e::xmx::dpas<8, 1, int32_t, int32_t, int32_t, int32_t,
            e::xmx::dpas_argument_type::s8, e::xmx::dpas_argument_type::s8>(
                I(0), b, a);
        const U sh = e::gather<uint16_t, 16>(
            reinterpret_cast<const uint16_t *>(w), wo + 2u);
        const U sl = e::gather<uint8_t, 16>(w, wo + uint32_t(4 + group / 2));
        const U scale = ((sl >> (4 * (group % 2))) & 15u) |
                        (((sh >> (2 * group)) & 3u) << 4);
        dot *= e::convert<int32_t>(scale) - 32;
        const F d = esimd_half_decode<16>(e::gather<uint16_t, 16>(
            reinterpret_cast<const uint16_t *>(w), wo));
        const auto dx = esimd_half_decode<1>(e::gather<uint16_t, 1>(
            reinterpret_cast<const uint16_t *>(x),
            e::simd<uint32_t, 1>(uint32_t(xo))));
        const F part = ex::fma(ex::fma(d, F(dx[0]), F(0.f)),
                               e::convert<float>(dot), F(0.f));
        values.select<16, 1>(cell * 16) += part;
      }
    }
    if (bin == 0) even = values;
    else if (bin == 1) odd = values;
    else if (bin % 2 == 0) even += values;
    else odd += values;
  }
  e::simd<float, 256> total = even + odd;
  e::simd<float, 128> s8 = total.select<128, 1>(0) + total.select<128, 1>(128);
  e::simd<float, 64> s4 = s8.select<64, 1>(0) + s8.select<64, 1>(64);
  e::simd<float, 32> s2 = s4.select<32, 1>(0) + s4.select<32, 1>(32);
  return s2.select<16, 1>(0) + s2.select<16, 1>(16);
}
e::simd<uint32_t, 8> lookup8(e::simd<uint32_t, 8> packed, int shift) SYCL_ESIMD_FUNCTION {
  e::simd<uint32_t, 8> out = 0u;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    const e::simd<uint32_t, 8> index = (packed >> (8 * byte + shift)) & 15u;
    e::simd<uint32_t, 8> low = 0xbfad9881u, high = 0x26190d01u;
    low.merge(e::simd<uint32_t, 8>(0xf6eaddcfu), (index & 4u) != 0u);
    high.merge(e::simd<uint32_t, 8>(0x71594535u), (index & 4u) != 0u);
    low.merge(high, (index & 8u) != 0u);
    out |= ((low >> (8u * (index & 3u))) & 255u) << (8 * byte);
  }
  return out;
}
template <int Blocks>
e::simd<float, 8> xmx_tile8(const uint8_t *w, const uint8_t *x) SYCL_ESIMD_FUNCTION {
  const e::simd<uint32_t, 8> row(0, 1);
  e::simd<float, 128> even = 0.f, odd = 0.f;
#pragma unroll
  for (int bin = 0; bin < 8; ++bin) {
    e::simd<float, 128> values = 0.f;
#pragma unroll 1
    for (int first = 0; first < Blocks; first += 16) {
#pragma unroll 1
      for (int cell = 0; cell < 16; ++cell) {
        const int block = first + bin * 2 + cell / 8, group = cell % 8;
        if (block >= Blocks) continue;
        const e::simd<uint32_t, 8> wo = row * uint32_t(Blocks * 136) + uint32_t(block * 136);
        e::simd<int32_t, 128> b;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const e::simd<uint32_t, 8> codes = e::gather<uint32_t, 8>(
              reinterpret_cast<const uint32_t *>(w),
              wo + uint32_t(8 + group * 16 + 4 * j));
          auto low = lookup8(codes, 0), high = lookup8(codes, 4);
          b.select<8, 1>(j * 16) = low.bit_cast_view<int32_t>();
          b.select<8, 1>(j * 16 + 8) = low.bit_cast_view<int32_t>();
          b.select<8, 1>((j + 4) * 16) = high.bit_cast_view<int32_t>();
          b.select<8, 1>((j + 4) * 16 + 8) = high.bit_cast_view<int32_t>();
        }
        const int xo = (block * 8 + group) * 36;
        e::simd<int32_t, 8> a = e::gather<int32_t, 8>(
            reinterpret_cast<const int32_t *>(x),
            e::simd<uint32_t, 8>(4, 4) + uint32_t(xo));
        I dot16 = e::xmx::dpas<8, 1, int32_t, int32_t, int32_t, int32_t,
            e::xmx::dpas_argument_type::s8, e::xmx::dpas_argument_type::s8>(
                I(0), b, a);
        e::simd<int32_t, 8> dot = dot16.select<8, 1>(0);
        const e::simd<uint32_t, 8> sh = e::gather<uint16_t, 8>(
            reinterpret_cast<const uint16_t *>(w), wo + 2u);
        const e::simd<uint32_t, 8> sl = e::gather<uint8_t, 8>(w, wo + uint32_t(4 + group / 2));
        const e::simd<uint32_t, 8> scale = ((sl >> (4 * (group % 2))) & 15u) |
                        (((sh >> (2 * group)) & 3u) << 4);
        dot *= e::convert<int32_t>(scale) - 32;
        const e::simd<float, 8> d = esimd_half_decode<8>(e::gather<uint16_t, 8>(
            reinterpret_cast<const uint16_t *>(w), wo));
        const auto dx = esimd_half_decode<1>(e::gather<uint16_t, 1>(
            reinterpret_cast<const uint16_t *>(x),
            e::simd<uint32_t, 1>(uint32_t(xo))));
        const e::simd<float, 8> part = ex::fma(ex::fma(d, e::simd<float, 8>(dx[0]), e::simd<float, 8>(0.f)),
                               e::convert<float>(dot), e::simd<float, 8>(0.f));
        values.select<8, 1>(cell * 8) += part;
      }
    }
    if (bin == 0) even = values;
    else if (bin == 1) odd = values;
    else if (bin % 2 == 0) even += values;
    else odd += values;
  }
  e::simd<float, 128> total = even + odd;
  e::simd<float, 64> s8 = total.select<64, 1>(0) + total.select<64, 1>(64);
  e::simd<float, 32> s4 = s8.select<32, 1>(0) + s8.select<32, 1>(32);
  e::simd<float, 16> s2 = s4.select<16, 1>(0) + s4.select<16, 1>(16);
  return s2.select<8, 1>(0) + s2.select<8, 1>(8);
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
