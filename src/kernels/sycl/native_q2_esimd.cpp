// Q2_0 native Q8_1 projections. Block layout and virtual lane mapping are
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
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {
namespace e = sycl::ext::intel::esimd;
namespace ex = sycl::ext::intel::experimental::esimd;
using namespace sycl_backend;
constexpr int V = 16;
using U = e::simd<uint32_t, V>;
using I = e::simd<int32_t, V>;
using F = e::simd<float, V>;

I unpack(U code) SYCL_ESIMD_FUNCTION {
  U bytes = (code & 3u) | ((code & 12u) << 6) |
            ((code & 48u) << 12) | ((code & 192u) << 18);
  // Subtract one independently in every byte, without borrowing between bytes.
  // Clearing the temporary high bit gives signed codes -1, 0, 1, 2.
  bytes = ((bytes | 0x80808080u) - 0x01010101u) ^ 0x80808080u;
  return bytes.bit_cast_view<int32_t>();
}
F block_half(const uint8_t *w, const uint8_t *x, U block, int blocks)
    SYCL_ESIMD_FUNCTION {
  const U lane(0, 1), half = lane % 2u;
  const e::simd_mask<V> valid = block < uint32_t(blocks);
  const U wo = block * 18u, xo = (block * 2u + half) * 36u;
  const F d = esimd_half_decode<V>(e::gather<uint16_t, V>(
      reinterpret_cast<const uint16_t *>(w), wo, valid));
  const F dx = esimd_half_decode<V>(e::gather<uint16_t, V>(
      reinterpret_cast<const uint16_t *>(x), xo, valid));
  I dot = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const U codes = e::gather<uint16_t, V>(
        reinterpret_cast<const uint16_t *>(w), wo + 2u + half * 8u + 2u * k,
        valid);
    const I low = e::gather<int32_t, V>(
        reinterpret_cast<const int32_t *>(x), xo + 4u + 8u * k, valid);
    const I high = e::gather<int32_t, V>(
        reinterpret_cast<const int32_t *>(x), xo + 8u + 8u * k, valid);
    dot = e::dp4a<int32_t>(dot, unpack(codes & 255u), low);
    dot = e::dp4a<int32_t>(dot, unpack(codes >> 8), high);
  }
  // Preserve both product boundaries before the virtual-lane accumulation.
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
  for (int first = 0; first < blocks; first += 64) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (first + j * 8 < blocks)
        acc[j] += block_half(w, x, U(first + j * 8) + lane / 2u, blocks);
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

// Sixteen weight rows form DPAS's columns. Each Q2_0 half already packs
// thirty-two unsigned two-bit values; subtract the integer activation sum
// in the initial accumulator to recover codes -1, 0, 1, 2 exactly.
template <int Blocks>
F xmx_tile(const uint8_t *w, const uint8_t *x) SYCL_ESIMD_FUNCTION {
  const U row(0, 1);
  e::simd<float, 256> even = 0.f, odd = 0.f;
#pragma unroll
  for (int bin = 0; bin < 8; ++bin) {
    e::simd<float, 256> values = 0.f;
#pragma unroll 1
    for (int cell = 0; cell < 16 && bin * 16 + cell < 2 * Blocks; ++cell) {
      const int half = bin * 16 + cell;
      const U wo = row * uint32_t(Blocks * 18) + uint32_t(half / 2 * 18);
      const U packed = wo + 2u + uint32_t(half % 2 * 8);
      e::simd<uint32_t, 32> b;
      b.select<16, 1>(0) = e::convert<uint32_t>(e::gather<uint16_t, 16>(
          reinterpret_cast<const uint16_t *>(w), packed)) |
          (e::convert<uint32_t>(e::gather<uint16_t, 16>(
              reinterpret_cast<const uint16_t *>(w), packed + 2u)) << 16);
      b.select<16, 1>(16) = e::convert<uint32_t>(e::gather<uint16_t, 16>(
          reinterpret_cast<const uint16_t *>(w), packed + 4u)) |
          (e::convert<uint32_t>(e::gather<uint16_t, 16>(
              reinterpret_cast<const uint16_t *>(w), packed + 6u)) << 16);
      e::simd<int32_t, 8> a = e::gather<int32_t, 8>(
          reinterpret_cast<const int32_t *>(x),
          e::simd<uint32_t, 8>(4, 4) + uint32_t(half * 36));
      const auto sums = e::dp4a<int32_t>(e::simd<int32_t, 8>(0),
          e::simd<int32_t, 8>(0x01010101), a);
      const I correction(-e::reduce<int32_t>(sums, std::plus<>()));
      const I dot = e::xmx::dpas<8, 1, int32_t, int32_t, uint32_t, int32_t,
          e::xmx::dpas_argument_type::u2, e::xmx::dpas_argument_type::s8>(
              correction, b, a);
      const F d = esimd_half_decode<16>(e::gather<uint16_t, 16>(
          reinterpret_cast<const uint16_t *>(w), wo));
      const auto dx = esimd_half_decode<1>(e::gather<uint16_t, 1>(
          reinterpret_cast<const uint16_t *>(x),
          e::simd<uint32_t, 1>(uint32_t(half * 36))));
      const F part = ex::fma(ex::fma(d, F(dx[0]), F(0.f)),
                             e::convert<float>(dot), F(0.f));
      values.select<16, 1>(cell * 16) = F(0.f) + part;
    }
    if (bin == 0) even = values;
    else if (bin == 1) odd = values;
    else if (bin % 2 == 0) even += values;
    else odd += values;
  }
  // Cells are contiguous sixteen-row vectors. Keep the original virtual
  // lane additions, including zero bins, then its XOR reduction tree.
  e::simd<float, 256> total = even + odd;
  e::simd<float, 128> s8 = total.select<128, 1>(0) + total.select<128, 1>(128);
  e::simd<float, 64> s4 = s8.select<64, 1>(0) + s8.select<64, 1>(64);
  e::simd<float, 32> s2 = s4.select<32, 1>(0) + s4.select<32, 1>(32);
  return s2.select<16, 1>(0) + s2.select<16, 1>(16);
}

template <int Blocks, bool Down>
void grouped_xmx(const NativeExpertLayout &layout,
                 const unsigned long long *pointers, const int32_t *starts,
                 const int32_t *count, const int32_t *destinations,
                 const int32_t *tokens, int groups, int entries,
                 const void *activation, float *gate, float *up, float *out,
                 void *stream) {
  const int rows = Down ? int(layout.n_embd) : int(2 * layout.n_ff);
  const auto *x = static_cast<const uint8_t *>(activation);
  const size_t tiles = size_t(rows / 16), padded = (tiles + 15) & ~size_t(15);
  queue_for(stream).parallel_for(
      sycl::nd_range<2>({size_t(groups), padded}, {1, 16}),
      [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
        const int group = int(item.get_global_id(0));
        const int row = int(item.get_global_id(1)) * 16;
        const int active = *count;
        if (row >= rows || active < 0 || active > groups || group >= active)
          return;
        const int begin = starts[group], end = starts[group + 1];
        if (begin < 0 || end < begin || end > entries || !pointers[group])
          return;
        const auto *blob = reinterpret_cast<const uint8_t *>(pointers[group]);
        const auto *w = blob + (Down ? layout.down_off : 0) +
                        size_t(row) * Blocks * 18;
        for (int entry = begin; entry < end; ++entry) {
          const int input = Down ? entry : tokens[entry];
          if (input < 0 || input >= entries) {
            if constexpr (!Down) {
              float *result = row < layout.n_ff ? gate : up;
              e::scatter<float, 16>(result + size_t(entry) * layout.n_ff + row % layout.n_ff,
                                    U(0, 4), F(0.f));
            }
            continue;
          }
          const F result = xmx_tile<Blocks>(w, x + size_t(input) * Blocks * 2 * 36);
          if constexpr (Down) {
            const int dst = destinations[entry];
            if (dst >= 0 && dst < entries)
              e::scatter<float, 16>(out + size_t(dst) * layout.n_embd + row,
                                    U(0, 4), result);
          } else {
            float *target = row < layout.n_ff ? gate : up;
            e::scatter<float, 16>(target + size_t(entry) * layout.n_ff + row % layout.n_ff,
                                  U(0, 4), result);
          }
        }
      });
}
} // namespace
void native_q2_0_esimd(const void *weights, const void *activation, float *out,
                       int width, int rows, int columns, void *stream) {
  const int blocks = width / 64;
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
            result[row] = row_dot(w + row * blocks * 18, input, blocks);
        });
  }
}
void native_q2_grouped_esimd(bool down, const NativeExpertLayout &layout,
                             const unsigned long long *pointers,
                             const int32_t *starts, const int32_t *count,
                             const int32_t *destinations, const int32_t *tokens,
                             int groups, int entries, const void *activation,
                             float *gate, float *up, float *out, void *stream) {
  static const bool xmx = [] {
    const char *value = std::getenv("STRATA_SYCL_Q2_XMX");
    return value == nullptr || std::strcmp(value, "0") != 0;
  }();
  if (xmx && layout.n_embd == 2560 && layout.n_ff == 640) {
    if (down)
      grouped_xmx<10, true>(layout, pointers, starts, count, destinations,
                            tokens, groups, entries, activation, gate, up, out, stream);
    else
      grouped_xmx<40, false>(layout, pointers, starts, count, destinations,
                             tokens, groups, entries, activation, gate, up, out, stream);
    return;
  }
  const int rows = down ? int(layout.n_embd) : int(2 * layout.n_ff);
  const int width = down ? int(layout.n_ff) : int(layout.n_embd);
  const int blocks = width / 64;
  const auto *x = static_cast<const uint8_t *>(activation);
  const size_t padded = (size_t(rows) + 15) & ~size_t(15);
  queue_for(stream).parallel_for(
      sycl::nd_range<2>({size_t(groups), padded}, {1, 16}),
      [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
        const int group = int(item.get_global_id(0));
        const int row = int(item.get_global_id(1));
        const int active = *count;
        if (row >= rows || active < 0 || active > groups || group >= active)
          return;
        const int begin = starts[group], end = starts[group + 1];
        if (begin < 0 || end < begin || end > entries || !pointers[group])
          return;
        const auto *blob = reinterpret_cast<const uint8_t *>(pointers[group]);
        const auto *w = blob + (down ? layout.down_off : 0) +
                        size_t(row) * blocks * 18;
        for (int entry = begin; entry < end; ++entry) {
          const int input = down ? entry : tokens[entry];
          if (input < 0 || input >= entries) {
            // The SPMD path writes zero projections for invalid token rows.
            if (!down) {
              float *result = row < layout.n_ff ? gate : up;
              result[size_t(entry) * layout.n_ff + row % layout.n_ff] = 0.f;
            }
            continue;
          }
          const float sum =
              row_dot(w, x + size_t(input) * (width / 32) * 36, blocks);
          if (down) {
            const int dst = destinations[entry];
            if (dst >= 0 && dst < entries)
              out[size_t(dst) * layout.n_embd + row] = sum;
          } else {
            float *result = row < layout.n_ff ? gate : up;
            result[size_t(entry) * layout.n_ff + row % layout.n_ff] = sum;
          }
        }
      });
}
} // namespace strata::kernels
