// IQ native Q8_1 expert projections. Block layout and virtual lane mapping are
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

// Packed IQ expert products with the original 128 virtual-lane reduction.
// Layouts/codebooks follow the pinned GGML definitions in iq_codebooks.hpp.
#include "strata/artifact/iq_codebooks.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/sycl/esimd_half.hpp"
#include "strata/sycl/launch.hpp"
#include "strata/sycl/native_mmq.hpp"
#include <cstdlib>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>

namespace strata::kernels {
namespace {
namespace e = sycl::ext::intel::esimd;
namespace ex = sycl::ext::intel::experimental::esimd;
using namespace sycl_backend;
using U = e::simd<uint32_t, 16>;
using I = e::simd<int32_t, 16>;
using F = e::simd<float, 16>;
bool enabled() {
  static const bool on = [] {
    const char *v = std::getenv("STRATA_SYCL_IQ_ESIMD");
    return v == nullptr || std::atoi(v) != 0;
  }();
  return on;
}
U word(const uint8_t *p, U offset, e::simd_mask<16> valid) SYCL_ESIMD_FUNCTION {
  return e::convert<uint32_t>(e::gather<uint16_t, 16>(
             reinterpret_cast<const uint16_t *>(p), offset, valid)) |
         (e::convert<uint32_t>(e::gather<uint16_t, 16>(
              reinterpret_cast<const uint16_t *>(p), offset + 2u, valid))
          << 16);
}
U signed_word(U positive, U signs) SYCL_ESIMD_FUNCTION {
  U mask = 0u;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte)
    mask |= ((0u - ((signs >> byte) & 1u)) & 255u) << (byte * 8);
  const U negative = (0x80808080u - positive) ^ 0x80808080u;
  return (positive & ~mask) | (negative & mask);
}
U nonlinear(U packed, int shift) SYCL_ESIMD_FUNCTION {
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
template <int Type>
constexpr int block_bytes = Type == 18   ? 98
                            : Type == 21 ? 110
                            : Type == 22 ? 82
                            : Type == 23 ? 136
                                         : 18;
template <int Type> constexpr int block_width = Type == 20 ? 32 : 256;
template <int Type>
F part(const uint8_t *w, const uint8_t *x, U tid,
       int blocks) SYCL_ESIMD_FUNCTION {
  constexpr int T = Type == 20 ? 2 : 8;
  const U block = tid / T, group = tid % T, wo = block * block_bytes<Type>;
  const U xo =
      (block * (block_width<Type> / 32) + (Type == 20 ? U(0u) : group)) * 36u;
  const auto valid = block < uint32_t(blocks);
  const F d = esimd_half_decode<16>(e::gather<uint16_t, 16>(
      reinterpret_cast<const uint16_t *>(w), wo, valid));
  const F dx = esimd_half_decode<16>(e::gather<uint16_t, 16>(
      reinterpret_cast<const uint16_t *>(x), xo, valid));
  I dot = 0;
  F scaled;
  if constexpr (Type == 20 || Type == 23) {
    constexpr int Words = Type == 20 ? 2 : 4;
    const U offset = wo + (Type == 20 ? 2u + group * 8u : 8u + group * 16u);
#pragma unroll
    for (int j = 0; j < Words; ++j) {
      const U code = word(w, offset + 4u * j, valid);
      const U a = xo + 4u + (Type == 20 ? group * 8u : U(0u)) + 4u * j;
      U wl = nonlinear(code, 0), wh = nonlinear(code, 4);
      dot =
          e::dp4a<int32_t>(dot, I(wl.bit_cast_view<int32_t>()),
                           e::gather<int32_t, 16>(
                               reinterpret_cast<const int32_t *>(x), a, valid));
      dot = e::dp4a<int32_t>(
          dot, I(wh.bit_cast_view<int32_t>()),
          e::gather<int32_t, 16>(reinterpret_cast<const int32_t *>(x), a + 16u,
                                 valid));
    }
    if constexpr (Type == 23) {
      const U hi = e::gather<uint16_t, 16>(
          reinterpret_cast<const uint16_t *>(w), wo + 2u, valid);
      const U lo = e::gather<uint8_t, 16>(w, wo + 4u + group / 2u, valid);
      const U scale = ((lo >> (4u * (group % 2u))) & 15u) |
                      (((hi >> (2u * group)) & 3u) << 4);
      dot *= e::convert<int32_t>(scale) - 32;
    }
    scaled = e::convert<float>(dot);
  } else if constexpr (Type == 22) {
    const U high = e::gather<uint8_t, 16>(w, wo + 66u + group, valid);
    const U sc = e::gather<uint8_t, 16>(w, wo + 74u + group, valid);
    I dots[2] = {I(0), I(0)};
#pragma unroll
    for (int cell = 0; cell < 4; ++cell) {
      const U index = e::convert<uint32_t>(e::gather<uint8_t, 16>(
                          w, wo + 2u + group * 4u + cell, valid)) |
                      (((high >> (2 * cell)) & 3u) << 8);
      const auto grid =
          e::gather<uint64_t, 16>(strata::iq2_s_grid, index * 8u, valid);
      const U signs =
          e::gather<uint8_t, 16>(w, wo + 34u + group * 4u + cell, valid);
      const U a = xo + 4u + cell * 8u;
      U wl = signed_word(e::convert<uint32_t>(grid), signs);
      U wh = signed_word(e::convert<uint32_t>(grid >> 32), signs >> 4);
      dots[cell / 2] =
          e::dp4a<int32_t>(dots[cell / 2], I(wl.bit_cast_view<int32_t>()),
                           e::gather<int32_t, 16>(
                               reinterpret_cast<const int32_t *>(x), a, valid));
      dots[cell / 2] = e::dp4a<int32_t>(
          dots[cell / 2], I(wh.bit_cast_view<int32_t>()),
          e::gather<int32_t, 16>(reinterpret_cast<const int32_t *>(x), a + 4u,
                                 valid));
    }
    dot = dots[0] * e::convert<int32_t>(2u * (sc & 15u) + 1u) +
          dots[1] * e::convert<int32_t>(2u * (sc >> 4) + 1u);
    scaled = e::convert<float>(dot) * .125f;
  } else {
    U extra = 0u, high = 0u, scale;
    if constexpr (Type == 18) {
      extra = word(w, wo + 66u + 4u * group, valid);
      scale = 2u * (extra >> 28) + 1u;
    } else {
      high = e::gather<uint8_t, 16>(w, wo + 66u + group, valid);
      const U sc = e::gather<uint8_t, 16>(w, wo + 106u + group / 2u, valid);
      scale = 2u * ((sc >> (4u * (group % 2u))) & 15u) + 1u;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      U index = e::gather<uint8_t, 16>(w, wo + 2u + group * 8u + j, valid);
      U signs, grid;
      if constexpr (Type == 18) {
        grid = e::gather<uint32_t, 16>(strata::iq3_xxs_grid, index * 4u, valid);
        signs = (extra >> (7 * (j / 2))) & 127u;
        signs |= (e::cbit(signs) & 1u) << 7;
      } else {
        index |= ((high >> j) & 1u) << 8;
        grid = e::gather<uint32_t, 16>(strata::iq3_s_grid, index * 4u, valid);
        signs = e::gather<uint8_t, 16>(w, wo + 74u + group * 4u + j / 2, valid);
      }
      U ws = signed_word(grid, signs >> (4 * (j % 2)));
      dot = e::dp4a<int32_t>(
          dot, I(ws.bit_cast_view<int32_t>()),
          e::gather<int32_t, 16>(reinterpret_cast<const int32_t *>(x),
                                 xo + 4u + j * 4u, valid));
    }
    scaled = e::convert<float>(dot * e::convert<int32_t>(scale));
    if constexpr (Type == 18)
      scaled *= .25f;
  }
  F result = ex::fma(ex::fma(d, dx, F(0.f)), scaled, F(0.f));
  result.merge(F(0.f), !valid);
  return result;
}
template <int Type>
float row_dot(const uint8_t *w, const uint8_t *x,
              int width) SYCL_ESIMD_FUNCTION {
  constexpr int T = Type == 20 ? 2 : 8;
  const int blocks = width / block_width<Type>;
  const U lane(0, 1);
  F acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    acc[j] = 0.f;
  for (int first = 0; first < blocks * T; first += 128) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (first + j * 16 < blocks * T)
        acc[j] += part<Type>(w, x, U(first + j * 16) + lane, blocks);
  }
  F a = ((acc[0] + acc[2]) + acc[4]) + acc[6];
  F b = ((acc[1] + acc[3]) + acc[5]) + acc[7];
  F s = a + b;
  e::simd<float, 8> s8 = s.select<8, 1>(0) + s.select<8, 1>(8);
  e::simd<float, 4> s4 = s8.select<4, 1>(0) + s8.select<4, 1>(4);
  e::simd<float, 2> s2 = s4.select<2, 1>(0) + s4.select<2, 1>(2);
  return s2[0] + s2[1];
}
template <int Type>
void grouped(bool down, NativeExpertLayout L, const unsigned long long *ptr,
             const int32_t *start, const int32_t *count, const int32_t *dest,
             const int32_t *tokens, int groups, int entries, const void *act,
             float *gate, float *up, float *out, void *stream) {
  const int rows = down ? L.n_embd : L.n_ff * 2,
            width = down ? L.n_ff : L.n_embd;
  const size_t stride = size_t(width / block_width<Type>) * block_bytes<Type>;
  queue_for(stream).parallel_for(
      sycl::nd_range<2>({size_t(groups), (size_t(rows) + 15) & ~size_t(15)},
                        {1, 16}),
      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const int g = it.get_global_id(0), row = it.get_global_id(1),
                  n = *count;
        if (row >= rows || n < 0 || n > groups || g >= n)
          return;
        const int first = start[g], last = start[g + 1];
        if (first < 0 || last < first || last > entries || !ptr[g])
          return;
        const auto *w = reinterpret_cast<const uint8_t *>(ptr[g]) +
                        (down ? L.down_off : 0) + row * stride;
        for (int r = first; r < last; ++r) {
          const int input = down ? r : tokens[r];
          float value = 0.f;
          if (input >= 0 && input < entries)
            value = row_dot<Type>(w,
                                  static_cast<const uint8_t *>(act) +
                                      size_t(input) * (width / 32) * 36,
                                  width);
          if (down) {
            const int d = dest[r];
            if (d >= 0 && d < entries)
              out[size_t(d) * L.n_embd + row] = value;
          } else
            (row < L.n_ff ? gate : up)[size_t(r) * L.n_ff + row % L.n_ff] =
                value;
        }
      });
}
template <int Type> void prompt(NativeMmq p, void *stream) {
  constexpr int Tile = 8;
  const size_t tiles = (p.max_rows + Tile - 1) / Tile;
  const size_t rows = (size_t(p.rows) + 15) & ~size_t(15);
  const size_t stride = size_t(p.cols / block_width<Type>) * block_bytes<Type>;
  const size_t xs = ((size_t(p.cols) + 511) / 512 * 512) / 32 * 36;
  queue_for(stream).parallel_for(
      sycl::nd_range<2>({size_t(p.experts) * tiles, rows}, {1, 16}),
      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const size_t g = it.get_global_id(0), row = it.get_global_id(1);
        if (row >= size_t(p.rows))
          return;
        const int expert = g / tiles;
        const int begin = p.bounds[expert], end = p.bounds[expert + 1];
        if (begin < 0 || end < begin || end > p.total_rows)
          return;
        const auto *w = static_cast<const uint8_t *>(p.weights) +
                        expert * p.expert_bytes + row * stride;
        const int64_t first = int64_t(begin) + int64_t(g % tiles) * Tile;
        for (int64_t r = first; r < end && r < first + Tile; ++r) {
          const int64_t dst = p.destinations ? p.destinations[r] : r;
          if (dst >= 0 && dst < p.total_rows)
            p.output[size_t(dst) * p.ld_output + row] = row_dot<Type>(
                w, static_cast<const uint8_t *>(p.activation) + size_t(r) * xs,
                p.cols);
        }
      });
}
} // namespace
bool native_iq_grouped_esimd(int type, bool down, const NativeExpertLayout &L,
                             const unsigned long long *ptr,
                             const int32_t *start, const int32_t *count,
                             const int32_t *dest, const int32_t *tokens,
                             int groups, int entries, const void *act,
                             float *gate, float *up, float *out, void *stream) {
  // Keep the existing gate/up weight reuse for wider verification windows.
  // Their entry capacities exceed the single-token router's ten entries.
  if (!enabled() || (!down && entries > 16)) return false;
#define RUN(T)                                                                 \
  case T:                                                                      \
    grouped<T>(down, L, ptr, start, count, dest, tokens, groups, entries, act, \
               gate, up, out, stream);                                         \
    return true
  switch (type) {
    RUN(18);
    RUN(20);
    RUN(21);
    RUN(22);
    RUN(23);
  default:
    return false;
  }
#undef RUN
}
bool native_iq_mmq_esimd(const NativeMmq &p, void *stream) {
  // On B570, a single routed row benefits from explicit SIMD. For multiple
  // rows the SPMD path reuses decoded weights across its eight-row tile.
  if (p.max_rows != 1 || p.type == 20 || p.type == 23)
    return false;
  if (!enabled()) return false;
#define RUN(T)                                                                 \
  case T:                                                                      \
    prompt<T>(p, stream);                                                      \
    return true
  switch (p.type) {
    RUN(18);
    RUN(20);
    RUN(21);
    RUN(22);
    RUN(23);
  default:
    return false;
  }
#undef RUN
}
} // namespace strata::kernels
