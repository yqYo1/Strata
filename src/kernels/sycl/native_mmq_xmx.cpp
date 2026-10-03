// Q2_0 and IQ native Q8_1 projections. Block layouts and codebooks are
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

// Experimental routed prompt tiles share sixteen weight rows across 1/2/4/8
// activations. Weights retain their GGUF layout. The default FP32 sum preserves
// the preceding virtual-lane tree; an optional linear sum compares its cost.
#include "strata/artifact/iq_codebooks.hpp"
#include "strata/sycl/esimd_half.hpp"
#include "strata/sycl/launch.hpp"
#include "strata/sycl/native_mmq.hpp"
#include <cstdlib>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace strata::kernels {
namespace {
namespace e = sycl::ext::intel::esimd;
namespace ex = sycl::ext::intel::experimental::esimd;
using namespace sycl_backend;
using U = e::simd<uint32_t, 16>;
using F = e::simd<float, 16>;
U word(const uint8_t *p, U offsets, e::simd_mask<16> mask) SYCL_ESIMD_FUNCTION {
  return e::convert<uint32_t>(e::gather<uint16_t, 16>(
             reinterpret_cast<const uint16_t *>(p), offsets, mask)) |
         (e::convert<uint32_t>(e::gather<uint16_t, 16>(
              reinterpret_cast<const uint16_t *>(p), offsets + 2u, mask))
          << 16);
}
U nonlinear(U packed, int shift) SYCL_ESIMD_FUNCTION {
  U out = 0;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    const U index = (packed >> (8 * byte + shift)) & 15u;
    U low = 0xbfad9881u, high = 0x26190d01u;
    low.merge(U(0xf6eaddcfu), (index & 4u) != 0);
    high.merge(U(0x71594535u), (index & 4u) != 0);
    low.merge(high, (index & 8u) != 0);
    out |= ((low >> (8u * (index & 3u))) & 255u) << (8 * byte);
  }
  return out;
}
U signed_word(U positive, U signs) SYCL_ESIMD_FUNCTION {
  U mask = 0;
#pragma unroll
  for (int byte = 0; byte < 4; ++byte)
    mask |= ((0u - ((signs >> byte) & 1u)) & 255u) << (byte * 8);
  const U negative = (0x80808080u - positive) ^ 0x80808080u;
  return (positive & ~mask) | (negative & mask);
}
template <int Type>
constexpr int width = Type == 42   ? 64
                      : Type == 20 ? 32
                                   : 256;
template <int Type>
constexpr int bytes = Type == 18   ? 98
                      : Type == 21 ? 110
                      : Type == 22 ? 82
                      : Type == 23 ? 136
                                   : 18;
template <int Type, int Tile>
__attribute__((always_inline)) inline e::simd<float, 16 * Tile>
unit_product(const uint8_t *w, const uint8_t *x, U rows, e::simd_mask<16> valid,
             size_t stride, size_t xs, int64_t first, int end, int unit,
             int split) SYCL_ESIMD_FUNCTION {
  e::simd<float, 16 * Tile> result;
  const int group = unit % (width<Type> / 32);
  const U wo = rows * uint32_t(stride) +
               uint32_t(unit / (width<Type> / 32) * bytes<Type>);
  const F scale = esimd_half_decode<16>(e::gather<uint16_t, 16>(
      reinterpret_cast<const uint16_t *>(w), wo, valid));
  e::simd<uint32_t, Type == 42 ? 32 : 128> weights;
  if constexpr (Type == 42) {
    weights.template select<16, 1>(0) =
        word(w, wo + uint32_t(2 + (unit % 2) * 8), valid);
    weights.template select<16, 1>(16) =
        word(w, wo + uint32_t(6 + (unit % 2) * 8), valid);
  } else if constexpr (Type == 20 || Type == 23) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const U codes = word(
          w, wo + uint32_t((Type == 20 ? 2 : 8 + group * 16) + k * 4), valid);
      weights.template select<16, 1>(k * 16) = nonlinear(codes, 0);
      weights.template select<16, 1>((k + 4) * 16) = nonlinear(codes, 4);
    }
  }
  e::simd<int32_t, 16> factor = 1, factor_hi = 1;
  if constexpr (Type == 23) {
    const U hi = e::gather<uint16_t, 16>(reinterpret_cast<const uint16_t *>(w),
                                         wo + 2u, valid);
    const U lo =
        e::gather<uint8_t, 16>(w, wo + 4u + uint32_t(group / 2), valid);
    factor = e::convert<int32_t>(((lo >> (4 * (group % 2))) & 15u) |
                                 (((hi >> (2 * group)) & 3u) << 4)) -
             32;
  } else if constexpr (Type == 22) {
    const U high = e::gather<uint8_t, 16>(w, wo + 66u + uint32_t(group), valid);
    const U sc = e::gather<uint8_t, 16>(w, wo + 74u + uint32_t(group), valid);
    factor = e::convert<int32_t>(2u * (sc & 15u) + 1u);
    factor_hi = e::convert<int32_t>(2u * (sc >> 4) + 1u);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const U index = e::convert<uint32_t>(e::gather<uint8_t, 16>(
                          w, wo + uint32_t(2 + group * 4 + k), valid)) |
                      (((high >> (2 * k)) & 3u) << 8);
      const auto grid =
          e::gather<uint64_t, 16>(strata::iq2_s_grid, index * 8u, valid);
      const U signs =
          e::gather<uint8_t, 16>(w, wo + uint32_t(34 + group * 4 + k), valid);
      weights.template select<16, 1>(k * 32) =
          signed_word(e::convert<uint32_t>(grid), signs);
      weights.template select<16, 1>(k * 32 + 16) =
          signed_word(e::convert<uint32_t>(grid >> 32), signs >> 4);
    }
  } else if constexpr (Type == 18 || Type == 21) {
    U extra = 0, high = 0;
    if constexpr (Type == 18) {
      extra = word(w, wo + uint32_t(66 + 4 * group), valid);
      factor = e::convert<int32_t>(2u * (extra >> 28) + 1u);
    } else {
      high = e::gather<uint8_t, 16>(w, wo + uint32_t(66 + group), valid);
      const U sc =
          e::gather<uint8_t, 16>(w, wo + uint32_t(106 + group / 2), valid);
      factor = e::convert<int32_t>(2u * ((sc >> (4 * (group % 2))) & 15u) + 1u);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      U index =
          e::gather<uint8_t, 16>(w, wo + uint32_t(2 + group * 8 + k), valid);
      U signs, grid;
      if constexpr (Type == 18) {
        grid = e::gather<uint32_t, 16>(strata::iq3_xxs_grid, index * 4u, valid);
        signs = (extra >> (7 * (k / 2))) & 127u;
        signs |= (e::cbit(signs) & 1u) << 7;
      } else {
        index |= ((high >> k) & 1u) << 8;
        grid = e::gather<uint32_t, 16>(strata::iq3_s_grid, index * 4u, valid);
        signs = e::gather<uint8_t, 16>(w, wo + uint32_t(74 + group * 4 + k / 2),
                                       valid);
      }
      weights.template select<16, 1>(k * 16) =
          signed_word(grid, signs >> (4 * (k % 2)));
    }
  }
  e::simd<int32_t, 8 * Tile> act = 0;
  e::simd<int32_t, 16 * Tile> initial = 0;
  e::simd<float, Tile> dx = 0.f;
#pragma unroll
  for (int t = 0; t < Tile; ++t) {
    if (first + t < end) {
      const auto *input = x + size_t(first + t) * xs + size_t(unit) * 36;
      const e::simd<int32_t, 8> a = e::gather<int32_t, 8>(
          reinterpret_cast<const int32_t *>(input), e::simd<uint32_t, 8>(4, 4));
      act.template select<8, 1>(t * 8) = a;
      if constexpr (Type == 42) {
        const auto v = e::dp4a<int32_t>(e::simd<int32_t, 8>(0),
                                        e::simd<int32_t, 8>(0x01010101), a);
        initial.template select<16, 1>(t * 16) =
            -e::reduce<int32_t>(v, std::plus<>());
      }
      dx[t] = esimd_half_decode<1>(
          e::gather<uint16_t, 1>(reinterpret_cast<const uint16_t *>(input),
                                 e::simd<uint32_t, 1>(0)))[0];
    }
  }
  if constexpr (Type == 20) {
    if (split >= 0) {
#pragma unroll
      for (int t = 0; t < Tile; ++t) {
        act.template select<2, 1>(t * 8 + (split == 0 ? 2 : 0)) = 0;
        act.template select<2, 1>(t * 8 + (split == 0 ? 6 : 4)) = 0;
      }
    }
  }
  e::simd<int32_t, 16 * Tile> dot;
  if constexpr (Type == 22) {
    e::simd<int32_t, 8 * Tile> lo = 0, hi = 0;
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
      lo.template select<4, 1>(t * 8) = act.template select<4, 1>(t * 8);
      hi.template select<4, 1>(t * 8 + 4) =
          act.template select<4, 1>(t * 8 + 4);
    }
    auto low =
        e::xmx::dpas<8, Tile, int32_t, int32_t, uint32_t, int32_t,
                     e::xmx::dpas_argument_type::s8,
                     e::xmx::dpas_argument_type::s8>(initial, weights, lo);
    auto high =
        e::xmx::dpas<8, Tile, int32_t, int32_t, uint32_t, int32_t,
                     e::xmx::dpas_argument_type::s8,
                     e::xmx::dpas_argument_type::s8>(initial, weights, hi);
#pragma unroll
    for (int t = 0; t < Tile; ++t)
      dot.template select<16, 1>(t * 16) =
          e::simd<int32_t, 16>(low.template select<16, 1>(t * 16)) * factor +
          e::simd<int32_t, 16>(high.template select<16, 1>(t * 16)) * factor_hi;
  } else {
    dot = e::xmx::dpas<8, Tile, int32_t, int32_t, uint32_t, int32_t,
                       Type == 42 ? e::xmx::dpas_argument_type::u2
                                  : e::xmx::dpas_argument_type::s8,
                       e::xmx::dpas_argument_type::s8>(initial, weights, act);
#pragma unroll
    for (int t = 0; t < Tile; ++t)
      dot.template select<16, 1>(t * 16) *= factor;
  }
#pragma unroll
  for (int t = 0; t < Tile; ++t) {
    const F part = ex::fma(ex::fma(scale, F(dx[t]), F(0.f)),
                           e::convert<float>(e::simd<int32_t, 16>(
                               dot.template select<16, 1>(t * 16))) *
                               (Type == 18   ? .25f
                                : Type == 22 ? .125f
                                             : 1.f),
                           F(0.f));
    result.template select<16, 1>(t * 16) = part;
  }
  return result;
}
template <int Type, int Tile, bool Exact>
void launch(NativeMmq p, void *stream) {
  const size_t tiles = (p.max_rows + Tile - 1) / Tile;
  const size_t row_tiles = (p.rows + 15) / 16;
  const size_t padded = (row_tiles + 15) / 16 * 16;
  const size_t stride = size_t(p.cols / width<Type>) * bytes<Type>;
  const size_t xs = ((size_t(p.cols) + 511) / 512 * 512) / 32 * 36;
  queue_for(stream).parallel_for(
      sycl::nd_range<2>({size_t(p.experts) * tiles, padded}, {1, 16}),
      sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::grf_size<256>},
      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const size_t tile = it.get_global_id(0), row = it.get_global_id(1) * 16;
        if (row >= size_t(p.rows))
          return;
        const int expert = tile / tiles;
        const int begin = p.bounds[expert], end = p.bounds[expert + 1];
        if (begin < 0 || end < begin || end > p.total_rows)
          return;
        const int64_t first = int64_t(begin) + int64_t(tile % tiles) * Tile;
        if (first >= end)
          return;
        const U rows = U(uint32_t(row), 1);
        const e::simd_mask<16> valid = rows < uint32_t(p.rows);
        const auto *w =
            static_cast<const uint8_t *>(p.weights) + expert * p.expert_bytes;
        const auto *x = static_cast<const uint8_t *>(p.activation);
        e::simd<float, 16 * Tile> sum = 0.f;
        if constexpr (!Exact) {
#pragma unroll 1
          for (int unit = 0; unit < p.cols / 32; ++unit)
            sum += unit_product<Type, Tile>(w, x, rows, valid, stride, xs,
                                            first, end, unit, -1);
        } else {
          // Fold warps for each of sixteen lanes first, retaining just one
          // lane vector before the original subgroup XOR tree.
          constexpr int N = 16 * Tile;
          using V = e::simd<float, N>;
          e::simd<float, 16 * N> total = 0.f;
          const int units = p.cols / (Type == 20 ? 16 : 32);
#pragma unroll 1
          for (int cell = 0; cell < 16; ++cell) {
            V even = 0.f, odd = 0.f;
#pragma unroll 1
            for (int bin = 0; bin < 8; ++bin) {
              V partial = 0.f;
#pragma unroll 1
              for (int v = cell + bin * 16; v < units; v += 128)
                partial += unit_product<Type, Tile>(
                    w, x, rows, valid, stride, xs, first, end,
                    Type == 20 ? v / 2 : v, Type == 20 ? v % 2 : -1);
              if (bin % 2 == 0)
                even += partial;
              else
                odd += partial;
            }
            total.template select<N, 1>(cell * N) = even + odd;
          }
          e::simd<float, 8 * N> s8 = total.template select<8 * N, 1>(0) +
                                     total.template select<8 * N, 1>(8 * N);
          e::simd<float, 4 * N> s4 = s8.template select<4 * N, 1>(0) +
                                     s8.template select<4 * N, 1>(4 * N);
          e::simd<float, 2 * N> s2 = s4.template select<2 * N, 1>(0) +
                                     s4.template select<2 * N, 1>(2 * N);
          sum = s2.template select<N, 1>(0) + s2.template select<N, 1>(N);
        }
#pragma unroll
        for (int t = 0; t < Tile; ++t) {
          if (first + t < end) {
            const int dst =
                p.destinations ? p.destinations[first + t] : first + t;
            if (dst >= 0 && dst < p.total_rows)
              e::scatter<float, 16>(
                  p.output + size_t(dst) * p.ld_output + row, U(0, 4),
                  F(sum.template select<16, 1>(t * 16)), valid);
          }
        }
      });
}
template <int Tile, bool Exact> void dispatch(NativeMmq p, void *stream) {
  switch (p.type) {
#define TYPE(T)                                                                \
  case T:                                                                      \
    launch<T, Tile, Exact>(p, stream);                                         \
    break
    TYPE(18);
    TYPE(20);
    TYPE(21);
    TYPE(22);
    TYPE(23);
    TYPE(42);
#undef TYPE
  }
}
} // namespace
bool native_mmq_xmx(const NativeMmq &p, void *stream) {
  static const bool enabled = [] {
    const char *v = std::getenv("STRATA_SYCL_MMQ_XMX");
    return v && std::atoi(v) != 0;
  }();
  if (!enabled)
    return false;
  if (p.type != 18 && p.type != 20 && p.type != 21 && p.type != 22 &&
      p.type != 23 && p.type != 42)
    return false;
  const int block_width = p.type == 42 ? 64 : p.type == 20 ? 32 : 256;
  const int block_bytes = p.type == 18   ? 98
                          : p.type == 21 ? 110
                          : p.type == 22 ? 82
                          : p.type == 23 ? 136
                                         : 18;
  const uint64_t stride = uint64_t(p.cols / block_width) * block_bytes;
  if (uint64_t(p.rows) * stride > UINT32_MAX)
    return false;
  static const int tile = [] {
    const char *v = std::getenv("STRATA_SYCL_MMQ_XMX_TILE");
    return v ? std::atoi(v) : 4;
  }();
  static const bool exact = [] {
    const char *v = std::getenv("STRATA_SYCL_MMQ_XMX_EXACT");
    return !v || std::atoi(v) != 0;
  }();
  switch (tile) {
  case 1:
    if (exact)
      dispatch<1, true>(p, stream);
    else
      dispatch<1, false>(p, stream);
    break;
  case 2:
    if (exact)
      dispatch<2, true>(p, stream);
    else
      dispatch<2, false>(p, stream);
    break;
  case 8:
    if (exact)
      dispatch<8, true>(p, stream);
    else
      dispatch<8, false>(p, stream);
    break;
  default:
    if (exact)
      dispatch<4, true>(p, stream);
    else
      dispatch<4, false>(p, stream);
  }
  return true;
}
} // namespace strata::kernels
