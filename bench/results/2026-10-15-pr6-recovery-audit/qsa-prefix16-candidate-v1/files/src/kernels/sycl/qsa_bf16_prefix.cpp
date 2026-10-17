// Authored for Strata; repository LICENSE (MIT) applies.
// Independent typed online-attention implementation. Numerical design source:
// bench/results/2026-10-15-pr6-recovery-audit/qsa-framework-precision-plan-v1.ja.md.
// Live logical K8/V8 prefix decodes once to BF16; padding/unmapped rows are zero.
// No FP32 Q/O tensor or high/low matrix decomposition. Opt-in Mode0 long-prefill caller.
// Diagnostic-only bits are local through the tile loop and reduced once at end.
// Matrix operand arrays explicitly satisfy the extension's 4-byte base alignment.
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <cstddef>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace strata::kernels {
namespace {
namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace mi = sycl::ext::intel::experimental::matrix;
using BFloat = sycl::ext::oneapi::bfloat16;
constexpr int Dim = 256, Heads = 12, DefaultCells = 32, Workgroup = 128;
template<int Count> struct alignas(4) MatrixTile { BFloat values[Count]; };
static_assert(std::is_trivially_copyable_v<MatrixTile<16 * Dim>>);
static_assert(alignof(MatrixTile<16 * Dim>) >= 4);
static_assert(sizeof(MatrixTile<16 * Dim>) == 16 * Dim * sizeof(BFloat));
static_assert(sizeof(BFloat) == 2);
template<int Count> constexpr bool storage_contract() {
  using Tile = MatrixTile<Count>;
  static_assert(std::is_standard_layout_v<Tile> && std::is_trivially_copyable_v<Tile>);
  static_assert(alignof(Tile) >= 4 && offsetof(Tile, values) == 0);
  static_assert(sizeof(Tile) == Count * sizeof(BFloat));
  return true;
}
static_assert(storage_contract<16 * Dim>());
static_assert(storage_contract<DefaultCells * Dim>());
static_assert(storage_contract<16 * DefaultCells>());
static_assert(16 * sizeof(BFloat) % 8 == 0 && 32 * sizeof(BFloat) % 8 == 0);

template<int Cells, int ScoreStride> constexpr size_t LocalPayload =
    (16 * Dim + Cells * Dim + 16 * Cells) * 2
    + (16 * ScoreStride + 3 * 16) * 4 + Cells * 8 + 16 * 4;
static_assert(LocalPayload<32, 32> == 28160);
static_assert(LocalPayload<32, 33> == 28224);
static_assert(LocalPayload<64, 65> == 47936);
static_assert(LocalPayload<16, 17> == 18368);
static_assert(storage_contract<64 * Dim>() && storage_contract<16 * 64>());

bool matrix_available(const sycl::device &device, size_t local_payload) {
  if (!device.has(sycl::aspect::ext_intel_matrix) || !device.is_gpu() || device.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
      device.get_info<sycl::info::device::max_work_group_size>() < Workgroup ||
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < Workgroup ||
      device.get_info<sycl::info::device::local_mem_size>() < local_payload + 4096)
    return false;
  const auto subgroups = device.get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(subgroups.begin(), subgroups.end(), 16) == subgroups.end()) return false;
  const auto fits = [](size_t fixed, size_t maximum, size_t needed) {
    return fixed == needed || (!fixed && maximum >= needed);
  };
  for (const auto &c : device.get_info<
           sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
    if (c.atype == mx::matrix_type::bf16 && c.btype == mx::matrix_type::bf16 &&
        c.ctype == mx::matrix_type::fp32 && c.dtype == mx::matrix_type::fp32 &&
        fits(c.msize, c.max_msize, 8) && fits(c.nsize, c.max_nsize, 16) &&
        fits(c.ksize, c.max_ksize, 16)) return true;
  return false;
}
} // namespace

template<int Cells, int ScoreStride, bool TilePacked = false, bool SubgroupSoftmax = false>
static bool prefix_batch_impl(const uint16_t *query, const QsaAttnPools &pools,
                             const int32_t *ids, const int32_t *steps,
                             int64_t cap, const QsaShapes &shape, uint16_t *output,
                             int64_t queries, const QsaPromptBf16Bounds &bounds,
                             int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  static_assert(!SubgroupSoftmax || !TilePacked);
  static_assert(Cells == 32 || ((Cells == 16 || Cells == 64) && !TilePacked && !SubgroupSoftmax));
  static_assert(Cells * sizeof(BFloat) % 8 == 0);
  // Reuse the qualified admission contract; it never enqueues or waits.
  if (!qsa_prompt_attn_bf16_validate(query, pools, ids, steps, cap, shape,
                                    output, queries, bounds, stream)) return false;
  if (!queries) return true;
  if (live_cells <= 0 || uint64_t(live_cells) >
      uint64_t(bounds.logical_pages) * uint64_t(shape.page_size))
    throw std::invalid_argument("QSA BF16 prefix exceeds logical capacity");
  auto &queue = sycl_backend::queue_for(stream);
  if (!matrix_available(queue.get_device(), LocalPayload<Cells, ScoreStride>)) return false;
  // Validate all writable scratch together with the final output before pack.
  // The old admission also checks the full original inputs and output domains.
  using namespace sycl_backend;
  const size_t rows_count = checked_count(checked_count(bounds.physical_pages, 2), shape.page_size);
  const size_t active_pages = (uint64_t(live_cells) + shape.page_size - 1) / shape.page_size;
  const size_t active_rows = checked_count(checked_count(active_pages, 2), shape.page_size);
  const size_t packed_bytes = checked_count(checked_count(active_rows, Dim), 2);
  const size_t query_bytes = checked_count(checked_count(checked_count(queries, 24), Dim), 2);
  const Span pk{scratch.k, packed_bytes}, pv{scratch.v, packed_bytes},
             bad{scratch.bad_rows, checked_count(active_rows, 4)};
  validate_spans({pk, pv, bad, {output, query_bytes}},
      {{query, query_bytes}, {ids, checked_count(checked_count(queries, cap), 4)},
       {steps, checked_count(checked_count(queries, kStepCount), 4)},
       {pools.page_table, checked_count(bounds.logical_pages, 4)},
       {pools.k_q, checked_count(rows_count, Dim)}, {pools.v_q, checked_count(rows_count, Dim)},
       {pools.k_scale, checked_count(checked_count(rows_count, 4), 2)},
       {pools.v_scale, checked_count(checked_count(rows_count, 4), 2)}}, 1);
  validate_spans({}, {pk, pv}, 2); validate_spans({}, {bad}, 4);
  for (auto span : {pk, pv, bad}) validate_device_span(span.pointer, span.bytes, queue);
  const auto packed = scratch;
  // A row in scratch is logical. The page table is consulted before ANY
  // source code/scale read, and final-page padding never reads an input row.
  auto prepared = queue.parallel_for(sycl::nd_range<1>(active_rows * 64, 64),
      [=](sycl::nd_item<1> item) {
    const size_t row = item.get_group_linear_id(), lane = item.get_local_linear_id();
    const size_t logical = row / (2 * shape.page_size),
                 head = row / shape.page_size % 2, offset = row % shape.page_size;
    const size_t cell = logical * shape.page_size + offset;
    const int page = pools.page_table[logical];
    const bool valid = cell < uint64_t(live_cells) && page >= 0 && page < bounds.physical_pages;
    const size_t source_row = valid ? (size_t(page) * 2 + head) * shape.page_size + offset : 0;
    if (!lane) {
      int invalid = 0;
      if (valid) for (int g = 0; g < 4; ++g)
        invalid |= !sycl::isfinite(f32_from_f16(pools.k_scale[source_row * 4 + g])) ||
                   !sycl::isfinite(f32_from_f16(pools.v_scale[source_row * 4 + g]));
      packed.bad_rows[row] = invalid;
    }
    for (size_t d = lane; d < Dim; d += 64) {
      float k = 0.f, v = 0.f;
      if (valid) {
        k = float(pools.k_q[source_row * Dim + d]) * f32_from_f16(pools.k_scale[source_row * 4 + d / 64]);
        v = float(pools.v_q[source_row * Dim + d]) * f32_from_f16(pools.v_scale[source_row * 4 + d / 64]);
      }
      packed.k[row * Dim + d] = sycl::bit_cast<uint16_t>(BFloat(k));
      packed.v[row * Dim + d] = sycl::bit_cast<uint16_t>(BFloat(v));
    }
  });
  // queue_for requires in_order; no host tensor work or wait is introduced for
  // the explicit stream. Both events remain covered by the caller's lifetime.
  sycl_backend::finish(stream, prepared);
  auto event = queue.submit([&](sycl::handler &handler) {
    // The whole operand is a real contiguous array in a 4-byte aligned object.
    // No type-punned uint32_t storage or pointer arithmetic across subarrays.
    sycl::local_accessor<MatrixTile<16 * Dim>, 1> qstorage(1, handler);
    sycl::local_accessor<MatrixTile<Cells * Dim>, 1> kvstorage(1, handler);
    sycl::local_accessor<MatrixTile<16 * Cells>, 1> pstorage(1, handler);
    sycl::local_accessor<float, 1> scores(16 * ScoreStride, handler), maximum(16, handler),
                                  normalizer(16, handler), alpha(16, handler);
    sycl::local_accessor<int64_t, 1> rows(Cells, handler);
    sycl::local_accessor<int, 1> bad_head(16, handler);
    handler.parallel_for(sycl::nd_range<1>(size_t(queries) * 2 * Workgroup, Workgroup),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      auto qtile = sycl::address_space_cast<sycl::access::address_space::local_space,
          sycl::access::decorated::no>(qstorage[0].values);
      auto kv = sycl::address_space_cast<sycl::access::address_space::local_space,
          sycl::access::decorated::no>(kvstorage[0].values);
      auto probability = sycl::address_space_cast<sycl::access::address_space::local_space,
          sycl::access::decorated::no>(pstorage[0].values);
      const int tid = item.get_local_linear_id();
      const auto sg = item.get_sub_group();
      const int subgroup = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
      const int64_t batch = int64_t(item.get_group_linear_id()) / 2;
      const int kv_head = int(item.get_group_linear_id() % 2);
      const auto *q = query + (batch * 24 + kv_head * Heads) * Dim;
      // All subgroup lanes participate for real and padded head rows alike.
      for (int head = subgroup; head < 16; head += 8) {
        int bad = 0;
        for (int d = lane; d < Dim; d += 16) {
          float value = head < Heads
              ? sycl::bit_cast<float>(uint32_t(q[head * Dim + d]) << 16) : 0.f;
          bad |= !sycl::isfinite(value);
          // Scale and round before Q.K, matching the configured BF16 contract.
          value = value * 0.0625f;
          value = value * 1.4426950408f;
          if constexpr (TilePacked)
            qtile[(d / 16) * 256 + head * 16 + d % 16] = BFloat(value);
          else qtile[head * Dim + d] = BFloat(value);
        }
        bad = sycl::reduce_over_group(sg, bad, sycl::maximum<int>());
        if (!lane) bad_head[head] = bad;
      }
      if (tid < 16) { maximum[tid] = -INFINITY; normalizer[tid] = 0.f; }
      using Acc = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16>;
      using Left = mx::joint_matrix<sycl::sub_group, BFloat, mx::use::a, 8, 16,
                                    mx::layout::row_major>;
      using Right = mx::joint_matrix<sycl::sub_group, BFloat, mx::use::b, 16, 16,
                                     TilePacked ? mx::layout::ext_intel_packed : mx::layout::row_major>;
      Acc accumulated[4];
      for (int j = 0; j < 4; ++j) mx::joint_matrix_fill(sg, accumulated[j], 0.f);
      const int width = steps[batch * kStepCount + kStepWidth];
      const int count = width >= 0 && int64_t(width) <= cap ? width : 0;
      int local_bad = 0, local_live = 0;
      item.barrier(sycl::access::fence_space::local_space);
      for (int64_t first = 0; first < count; first += Cells) {
        if (tid < Cells) {
          int64_t row = -1;
          if (first + tid < count) {
            const int64_t cell = ids[batch * cap + first + tid];
            const int64_t logical = cell >= 0 ? cell / shape.page_size : -1;
            const int page = logical >= 0 && logical < bounds.logical_pages
                ? pools.page_table[logical] : -1;
            if (cell < live_cells && page >= 0 && page < bounds.physical_pages)
              row = (logical * 2 + kv_head) * shape.page_size + cell % shape.page_size;
          }
          rows[tid] = row;
          if (row >= 0) local_bad |= packed.bad_rows[row];
        }
        item.barrier(sycl::access::fence_space::local_space);
        for (int i = tid; i < Cells * Dim; i += Workgroup) {
          const int cell = i % Cells, d = i / Cells;
          const int64_t row = rows[cell];
          const uint16_t bits = row >= 0 ? packed.k[row * Dim + d] : uint16_t(0);
          if constexpr (TilePacked)
            kv[((d / 16) * 2 + cell / 16) * 256 + (d % 16 / 2) * 32 +
               (cell % 16) * 2 + d % 2] = sycl::bit_cast<BFloat>(bits);
          else kv[i] = sycl::bit_cast<BFloat>(bits);
        }
        if (tid < Cells) local_live |= rows[tid] >= 0;
        item.barrier(sycl::access::fence_space::local_space);
        if (subgroup < 2 * (Cells / 16)) {
          const int head = (subgroup / (Cells / 16)) * 8,
                    cell = (subgroup % (Cells / 16)) * 16;
          Acc product; Left a; Right b;
          mx::joint_matrix_fill(sg, product, 0.f);
          for (int d = 0; d < Dim; d += 16) {
            if constexpr (TilePacked) {
              mx::joint_matrix_load(sg, a, qtile + (d / 16) * 256 + head * 16, 16);
              mx::joint_matrix_load(sg, b, kv + ((d / 16) * 2 + cell / 16) * 256, 32);
            } else {
              mx::joint_matrix_load(sg, a, qtile + head * Dim + d, Dim);
              mx::joint_matrix_load(sg, b, kv + d * Cells + cell, Cells);
            }
            mx::joint_matrix_mad(sg, product, a, b, product);
          }
          mi::joint_matrix_apply(sg, product, [=](float &value, size_t r, size_t c) {
            scores[(head + r) * ScoreStride + cell + c] = value;
          });
        }
        item.barrier(sycl::access::fence_space::local_space);
        if constexpr (SubgroupSoftmax) {
          // Each SIMD16 subgroup owns two head rows. Every lane participates,
          // including missing cells and padded heads; FP32 state stays local.
          for (int h = subgroup; h < 16; h += 8) {
            const int c0 = lane, c1 = lane + 16;
            const float s0 = scores[h * ScoreStride + c0];
            const float s1 = scores[h * ScoreStride + c1];
            const bool row0 = rows[c0] >= 0, row1 = rows[c1] >= 0;
            const bool valid0 = row0 && sycl::isfinite(s0);
            const bool valid1 = row1 && sycl::isfinite(s1);
            const int bad = (row0 && !valid0) || (row1 && !valid1);
            const int score_bad = sycl::reduce_over_group(sg, bad, sycl::maximum<int>());
            const float lane_max = sycl::fmax(valid0 ? s0 : -INFINITY,
                                             valid1 ? s1 : -INFINITY);
            const float tile_max = sycl::reduce_over_group(sg, lane_max, sycl::maximum<float>());
            const float prior = !lane ? maximum[h] : -INFINITY;
            const float old_m = sycl::group_broadcast(sg, prior, 0);
            const float m = sycl::fmax(old_m, tile_max);
            const float p0 = valid0 ? sycl::exp2(s0 - m) : 0.f;
            const float p1 = valid1 ? sycl::exp2(s1 - m) : 0.f;
            const float sum = sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());
            probability[h * Cells + c0] = BFloat(p0);
            probability[h * Cells + c1] = BFloat(p1);
            if (!lane) {
              const float old_norm = normalizer[h];
              const float a = old_norm > 0.f ? sycl::exp2(old_m - m) : 0.f;
              bad_head[h] |= score_bad;
              maximum[h] = m; alpha[h] = a;
              normalizer[h] = sycl::fma(old_norm, a, sum);
            }
          }
        } else if (tid < 16) {
          float m = maximum[tid];
          for (int c = 0; c < Cells; ++c) if (rows[c] >= 0) {
            const float score = scores[tid * ScoreStride + c];
            bad_head[tid] |= !sycl::isfinite(score);
            if (sycl::isfinite(score)) m = sycl::fmax(m, score);
          }
          const float a = normalizer[tid] > 0.f ? sycl::exp2(maximum[tid] - m) : 0.f;
          float sum = 0.f;
          for (int c = 0; c < Cells; ++c) {
            const float score = scores[tid * ScoreStride + c];
            const float p = rows[c] >= 0 && sycl::isfinite(score)
                ? sycl::exp2(score - m) : 0.f;
            sum += p;
            if constexpr (TilePacked)
              probability[(c / 16) * 256 + tid * 16 + c % 16] = BFloat(p);
            else probability[tid * Cells + c] = BFloat(p);
          }
          maximum[tid] = m; alpha[tid] = a;
          normalizer[tid] = sycl::fma(normalizer[tid], a, sum);
        }
        for (int i = tid; i < Cells * Dim; i += Workgroup) {
          const int cell = i / Dim, d = i % Dim;
          const int64_t row = rows[cell];
          const uint16_t bits = row >= 0 ? packed.v[row * Dim + d] : uint16_t(0);
          if constexpr (TilePacked)
            kv[((cell / 16) * 16 + d / 16) * 256 + (cell % 16 / 2) * 32 +
               (d % 16) * 2 + cell % 2] = sycl::bit_cast<BFloat>(bits);
          else kv[i] = sycl::bit_cast<BFloat>(bits);
        }
        item.barrier(sycl::access::fence_space::local_space);
        const int head = (subgroup / 4) * 8, dim_base = (subgroup % 4) * 64;
        for (int j = 0; j < 4; ++j) {
          Left a; Right b;
          mi::joint_matrix_apply(sg, accumulated[j], [=](float &v, size_t r, size_t) {
            v *= alpha[head + r];
          });
          for (int c = 0; c < Cells; c += 16) {
            if constexpr (TilePacked) {
              mx::joint_matrix_load(sg, a, probability + (c / 16) * 256 + head * 16, 16);
              mx::joint_matrix_load(sg, b, kv + ((c / 16) * 16 + dim_base / 16 + j) * 256, 32);
            } else {
              mx::joint_matrix_load(sg, a, probability + head * Cells + c, Cells);
              mx::joint_matrix_load(sg, b, kv + c * Dim + dim_base + j * 16, Dim);
            }
            mx::joint_matrix_mad(sg, accumulated[j], a, b, accumulated[j]);
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      const bool invalid_scales = sycl::any_of_group(item.get_group(), local_bad != 0);
      const bool any_live = sycl::any_of_group(item.get_group(), local_live != 0);
      const int head = (subgroup / 4) * 8, dim_base = (subgroup % 4) * 64;
      for (int j = 0; j < 4; ++j)
        mi::joint_matrix_apply(sg, accumulated[j], [=](float &value, size_t r, size_t c) {
          const int h = head + int(r);
          if (h >= Heads) return;
          uint16_t result = 0;
          if (any_live && (bad_head[h] || invalid_scales)) result = 0x7fc0;
          else if (normalizer[h] > 0.f)
            result = sycl::bit_cast<uint16_t>(BFloat(value / normalizer[h]));
          output[(batch * 24 + kv_head * Heads + h) * Dim + dim_base + j * 16 + c] = result;
        });
    });
  });
  sycl_backend::finish(stream, event);
  return true;
}
bool qsa_bf16_prefix_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  return prefix_batch_impl<DefaultCells, DefaultCells>(query, pools, ids, steps, cap, shape,
      output, queries, bounds, live_cells, scratch, stream);
}
bool qsa_bf16_prefix_optimized_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  return prefix_batch_impl<DefaultCells, DefaultCells + 1>(query, pools, ids, steps, cap, shape,
      output, queries, bounds, live_cells, scratch, stream);
}
// FP32 subgroup max/sum; current score33 row-major operands and state layout.
bool qsa_bf16_prefix_subgroup_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  return prefix_batch_impl<DefaultCells, DefaultCells + 1, false, true>(query, pools, ids, steps, cap, shape,
      output, queries, bounds, live_cells, scratch, stream);
}
// Dense16 A tiles and BF16 pair-packed B tiles; same numerical contract.
bool qsa_bf16_prefix_tilepack_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  return prefix_batch_impl<DefaultCells, DefaultCells + 1, true>(query, pools, ids, steps, cap, shape,
      output, queries, bounds, live_cells, scratch, stream);
}
// Experimental 64-cell grouping: same BF16 operands and FP32 online state,
// with probability rounding relative to each 64-cell tile maximum. This can
// differ numerically from the existing 32-cell grouping; no exact-match claim.
bool qsa_bf16_prefix64_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  return prefix_batch_impl<64, 65>(query, pools, ids, steps, cap, shape,
      output, queries, bounds, live_cells, scratch, stream);
}
// Experimental lower-SLM 16-cell path: same query ownership and types,
// probability rounding now uses each 16-cell tile maximum. Not bit-identical
// to 32/64 grouping. More barriers may offset lower SLM and less padding.
bool qsa_bf16_prefix16_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  return prefix_batch_impl<16, 17>(query, pools, ids, steps, cap, shape,
      output, queries, bounds, live_cells, scratch, stream);
}
} // namespace strata::kernels
