// Authored for Strata; repository LICENSE (MIT) applies.
// Independent typed online-attention implementation. Numerical design source:
// bench/results/2026-10-15-pr6-recovery-audit/qsa-framework-precision-plan-v1.ja.md.
// No global decoded KV or FP32 Q/O, no high/low matrix decomposition.
#include "strata/kernels/qsa_bf16_attention.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <cmath>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace strata::kernels {
namespace {
namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace mi = sycl::ext::intel::experimental::matrix;
using BFloat = sycl::ext::oneapi::bfloat16;
constexpr int Dim = 256, Heads = 12, Cells = 32, Workgroup = 128;
constexpr size_t LocalPayload = (16 * Dim + Cells * Dim + 16 * Cells) * 2
    + (16 * Cells + 3 * 16) * 4 + Cells * 8 + 16 * 4 + 2 * 4;
static_assert(LocalPayload == 28168);

bool matrix_available(const sycl::device &device) {
  if (!device.is_gpu() || device.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
      device.get_info<sycl::info::device::max_work_group_size>() < Workgroup ||
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < Workgroup ||
      device.get_info<sycl::info::device::local_mem_size>() < LocalPayload + 4096)
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

bool qsa_bf16_attention_batch(const uint16_t *query, const QsaAttnPools &pools,
                             const int32_t *ids, const int32_t *steps,
                             int64_t cap, const QsaShapes &shape, uint16_t *output,
                             int64_t queries, const QsaPromptBf16Bounds &bounds,
                             void *stream) {
  // Reuse the qualified admission contract; it never enqueues or waits.
  if (!qsa_prompt_attn_bf16_validate(query, pools, ids, steps, cap, shape,
                                    output, queries, bounds, stream)) return false;
  if (!queries) return true;
  auto &queue = sycl_backend::queue_for(stream);
  if (!matrix_available(queue.get_device())) return false;
  auto event = queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<BFloat, 1> qtile(16 * Dim, handler), kv(Cells * Dim, handler),
                                   probability(16 * Cells, handler);
    sycl::local_accessor<float, 1> scores(16 * Cells, handler), maximum(16, handler),
                                  normalizer(16, handler), alpha(16, handler);
    sycl::local_accessor<int64_t, 1> rows(Cells, handler);
    sycl::local_accessor<int, 1> bad_head(16, handler), flags(2, handler);
    handler.parallel_for(sycl::nd_range<1>(size_t(queries) * 2 * Workgroup, Workgroup),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const int tid = item.get_local_linear_id();
      const auto sg = item.get_sub_group();
      const int subgroup = sg.get_group_linear_id(), lane = sg.get_local_linear_id();
      const int64_t batch = int64_t(item.get_group_linear_id()) / 2;
      const int kv_head = int(item.get_group_linear_id() % 2);
      const auto *q = query + (batch * 24 + kv_head * Heads) * Dim;
      if (tid < 2) flags[tid] = 0;
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
          qtile[head * Dim + d] = BFloat(value);
        }
        bad = sycl::reduce_over_group(sg, bad, sycl::maximum<int>());
        if (!lane) bad_head[head] = bad;
      }
      if (tid < 16) { maximum[tid] = -INFINITY; normalizer[tid] = 0.f; }
      using Acc = mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16>;
      using Left = mx::joint_matrix<sycl::sub_group, BFloat, mx::use::a, 8, 16,
                                    mx::layout::row_major>;
      using Right = mx::joint_matrix<sycl::sub_group, BFloat, mx::use::b, 16, 16,
                                     mx::layout::row_major>;
      Acc accumulated[4];
      for (int j = 0; j < 4; ++j) mx::joint_matrix_fill(sg, accumulated[j], 0.f);
      const int width = steps[batch * kStepCount + kStepWidth];
      const int count = width >= 0 && int64_t(width) <= cap ? width : 0;
      item.barrier(sycl::access::fence_space::local_space);
      for (int64_t first = 0; first < count; first += Cells) {
        if (tid < Cells) {
          int64_t row = -1;
          if (first + tid < count) {
            const int64_t cell = ids[batch * cap + first + tid];
            const int64_t logical = cell >= 0 ? cell / shape.page_size : -1;
            const int page = logical >= 0 && logical < bounds.logical_pages
                ? pools.page_table[logical] : -1;
            if (page >= 0 && page < bounds.physical_pages)
              row = (int64_t(page) * 2 + kv_head) * shape.page_size + cell % shape.page_size;
          }
          rows[tid] = row;
        }
        item.barrier(sycl::access::fence_space::local_space);
        int bad_scale = 0;
        for (int i = tid; i < Cells * Dim; i += Workgroup) {
          const int cell = i % Cells, d = i / Cells;
          const int64_t row = rows[cell];
          float value = 0.f;
          if (row >= 0) {
            const float scale = f32_from_f16(pools.k_scale[row * 4 + d / 64]);
            bad_scale |= !sycl::isfinite(scale);
            value = float(pools.k_q[row * Dim + d]) * scale;
          }
          kv[i] = BFloat(value); // K is transposed locally for Q.K.
        }
        const bool invalid_k = sycl::any_of_group(item.get_group(), bad_scale != 0);
        const bool tile_live = sycl::any_of_group(item.get_group(), tid < Cells && rows[tid] >= 0);
        if (!tid) { flags[0] |= invalid_k; flags[1] |= tile_live; }
        item.barrier(sycl::access::fence_space::local_space);
        if (subgroup < 4) {
          const int head = (subgroup / 2) * 8, cell = (subgroup % 2) * 16;
          Acc product; Left a; Right b;
          mx::joint_matrix_fill(sg, product, 0.f);
          for (int d = 0; d < Dim; d += 16) {
            mx::joint_matrix_load(sg, a, qtile.template get_multi_ptr<
                sycl::access::decorated::no>() + head * Dim + d, Dim);
            mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<
                sycl::access::decorated::no>() + d * Cells + cell, Cells);
            mx::joint_matrix_mad(sg, product, a, b, product);
          }
          mi::joint_matrix_apply(sg, product, [=](float &value, size_t r, size_t c) {
            scores[(head + r) * Cells + cell + c] = value;
          });
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (tid < 16) {
          float m = maximum[tid];
          for (int c = 0; c < Cells; ++c) if (rows[c] >= 0) {
            const float score = scores[tid * Cells + c];
            bad_head[tid] |= !sycl::isfinite(score);
            if (sycl::isfinite(score)) m = sycl::fmax(m, score);
          }
          const float a = normalizer[tid] > 0.f ? sycl::exp2(maximum[tid] - m) : 0.f;
          float sum = 0.f;
          for (int c = 0; c < Cells; ++c) {
            const float score = scores[tid * Cells + c];
            const float p = rows[c] >= 0 && sycl::isfinite(score)
                ? sycl::exp2(score - m) : 0.f;
            sum += p;
            probability[tid * Cells + c] = BFloat(p);
          }
          maximum[tid] = m; alpha[tid] = a;
          normalizer[tid] = sycl::fma(normalizer[tid], a, sum);
        }
        bad_scale = 0;
        for (int i = tid; i < Cells * Dim; i += Workgroup) {
          const int cell = i / Dim, d = i % Dim;
          const int64_t row = rows[cell];
          float value = 0.f;
          if (row >= 0) {
            const float scale = f32_from_f16(pools.v_scale[row * 4 + d / 64]);
            bad_scale |= !sycl::isfinite(scale);
            value = float(pools.v_q[row * Dim + d]) * scale;
          }
          kv[i] = BFloat(value); // V stays [cell, dimension].
        }
        const bool invalid_v = sycl::any_of_group(item.get_group(), bad_scale != 0);
        if (!tid) flags[0] |= invalid_v;
        item.barrier(sycl::access::fence_space::local_space);
        const int head = (subgroup / 4) * 8, dim_base = (subgroup % 4) * 64;
        for (int j = 0; j < 4; ++j) {
          Left a; Right b;
          mi::joint_matrix_apply(sg, accumulated[j], [=](float &v, size_t r, size_t) {
            v *= alpha[head + r];
          });
          for (int c = 0; c < Cells; c += 16) {
            mx::joint_matrix_load(sg, a, probability.template get_multi_ptr<
                sycl::access::decorated::no>() + head * Cells + c, Cells);
            mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<
                sycl::access::decorated::no>() + c * Dim + dim_base + j * 16, Dim);
            mx::joint_matrix_mad(sg, accumulated[j], a, b, accumulated[j]);
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      const int head = (subgroup / 4) * 8, dim_base = (subgroup % 4) * 64;
      for (int j = 0; j < 4; ++j)
        mi::joint_matrix_apply(sg, accumulated[j], [=](float &value, size_t r, size_t c) {
          const int h = head + int(r);
          if (h >= Heads) return;
          uint16_t result = 0;
          if (flags[1] && (bad_head[h] || flags[0])) result = 0x7fc0;
          else if (normalizer[h] > 0.f)
            result = sycl::bit_cast<uint16_t>(BFloat(value / normalizer[h]));
          output[(batch * 24 + kv_head * Heads + h) * Dim + dim_base + j * 16 + c] = result;
        });
    });
  });
  sycl_backend::finish(stream, event);
  return true;
}
} // namespace strata::kernels
