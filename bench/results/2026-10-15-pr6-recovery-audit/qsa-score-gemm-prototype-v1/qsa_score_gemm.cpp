// Main-authored from this fork's QSA contract and existing Gemm::f16 API.
// No XeStrata implementation, helper or proposal used. Uncompiled/unqualified.
#include "qsa_score_gemm.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>

namespace strata::experimental::qsa_score_gemm {
namespace {
void invalid(uint32_t *status, uint32_t reason) {
  sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space>(*status)
      .fetch_or(reason);
}
uint16_t half_bits(float value, uint32_t *status) {
  if (!sycl::isfinite(value) || sycl::fabs(value) > 65504.f) {
    invalid(status, 1);
    // The result is invalid, not clamped or accepted. Initialize the element
    // so that the owned diagnostic can complete and drain deterministically.
    value = 0.f;
  }
  return sycl::bit_cast<uint16_t>(sycl::half(value));
}
} // namespace

void scores(prefill::Gemm &gemm, const float *pooled, const float *dead,
            const float *query, const int32_t *steps, int64_t nq,
            int64_t blocks, const kernels::QsaShapes &shape, float *output,
            Scratch s, void *stream, int64_t active) {
  using namespace sycl_backend;
  using namespace kernels;
  if (!stream || gemm.stream() != stream || nq <= 0 || nq > 256 ||
      blocks <= 0 || blocks > 65538 || active <= 0 || active > blocks ||
      shape.idx_n_head != 4 || shape.idx_dim != 128 || shape.idx_block != 4 ||
      s.tile_blocks <= 0 || s.tile_blocks > 8192 || s.tile_blocks % 64)
    throw std::invalid_argument("invalid private QSA score GEMM geometry/owner");
  auto &queue = queue_for(stream);
  if (!queue.get_device().is_gpu() ||
      queue.get_backend() != sycl::backend::ext_oneapi_level_zero ||
      !queue.get_device().has(sycl::aspect::fp16))
    throw std::invalid_argument("private QSA score GEMM needs Level Zero FP16 GPU");
  const int64_t padded = (active + 63) / 64 * 64;
  validate_spans({{output, checked_count(nq, blocks) * 4},
                  {s.pooled_fp16, checked_count(padded, 128) * 2},
                  {s.query_fp16, checked_count(nq, 512) * 2},
                  {s.heads_fp32, checked_count(nq * 4, s.tile_blocks) * 4},
                  {s.status, 4}},
                 {{pooled, checked_count(blocks, 128) * 4}, {dead, 512},
                  {query, checked_count(nq, 512) * 4},
                  {steps, checked_count(nq, kStepCount) * 4}});
  finish(stream, queue.memset(s.status, 0, 4));
  finish(stream, queue.parallel_for(sycl::range<1>(size_t(padded * 128)),
      [=](sycl::id<1> id) {
        const size_t i = id[0];
        s.pooled_fp16[i] = i / 128 < size_t(active)
            ? half_bits(pooled[i], s.status) : uint16_t(0);
      }));
  finish(stream, queue.parallel_for(sycl::range<1>(size_t(nq * 512)),
      [=](sycl::id<1> id) {
        const size_t i = id[0];
        s.query_fp16[i] = half_bits(query[i], s.status);
      }));
  // W is compact [padded,128]. Flattening query[nq,4,128] introduces no
  // transpose or host grouping. Each tile reuses the same FP32 head buffer;
  // the in-order queue retires its readers before the next GEMM overwrites it.
  for (int64_t first = 0; first < padded; first += s.tile_blocks) {
    const int64_t columns = std::min(s.tile_blocks, padded - first);
    gemm.f16(s.query_fp16, s.pooled_fp16 + first * 128, s.heads_fp32,
             nq * 4, columns, 128, s.tile_blocks, 0.f);
    finish(stream, queue.parallel_for(sycl::range<1>(size_t(nq * columns)),
        [=](sycl::id<1> id) {
          const int64_t qi = id[0] / columns, column = id[0] % columns;
          const int64_t b = first + column;
          const auto step = steps + qi * kStepCount;
          const int64_t n = step[kStepNKv], nb = step[kStepNBid];
          if (n < 0 || nb != n / 4 || nb >= active || b >= active || b >= nb)
            return; // Dead block is handled once by the original FP32 tree.
          float total = 0;
          for (int h = 0; h < 4; ++h) {
            const float dot = s.heads_fp32[(qi * 4 + h) * s.tile_blocks + column];
            if (!sycl::isfinite(dot)) invalid(s.status, 2);
            total += dot > 0 ? dot : 0.f;
          }
          if (!sycl::isfinite(total)) invalid(s.status, 2);
          output[qi * blocks + b] = total;
        }));
  }
  finish(stream, queue.parallel_for(
      sycl::nd_range<1>(size_t(nq) * 32, 32),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t qi = it.get_group_linear_id();
        const auto step = steps + qi * kStepCount;
        const int64_t n = step[kStepNKv], nb = step[kStepNBid];
        if (n < 0 || nb != n / 4 || nb >= active) return;
        const int lane = it.get_local_linear_id();
        const auto q = query + qi * 512;
        float total = 0;
        for (int h = 0; h < 4; ++h) {
          float dot = dead[lane * 4] * q[h * 128 + lane * 4];
          for (int d = 1; d < 4; ++d)
            dot += dead[lane * 4 + d] * q[h * 128 + lane * 4 + d];
          for (int off = 16; off; off /= 2)
            dot += sycl::permute_group_by_xor(it.get_sub_group(), dot, off);
          if (!sycl::isfinite(dot)) invalid(s.status, 2);
          total += dot > 0 ? dot : 0.f;
        }
        if (lane == 0) {
          if (n % 4) total += 1e9f;
          if (!sycl::isfinite(total)) invalid(s.status, 2);
          output[qi * blocks + nb] = total;
        }
      }));
}
} // namespace strata::experimental::qsa_score_gemm
