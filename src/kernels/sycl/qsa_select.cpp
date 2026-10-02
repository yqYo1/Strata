#include "strata/kernels/qsa_select.hpp"
#include "strata/sycl/launch.hpp"
#include <climits>

namespace strata::kernels {
namespace {
using namespace sycl_backend;
int64_t validate(int64_t nq, int64_t blocks, int64_t active,
                 const QsaShapes &s) {
  if (nq <= 0 || nq > 65535 || blocks <= 0 || blocks > INT_MAX ||
      s.idx_dim != 128 || s.idx_n_head != 4 || s.idx_block != 4 ||
      s.idx_top_k < 0 || s.idx_top_k > INT_MAX - 3 ||
      (active != -1 && (active <= 0 || active > blocks)))
    throw std::invalid_argument("invalid SYCL block indexer geometry/capacity");
  return active == -1 ? blocks : active;
}
uint32_t order(float value) {
  uint32_t b = sycl::bit_cast<uint32_t>(value);
  if ((b & 0x7fffffffu) > 0x7f800000u)
    return 0;
  if ((b & 0x7fffffffu) == 0)
    b = 0;
  return b & 0x80000000u ? ~b : b | 0x80000000u;
}
} // namespace
void qsa_block_scores(const float *pooled, const float *dead,
                      const float *query, const int32_t *steps, int64_t nq,
                      int64_t blocks, const QsaShapes &s, float *scores,
                      void *stream, int64_t active) {
  const int64_t launched = validate(nq, blocks, active, s);
  validate_spans({{scores, checked_count(nq, blocks) * 4}},
                 {{pooled, checked_count(blocks, 128) * 4},
                  {dead, 512},
                  {query, checked_count(nq, 512) * 4},
                  {steps, checked_count(nq, kStepCount) * 4}});
  auto event = queue_for(stream).parallel_for(
      sycl::nd_range<1>(size_t(nq * launched) * 32, 32),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t qi = it.get_group_linear_id() / launched,
                      b = it.get_group_linear_id() % launched;
        const auto st = steps + qi * kStepCount;
        const int64_t n = st[kStepNKv], nb = st[kStepNBid];
        if (n < 0 || nb != n / 4 || nb >= launched || b > nb)
          return;
        const int lane = it.get_local_linear_id();
        const float *key = b == nb ? dead : pooled + b * 128;
        const float *q = query + qi * 512;
        float total = 0;
        for (int h = 0; h < 4; ++h) {
          float dot = key[lane * 4] * q[h * 128 + lane * 4];
          for (int d = 1; d < 4; ++d)
            dot += key[lane * 4 + d] * q[h * 128 + lane * 4 + d];
          for (int off = 16; off; off /= 2)
            dot += sycl::permute_group_by_xor(it.get_sub_group(), dot, off);
          total += dot > 0 ? dot : 0.f;
        }
        if (lane == 0) {
          if (b == nb && n % 4)
            total += 1e9f;
          scores[qi * blocks + b] = total;
        }
      });
  finish(stream, event);
}
bool qsa_block_scores_tc(const float *, const float *, const float *,
                         const int32_t *, int64_t, int64_t, const QsaShapes &,
                         float *, void *, int64_t) {
  // The caller's documented fallback is qsa_block_scores. No tensor-core
  // accuracy/performance contract has been established on this backend yet.
  return false;
}
void qsa_block_topk(const float *scores, const int32_t *steps, int64_t nq,
                    int64_t blocks, int64_t cap, const QsaShapes &s,
                    int32_t *ids, void *stream, int64_t active) {
  const int64_t launched = validate(nq, blocks, active, s);
  if (cap <= 0 || cap > INT_MAX)
    throw std::invalid_argument("invalid SYCL block selection capacity");
  validate_spans({{ids, checked_count(nq, cap) * 4}},
                 {{scores, checked_count(nq, blocks) * 4},
                  {steps, checked_count(nq, kStepCount) * 4}});
  auto event = queue_for(stream).parallel_for(
      sycl::nd_range<1>(size_t(nq) * 256, 256), [=](sycl::nd_item<1> it) {
        const int64_t qi = it.get_group_linear_id();
        const auto st = steps + qi * kStepCount;
        const int64_t n = st[kStepNKv], nb = st[kStepNBid],
                      width = st[kStepWidth];
        if (n < 0 || nb != n / 4 || nb >= launched || width < 0 ||
            width > cap || width > n)
          return;
        const int t = it.get_local_linear_id();
        auto out = ids + qi * cap;
        if (n == width) {
          for (int64_t j = t; j < n; j += 256)
            out[j] = int32_t(j);
          return;
        }
        if (!width)
          return;
        const auto sc = scores + qi * blocks;
        const int64_t chunk = (nb + 1 + 255) / 256, lo = t * chunk,
                      hi = sycl::min(nb + 1, lo + chunk);
        auto weight = [=](int64_t b) { return b < nb ? 4 : int(n % 4); };
        uint32_t threshold = 0;
        for (int bit = 31; bit >= 0; --bit) {
          const uint32_t candidate = threshold | (1u << bit);
          int count = 0;
          for (int64_t b = lo; b < hi; ++b)
            if (order(sc[b]) >= candidate)
              count += weight(b);
          if (sycl::reduce_over_group(it.get_group(), count,
                                      sycl::plus<int>()) >= width)
            threshold = candidate;
        }
        int gt = 0, eq = 0;
        for (int64_t b = lo; b < hi; ++b) {
          const auto k = order(sc[b]);
          if (k > threshold)
            gt += weight(b);
          if (k == threshold)
            eq += weight(b);
        }
        const int budget =
            int(width) -
            sycl::reduce_over_group(it.get_group(), gt, sycl::plus<int>());
        const int eqstart = sycl::exclusive_scan_over_group(it.get_group(), eq,
                                                            sycl::plus<int>());
        const int chosen = gt + sycl::min(eq, sycl::max(0, budget - eqstart));
        int offset = sycl::exclusive_scan_over_group(it.get_group(), chosen,
                                                     sycl::plus<int>());
        int eqrank = eqstart;
        for (int64_t b = lo; b < hi; ++b) {
          const auto k = order(sc[b]);
          int take = k > threshold ? weight(b) : 0;
          if (k == threshold) {
            take = sycl::min(weight(b), sycl::max(0, budget - eqrank));
            eqrank += weight(b);
          }
          for (int j = 0; j < take; ++j)
            out[offset++] = int32_t(b * 4 + j);
        }
      });
  finish(stream, event);
}
void qsa_block_topk_ref(const float *scores, const int32_t *steps, int64_t nq,
                        int64_t blocks, int64_t cap, const QsaShapes &s,
                        int32_t *ids, void *stream) {
  qsa_block_topk(scores, steps, nq, blocks, cap, s, ids, stream);
}
} // namespace strata::kernels
