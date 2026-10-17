// Main-authored experiment derived only from this repository's FP32 scorer.
// Same lane-local sums, XOR reduction and head summation; reuse key registers
// across a small number of adjacent queries. No local memory or matrix math.
#include "strata/kernels/qsa_select.hpp"
#include "strata/sycl/launch.hpp"
#include <climits>
namespace strata::kernels {
namespace {
using namespace sycl_backend;
template <int Tile>
sycl::event submit(sycl::queue &queue, const float *pooled, const float *dead,
                   const float *query, const int32_t *steps, int64_t nq,
                   int64_t blocks, float *scores, int64_t active) {
  return queue.parallel_for(
      sycl::nd_range<1>(size_t(((nq + Tile - 1) / Tile) * active) * 32, 32),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t q0 = (it.get_group_linear_id() / active) * Tile;
        const int64_t b = it.get_group_linear_id() % active;
        const int lane = it.get_local_linear_id();
        float common[4];
        for (int d = 0; d < 4; ++d) common[d] = pooled[b * 128 + lane * 4 + d];
        // A condition on qi/b/step is uniform across the entire subgroup.
        // Even a partial last tile converges at every XOR collective.
        #pragma unroll
        for (int row = 0; row < Tile; ++row) {
          const int64_t qi = q0 + row;
          if (qi >= nq) continue;
          const auto st = steps + qi * kStepCount;
          const int64_t n = st[kStepNKv], nb = st[kStepNBid];
          if (n < 0 || nb != n / 4 || nb >= active || b > nb) continue;
          float key[4];
          for (int d = 0; d < 4; ++d)
            key[d] = b == nb ? dead[lane * 4 + d] : common[d];
          const float *q = query + qi * 512;
          float total = 0;
          for (int h = 0; h < 4; ++h) {
            float dot = key[0] * q[h * 128 + lane * 4];
            for (int d = 1; d < 4; ++d) dot += key[d] * q[h * 128 + lane * 4 + d];
            for (int off = 16; off; off /= 2)
              dot += sycl::permute_group_by_xor(it.get_sub_group(), dot, off);
            total += dot > 0 ? dot : 0.f;
          }
          if (lane == 0) {
            if (b == nb && n % 4) total += 1e9f;
            scores[qi * blocks + b] = total;
          }
        }
      });
}
} // namespace
void qsa_block_scores_querytile(const float *pooled, const float *dead,
                      const float *query, const int32_t *steps, int64_t nq,
                      int64_t blocks, const QsaShapes &s, float *scores,
                      void *stream, int64_t active, int tile) {
  if (nq <= 0 || nq > 65535 || blocks <= 0 || blocks > INT_MAX ||
      s.idx_dim != 128 || s.idx_n_head != 4 || s.idx_block != 4 ||
      s.idx_top_k < 0 || s.idx_top_k > INT_MAX - 3 || active <= 0 || active > blocks)
    throw std::invalid_argument("invalid private query-tile scorer geometry");
  validate_spans({{scores, checked_count(nq, blocks) * 4}},
                 {{pooled, checked_count(blocks, 128) * 4}, {dead, 512},
                  {query, checked_count(nq, 512) * 4}, {steps, checked_count(nq, kStepCount) * 4}});
  auto &q = queue_for(stream);
  sycl::event event;
  if (tile == 2) event = submit<2>(q, pooled, dead, query, steps, nq, blocks, scores, active);
  else if (tile == 4) event = submit<4>(q, pooled, dead, query, steps, nq, blocks, scores, active);
  else if (tile == 8) event = submit<8>(q, pooled, dead, query, steps, nq, blocks, scores, active);
  else throw std::invalid_argument("unsupported private query tile");
  finish(stream, event);
}
} // namespace strata::kernels
