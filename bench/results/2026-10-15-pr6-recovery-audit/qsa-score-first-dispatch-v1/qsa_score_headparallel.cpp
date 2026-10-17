// Main-authored exact-FP32 work mapping experiment, based only on this
// repository's scorer. The four heads share SG32 as four groups of eight lanes.
// Each lane computes the original contributions at l,l+8,l+16,l+24, then
// combines (p0+p2)+(p1+p3), exactly the original XOR16/XOR8 root subtree.
// XOR4/2/1 stay inside each eight-lane head; select lanes0/8/16/24 and add
// heads in the original order. No SLM, XMX, dtype change or reassociation.
#include "strata/kernels/qsa_select.hpp"
#include "strata/sycl/launch.hpp"
#include <climits>
namespace strata::kernels {
void qsa_block_scores_headparallel(const float *pooled, const float *dead,
    const float *query, const int32_t *steps, int64_t nq, int64_t blocks,
    const QsaShapes &shape, float *scores, void *stream, int64_t active) {
  using namespace sycl_backend;
  if (nq <= 0 || nq > 65535 || blocks <= 0 || blocks > INT_MAX ||
      active <= 0 || active > blocks || shape.idx_dim != 128 ||
      shape.idx_n_head != 4 || shape.idx_block != 4 ||
      shape.idx_top_k < 0 || shape.idx_top_k > INT_MAX - 3)
    throw std::invalid_argument("invalid private head-parallel QSA geometry");
  validate_spans({{scores, checked_count(nq, blocks) * 4}},
      {{pooled, checked_count(blocks, 128) * 4}, {dead, 512},
       {query, checked_count(nq, 512) * 4}, {steps, checked_count(nq, kStepCount) * 4}});
  auto event = queue_for(stream).parallel_for(
      sycl::nd_range<1>(size_t(((nq + 7) / 8) * active) * 32, 32),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t q0 = (it.get_group_linear_id() / active) * 8;
        const int64_t b = it.get_group_linear_id() % active;
        const int lane = it.get_local_linear_id(), head = lane / 8, l = lane % 8;
        float common[4][4];
        for (int j = 0; j < 4; ++j)
          for (int d = 0; d < 4; ++d)
            common[j][d] = pooled[b * 128 + (l + j * 8) * 4 + d];
#pragma unroll
        for (int row = 0; row < 8; ++row) {
          const int64_t qi = q0 + row;
          if (qi >= nq) continue;
          const auto st = steps + qi * kStepCount;
          const int64_t n = st[kStepNKv], nb = st[kStepNBid];
          if (n < 0 || nb != n / 4 || nb >= active || b > nb) continue;
          const float *q = query + qi * 512 + head * 128;
          float part[4];
          for (int j = 0; j < 4; ++j) {
            const int base = (l + j * 8) * 4;
            const float a = b == nb ? dead[base] : common[j][0];
            part[j] = a * q[base];
            for (int d = 1; d < 4; ++d) {
              const float k = b == nb ? dead[base + d] : common[j][d];
              part[j] += k * q[base + d];
            }
          }
          float dot = (part[0] + part[2]) + (part[1] + part[3]);
          for (int off = 4; off; off /= 2)
            dot += sycl::permute_group_by_xor(it.get_sub_group(), dot, off);
          float total = 0;
          for (int h = 0; h < 4; ++h) {
            const float hd = sycl::select_from_group(it.get_sub_group(), dot, h * 8);
            total += hd > 0 ? hd : 0.f;
          }
          if (lane == 0) {
            if (b == nb && n % 4) total += 1e9f;
            scores[qi * blocks + b] = total;
          }
        }
      });
  finish(stream, event);
}
} // namespace strata::kernels
