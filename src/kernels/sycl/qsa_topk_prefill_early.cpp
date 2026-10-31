// Main-authored prefill-only route; repository LICENSE.
// Integer kernel derives from selected6e40 plus the exact-completion insertion
// qualified by source44636e26c/device result1a752d9a3. No external implementation.
#include "strata/kernels/qsa_topk_prefill_early.hpp"
#include "strata/sycl/launch.hpp"
#include <climits>
#include <cstdlib>

namespace strata::kernels {
namespace {
namespace prefill_topk_early_detail {
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
sycl::event submit_topk_prefill_early(sycl::queue &queue, const float *scores,
                        const int32_t *steps, int64_t nq, int64_t blocks,
                        int64_t cap, int32_t *ids, int64_t launched) {
  return queue.parallel_for(
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
        {
          // Rank is in cells, including the partial final block. All counts
          // are subsets of nonnegative int32 n, hence fit int without adding
          // a physical-context restriction to the existing API.
          int rank = int(width);
          uint32_t mask = 0;
          for (int shift = 28; shift >= 0; shift -= 4) {
            int histogram[16] = {};
            // Every round loads every key before checking the prefix.
            for (int64_t b = lo; b < hi; ++b) {
              const uint32_t key = order(sc[b]);
              if ((key & mask) == threshold)
                histogram[(key >> shift) & 15u] += weight(b);
            }
            for (int digit = 0; digit < 16; ++digit)
              histogram[digit] = sycl::reduce_over_group(
                  it.get_group(), histogram[digit], sycl::plus<int>());
            int digit = 15;
            while (digit > 0 && rank > histogram[digit]) {
              rank -= histogram[digit];
              --digit;
            }
            threshold |= uint32_t(digit) << shift;
            mask |= 15u << shift;
            // Higher digits are already selected. If the remaining rank
            // equals this whole bucket, every cell in it is selected: lower
            // digits cannot change the selected set or ascending tie order.
            // rank and histogram are replicated WG reductions, so this break
            // is workgroup-uniform, including zero-weight final blocks.
            if (rank == histogram[digit]) break;
            // Every completed block weighs4; only the final block can weigh
            // 1..3. A nonempty bucket weighing<=4 therefore contains exactly
            // one positive-weight block. Fetch its full key before stopping:
            // using only the prefix would overselect a partial block budget.
            if (histogram[digit] <= 4) {
              uint32_t unique_key = 0;
              for (int64_t b = lo; b < hi; ++b) {
                const uint32_t key = order(sc[b]);
                if (weight(b) > 0 && (key & mask) == threshold)
                  unique_key = key;
              }
              threshold = sycl::reduce_over_group(it.get_group(), unique_key,
                                                    sycl::maximum<uint32_t>());
              break;
            }
          }
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
}
int64_t validate_topk(const float *scores, const int32_t *steps, int64_t nq,
                      int64_t blocks, int64_t cap, const QsaShapes &s,
                      int32_t *ids, int64_t active) {
  const int64_t launched = validate(nq, blocks, active, s);
  if (cap <= 0 || cap > INT_MAX)
    throw std::invalid_argument("invalid SYCL block selection capacity");
  validate_spans({{ids, checked_count(nq, cap) * 4}},
                 {{scores, checked_count(nq, blocks) * 4},
                  {steps, checked_count(nq, kStepCount) * 4}});
  return launched;
}

bool one(const char *raw) {
  return raw && raw[0] == '1' && raw[1] == '\0';
}
bool policy() {
  const char *raw = std::getenv("STRATA_SYCL_PREFILL_TOPK_EARLY");
  if (!raw || (raw[0] == '0' && raw[1] == '\0'))
    return false;
  if (!one(raw))
    throw std::invalid_argument("STRATA_SYCL_PREFILL_TOPK_EARLY requires 0 or 1");
  if (!one(std::getenv("STRATA_SYCL_SELECT_RADIX4")) ||
      one(std::getenv("STRATA_SYCL_SELECT_CYCLIC")))
    throw std::invalid_argument("prefill early Top-k requires radix4=1 and cyclic disabled");
  return true;
}
} // namespace prefill_topk_early_detail
} // namespace

bool qsa_prefill_topk_early_enabled() {
  static const bool enabled = prefill_topk_early_detail::policy();
  return enabled;
}
void qsa_block_topk_prefill_early(const float *scores, const int32_t *steps,
    int64_t nq, int64_t blocks, int64_t cap, const QsaShapes &shape,
    int32_t *ids, void *stream, int64_t active) {
  const int64_t launched = prefill_topk_early_detail::validate_topk(
      scores, steps, nq, blocks, cap, shape, ids, active);
  auto &queue = sycl_backend::queue_for(stream);
  const auto event = prefill_topk_early_detail::submit_topk_prefill_early(
      queue, scores, steps, nq, blocks, cap, ids, launched);
  sycl_backend::finish(stream, event);
}
} // namespace strata::kernels
