// Main-authored Strata experiment; repository LICENSE (MIT) applies.
// Rank/emission contract: selected6e40 src/kernels/sycl/qsa_select.cpp,
// SHA256 463a9ec02ed28125a5cd38b7b5d28784b849a4fbb199cb017042cd99078efee1.
// Only the sixteen per-digit WG reductions are replaced, by a batched two-level
// integer reduction. No inference caller/build target selects this candidate.
#include "strata/kernels/qsa_radix_batched.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <climits>

namespace strata::kernels {
namespace {
constexpr int WG = 256, SG = 32, Bins = 16, Groups = WG / SG;
static_assert(Bins % Groups == 0 && Groups <= SG && Groups == 8);

uint32_t key(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return 0;
  if ((bits & 0x7fffffffu) == 0) bits = 0;
  return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
template<bool Batched>
void radix4(const float *scores, const int32_t *steps,
    int64_t nq, int64_t blocks, int64_t cap, const QsaShapes &shape,
    int32_t *ids, void *stream, int64_t active) {
  using namespace sycl_backend;
  // Same public shape/span contract as the original selector, before enqueue.
  if (nq <= 0 || nq > 65535 || blocks <= 0 || blocks > INT_MAX ||
      shape.idx_dim != 128 || shape.idx_n_head != 4 || shape.idx_block != 4 ||
      shape.idx_top_k < 0 || shape.idx_top_k > INT_MAX - 3 || cap <= 0 ||
      cap > INT_MAX || (active != -1 && (active <= 0 || active > blocks)))
    throw std::invalid_argument("invalid batched SYCL block selection geometry");
  const int64_t launched = active == -1 ? blocks : active;
  validate_spans({{ids, checked_count(nq, cap) * 4}},
      {{scores, checked_count(nq, blocks) * 4},
       {steps, checked_count(nq, kStepCount) * 4}});
  auto &queue = queue_for(stream);
  const auto device = queue.get_device();
  const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
  if (device.get_info<sycl::info::device::max_work_group_size>() < WG ||
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < WG ||
      std::find(sizes.begin(), sizes.end(), SG) == sizes.end())
    throw std::invalid_argument("batched selection requires WG256/SG32");
  auto event = queue.submit([&](sycl::handler &handler) {
    sycl::local_accessor<int, 1> partial(Bins * Groups, handler);
    sycl::local_accessor<int, 1> totals(Bins, handler);
    handler.parallel_for(sycl::nd_range<1>(size_t(nq) * WG, WG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
      const int64_t qi = item.get_group_linear_id();
      const auto st = steps + qi * kStepCount;
      const int64_t n = st[kStepNKv], nb = st[kStepNBid], width = st[kStepWidth];
      // Every branch here is uniform for the whole workgroup.
      if (n < 0 || nb != n / 4 || nb >= launched || width < 0 ||
          width > cap || width > n) return;
      const int tid = item.get_local_linear_id();
      auto out = ids + qi * cap;
      if (n == width) {
        for (int64_t j = tid; j < n; j += WG) out[j] = int32_t(j);
        return;
      }
      if (!width) return;
      const auto subgroup = item.get_sub_group();
      const int group = subgroup.get_group_linear_id();
      const int lane = subgroup.get_local_linear_id();
      const auto sc = scores + qi * blocks;
      const int64_t chunk = (nb + 1 + WG - 1) / WG, lo = tid * chunk;
      const int64_t hi = sycl::min(nb + 1, lo + chunk);
      auto weight = [=](int64_t block) { return block < nb ? 4 : int(n % 4); };
      int rank = int(width);
      uint32_t threshold = 0, mask = 0;
      for (int shift = 28; shift >= 0; shift -= 4) {
        int histogram[Bins] = {};
        for (int64_t block = lo; block < hi; ++block) {
          const uint32_t value = key(sc[block]);
          if ((value & mask) == threshold)
            histogram[(value >> shift) & 15u] += weight(block);
        }
        if constexpr (Batched) {
          // All SG lanes execute all sixteen reductions. Integer sums cover
          // disjoint cells and are <=n<=INT_MAX, so reassociation is exact.
          for (int digit = 0; digit < Bins; ++digit) {
            const int sum = sycl::reduce_over_group(subgroup, histogram[digit], sycl::plus<int>());
            if (!lane) partial[digit * Groups + group] = sum;
          }
          item.barrier(sycl::access::fence_space::local_space);
          // Eight SGs each own two digits. All32 lanes use a bounded address;
          // only lanes0..7 contribute their distinct subgroup partials. The
          // other24lanes supply zero and still participate in both reductions.
          for (int digit = group; digit < Bins; digit += Groups) {
            const int value = partial[digit * Groups + lane % Groups];
            const int sum = sycl::reduce_over_group(
                subgroup, lane < Groups ? value : 0, sycl::plus<int>());
            if (!lane) totals[digit] = sum;
          }
          item.barrier(sycl::access::fence_space::local_space);
        } else {
          // Private control: original radix4 WG reductions, same forced SG32
          // and guarded public entry. Do not infer baseline subgroup size.
          for (int digit = 0; digit < Bins; ++digit)
            histogram[digit] = sycl::reduce_over_group(
                item.get_group(), histogram[digit], sycl::plus<int>());
        }
        int digit = Bins - 1;
        const auto count = [&](int d) {
          if constexpr (Batched) return totals[d];
          else return histogram[d];
        };
        while (digit > 0 && rank > count(digit)) {
          rank -= count(digit);
          --digit;
        }
        threshold |= uint32_t(digit) << shift;
        mask |= 15u << shift;
        // The next iteration's first WG barrier separates all current totals
        // readers from its next totals writes. No third barrier is needed.
      }
      int gt = 0, eq = 0;
      for (int64_t block = lo; block < hi; ++block) {
        const uint32_t value = key(sc[block]);
        if (value > threshold) gt += weight(block);
        if (value == threshold) eq += weight(block);
      }
      const int budget = int(width) - sycl::reduce_over_group(item.get_group(), gt, sycl::plus<int>());
      const int eqstart = sycl::exclusive_scan_over_group(item.get_group(), eq, sycl::plus<int>());
      const int chosen = gt + sycl::min(eq, sycl::max(0, budget - eqstart));
      int offset = sycl::exclusive_scan_over_group(item.get_group(), chosen, sycl::plus<int>());
      int eqrank = eqstart;
      for (int64_t block = lo; block < hi; ++block) {
        const uint32_t value = key(sc[block]);
        int take = value > threshold ? weight(block) : 0;
        if (value == threshold) {
          take = sycl::min(weight(block), sycl::max(0, budget - eqrank));
          eqrank += weight(block);
        }
        for (int j = 0; j < take; ++j) out[offset++] = int32_t(block * 4 + j);
      }
    });
  });
  finish(stream, event);
}
} // namespace

void qsa_block_topk_batched_prefill(const float *scores, const int32_t *steps,
    int64_t nq, int64_t blocks, int64_t cap, const QsaShapes &shape,
    int32_t *ids, void *stream, int64_t active) {
  radix4<true>(scores, steps, nq, blocks, cap, shape, ids, stream, active);
}
void qsa_block_topk_sg32_control(const float *scores, const int32_t *steps,
    int64_t nq, int64_t blocks, int64_t cap, const QsaShapes &shape,
    int32_t *ids, void *stream, int64_t active) {
  radix4<false>(scores, steps, nq, blocks, cap, shape, ids, stream, active);
}
} // namespace strata::kernels
