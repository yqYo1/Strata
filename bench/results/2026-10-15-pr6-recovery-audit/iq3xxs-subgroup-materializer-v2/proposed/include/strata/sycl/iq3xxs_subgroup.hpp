#pragma once
#include "strata/sycl/iq_parallel_decode.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace strata::sycl_backend {
// Read only during static initialization; environment is immutable after exec.
inline const unsigned iq3xxs_subgroup_width = [] {
  const char *v = std::getenv("STRATA_SYCL_IQ3XXS_SUBGROUP");
  if (v && std::strcmp(v, "sg16") == 0) return 16u;
  if (v && std::strcmp(v, "sg32") == 0) return 32u;
  return 0u;
}();

template <unsigned S, bool GU>
bool try_iq3xxs_subgroup(const void *gate, const void *up, size_t count,
                         size_t width, uint16_t *out, void *stream) {
  static_assert(S == 16 || S == 32);
  // Caller has validated complete spans, counts, width and alignment.
  auto &q = queue_for(stream);
  const auto sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  constexpr size_t groups_per_wg = 4, local = groups_per_wg * S;
  if (std::find(sizes.begin(), sizes.end(), S) == sizes.end() ||
      q.get_device().get_info<sycl::info::device::max_work_group_size>() < local ||
      q.get_device().get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < local)
    return false;
  const size_t blocks = count / 256;
  const size_t workgroups = blocks / groups_per_wg + (blocks % groups_per_wg != 0);
  if (workgroups > SIZE_MAX / local) return false;
  auto event = q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(workgroups * local), sycl::range<1>(local)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(S)]] {
        const auto sg = item.get_sub_group();
        const unsigned lane = sg.get_local_linear_id();
        const size_t bid = item.get_group_linear_id() * groups_per_wg +
                           sg.get_group_linear_id();
        // Subgroup-uniform guard before all collectives, including padded SGs.
        if (bid >= blocks) return;
        const uint8_t *block;
        if constexpr (GU) {
          const size_t bpr = width / 256;
          const size_t row = bid / bpr, col = bid % bpr;
          const auto *base = static_cast<const uint8_t *>(row % 2 ? up : gate);
          block = base + ((row / 2) * bpr + col) * 98;
        } else block = static_cast<const uint8_t *>(gate) + bid * 98;
        float d = 0.f;
        if (lane == 0) d = fp16_to_fp32(read_u16(block));
        d = sycl::group_broadcast(sg, d, 0);
        // Only d is constant over the block. aux and db change every 32 values.
        for (unsigned group = 0; group < 8; ++group) {
          uint32_t aux = 0;
          float db = 0.f;
          if (lane == 0) {
            const auto *p = block + 66 + group * 4;
            aux = uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
                  (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
            db = d * (0.5f + (aux >> 28)) * 0.5f;
          }
          aux = sycl::group_broadcast(sg, aux, 0);
          db = sycl::group_broadcast(sg, db, 0);
          for (unsigned half = 0; half < 32 / S; ++half) {
            // Equivalent to j = iteration*S + lane, iteration=group*32/S+half.
            const unsigned j = group * 32 + half * S + lane;
            const unsigned word = (j % 32) / 4, byte = j % 4;
            uint32_t grid = 0, signs = 0;
            if (lane % 4 == 0)
              grid = iq3_xxs_grid[block[2 + group * 8 + word]];
            if (lane % 8 == 0)
              signs = iq_even_parity_signs(aux >> (7 * (word / 2)));
            // All lanes participate. Remote ids vary by cluster, legally.
            grid = sycl::select_from_group(sg, grid, sycl::id<1>((lane / 4) * 4));
            signs = sycl::select_from_group(sg, signs, sycl::id<1>((lane / 8) * 8));
            const float value = db * int((grid >> (8 * byte)) & 255) *
                ((signs >> (4 * (word % 2) + byte)) & 1 ? -1.f : 1.f);
            out[bid * 256 + j] = strata::kernels::f16_from_f32(value);
          }
        }
      });
  finish(stream, event);
  return true;
}

template <bool GU>
bool dispatch_iq3xxs_subgroup(const void *gate, const void *up, size_t count,
                              size_t width, uint16_t *out, void *stream) {
  switch (iq3xxs_subgroup_width) {
  case 16: return try_iq3xxs_subgroup<16, GU>(gate, up, count, width, out, stream);
  case 32: return try_iq3xxs_subgroup<32, GU>(gate, up, count, width, out, stream);
  default: return false;
  }
}
} // namespace strata::sycl_backend
