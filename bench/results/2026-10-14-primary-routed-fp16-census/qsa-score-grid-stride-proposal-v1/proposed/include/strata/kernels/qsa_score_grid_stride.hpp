#pragma once

#include <cstdint>

namespace strata::kernels::detail {

// Host policy only: missing, empty and every spelling except exactly "1" are off.
inline bool score_grid_stride_enabled(const char *value) {
  return value && value[0] == '1' && value[1] == 0;
}

// Construct only with the original positive bound returned by validate().
// Capacity/active bound and launch group count have different meanings.
struct ScoreGridStride {
  const int64_t active_bound;
  const int64_t group_count;

  constexpr explicit ScoreGridStride(int64_t bound)
      : active_bound(bound), group_count(bound < 4096 ? bound : 4096) {}

  constexpr int64_t query_for_group(int64_t group) const {
    return group / group_count;
  }
  constexpr int64_t first_block(int64_t group) const {
    return group % group_count;
  }

  template <typename Function>
  void for_each_block(int64_t n, int64_t nb, int64_t first,
                      Function function) const {
    // Read n/nb from the device step, just as in the original score kernel.
    // nb >= group_count would incorrectly reject long-context queries.
    if (n < 0 || nb != n / 4 || nb >= active_bound)
      return;
    for (int64_t b = first; b <= nb; b += group_count)
      function(b);
  }
};

} // namespace strata::kernels::detail
