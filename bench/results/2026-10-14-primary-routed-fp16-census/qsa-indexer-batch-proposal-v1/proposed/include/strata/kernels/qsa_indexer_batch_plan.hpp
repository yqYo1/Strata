#pragma once
#include <cstdint>

namespace strata::kernels::detail {
// Host/device integer planning only. Caller validates 0 <= p0, n <= capacity-p0,
// capacity <= INT32_MAX before use; no pointer, allocation, or queue ownership.
struct QsaIndexerBatchPlan {
  int64_t p0;
  int64_t end;
  int64_t first_block;
  int64_t completed;
  bool initialize_dead;

  // Completed block rows before p0 are the existing tail slots. All other
  // sources are raw[input_row,128], rounded to F16 by the consuming kernel.
  constexpr int64_t input_row(int64_t block, int row) const {
    const int64_t pos = 4 * block + row;
    return pos < p0 ? -1 : pos - p0;
  }

  // Last serial write to tail slot 0..2. -1 retains the old slot unchanged,
  // including when a short continuation never touches that slot.
  constexpr int64_t tail_input_row(int slot) const {
    if (end <= slot)
      return -1;
    const int64_t last = end - 1 - (end - 1 - slot) % 4;
    return last < p0 ? -1 : last - p0;
  }

  constexpr int64_t last_completed_block() const {
    return first_block + completed - 1;
  }
  constexpr int64_t spare_block() const {
    return first_block + completed;
  }
};

constexpr QsaIndexerBatchPlan qsa_indexer_batch_plan(int64_t p0, int64_t n) {
  return {p0, p0 + n, p0 / 4, (p0 + n) / 4 - p0 / 4,
          p0 == 0 && n > 0};
}
} // namespace strata::kernels::detail
