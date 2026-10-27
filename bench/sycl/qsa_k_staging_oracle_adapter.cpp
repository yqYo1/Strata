// Authored for Strata; repository LICENSE (MIT) applies.
// Route the existing independent oracle/guard fixture to this private candidate.
// This adapter has no production caller or selection/environment mutation.
#include "qsa-k-staging.hpp"

namespace strata::kernels {
bool qsa_k_staging_test_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  return kstaging_candidate::qsa_bf16_prefix_ordered_exp_batch(query, pools,
      ids, steps, cap, shape, output, queries, bounds, live_cells, scratch,
      stream, nullptr);
}
} // namespace strata::kernels
