// Main-authored same-build production oracle adapter; repository LICENSE applies.
#include "strata/kernels/qsa_bf16_prefix.hpp"

namespace strata::kernels {
bool qsa_k_model_test_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream) {
  return qsa_bf16_prefix_ordered_exp_batch(query, pools, ids, steps, cap, shape,
      output, queries, bounds, live_cells, scratch, stream);
}
} // namespace strata::kernels
