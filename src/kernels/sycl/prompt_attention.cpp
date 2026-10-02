#include "strata/kernels/qsa_prompt_attn.hpp"
namespace strata::kernels {
bool qsa_prompt_attn_batch(const float *, const QsaAttnPools &, const int32_t *,
                           const int32_t *, int64_t, const QsaShapes &, float *,
                           int64_t, void *) {
  // Matrix attention is not implemented yet. The caller uses the validated
  // batched split-attention kernel with its caller-owned scratch.
  return false;
}
} // namespace strata::kernels
