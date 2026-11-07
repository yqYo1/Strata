#pragma once
#include "strata/kernels/qsa_select.hpp"

namespace strata::kernels {
// Cached prefill-only experimental policy. Set before exec; never mutate the
// environment after provider initialization. Default off; enabled requires the
// existing radix4 arm and excludes the cyclic arm.
bool qsa_prefill_topk_early_enabled();

// Dedicated WG256 radix4 exact-completion kernel. The caller must select this
// only for layer-major prefill. Decode and short-input routes keep their entry.
void qsa_block_topk_prefill_early(const float *scores, const int32_t *steps,
    int64_t nq, int64_t blocks, int64_t cap, const QsaShapes &shape,
    int32_t *ids, void *stream, int64_t active = -1);
} // namespace strata::kernels
