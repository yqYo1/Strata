// Main-authored Strata experiment; repository LICENSE (MIT) applies.
#pragma once
#include "strata/kernels/qsa_select.hpp"

namespace strata::kernels {
// Isolated prefill candidate, with no production caller. Preserves the existing
// radix4 selector's weighted-cell rank, tie break and ascending output IDs.
// Requires SG16/WG256. Main must qualify this exact entrypoint before adoption.
void qsa_block_topk_batched_prefill(const float *scores, const int32_t *steps,
    int64_t nq, int64_t blocks, int64_t cap, const QsaShapes &shape,
    int32_t *ids, void *stream, int64_t active = -1);
} // namespace strata::kernels
