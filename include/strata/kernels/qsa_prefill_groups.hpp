#pragma once

#include "strata/kernels/qsa_bf16_prefix.hpp"

#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels {

// Paired long-prefill component entrypoints. Both arms currently reserve nq*cap
// int32 words, even though group rows use a shorter physical stride. The
// selector writes ordered block IDs and a cut header. Only the grouped
// attention entrypoint consumes this representation. Use the same in-order
// queue and unchanged steps. A rejected attention call requires explicit
// selection again before an ordinary attention fallback can read IDs.
void qsa_prefill_select_groups(const float *scores, const int32_t *steps,
    int64_t nq, int64_t blocks, int64_t cap, const QsaShapes &shape,
    int32_t *groups, void *stream, int64_t active);
bool qsa_prefill_attend_groups(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *groups, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);

} // namespace strata::kernels
#endif
