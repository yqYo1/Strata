// Authored for Strata; repository LICENSE applies.
#pragma once
#include "strata/kernels/qsa_bf16_prefix.hpp"
#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels::prefill {
// Component-only long-prefill candidate. The independent128-thread production
// export stays unchanged. Admission,rounding,32-cell tiles and ordered FP32
// state match it;SG16 ownership changes coherently with the64-thread workgroup.
bool qsa_bf16_prefix_workgroup64_batch(const uint16_t *query, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
} // namespace strata::kernels::prefill
#endif
