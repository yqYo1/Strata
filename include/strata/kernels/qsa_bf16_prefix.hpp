// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/kernels/qsa_bf16_packed.hpp"
#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels {
// Transient BF16 storage uses logical [ceil(live/page_size),2,page_size,256].
// Only live logical cells with valid page mappings are loaded from the input.
// Unmapped pages and final-page padding are written as zero. Original input
// pools need initialization ONLY at valid mapped cells in [0, live_cells).
// Scratch is refreshed every call, on the same in-order GPU queue as attention;
// no persistent-cache invalidation, host tensor math, allocation or host wait.
// Selected cells outside [0, live_cells) are missing, as are unmapped pages.
// Caller owns all spans through completion. Admission rejects invalid live
// extent and all scratch/input/output aliases before either GPU submission.
bool qsa_bf16_prefix_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
// Same arithmetic and update order, FP32 score row pitch33 (+64 local bytes).
bool qsa_bf16_prefix_optimized_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
// Experimental score33: retain BF16 Q matrix fragments across all cell tiles,
// then reuse their initial SLM arena for K/V. Unchanged DPAS/rounding order.
bool qsa_bf16_prefix_qcache_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
// Cached Q with four adjacent typed K values per lane, parallel row ownership
// and the same local layout. Diagnostic only; no extra alignment requirement.
bool qsa_bf16_prefix_kvec4_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
// Score33 row-major storage; subgroup FP32 max/sum changes sum association.
// BF16 probability rounding and FP32 online state/accumulation are preserved.
bool qsa_bf16_prefix_subgroup_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
// Same score33 arithmetic/order/payload; dense16 A and BF16 pair-packed B.
bool qsa_bf16_prefix_tilepack_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
} // namespace strata::kernels
#endif
