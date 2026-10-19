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
// Cached Q and typed four-value K/V staging. Same scalar local layout,
// arithmetic/barriers and two-byte packed-scratch alignment contract.
bool qsa_bf16_prefix_kvvec4_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
// Experimental cached-Q/KVvec4 route: parallel exp2, ascending FP32 sum.
// All SIMD16 lanes execute each indexed exchange; no tree-sum reassociation,
// extra workgroup barrier, local memory or CPU arithmetic is introduced.
bool qsa_bf16_prefix_ordered_exp_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);
// Component-only cached-Q/KVvec4 variant: each lane adds cells lane/lane+16,
// then SIMD16 reduces the FP32 sums. Reduction order is not a bitwise contract.
// The maximum, exp2, alpha, BF16 probability and FP32 FMA update are unchanged.
bool qsa_bf16_prefix_parallel_sum_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch, void *stream);

// Diagnostic-only online state for one query, KV group and head row. Includes
// padded head rows12..15; they follow the original state update too. Each
// ceil(cap/32) row stores FP32 [maximum, alpha, normalizer, tile_sum]. Caller
// owns and initializes the registered disjoint device span through checked
// completion. Entries beyond the selected width stay unchanged. Zero queries
// follow the normal no-op contract.
struct QsaPrefixStateProbe {
  int64_t query = 0;
  int kv_head = 0, head = 0;
  float *states = nullptr;
};
// Diagnostic counterpart of parallel_sum; same state-span admission contract.
// Do not collect performance from this specialization.
bool qsa_bf16_prefix_parallel_sum_probe_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch,
    const QsaPrefixStateProbe &probe, void *stream);
// Explicit bool selects matched serial or ordered-exp diagnostic kernels.
// It does not mutate environment or enable a model dispatch route. Compare
// instrumented output to each corresponding normal export before interpreting
// these additional state stores. Performance uses the non-probe exports only.
bool qsa_bf16_prefix_state_probe_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *output, int64_t queries, const QsaPromptBf16Bounds &bounds,
    int64_t live_cells, const QsaBf16PackedScratch &scratch,
    bool ordered_exp, const QsaPrefixStateProbe &probe, void *stream);
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
