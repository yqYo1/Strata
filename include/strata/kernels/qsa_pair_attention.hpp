// Main-authored Strata experiment; repository LICENSE (MIT) applies.
#pragma once
#include "strata/kernels/qsa_bf16_prefix.hpp"
#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels {
// Additional pair producer/consumer device requirements;no allocation or
// enqueue. Common BF16 input/device validation remains required separately.
// Model admission and the composed wrapper use this same predicate.
bool qsa_pair_prefill_device_supported(void *stream);

// Attention-only two-query kernel;the pair-prefix wrapper refreshes its inputs.
// packed must already contain the original logical BF16 prefix and bad-row
// flags, produced after the current KV/page-table writes on this in-order
// queue. This function does NOT refresh it. All inputs stay immutable/live
// through completion; a different producing queue requires explicit ordering.
// Pair metadata follows qsa_pair_union_prefill. Counts<0 signal invalid input;
// no uninitialized payload is read. original_ids is retained for the existing
// admission contract; the device consumes only the explicit union/membership.
// BF16 Q/P/O storage, local FP32 dot/softmax/output accumulation. Each query
// and KV head has independent masks and online state. A new tile grouping and
// tile boundary need numerical/task qualification, not bit equality. Retains
// the selected ascending FP32 tile sum and typed four-value K/V staging.
// Existing admission permits at most65535 query rows per call;262144 refers
// to physical KV extent, not a claim that a full model context is qualified.
// Packed rows/diagnostic flags must be exactly those of the logical-prefix
// producer, including flags for all nonfinite scales. Bad rows stage zeros
// while their member queries receive NaNs, isolating0*NaN across queries.
bool qsa_pair_attention_prefill(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *original_ids, const int32_t *steps, int64_t cap,
    const QsaShapes &shape, uint16_t *out, int64_t queries,
    const QsaPromptBf16Bounds &bounds, int64_t live_cells,
    const QsaBf16PackedScratch &packed, const int32_t *unions,
    const uint8_t *membership, const int32_t *counts, void *stream);
} // namespace strata::kernels
#endif
