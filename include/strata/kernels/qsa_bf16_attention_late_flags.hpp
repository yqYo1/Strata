// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/kernels/qsa_bf16_attention.hpp"
#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels {
// Experimental scheduling control for v1: identical operands/FP32 online math;
// diagnostic-only workgroup reductions move from every tile to kernel end.
// Same admission, missing-page, NaN, lifetime and output contracts as v1.
bool qsa_bf16_attention_late_flags_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds, void *stream);
} // namespace strata::kernels
#endif
