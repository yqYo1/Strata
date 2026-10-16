// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/kernels/qsa_prompt_attn.hpp"

#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels {
// Experimental BF16 matrix operands, FP32 online softmax/accumulation.
// Q/O are BF16. Selected K8/V8 codes times FP16 per-64 scales round to
// BF16 in local tiles; P rounds to BF16 before P.V. Same complete-span,
// registered-device, nonaliasing, shape and lifetime contract as the old
// typed entry. No production caller selects this entry yet.
bool qsa_bf16_attention_batch(const uint16_t *q, const QsaAttnPools &pools,
                             const int32_t *ids, const int32_t *steps,
                             int64_t cap, const QsaShapes &shape, uint16_t *out,
                             int64_t queries, const QsaPromptBf16Bounds &bounds,
                             void *stream);
} // namespace strata::kernels
#endif
