// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/kernels/qsa_bf16_attention.hpp"
#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels {
struct QsaBf16PackedScratch {
  // Both tensors have [physical_pages,2,page_size,256] BF16 bit patterns.
  // bad_rows has [physical_pages,2,page_size] INT32 diagnostic bits.
  uint16_t *k = nullptr, *v = nullptr;
  int32_t *bad_rows = nullptr;
};
// Experimental complete physical-pool pack + attention on the same in-order
// GPU queue, with the late-flags BF16/FP32 numerical contract. Each invocation
// refreshes scratch before attention; scratch never represents persistent KV.
// Caller must initialize every physical K8/V8/scale element, including unused
// rows. This component API does not establish that a model session satisfies
// that requirement; production integration must prove or narrow the live extent.
// Same selected-ID/missing-page/diagnostic semantics as the original entry.
// All scratch spans are live registered device allocations and disjoint from
// each other, output and every input. Every check precedes both submissions.
// Caller retains all input/output/scratch through completion. No allocation,
// CPU tensor arithmetic, global FP32 tensor, fallback or environment mutation.
bool qsa_bf16_packed_batch(const uint16_t *q, const QsaAttnPools &pools,
    const int32_t *ids, const int32_t *steps, int64_t cap, const QsaShapes &shape,
    uint16_t *out, int64_t queries, const QsaPromptBf16Bounds &bounds,
    const QsaBf16PackedScratch &scratch, void *stream);
} // namespace strata::kernels
#endif
