// Main-authored Strata prefill path; repository LICENSE (MIT) applies.
#pragma once
#include "strata/kernels/qsa_pair_attention.hpp"
#if defined(STRATA_ENABLE_SYCL)
namespace strata::kernels {
struct QsaPairScratch {
  int32_t *unions = nullptr;
  uint8_t *membership = nullptr;
  int32_t *counts = nullptr;
};
// Dedicated prefill pipeline: refresh logical BF16 K/V,produce exact pair
// metadata,consume it on the same in-order queue. No host tensor computation,
// readback,allocation or nonnull-stream wait. All output/scratch/input spans
// are checked together before the first enqueue;caller owns them through drain.
// union/membership each have ceil(queries/2)*2*cap elements;counts has
// ceil(queries/2). The initialized packed prefix and metadata are transient,
// have no decode consumers,and must not survive their owning prefill lease.
bool qsa_bf16_pair_prefix_batch(const uint16_t *q,const QsaAttnPools &pools,
    const int32_t *ids,const int32_t *steps,int64_t cap,const QsaShapes &shape,
    uint16_t *out,int64_t queries,const QsaPromptBf16Bounds &bounds,
    int64_t live_cells,const QsaBf16PackedScratch &packed,
    const QsaPairScratch &pair,void *stream);
} // namespace strata::kernels
#endif
