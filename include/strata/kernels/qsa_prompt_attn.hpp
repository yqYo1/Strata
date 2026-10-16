// include/strata/kernels/qsa_prompt_attn.hpp - perf-review D-1: the prompt path's QSA attention on tensor cores.
//
// `qsa_decode_attn_batch` serves a prompt one query at a time with the decode kernel: FP32 dot products with a warp
// reduction per head and cell, 64-cell chunks whose partial sums go through global scratch to a second (merge)
// kernel. At 32K that is 21% of the prompt (5.6 s of 26 s on a 5070, Q2_0, int8 KV), and it is limited by
// instruction issue, not memory (~150 GB/s of logical reads, 3.4 TFLOP/s).
//
// This kernel keeps the same per-query selection (no masking, no union) and reads the same pools, but:
//   * one block per (query, KV head) walks all its selected cells in chunks of 32 with an online softmax: no
//     split-K scratch and no merge kernel;
//   * q.k and p.v are m16n8k16 FP16 MMAs with FP32 accumulation over the 12 query heads of the KV head (+4 pad rows);
//   * the stored values enter exactly: int8 codes are exact in FP16 and their per-64 scales are applied in FP32
//     (to the q.k partial of each 64-dim group, and folded into p for p.v); FP16 KV is used as is;
//   * q and p are split into FP16 hi + lo parts (two MMAs each), so they keep ~22 bits: the result differs from
//     the FP32 kernel by summation order and the exp2 rounding, not by an FP16 cast.
// Not bitwise equal to `qsa_decode_attn_batch`; `qsa_prompt_attn_parity` bounds the difference and the prompt
// quality gate (needles, teacher-forced top-1) checks it end to end. Q4_0 KV (mode 4): each block's codes enter
// exactly as int8 and its scale (one per 32 values) in FP32, as int8 KV's do (STRATA_PROMPT_ATTN_Q4=0: the old kernel).
// SYCL uses Intel joint_matrix m8n16k16 with the same FP32 scale/hi+lo contract,
// including K8V4. Devices without those matrix shapes or enough local memory
// return false before launching, so the caller can use split attention.
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// Same arguments and output as `qsa_decode_attn_batch` minus the scratch. Returns false (nothing launched) when the
/// geometry is not 24 heads / 2 KV heads / 256: the caller then uses the old kernel.
bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream);

#if defined(STRATA_ENABLE_SYCL)
/// Bounds of the allocations behind pools; both counts are positive, and the
/// caller must provide the actual accessible extents (not the selected width).
/// Paged pools hold [physical_pages, 2, page_size, 256] INT8 codes and
/// [physical_pages, 2, page_size, 4] FP16 scales. Table has logical_pages entries.
struct QsaPromptBf16Bounds {
    int64_t logical_pages = 0;
    int64_t physical_pages = 0;
};

/// Explicit native-16 candidate, independent of all prompt environment flags.
/// q/attn are BF16 bit patterns [n_q,24,256]; INT8 K/V and FP16 scales only.
/// BF16 -> one scaled FP16 Q part, one FP16 probability part, FP32 scores,
/// normalizers and matrix accumulators, then BF16 round-to-nearest-even output.
/// Same selected IDs/width semantics as the FP32 prompt entry: masking is the
/// selector's responsibility. Negative/out-of-bounds pages/IDs are missing.
/// Invalid widths produce zero; cap <= 0 and unsupported geometry/mode/device
/// return false without submission. n_q==0 is a successful no-op.
/// Invalid/null/foreign/host/unregistered/overlapping spans throw before submission.
/// Full spans must fit live Strata device allocations in the queue context/device. Caller
/// owns truthful extents and allocation lifetimes through stream completion.
/// Nonfinite Q or live KV scales produce BF16 quiet NaN when rows contribute;
/// no selected/resident rows produce zero, including for nonfinite Q.
/// No global FP32 query/output or scratch; no default route is changed.
bool qsa_prompt_attn_bf16_batch(const uint16_t* q, const QsaAttnPools& pools,
                                const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, uint16_t* attn,
                                int64_t n_q, const QsaPromptBf16Bounds& bounds,
                                void* stream);
/// Same complete validation/capability checks; never submits or waits.
bool qsa_prompt_attn_bf16_validate(const uint16_t* q, const QsaAttnPools& pools,
                                const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, uint16_t* attn,
                                int64_t n_q, const QsaPromptBf16Bounds& bounds,
                                void* stream);
#endif

}  // namespace strata::kernels
