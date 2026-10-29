// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors
#pragma once
#include "strata/kernels/qsa_prompt_attn.hpp"
#include <cstdint>

namespace strata::prefill::native16::qsa {
// Explicit writable owner view. The attention pool API is intentionally const;
// append must never cast away its constness.
struct Q8WritePools {
    int8_t* k_q = nullptr;
    int8_t* v_q = nullptr;
    uint16_t* k_scale = nullptr;
    uint16_t* v_scale = nullptr;
    const int32_t* page_table = nullptr;
};
// Startup-only opt-in. A short request does not activate this route.
bool requested();
// Startup-only arithmetic selector. Default preserves the qualified typed path.
// BF16 operands follow the pinned same-model SGLang prefill design; prefix
// packing is transient and belongs to the dedicated long-prefill lease.
enum class AttentionMath { existing, bf16, bf16_prefix, bf16_prefix_optimized, bf16_prefix_tilepack, bf16_prefix_subgroup, bf16_prefix_qcache, bf16_prefix_kvec4, bf16_prefix_kvvec4, bf16_prefix_ordered_exp, bf16_prefix_pair };
AttentionMath attention_math();
// No submit. Required before enabling this path for any layer.
void validate_device(void* stream);
// BF16 input/output; local FP32 RMS and rotation, first 64 dimensions rotary.
// One head has 256 values. input_head_stride=512 skips the gate half of Q.
// cos/sin are the session-owned [table_rows,32] FP32 tables, not new tables.
// Disjoint output; writes only heads*256 values per row, preserving padding.
void norm_rope_split(const uint16_t* input, const float* weight,
                     uint16_t* output, int64_t tokens, int64_t heads,
                     int64_t input_head_stride, int64_t input_ld,
                     int64_t output_ld, int64_t pos0, int64_t table_rows,
                     const float* cos, const float* sin, float eps, void* stream);
// BF16 attention and full BF16 [head,query256,gate256] projection -> FP16
// projection input. No saturation, host math, FP32 intermediate or allocation.
void gate(const uint16_t* attention, const uint16_t* query_and_gate,
          uint16_t* output_fp16, int64_t tokens, int64_t attention_ld,
          int64_t query_ld, int64_t output_ld, void* stream);
// BF16 K/V -> paged INT8 codes and FP16 per-64 scales, directly on device.
// Resident only; no mirror, stage, ring, rotation or folded pools. Capacities
// are actual owner counts. Invalid table pages are skipped without access.
// Caller owns an injective mapping for written cells (production mode0 uses
// the identity table); concurrent aliases/mutation of that table are invalid.
// Nonfinite input groups emit NaN scale + zero codes, retaining a diagnostic.
void append_q8(const uint16_t* k, const uint16_t* v, int64_t tokens,
               int64_t input_ld, int64_t pos0,
               const Q8WritePools& pools,
               const strata::kernels::QsaShapes& shapes,
               const strata::kernels::QsaPromptBf16Bounds& bounds, void* stream);
} // namespace strata::prefill::native16::qsa
