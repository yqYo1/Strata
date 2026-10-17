// Main-authored private, uncompiled experiment; repository MIT license.
#pragma once
#include "strata/kernels/qsa_select.hpp"
#include <cstdint>
namespace strata::prefill { class Gemm; }
namespace strata::experimental::qsa_score_gemm {
struct Scratch {
  uint16_t *pooled_fp16 = nullptr; // [ceil(active/64)*64,128]
  uint16_t *query_fp16 = nullptr;  // [nq*4,128]
  float *heads_fp32 = nullptr;    // [nq*4,tile_blocks]
  uint32_t *status = nullptr;     // one word; checked after the final drain
  int64_t tile_blocks = 8192;     // positive multiple64, at most8192
};
// All allocations are registered device memory in the explicit in-order stream.
// Caller owns initialized/immutable full active pooled rows and all inputs, the
// Gemm, stream, runtime and scratch through a checked completion. There is no
// tensor readback, wrapper allocation, fallback or synchronization here.
// Library-internal allocation and actual hardware lowering are unmeasured.
// status!=0 INVALIDATES the result: mask1 input cannot be represented in FP16;
// mask2 GEMM/head reduction or dead-key arithmetic is nonfinite. A caller must
// inspect it after draining before using results. This is NOT a production API.
// FP16 operand rounding and GEMM summation change completed-block scores;
// quality/selection overlap and actual GPU execution must be qualified first.
void scores(prefill::Gemm &gemm, const float *pooled, const float *dead,
            const float *query, const int32_t *steps, int64_t nq,
            int64_t blocks, const kernels::QsaShapes &shape, float *output,
            Scratch scratch, void *stream, int64_t active);
} // namespace strata::experimental::qsa_score_gemm
