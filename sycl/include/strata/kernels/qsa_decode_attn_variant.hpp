#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

namespace strata::kernels {

// SYCL launch/layout experiments on the original split-K attention. Same scratch size.
// 1: transpose the shared query tiles; 2: pair the original 32 lanes on subgroup 16;
// 3: both; 4: both with neighboring queries along the innermost workgroup dimension.
// The paired reduction performs the original XOR-16 addition first.
void qsa_decode_attn_batch_variant(const float* q, const QsaAttnPools& pools,
                                   const int32_t* ids, const int32_t* steps, int64_t cap,
                                   const QsaShapes& s, float* scratch, float* attn,
                                   int64_t n_q, int variant, void* stream);

}  // namespace strata::kernels
