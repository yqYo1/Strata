// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/kernels/qsa_select.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::prefill {
class Gemm;
namespace qsa_score_gemm {

bool requested(); // Immutable preexec STRATA_SYCL_PREFILL_SCORE_GEMM=0|1; default0.
// Allocation/submission-free capability check. The engine calls this at scope
// admission, before any persistent KV/indexer state can be modified.
void validate_device(void *stream);

struct Scratch {
  uint16_t *pooled_fp16 = nullptr; // [round_up(max_blocks,64),128]
  uint16_t *query_fp16 = nullptr;  // [max_queries*4,128]
  float *heads_fp32 = nullptr;    // [max_queries*4,tile_blocks]
  uint32_t *status = nullptr;     // INDEPENDENT of any reused model arena
  int64_t tile_blocks = 8192;
};

// No allocation or submission. The whole returned layout must fit the actual
// dead device-arena reservation. Half/query/head views are aligned/disjoint.
// status is a separately owned four-byte device word, not counted here.
size_t workspace_bytes(int64_t max_queries, int64_t max_blocks,
                       int64_t tile_blocks = 8192);
Scratch carve(void *arena, size_t bytes, uint32_t *status,
              int64_t max_queries, int64_t max_blocks,
              int64_t tile_blocks = 8192);

// GPU FP16-rounded operands -> existing GEMM FP32 output -> ReLU/head sum.
// Original FP32 dead-key tree and partial-tail bonus are retained. Completed
// scores and selected IDs may differ from FP32; this is an opt-in experiment.
//
// Caller validates registered device spans/context, serializes host ownership
// and keeps the Gemm, in-order stream and every input/scratch owner alive through
// checked completion. Pooled [0,active) must be initialized and immutable.
// No wrapper allocation, host tensor round trip, fallback or wait is performed.
// Library-internal behavior is separate from that wrapper contract.
//
// Zero status ONCE before the long scope, then retain it across every call.
// This function NEVER clears it. Atomic mask1 means an input is nonfinite or
// outside FP16 range; mask2 means nonfinite GEMM/reduction/dead arithmetic.
// A nonzero word invalidates the scope. Inspect after checked drain before
// exposing results to decode; a later healthy call cannot erase an earlier fault.
void scores(Gemm &gemm, const float *pooled, const float *dead,
            const float *query, const int32_t *steps, int64_t nq,
            int64_t blocks, const kernels::QsaShapes &shape, float *output,
            Scratch scratch, void *stream, int64_t active);

} // namespace qsa_score_gemm
} // namespace strata::prefill
