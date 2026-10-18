// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/sycl/runtime.hpp"
#include "strata/prefill/moe_window_routes.hpp"
#include <cstdint>

namespace strata::experimental::vllm_xe2_native16 {

enum class Storage { fp16, bf16 };
enum class Tile { compact_16x64, wide_128x128 };
enum class ScatterLayout { matrix, output_copy };

struct Problem {
  const uint16_t *x = nullptr; // compact X[T,K], same input type as W
  const uint16_t *w = nullptr; // compact W[N,K], no transpose or packing
  uint16_t *y = nullptr;       // Y[T,ldy], independent output type
  int64_t t = 0, n = 0, k = 0, ldy = 0; // ldy=0 means N
  Storage input = Storage::fp16, output = Storage::bf16;
  float beta = 0; // only +/-0 accepted; no alpha, bias or activation
};

// Fixed production MoE Down shape: X[job.rows,640], W[2560,640], FP16.
// The published plan and its uploaded maps/weights must remain immutable and
// alive until the returned event is checked and the queue is drained. Maps
// must contain that exact plan, not arbitrary token IDs. Each job has one
// expert and at most one row per token; an in-order queue serializes jobs.
struct DownScatterProblem {
  const strata::prefill::detail::MoeWindowRoutes *plan = nullptr;
  size_t job_index = 0;
  const uint16_t *x = nullptr, *w = nullptr;
  const int32_t *source = nullptr, *original = nullptr;
  const float *job_weights = nullptr;
  float *accumulator = nullptr; // [plan.rows,2560], explicit FP32 FMA
  uint32_t *status = nullptr;   // invalid slot bit1, token bit2
  // Optional correctness-only BF16 capture[job.rows,2560]. Null for timing;
  // no Down tensor is written or read by the null-capture specialization.
  uint16_t *rounded_diagnostic = nullptr;
};

struct DownOwnership {
  size_t cells = 0, missing = 0, multiple = 0, masked = 0;
};
// Host-only inspection of the selected epilogue's actual TLA coordinates.
// Exhaustively counts every tile/lane/fragment element, including M tails.
DownOwnership inspect_down_ownership(int64_t rows, Tile tile,
                                     ScatterLayout layout = ScatterLayout::matrix);

// Source proposal only. Does not alter Gemm::*_out16 or select a runtime/device.
// Runtime, queue and all registered Allocation owners must outlive submitted
// work. Caller serializes owner destruction/reallocation with submissions.
// Queue must have been made by this Runtime (compute/transfer/make_queue), be
// in order, and remain alive at its original address for this object's life.
// This object borrows them; it owns no runtime, queue, USM or scratch buffer.
class Gemm {
public:
  Gemm(sycl_backend::Runtime &runtime, sycl::queue &queue);
  Gemm(const Gemm &) = delete;
  Gemm &operator=(const Gemm &) = delete;

  // Returns the actual GEMM event, without a wait, copy, retry or fallback.
  // All synchronous guards run before enqueue. An enqueue/asynchronous error
  // still requires the caller to drain the queue before releasing any owner.
  // Explicit tile makes later operator comparisons possible without env edits.
  sycl::event beta0(const Problem &problem, Tile tile = Tile::wide_128x128);
  // Component candidate only; no engine dispatch selects this API yet.
  // Same DPAS order, then BF16 rounding/widening, then FP32 sycl::fma.
  sycl::event down_scatter(const DownScatterProblem &problem,
                         Tile tile = Tile::wide_128x128,
                         ScatterLayout layout = ScatterLayout::matrix);
  sycl::queue &queue() const { return queue_; }

private:
  sycl_backend::Runtime &runtime_;
  sycl::queue &queue_;
  // Immutable per-object query result; constructor either succeeds or throws.
  const bool has_fp16_;
};

} // namespace strata::experimental::vllm_xe2_native16
