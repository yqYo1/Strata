// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/sycl/runtime.hpp"
#include "strata/prefill/moe_window_routes.hpp"
#include <cstdint>

namespace strata::experimental::vllm_xe2_native16 {

enum class Storage { fp16, bf16 };
enum class Tile { compact_16x64, wide_128x128 };
enum class ScatterLayout { matrix, output_copy, matrix_row_cache };
enum class GuLayout { matrix, output_copy };

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
  size_t metadata_slots = 0, metadata_missing = 0, metadata_multiple = 0;
  size_t metadata_outside_tile = 0;
};
// Host-only inspection of the selected epilogue's actual TLA coordinates.
// Exhaustively counts every tile/lane/fragment element, including M tails.
DownOwnership inspect_down_ownership(int64_t rows, Tile tile,
                                     ScatterLayout layout = ScatterLayout::matrix);

// Host-only proof for the independent Down M64/N128 matrix/row-cache layout.
DownOwnership inspect_down64_ownership(int64_t rows);

struct GuPairOwnership {
  size_t gu_values = 0, hidden_values = 0, masked = 0;
  size_t gu_missing = 0, gu_multiple = 0, hidden_missing = 0, hidden_multiple = 0;
  size_t checked_pairs = 0, partner_mismatch = 0;
  size_t packed_missing = 0, packed_multiple = 0, packed_orientation_mismatch = 0;
};
// Host-only proof for interleaved GU[N1280,K2560] -> H[N640]. Tests whether
// each even column's adjacent odd column is held at the same fragment index
// by local lane XOR1 in its SIMD16 subgroup, for both actual TLA layouts.
// A nonzero partner_mismatch is evidence against that schedule, not a GPU run.
GuPairOwnership inspect_gu_pair_ownership(int64_t rows, Tile tile, GuLayout layout);

// Fixed interleaved GU shape: FP16 X[rows,2560], W[1280,2560], FP16 H[rows,640].
// FP32 DPAS -> per-operand BF16 RNE -> FP32 SwiGLU -> finite FP16 saturation.
// All pointers are disjoint registered device spans. The optional BF16 capture
// is correctness-only [rows,1280]; the timed specialization does not store GU.
struct GuSwiGLUProblem {
  const uint16_t *x = nullptr, *w = nullptr;
  uint16_t *y = nullptr;
  int64_t rows = 0;
  uint16_t *rounded_diagnostic = nullptr;
};

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
  // Component candidate and explicit large-job engine hook; default off.
  // Same DPAS order, then BF16 rounding/widening, then FP32 sycl::fma.
  sycl::event down_scatter(const DownScatterProblem &problem,
                         Tile tile = Tile::wide_128x128,
                         ScatterLayout layout = ScatterLayout::matrix);
  // Separate split-accumulator candidate, fixed production wide128 tile and
  // row-cached metadata. Maps remain global window-row IDs; both disjoint
  // FP32 banks survive the same checked queue drain. No implicit allocation.
  sycl::event down_scatter_split(const DownScatterProblem &problem,
                                float *tail, int64_t primary_rows);
  // Experimental component only: same FP32 updates with per-row destination
  // addresses cached in SLM. No production dispatcher calls this method.
  sycl::event down_scatter_split_row_address(const DownScatterProblem &problem,
                                            float *tail, int64_t primary_rows);
  // Down-only component policies; no production caller or shared Tile selector.
  sycl::event down_scatter_split_row64(const DownScatterProblem &problem,
                                     float *tail, int64_t primary_rows);
  sycl::event beta0_down64(const Problem &problem);
  // Explicit large-job long-prefill hook; default off pending model qualification.
  sycl::event gu_swiglu(const GuSwiGLUProblem &problem,
                       Tile tile = Tile::wide_128x128);
  sycl::queue &queue() const { return queue_; }

private:
  sycl_backend::Runtime &runtime_;
  sycl::queue &queue_;
  // Immutable per-object query result; constructor either succeeds or throws.
  const bool has_fp16_;
  const size_t local_mem_bytes_;
};

} // namespace strata::experimental::vllm_xe2_native16
