// Authored for Strata; repository LICENSE (MIT) applies.
#pragma once
#include "strata/sycl/runtime.hpp"
#include <cstdint>

namespace strata::experimental::vllm_xe2_native16 {

enum class Storage { fp16, bf16 };
enum class Tile { compact_16x64, wide_128x128 };

struct Problem {
  const uint16_t *x = nullptr; // compact X[T,K], same input type as W
  const uint16_t *w = nullptr; // compact W[N,K], no transpose or packing
  uint16_t *y = nullptr;       // Y[T,ldy], independent output type
  int64_t t = 0, n = 0, k = 0, ldy = 0; // ldy=0 means N
  Storage input = Storage::fp16, output = Storage::bf16;
  float beta = 0; // only +/-0 accepted; no alpha, bias or activation
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
  sycl::queue &queue() const { return queue_; }

private:
  sycl_backend::Runtime &runtime_;
  sycl::queue &queue_;
  // Immutable per-object query result; constructor either succeeds or throws.
  const bool has_fp16_;
};

} // namespace strata::experimental::vllm_xe2_native16
