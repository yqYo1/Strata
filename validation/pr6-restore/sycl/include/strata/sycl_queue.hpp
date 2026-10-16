// include/strata/sycl_queue.hpp - the SYCL port's one addition to the engine's API surface.
//
// Every launcher takes `void* stream`, a cudaStream_t where null means the default stream. dpct migrates the
// cast to `(dpct::queue_ptr) stream` and dereferences it, so a null stream is a null sycl::queue* and a crash.
// q_of() is that cast with CUDA's null-stream meaning restored: the default in-order queue.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_async_contract.hpp"
#ifndef STRATA_PR6_CHECKED_DPCT
#error "Build PR6 through its checked validation wrapper"
#endif

namespace strata {
inline sycl::queue* q_of(const void* stream) {
    return pr6_require_checked_queue(stream ? (sycl::queue*) stream : &dpct::get_in_order_queue());
}
}  // namespace strata
