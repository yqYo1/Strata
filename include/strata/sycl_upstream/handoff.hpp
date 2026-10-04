#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::sycl_upstream {
// The pinned upstream uses volatile mapped-memory flags and system fences, not
// atomic RMWs. This implementation is validated on B570/x86 with host USM; it
// does not imply usm_atomic_host_allocations support or portable SYCL behavior.
// Flag/payload storage remains caller-owned through all queued users.
struct HandoffWaitLimit {
    // Diagnostic only: zero preserves the upstream unbounded wait. With a
    // limit, exhaustion writes 1 and exits; callers MUST check timed_out before
    // accepting subsequent results. Each pending wait needs a separate status.
    uint32_t max_polls = 0;
    uint32_t* timed_out = nullptr;
    uint32_t* observed_polls = nullptr; // Optional bounded-probe evidence; written on exit.
};
sycl::event doorbell_ring(sycl::queue&, uint32_t* seq);
sycl::event doorbell_publish(sycl::queue&, const float* x, const int32_t* ids,
    const float* weights, int64_t n, int64_t k, float* x_out, int32_t* ids_out,
    float* weights_out, uint32_t* seq);
sycl::event doorbell_wait(sycl::queue&, const uint32_t* flag, const uint32_t* seq,
    HandoffWaitLimit = {});
sycl::event wait_flag_ge(sycl::queue&, const uint32_t* flag, uint32_t value,
    HandoffWaitLimit = {});
sycl::event wait_flag_ge_or(sycl::queue&, const uint32_t* flag, uint32_t value,
    const uint32_t* skip, HandoffWaitLimit = {});
sycl::event copy_from_mapped(sycl::queue&, float* dst, const float* src, int64_t n);
sycl::event copy_i32_from_mapped(sycl::queue&, int32_t* dst, const int32_t* src, int64_t n);
} // namespace strata::sycl_upstream
