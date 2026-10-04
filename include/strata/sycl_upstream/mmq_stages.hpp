#pragma once
#include "strata/prefill/moe_mmq.hpp"
#include <sycl/sycl.hpp>

namespace strata::sycl_upstream {
// Direct equivalents of src/prefill/moe_mmq.cu's surrounding operations.
// Pointers are device accessible, non-overlapping, and valid through completion.
// Queues must be in-order. No internal allocation or host synchronization.
sycl::event mmq_gather_native(sycl::queue&, const void* gate, const void* up,
    size_t half_bytes, const void* down, size_t down_bytes, void* gu, void* dn);
// Preserves upstream's [first,n) indexing and false/no-launch fallback contract.
bool mmq_gather_native_group(sycl::queue&, const prefill::mmq::GatherGroup&,
    size_t up_offset, size_t half_bytes, size_t down_offset, size_t down_bytes,
    void* gu, size_t gu_stride, void* dn, size_t dn_stride);
sycl::event mmq_gather_strata_q2(sycl::queue&, const uint8_t* blob, void* gu, void* dn);
sycl::event mmq_swiglu(sycl::queue&, const float* gu, float* h,
    int64_t rows, int64_t n_ff, bool interleaved);
sycl::event mmq_iota(sycl::queue&, int32_t* dst, int64_t n);
} // namespace strata::sycl_upstream
