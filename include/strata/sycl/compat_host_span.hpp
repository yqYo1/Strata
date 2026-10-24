#pragma once
#include "strata/sycl/runtime.hpp"

namespace strata::sycl_backend::compat {
// Query the existing compatibility allocator registry, without changing its
// allocation/free behavior. Accept only a live cudaHostAlloc-owned host-USM
// subspan in runtime0's queue domain. No enqueue/wait; caller retains the owner
// and excludes concurrent CPU access/free until checked queue completion.
void validate_host_span(const void* pointer, size_t bytes, const sycl::queue& queue);
} // namespace strata::sycl_backend::compat
