#pragma once
#include "strata/prefill/moe_mmq.hpp"
#include <sycl/sycl.hpp>

namespace strata::sycl_upstream {
// When linked with strata_sycl_mmq, the original prefill::mmq API takes a
// non-null pointer to an in-order sycl::queue as its opaque stream. This is a
// component-test contract. Engine-facing strata_sycl_mmq_frontend instead uses
// opaque cudaStream_t handles (null is the legacy default stream). Link exactly
// one of these adapters; both share the same kernel implementation.
// Context retains queue copies and scratch until teardown. Callers retain all
// Product inputs/outputs until queue completion, as with the upstream API.
// Warm Context::run on each stream before recording a graph that needs fixup
// scratch. Context and its scratch must outlive all graphs using them.
// Public fits() checks all visible GPUs; this helper checks a supplied device.
bool mmq_device_fits(const sycl::device& device, int type, int64_t weight_rows);
} // namespace strata::sycl_upstream
