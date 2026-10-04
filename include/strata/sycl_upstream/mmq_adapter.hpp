#pragma once
#include "strata/prefill/moe_mmq.hpp"
#include <sycl/sycl.hpp>

namespace strata::sycl_upstream {
// When linked with strata_sycl_mmq, the original prefill::mmq API takes a
// non-null pointer to an in-order sycl::queue as its opaque stream. This is a
// component contract; a CUDA compatibility runtime has not been integrated.
// Context retains queue copies and scratch until teardown. Callers retain all
// Product inputs/outputs until queue completion, as with the upstream API.
// Public fits() checks all visible GPUs; this helper checks a supplied device.
bool mmq_device_fits(const sycl::device& device, int type, int64_t weight_rows);
} // namespace strata::sycl_upstream
