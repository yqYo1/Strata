#pragma once
#include <cstddef>
namespace strata::sycl_backend::compat {
// Read-only check against the existing cudaMalloc/Allocation registry. No queue
// submission, synchronization, new allocation or TLS error state mutation.
void validate_registered_device_span(const void*,std::size_t);
}
