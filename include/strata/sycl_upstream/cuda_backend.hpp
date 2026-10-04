#pragma once
#include "strata/sycl_upstream/cuda/cuda_runtime.h"
#include "strata/sycl_upstream/memory.hpp"
namespace strata::sycl_upstream::cuda {
// Kernel adapters resolve the opaque frontend stream through this function.
// A cudaStream_t is never a sycl::queue*. The Runtime callback rules apply.
cudaError_t submit(cudaStream_t, const Runtime::Submit&) noexcept;
Memory::Stats memory_stats();
std::optional<Memory::Info> allocation_info(const void*);
// Diagnostic detail for the last caught backend exception on this host thread.
// CUDA's public error string stays stable for each symbolic error code.
const char* backend_error_detail() noexcept;
}
