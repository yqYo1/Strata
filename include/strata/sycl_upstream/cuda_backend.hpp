#pragma once
#include "strata/sycl_upstream/cuda/cuda_runtime.h"
#include "strata/sycl_upstream/memory.hpp"
namespace strata::sycl_upstream::cuda {
// Kernel adapters resolve the opaque frontend stream through this function.
// A cudaStream_t is never a sycl::queue*. The Runtime callback rules apply.
cudaError_t submit(cudaStream_t, const Runtime::Submit&) noexcept;
// Native library adapters inspect the owning device without submitting work.
cudaError_t stream_device(cudaStream_t, sycl::device*) noexcept;
// Read native ordinal metadata without changing the current device. Queries
// retain the frontend's sticky-error and output-lifetime rules.
cudaError_t inspect_device(int, const std::function<void(const sycl::device&)>&) noexcept;
cudaError_t device_context(int, sycl::context*) noexcept;
cudaError_t validate_device_buffer(cudaStream_t, const void*, size_t) noexcept;
Memory::Stats memory_stats();
std::optional<Memory::Info> allocation_info(const void*);
// Diagnostic detail for the last caught backend exception on this host thread.
// CUDA's public error string stays stable for each symbolic error code.
const char* backend_error_detail() noexcept;
}
