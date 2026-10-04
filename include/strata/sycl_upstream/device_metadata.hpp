#pragma once
#include "strata/sycl_upstream/cuda_backend.hpp"
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <string>
#include <vector>

namespace strata::sycl_upstream {
// Named original poison kernel: query its executable image without launching
// the arena's initializer or pretending to expose CUDA function attributes.
class OriginalDevicePoison;

struct DeviceFacts {
    std::string name, architecture, driver_version, device_version, platform_version;
    uint64_t total_memory_bytes = 0;
    uint64_t local_memory_bytes = 0;
    uint64_t architecture_id = 0;
    uint32_t compute_units = 0, max_clock_mhz = 0, vendor_id = 0;
    size_t max_workgroup_size = 0;
    std::vector<size_t> subgroup_sizes;
    bool gpu = false, level_zero = false;
    bool fp16 = false, matrix = false;
    bool usm_device = false, usm_host = false, usm_shared = false;
};

// These values are native SYCL metadata, not CUDA properties or version codes.
cudaError_t query_device_facts(int ordinal, DeviceFacts* output) noexcept;
const char* compiled_device_archs() noexcept;
std::string device_arch_problem(int ordinal);
std::string device_summary_detail(const DeviceFacts&);
} // namespace strata::sycl_upstream
