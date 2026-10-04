#include "strata/sycl_upstream/device_metadata.hpp"
#include <algorithm>
#include <cstdio>
#include <sstream>

namespace strata::sycl_upstream {
namespace {
namespace experimental = sycl::ext::oneapi::experimental;
std::string architecture_name(experimental::architecture arch) {
    // Use the installed SDK's complete native architecture enum, including
    // its unknown value, without a CUDA/HIP marketing-name mapping table.
    switch (arch) {
#define __SYCL_ARCHITECTURE(name, id) case experimental::architecture::name: return #name;
#define __SYCL_ARCHITECTURE_ALIAS(name, arch)
#include <sycl/ext/oneapi/experimental/device_architecture.def>
#undef __SYCL_ARCHITECTURE_ALIAS
#undef __SYCL_ARCHITECTURE
    }
    std::ostringstream text;
    text << "SYCL architecture 0x" << std::hex << uint64_t(arch);
    return text.str();
}
bool subgroup(const DeviceFacts& facts, size_t width) {
    return std::find(facts.subgroup_sizes.begin(), facts.subgroup_sizes.end(), width)
        != facts.subgroup_sizes.end();
}
} // namespace

cudaError_t query_device_facts(int ordinal, DeviceFacts* output) noexcept {
    return cuda::inspect_device(ordinal, [&](const sycl::device& device) {
        if (!output) throw std::invalid_argument("null native device facts output");
        DeviceFacts facts;
        facts.name = device.get_info<sycl::info::device::name>();
        const auto architecture = device.get_info<experimental::info::device::architecture>();
        facts.architecture = architecture_name(architecture);
        facts.architecture_id = uint64_t(architecture);
        facts.driver_version = device.get_info<sycl::info::device::driver_version>();
        facts.device_version = device.get_info<sycl::info::device::version>();
        facts.platform_version = device.get_platform().get_info<sycl::info::platform::version>();
        facts.total_memory_bytes = device.get_info<sycl::info::device::global_mem_size>();
        facts.local_memory_bytes = device.get_info<sycl::info::device::local_mem_size>();
        facts.compute_units = device.get_info<sycl::info::device::max_compute_units>();
        facts.max_clock_mhz = device.get_info<sycl::info::device::max_clock_frequency>();
        facts.vendor_id = device.get_info<sycl::info::device::vendor_id>();
        facts.max_workgroup_size = device.get_info<sycl::info::device::max_work_group_size>();
        facts.subgroup_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
        facts.gpu = device.is_gpu();
        facts.level_zero = device.get_backend() == sycl::backend::ext_oneapi_level_zero;
        facts.fp16 = device.has(sycl::aspect::fp16);
        facts.matrix = device.has(sycl::aspect::ext_intel_matrix);
        facts.usm_device = device.has(sycl::aspect::usm_device_allocations);
        facts.usm_host = device.has(sycl::aspect::usm_host_allocations);
        facts.usm_shared = device.has(sycl::aspect::usm_shared_allocations);
        *output = std::move(facts);
    });
}

const char* compiled_device_archs() noexcept {
#ifdef STRATA_SYCL_DEVICE_ARCH_NAME
    return STRATA_SYCL_DEVICE_ARCH_NAME;
#else
    return "spir64 JIT";
#endif
}

std::string device_arch_problem(int ordinal) {
    DeviceFacts facts;
    if (query_device_facts(ordinal, &facts) != cudaSuccess) {
        cudaGetLastError();
        return {}; // Match original enumeration: invalid ordinals use caller errors.
    }
    const std::string card = "GPU " + std::to_string(ordinal) + " (" + facts.name + ", "
        + facts.architecture + ")";
#ifdef STRATA_SYCL_DEVICE_ARCH_ID
    if (facts.architecture_id != uint64_t(STRATA_SYCL_DEVICE_ARCH_ID))
        return card + " differs from this binary's SYCL AOT target (" + compiled_device_archs() + ")";
#endif
    if (!facts.gpu || !facts.level_zero || facts.vendor_id != 0x8086)
        return card + " does not provide this port's native Intel Level Zero GPU backend";
    if (!facts.fp16 || !facts.matrix || !facts.usm_device || !facts.usm_host || !facts.usm_shared)
        return card + " lacks native FP16, matrix or USM allocation support used by this port";
    if (!subgroup(facts, 16) || !subgroup(facts, 32))
        return card + " lacks the native subgroup sizes 16 and 32 used by this port";
    // Existing original GR/Ple kernels use 1024 work-items. Short attention
    // uses 4096+32+32 F32 local elements; MMQ requires fewer local bytes.
    if (facts.max_workgroup_size < 1024 || facts.local_memory_bytes < (4096 + 32 + 32) * 4)
        return card + " lacks the workgroup/local-memory capacity used by this port";
    return {};
}

std::string device_summary_detail(const DeviceFacts& facts) {
    char buffer[160];
    std::snprintf(buffer, sizeof buffer, "SYCL arch %s, %.1f GiB, %u compute units",
                  facts.architecture.c_str(), double(facts.total_memory_bytes) / (1024. * 1024 * 1024),
                  facts.compute_units);
    return buffer;
}
} // namespace strata::sycl_upstream
