// SPDX-FileCopyrightText: 2026 the Strata contributors
// SPDX-License-Identifier: MIT
// Query only: no queue, context, allocation, kernel or host/device polling.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <iostream>
#include <stdexcept>
#include <string>

const char* order_name(sycl::memory_order o) {
    switch (o) {
    case sycl::memory_order::relaxed: return "relaxed";
    case sycl::memory_order::acquire: return "acquire";
    case sycl::memory_order::release: return "release";
    case sycl::memory_order::acq_rel: return "acq_rel";
    case sycl::memory_order::seq_cst: return "seq_cst";
    default: return "unknown";
    }
    return "unknown";
}
const char* scope_name(sycl::memory_scope s) {
    switch (s) {
    case sycl::memory_scope::work_item: return "work_item";
    case sycl::memory_scope::sub_group: return "sub_group";
    case sycl::memory_scope::work_group: return "work_group";
    case sycl::memory_scope::device: return "device";
    case sycl::memory_scope::system: return "system";
    }
    return "unknown";
}
int main(int argc, char** argv) try {
    if (argc != 2 || std::string(argv[1]) != "0000:05:00.0")
        throw std::runtime_error("one fixed B570 BDF required");
    bool found = false;
    for (const auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (d.get_backend() != sycl::backend::ext_oneapi_level_zero ||
            d.get_info<sycl::ext::intel::info::device::pci_address>() != argv[1]) continue;
        if (found) throw std::runtime_error("ambiguous PCI device");
        found = true;
        const auto ze = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d);
        const auto status = zeDeviceGetStatus(ze);
        std::cout << "STATUS " << argv[1] << ' ' << static_cast<uint32_t>(status) << std::endl;
        if (status != ZE_RESULT_SUCCESS) throw std::runtime_error("device status failed");
        std::cout << "HOST_USM " << d.has(sycl::aspect::usm_host_allocations) << '\n';
        std::cout << "ATOMIC_HOST_USM " << d.has(sycl::aspect::usm_atomic_host_allocations) << '\n';
        std::cout << "SHARED_USM " << d.has(sycl::aspect::usm_shared_allocations) << '\n';
        std::cout << "ATOMIC_SHARED_USM " << d.has(sycl::aspect::usm_atomic_shared_allocations) << '\n';
        for (auto v : d.get_info<sycl::info::device::atomic_memory_order_capabilities>()) std::cout << "ATOMIC_ORDER " << order_name(v) << '\n';
        for (auto v : d.get_info<sycl::info::device::atomic_memory_scope_capabilities>()) std::cout << "ATOMIC_SCOPE " << scope_name(v) << '\n';
        for (auto v : d.get_info<sycl::info::device::atomic_fence_order_capabilities>()) std::cout << "FENCE_ORDER " << order_name(v) << '\n';
        for (auto v : d.get_info<sycl::info::device::atomic_fence_scope_capabilities>()) std::cout << "FENCE_SCOPE " << scope_name(v) << '\n';
        ze_device_memory_access_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
        const auto result = zeDeviceGetMemoryAccessProperties(ze, &props);
        if (result != ZE_RESULT_SUCCESS) throw std::runtime_error("ze memory access query failed");
        std::cout << "ZE_HOST_CAPS " << props.hostAllocCapabilities << '\n';
        std::cout << "ZE_DEVICE_CAPS " << props.deviceAllocCapabilities << '\n';
        std::cout << "ZE_SHARED_SINGLE_CAPS " << props.sharedSingleDeviceAllocCapabilities << '\n';
        std::cout << "ZE_SHARED_CROSS_CAPS " << props.sharedCrossDeviceAllocCapabilities << '\n';
        std::cout << "ZE_SHARED_SYSTEM_CAPS " << props.sharedSystemAllocCapabilities << '\n';
        std::cout << "PASS query only" << std::endl;
    }
    if (!found) throw std::runtime_error("expected device absent");
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << std::endl;
    return 1;
}
