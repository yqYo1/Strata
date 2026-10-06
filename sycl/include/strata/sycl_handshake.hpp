#pragma once
#include <sycl/sycl.hpp>
#include <algorithm>
#include <string>

namespace strata {
inline bool require_host_handshake(const sycl::queue& queue, std::string& error) {
    const auto device = queue.get_device();
    if (!device.has(sycl::aspect::usm_atomic_host_allocations)) {
        error = "SYCL mapped handshake requires usm_atomic_host_allocations";
        return false;
    }
    const auto orders = device.get_info<sycl::info::device::atomic_memory_order_capabilities>();
    const auto scopes = device.get_info<sycl::info::device::atomic_memory_scope_capabilities>();
    if (std::find(orders.begin(), orders.end(), sycl::memory_order::acquire) == orders.end() ||
        std::find(orders.begin(), orders.end(), sycl::memory_order::release) == orders.end() ||
        std::find(scopes.begin(), scopes.end(), sycl::memory_scope::system) == scopes.end()) {
        error = "SYCL mapped handshake requires system-scope acquire/release atomics";
        return false;
    }
    return true;
}
} // namespace strata
