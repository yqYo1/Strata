// include/strata/sycl_doorbell.hpp - the SYCL port: host<->device flag traffic that must bypass the GPU's caches.
//
// Strata's decode loop is a handshake through host-mapped memory: the GPU rings a sequence number the host polls,
// and the host waits for completion before submitting dependent work. In CUDA `volatile` loads and stores
// nvcc turns into cache-bypassing accesses. A `volatile` in SYCL device code carries no such meaning on Intel
// GPUs: the spin read its first value from L3 forever (measured: the GPU at 100% and the host seeing no ring).
// Pair system-scope acquire/release device atomics with host std::atomic_ref.
// Concurrent host-USM use additionally requires usm_atomic_host_allocations.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata {
using sys_atomic_u32 = sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system>;

inline uint32_t sys_load(const volatile uint32_t* p) {
    return sys_atomic_u32(*const_cast<uint32_t*>(p)).load(sycl::memory_order::acquire);
}
inline void sys_store(volatile uint32_t* p, uint32_t v) {
    sys_atomic_u32(*const_cast<uint32_t*>(p)).store(v, sycl::memory_order::release);
}

}  // namespace strata
