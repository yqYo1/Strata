#pragma once
#include <atomic>

namespace strata {
// SYCL 2020 4.15.5 pairs system-scope device atomic_ref with std::atomic_ref
// on the same host-USM word. Volatile CPU loads/stores do not form that pair.
template <class T> T host_atomic_load(const T* pointer) {
    return std::atomic_ref<T>(*const_cast<T*>(pointer)).load(std::memory_order_acquire);
}
template <class T> void host_atomic_store(T* pointer, T value) {
    std::atomic_ref<T>(*pointer).store(value, std::memory_order_release);
}
template <class T> void host_atomic_raise(T* pointer, T value) {
    auto word = std::atomic_ref<T>(*pointer);
    T current = word.load(std::memory_order_relaxed);
    while (current < value && !word.compare_exchange_weak(
               current, value, std::memory_order_release, std::memory_order_relaxed)) {}
}
} // namespace strata
