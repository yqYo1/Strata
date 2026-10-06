#pragma once
#include <new>

namespace strata {
// SYCL USM allocation may return nullptr without throwing. DPCT_CHECK_ERROR
// checks exceptions only, so turn that allocation failure into its error path.
// These call sites require usable storage; nullptr takes their failure path.
template <class T> T* checked_usm(T* pointer) {
    if (!pointer) throw std::bad_alloc{};
    return pointer;
}
} // namespace strata
