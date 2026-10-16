#pragma once

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace strata::sycl_backend {

// A failed checked drain cannot authorize releasing an allocation. Stop before
// unwinding its owners; process exit retires the failed runtime as a whole.
[[noreturn]] inline void cleanup_fail_stop(const char *message) noexcept {
  std::fprintf(stderr, "Strata SYCL cleanup: %s; retaining owners until process exit\n",
               message);
  std::fflush(stderr);
  std::_Exit(EXIT_FAILURE);
}

template <typename Drain, typename Release>
void checked_cleanup(Drain &&drain, Release &&release,
                     const char *message) noexcept {
  try {
    std::forward<Drain>(drain)();
  } catch (...) {
    cleanup_fail_stop(message);
  }
  try {
    std::forward<Release>(release)();
  } catch (...) {
    cleanup_fail_stop("resource release failed after a checked drain");
  }
}

} // namespace strata::sycl_backend
