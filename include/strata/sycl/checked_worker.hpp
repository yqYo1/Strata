#pragma once

// Main-authored; repository LICENSE. Adaptive work may already own queued DMA
// when an allocation or submission throws. Convert that exception into the
// existing failure result only after a checked drain, with all owners live.
#include "strata/sycl/checked_cleanup.hpp"

#include <exception>
#include <thread>
#include <utility>

namespace strata::sycl_backend {

template <typename Work, typename Drain>
bool checked_worker(Work &&work, Drain &&drain, const char *message) noexcept {
  bool ok = false;
  try {
    ok = std::forward<Work>(work)();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Strata SYCL worker: %s: %s\n", message, error.what());
  } catch (...) {
    std::fprintf(stderr, "Strata SYCL worker: %s: unknown exception\n", message);
  }
  if (!ok)
    checked_cleanup(std::forward<Drain>(drain), [] {}, message);
  return ok;
}

// Declare after the thread's result and every owner captured by its work. This
// also joins if the main thread throws while committing or preparing a draft.
class CheckedWorkerJoin {
public:
  explicit CheckedWorkerJoin(std::thread &worker) noexcept : worker_(worker) {}
  CheckedWorkerJoin(const CheckedWorkerJoin &) = delete;
  CheckedWorkerJoin &operator=(const CheckedWorkerJoin &) = delete;
  ~CheckedWorkerJoin() noexcept {
    try {
      if (worker_.joinable()) worker_.join();
    } catch (...) {
      cleanup_fail_stop("adaptive worker join failed");
    }
  }

private:
  std::thread &worker_;
};

} // namespace strata::sycl_backend
