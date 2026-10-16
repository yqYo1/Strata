#pragma once
// New project code under repository LICENSE. No external implementation copied.
#include <sycl/sycl.hpp>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace strata {
inline void pr6_async_fail_stop(sycl::exception_list errors) noexcept {
    // Never consume an async error as success. No unwinding, stdio locks or USM
    // cleanup: exit 87 preserves the source/owner lifetimes of accepted work.
    if (errors.begin() != errors.end()) std::_Exit(87);
}
inline std::mutex pr6_queue_mutex;
inline std::vector<std::weak_ptr<sycl::queue>> pr6_checked_queues;
inline void pr6_register_queue(const std::shared_ptr<sycl::queue>& queue) {
    std::lock_guard<std::mutex> lock(pr6_queue_mutex);
    pr6_checked_queues.emplace_back(queue);
}
inline sycl::queue* pr6_require_checked_queue(sycl::queue* queue) {
    std::lock_guard<std::mutex> lock(pr6_queue_mutex);
    for (auto it = pr6_checked_queues.begin(); it != pr6_checked_queues.end();) {
        if (auto owner = it->lock()) {
            if (owner.get() == queue) return queue;
            ++it;
        } else it = pr6_checked_queues.erase(it);
    }
    throw std::invalid_argument("PR6: queue lacks the owned fail-stop async contract");
}
} // namespace strata
