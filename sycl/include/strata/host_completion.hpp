#pragma once
#include <atomic>
#include <cstdint>
#include <memory>

namespace strata {
// This acknowledgement is ordinary CPU memory, accessed only by host threads.
// The in-order queue schedules the host task after its preceding DMA. No GPU
// fill/volatile host poll, atomic-host-USM capability, or device spin is needed.
using HostCompletion = std::shared_ptr<std::atomic<uint64_t>>;
inline HostCompletion make_host_completion() {
    return std::make_shared<std::atomic<uint64_t>>(0);
}
template <class Queue>
void enqueue_host_completion(Queue& queue, HostCompletion completed, uint64_t value) {
    queue.submit([completed, value](auto& handler) {
        // Only ordinary C++ objects are captured: SYCL reference-semantic
        // objects must not be captured or accessed inside a host task.
        handler.host_task([completed, value] {
            completed->store(value, std::memory_order_release);
        });
    });
}
inline bool host_completed(const HostCompletion& completed, uint64_t value) {
    return completed->load(std::memory_order_acquire) >= value;
}
} // namespace strata
