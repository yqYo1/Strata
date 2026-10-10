// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
namespace strata::prefill::publication {
template<class ErrorText> void require(bool ok, const char* operation, ErrorText text) {
    if (!ok) throw std::runtime_error(std::string(operation) + ": " + text());
}
class Failure {
    mutable std::mutex mutex_;
    std::exception_ptr first_;
public:
    void capture() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!first_) first_ = std::current_exception();
    }
    void rethrow() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (first_) std::rethrow_exception(first_);
    }
};
inline void wait_issued(const std::atomic<size_t>& issued, const std::atomic<bool>& stop,
                        const Failure& failure, size_t k) {
    for (;;) {
        failure.rethrow();
        if (stop.load(std::memory_order_acquire)) {
            failure.rethrow();
            throw std::runtime_error("issuer aborted before publication");
        }
        if (issued.load(std::memory_order_acquire) > k) return;
        std::this_thread::yield();
    }
}
// Cancellation uses a high terminal value. A racing normal publication must
// never lower it: atomic::wait is value-based, so notify alone cannot fix ABA.
inline void publish_staged(std::atomic<int>& issued, int value) {
    int previous = issued.load(std::memory_order_acquire);
    while (previous < value &&
           !issued.compare_exchange_weak(previous, value, std::memory_order_release,
                                         std::memory_order_acquire)) {}
    issued.notify_all();
}
// Used on every exceptional/early consumer exit; callbacks run in this order.
template<class Cancel, class Drain>
void stop_join(std::atomic<bool>& stop, std::thread& producer, Cancel cancel, Drain drain) {
    if (!producer.joinable()) return;
    stop.store(true, std::memory_order_release);
    cancel();
    producer.join();
    drain();
}
// Shared production ordering: flush queued readers before waiting for the producer;
// wait for its metadata read before changing the reusable event or plain slot metadata.
template<class Flush, class Wait, class Release>
void skipped_entry(Flush flush, Wait wait, Release release) {
    flush();
    wait();
    release();
}
}
