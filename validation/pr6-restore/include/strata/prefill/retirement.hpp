#pragma once
// New project code under repository LICENSE. No external implementation source.
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace strata::prefill::detail {
[[noreturn]] inline void retirement_fail_stop(const char* reason) noexcept {
    // No stdio locks: a stalled provider/issuer may hold them. Exit 86 means
    // completion/error proof was unavailable; Main records the active operation.
    (void) reason;
    std::_Exit(86);
}
// Also bounds a provider call which blocks inside status/wait/submission itself.
// This is a process deadline, never a timeout that unwinds live DMA owners.
class RetirementDeadline {
    std::mutex mu_;
    std::condition_variable cv_;
    bool done_ = false;
    std::thread watcher_;
public:
    explicit RetirementDeadline(std::chrono::seconds budget = std::chrono::seconds(60))
        : watcher_([this, budget] {
            std::unique_lock<std::mutex> lock(mu_);
            if (!cv_.wait_for(lock, budget, [this] { return done_; }))
                retirement_fail_stop("deadline expired");
        }) {}
    RetirementDeadline(const RetirementDeadline&) = delete;
    RetirementDeadline& operator=(const RetirementDeadline&) = delete;
    ~RetirementDeadline() {
        { std::lock_guard<std::mutex> lock(mu_); done_ = true; }
        cv_.notify_one();
        watcher_.join();
    }
};
template<class Predicate>
inline void retirement_until(Predicate ready, const char* reason) noexcept {
    try {
        RetirementDeadline deadline;
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!ready()) {
            if (std::chrono::steady_clock::now() >= end) retirement_fail_stop(reason);
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    } catch (...) { retirement_fail_stop(reason); }
}
} // namespace strata::prefill::detail
