// Independent project fixture under repository LICENSE. No SYCL/runtime init.
#include "strata/prefill/retirement.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <atomic>
#include <cassert>
#include <stdexcept>

using strata::prefill::detail::bounded_join;

struct JoinOnExit {
    std::atomic<bool>& stop;
    std::thread& worker;
    ~JoinOnExit() { stop.store(true, std::memory_order_release); bounded_join(worker); }
};

int main() {
    std::thread absent;
    bounded_join(absent);
    int result = 0;
    std::mutex owner_mutex;
    // The worker needs this mutex to finish: joining is outside its lock scope.
    std::unique_lock<std::mutex> held(owner_mutex);
    std::thread normal([&] { std::lock_guard<std::mutex> lock(owner_mutex); result = 41; });
    held.unlock();
    bounded_join(normal);
    assert(!normal.joinable() && result == 41);
    bounded_join(normal); // explicit join followed by owner cleanup is harmless

    std::atomic<bool> stop{false};
    std::thread issuer;
    try {
        // All referenced owners precede the guard, matching the real issuer.
        JoinOnExit guard{stop, issuer};
        issuer = std::thread([&] {
            while (!stop.load(std::memory_order_acquire)) std::this_thread::yield();
            std::lock_guard<std::mutex> lock(owner_mutex);
            result = 42;
        });
        throw std::runtime_error("owner unwind");
    } catch (const std::runtime_error&) {}
    assert(!issuer.joinable() && result == 42);
    // No hanging provider or deadline injection here. Main must check exit86
    // with a real owned, externally supervised process before native admission.
}
