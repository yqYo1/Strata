// CPU-only deferred task, ring-reuse and error-reporting regression checks.
#include "strata/host_completion.hpp"
#include "strata/sycl_error.hpp"
#include "strata/sycl_allocation.hpp"
#include "strata/host_atomic.hpp"
#include <cassert>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

struct DeferredQueue {
    struct Handler {
        std::function<void()> task;
        template <class F> void host_task(F task) { this->task = std::move(task); }
    };
    std::vector<std::function<void()>> tasks;
    template <class F> void submit(F f) {
        Handler handler; f(handler); tasks.push_back(std::move(handler.task));
    }
};

int main() {
    // Null allocations take the existing fallback/error path before any
    // memset, memcpy or device submission can dereference the result.
    for (int failure = 0; failure != 2; ++failure) {
        try {
            if (failure) strata::checked_usm(static_cast<float*>(nullptr));
            else strata::checked_usm(static_cast<void*>(nullptr));
            assert(false);
        } catch (const std::bad_alloc&) {}
    }
    float allocation = 1.0f;
    assert(strata::checked_usm(&allocation) == &allocation);
    // Check publication of ordinary payload by a release/acquire handshake.
    uint32_t published = 0, consumed = 0;
    int row = -1;
    std::thread producer([&] {
        for (uint32_t i = 1; i <= 4096; ++i) {
            while (strata::host_atomic_load(&consumed) != i - 1) std::this_thread::yield();
            row = (int) i;
            strata::host_atomic_store(&published, i);
        }
    });
    for (uint32_t i = 1; i <= 4096; ++i) {
        while (strata::host_atomic_load(&published) != i) std::this_thread::yield();
        assert(row == (int) i);
        strata::host_atomic_store(&consumed, i);
    }
    producer.join();
    // Concurrent callbacks must never undo a cancellation/release sentinel.
    uint32_t flag = 0;
    std::vector<std::thread> raisers;
    for (uint32_t j = 0; j != 4; ++j) raisers.emplace_back([&, j] {
        for (uint32_t i = 1; i <= 4096; ++i) strata::host_atomic_raise(&flag, i + j * 4096);
    });
    strata::host_atomic_raise(&flag, UINT32_MAX);
    for (auto& thread : raisers) thread.join();
    assert(strata::host_atomic_load(&flag) == UINT32_MAX);
    DeferredQueue queue;
    auto ack = strata::make_host_completion();
    assert(strata::host_completed(ack, 0));
    strata::enqueue_host_completion(queue, ack, 1);
    assert(!strata::host_completed(ack, 1));
    std::thread callback([&] { queue.tasks[0](); });
    while (!strata::host_completed(ack, 1)) std::this_thread::yield();
    callback.join();
    // A queued callback retains its acknowledgement even if the issuer exits.
    std::weak_ptr<std::atomic<uint64_t>> retained = ack;
    ack.reset();
    assert(!retained.expired());
    queue.tasks.clear();
    assert(retained.expired());

    constexpr int ring = 2, jobs = 2048;
    strata::HostCompletion done[ring] = {strata::make_host_completion(), strata::make_host_completion()};
    uint64_t wanted[ring]{};
    int payload[ring]{};
    queue.tasks.reserve(jobs * 2);
    for (int i = 0; i != jobs; ++i) {
        const int b = i % ring;
        // Simulate the DMA immediately before its acknowledgement. Workers
        // cannot refill the slot until this deferred read has completed.
        if (i >= ring) {
            assert(!strata::host_completed(done[b], wanted[b]));
            queue.tasks[(i - ring) * 2]();
            queue.tasks[(i - ring) * 2 + 1]();
        }
        assert(strata::host_completed(done[b], wanted[b]));
        payload[b] = i;
        queue.tasks.push_back([&, i, b] { assert(payload[b] == i); });
        wanted[b] = i + 1;
        strata::enqueue_host_completion(queue, done[b], wanted[b]);
    }
    for (int i = jobs - ring; i != jobs; ++i) { queue.tasks[i * 2](); queue.tasks[i * 2 + 1](); }
    for (int b = 0; b < ring; ++b) assert(strata::host_completed(done[b], wanted[b]));

    std::ostringstream logs;
    auto* previous = std::cerr.rdbuf(logs.rdbuf());
    const std::vector<std::exception_ptr> none;
    strata::rethrow_sycl_errors(none);
    const std::vector<std::exception_ptr> errors = {
        std::make_exception_ptr(std::runtime_error("first-device-loss")),
        std::make_exception_ptr(std::runtime_error("second-copy-error"))};
    try { strata::rethrow_sycl_errors(errors); assert(false); }
    catch (const std::runtime_error& e) { assert(std::string(e.what()) == "first-device-loss"); }
    assert(logs.str().find("second-copy-error") != std::string::npos);
    assert(strata::finish_sycl_serve([] {}) == 0);
    assert(strata::finish_sycl_serve([] { throw std::runtime_error("shutdown-device-loss"); }) == 1);
    assert(strata::finish_sycl_serve([] { throw 7; }) == 1);
    std::cerr.rdbuf(previous);
    std::cout << "PASS: null allocation rejection, 4096 payload handshakes, concurrent monotonic flags, deferred acknowledgement, 2048 two-slot reuses, callback lifetime, async errors, shutdown status\n";
}
