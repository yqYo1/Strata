#include "strata/sycl_upstream/runtime.hpp"
#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
using strata::sycl_upstream::Runtime;
using namespace std::chrono_literals;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Gate {
    std::promise<void> release, entered;
    std::shared_future<void> opened = release.get_future().share();
    std::future<void> started = entered.get_future();
    bool released = false;
    void open() { if (!released) { released = true; release.set_value(); } }
    ~Gate() { open(); }
    void block(Runtime& rt, Runtime::Stream s) {
        rt.host_function(s, [this]{ entered.set_value(); opened.wait(); });
    }
};
template<class F> bool eventually(F f) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    do { if (f()) return true; std::this_thread::yield(); } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
void write(Runtime& rt, Runtime::Stream s, int* dst, int value) {
    rt.enqueue(s, [=](sycl::queue& q) { return q.single_task([=]{ *dst = value; }); });
}
void add(Runtime& rt, Runtime::Stream s, const int* src, int* dst, int value) {
    rt.enqueue(s, [=](sycl::queue& q) { return q.single_task([=]{ *dst = *src + value; }); });
}
int main() try {
    Runtime rt{sycl::device{sycl::gpu_selector_v}};
    std::cout << "device=" << rt.device().get_info<sycl::info::device::name>() << '\n';
    auto a = rt.create_stream(), b = rt.create_stream(), free = rt.create_stream(true), consumer = rt.create_stream(true);
    int* values = sycl::malloc_shared<int>(16, rt.device(), rt.context());
    check(values, "allocation");
    // Warm the two kernel shapes before bounded observations of queue ordering.
    write(rt, free, values, 1); add(rt, free, values, values + 1, 2); rt.synchronize(free);
    int cases = 0;
    {
        Gate gate; gate.block(rt, a);
        std::atomic<int> phase{0};
        auto submitted = std::async(std::launch::async, [&] {
            phase = 1; write(rt, a, values, 10);
            phase = 2; add(rt, 0, values, values + 1, 20);
            phase = 3; add(rt, b, values + 1, values + 2, 30);
            phase = 4; write(rt, free, values + 3, 77);
            phase = 5;
        });
        const bool returned = submitted.wait_for(2s) == std::future_status::ready;
        bool independent = false, blocked = false;
        if (returned) {
            submitted.get(); independent = eventually([&]{ return rt.query(free); });
            blocked = !rt.query(0) && !rt.query(a) && !rt.query(b);
        }
        const int stopped_phase = phase.load();
        gate.open(); if (!returned) submitted.get(); rt.synchronize_device();
        if (!(returned && independent && blocked))
            std::cerr << "returned=" << returned << " independent=" << independent << " blocked=" << blocked << " phase=" << stopped_phase << '\n';
        check(returned && independent && blocked, "legacy dependencies/nonblocking exclusion");
        check(values[0] == 10 && values[1] == 30 && values[2] == 60 && values[3] == 77, "legacy chain values");
        ++cases;
    }
    {
        Gate gate; gate.block(rt, free);
        write(rt, 0, values, 41); add(rt, a, values, values + 1, 1);
        const bool independent = eventually([&]{ return rt.query(a); });
        const bool pending = !rt.query(free);
        gate.open(); rt.synchronize_device();
        check(independent && pending && values[1] == 42, "nonblocking must not hold legacy stream"); ++cases;
    }
    {
        Gate gate; gate.block(rt, a); write(rt, b, values, 63);
        const bool independent = eventually([&]{ return rt.query(b); });
        gate.open(); rt.synchronize_device();
        check(independent && values[0] == 63, "ordinary streams spuriously serialized"); ++cases;
    }
    {
        Runtime::Event event;
        check(rt.query(event), "unrecorded event not complete");
        rt.synchronize(event); rt.wait_event(consumer, event);
        write(rt, free, values, 9); rt.record(event, free);
        rt.wait_event(consumer, event); add(rt, consumer, values, values + 1, 1);
        Gate gate; gate.block(rt, free);
        write(rt, free, values, 99); rt.record(event, free);
        const bool first_complete = eventually([&]{ return rt.query(consumer); });
        const bool second_pending = !rt.query(event);
        const int first_value = first_complete ? values[1] : -1;
        rt.wait_event(consumer, event);
        // Destroy/reassign the Event object after enqueueing the dependency.
        event = Runtime::Event{};
        add(rt, consumer, values, values + 2, 1);
        const bool second_wait_pending = !rt.query(consumer);
        gate.open(); rt.synchronize_device();
        check(first_complete && second_pending && second_wait_pending && first_value == 10 && values[2] == 100,
              "event generations or event destruction retargeted a wait"); ++cases;
    }
    {
        auto gone = rt.create_stream(true);
        Gate gate; gate.block(rt, gone); write(rt, gone, values, 123);
        Runtime::Event finished; rt.record(finished, gone);
        auto destroyed = std::async(std::launch::async, [&]{ rt.destroy_stream(gone); });
        const bool returned = destroyed.wait_for(2s) == std::future_status::ready;
        if (!returned) { gate.open(); destroyed.get(); throw std::runtime_error("stream destroy synchronized"); }
        destroyed.get();
        bool invalid = false;
        try { rt.query(gone); } catch (const std::invalid_argument&) { invalid = true; }
        std::promise<void> entered;
        auto started = entered.get_future();
        auto sync = std::async(std::launch::async, [&]{ entered.set_value(); rt.synchronize_device(); });
        started.wait();
        const bool waits_retired = sync.wait_for(100ms) == std::future_status::timeout;
        const bool still_pending = !rt.query(finished);
        gate.open(); sync.get(); rt.synchronize(finished);
        check(invalid && waits_retired && still_pending && values[0] == 123, "retired stream lost pending work"); ++cases;
    }
    {
        // Original prefill ring: each slot's next copy waits for its previous
        // consumer, and each consumer waits for the corresponding DMA event.
        constexpr int slots = 2, batches = 40, width = 257;
        auto* host = sycl::malloc_host<uint8_t>(batches * width, rt.context());
        auto* ring = sycl::malloc_device<uint8_t>(slots * width, rt.device(), rt.context());
        auto* sums = sycl::malloc_shared<int>(batches, rt.device(), rt.context());
        check(host && ring && sums, "ring allocation");
        for (int i = 0; i < batches * width; ++i) host[i] = uint8_t((i * 19) ^ (i >> 8));
        Runtime::Event copied[slots], used[slots];
        for (int batch = 0; batch < batches; ++batch) {
            const int slot = batch % slots;
            if (batch >= slots) rt.wait_event(free, used[slot]);
            rt.copy(free, ring + slot * width, host + batch * width, width);
            rt.record(copied[slot], free); rt.wait_event(consumer, copied[slot]);
            rt.enqueue(consumer, [=](sycl::queue& q) {
                return q.single_task([=]{ int sum = 0; for (int j = 0; j < width; ++j) sum += ring[slot * width + j]; sums[batch] = sum; });
            });
            rt.record(used[slot], consumer);
        }
        rt.synchronize(consumer);
        for (int batch = 0; batch < batches; ++batch) {
            int ref = 0; for (int j = 0; j < width; ++j) ref += host[batch * width + j];
            check(sums[batch] == ref, "prefill ring slot reused before consumption");
        }
        // DMA -> host callback -> subsequent device work ordering.
        int copied_sum = -1; std::atomic<int> flag{-1};
        rt.copy(consumer, &copied_sum, sums + batches - 1, sizeof(int));
        rt.host_function(consumer, [&]{ flag.store(copied_sum, std::memory_order_release); });
        write(rt, consumer, values, 234); rt.synchronize(consumer);
        check(flag.load(std::memory_order_acquire) == sums[batches - 1] && values[0] == 234, "host callback ordering");
        sycl::free(host, rt.context()); sycl::free(ring, rt.context()); sycl::free(sums, rt.context()); ++cases;
    }
    {
        // Failed host-side submission must not erase work already queued by it.
        bool caught = false;
        try {
            rt.enqueue(a, [&](sycl::queue& q) -> sycl::event {
                q.single_task([=]{ values[0] = 345; });
                throw std::runtime_error("intentional submission exception");
            });
        } catch (const std::runtime_error&) { caught = true; }
        add(rt, 0, values, values + 1, 1); rt.synchronize(0);
        check(caught && values[1] == 346, "submission exception lost queue tail"); ++cases;
    }
    rt.destroy_stream(a); rt.destroy_stream(b); rt.destroy_stream(free); rt.destroy_stream(consumer);
    auto reused = rt.create_stream(true); // Reap completed retired queues.
    write(rt, reused, values, 456); rt.synchronize_device(); check(values[0] == 456, "reaped queue regression");
    sycl::free(values, rt.context());
    std::cout << "PASS runtime_cases=" << cases << " prefill_ring_batches=40 legacy_and_event_ordering=1 async_destroy=1\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
