// Build manually: c++ -std=c++20 -pthread -Iinclude tests/cpu/watchdog_lifecycle.cpp -o /tmp/watchdog-lifecycle-test
// Host-only contracts; does not claim GPU/kernel ordering coverage.
#include "strata/core/watchdog_lifecycle.hpp"
#include <cassert>
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <vector>
#undef assert
#define assert(expression) do { if (!(expression)) { std::fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #expression); std::abort(); } } while (false)
using namespace strata::core;
int main() {
    VerifyWindowSnapshot snapshot;
    std::atomic<bool> done{false};
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::thread writer([&] {
        gate.wait();
        for (int i = 1; i <= 10000; ++i) {
            snapshot.begin(i, int64_t(i) * 101);
            snapshot.phase(VerifyPhase::running);
            snapshot.phase(VerifyPhase::completed);
        }
        done.store(true);
    });
    std::thread reader([&] {
        gate.wait();
        do {
            auto v = snapshot.read();
            assert(v.position == int64_t(v.tokens) * 101);
            assert(v.generation == uint64_t(v.tokens));
        } while (!done.load());
    });
    start.set_value(); writer.join(); reader.join();
    assert(snapshot.read().phase == VerifyPhase::completed);
    // Deterministic rendezvous ensures the reader sees every publication,
    // independently of how the earlier overlapping stress threads were scheduled.
    VerifyWindowSnapshot staged_snapshot;
    std::barrier rendezvous(2);
    std::thread staged_writer([&] {
        for (int i = 1; i <= 256; ++i) {
            staged_snapshot.begin(i, int64_t(i) * 101);
            rendezvous.arrive_and_wait();
            rendezvous.arrive_and_wait();
        }
    });
    for (int i = 1; i <= 256; ++i) {
        rendezvous.arrive_and_wait();
        const auto v = staged_snapshot.read();
        assert(v.generation == uint64_t(i) && v.tokens == i && v.position == int64_t(i) * 101);
        assert(v.phase == VerifyPhase::preparing);
        rendezvous.arrive_and_wait();
    }
    staged_writer.join();
    snapshot.begin(7, 707); snapshot.phase(VerifyPhase::failed);
    assert(snapshot.read().tokens == 7 && snapshot.read().phase == VerifyPhase::failed);

    struct Stage { VerifyWindowSnapshot window; int releases = 0; } a, b, extra;
    CallbackRegistry<Stage, 2> registry;
    assert(registry.add(&a) && registry.add(&b) && !registry.add(&extra));
    auto rescue = [&] { registry.visit([](Stage& s) {
        s.window.if_running([&] { ++s.releases; });
    }); };
    rescue(); assert(a.releases == 0 && b.releases == 0); // idle/prefill
    a.window.begin(4, 123); b.window.begin(4, 123);
    rescue(); assert(a.releases == 0 && b.releases == 0); // capture/staging
    a.window.phase(VerifyPhase::running); b.window.phase(VerifyPhase::running);
    rescue(); assert(a.releases == 1 && b.releases == 1); // all active stages
    a.window.phase(VerifyPhase::completed); b.window.phase(VerifyPhase::failed);
    rescue(); assert(a.releases == 1 && b.releases == 1);

    // A second free slot must not permit re-registration of a closing object.
    CallbackRegistry<int, 2> leases;
    int object = 42, replacement = 1;
    assert(!leases.add(nullptr));
    leases.remove(nullptr);
    assert(leases.add(&object));
    std::promise<void> acquired, allow_return, removing;
    auto return_gate = allow_return.get_future().share();
    std::thread callback([&] { leases.visit([&](int& value) {
        acquired.set_value(); return_gate.wait(); assert(value == 42);
    }); });
    acquired.get_future().wait();
    std::atomic<bool> removed{false};
    std::thread destroyer([&] { removing.set_value(); leases.remove(&object); removed.store(true); });
    removing.get_future().wait();
    // Observe closing, rather than assuming the destroyer has been scheduled.
    while (leases.add(&object)) std::this_thread::yield();
    assert(!removed.load()); // Acquired lease cannot be bypassed.
    assert(leases.add(&replacement));
    assert(!leases.add(&object));
    leases.remove(&replacement);
    allow_return.set_value(); callback.join(); destroyer.join();
    assert(removed.load() && leases.add(&replacement));
    leases.remove(&replacement);

    // Ordinary stop wakes the periodic wait and joins before observer storage dies.
    std::atomic<int> calls{0};
    {
        ScopedWatchdog watchdog;
        watchdog.start([&] { ++calls; });
        bool rejected = false;
        try { watchdog.start([] {}); } catch (const std::logic_error&) { rejected = true; }
        assert(rejected);
        watchdog.stop(); watchdog.stop();
    }
    const int stopped_calls = calls.load();
    assert(calls.load() == stopped_calls);
    std::promise<void> tick_entered, tick_return;
    auto tick_gate = tick_return.get_future().share();
    std::atomic<bool> stop_returned{false};
    ScopedWatchdog active_observer;
    std::atomic<bool> first_tick{true};
    active_observer.start([&] { if (first_tick.exchange(false)) { tick_entered.set_value(); tick_gate.wait(); } }, std::chrono::milliseconds(1));
    tick_entered.get_future().wait();
    std::thread stopper([&] { active_observer.stop(); stop_returned.store(true); });
    assert(!stop_returned.load()); // A callback that already started must finish before join returns.
    tick_return.set_value(); stopper.join();
    assert(stop_returned.load());
    // Atomic monotonic rescue cannot be overwritten by a late host callback.
    alignas(std::atomic_ref<uint32_t>::required_alignment) uint32_t flag = 0;
    host_flag_raise(&flag, UINT32_MAX); host_flag_raise(&flag, 12);
    assert(host_flag_load(&flag) == UINT32_MAX);
}
