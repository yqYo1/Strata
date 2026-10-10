// CPU-only qualification of production publication.hpp; no GPU or model required.
#include "strata/prefill/publication.hpp"
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
using namespace strata::prefill::publication;
const char* error_text() { return "injected failure"; }
int main() {
    // Old ordering is the expected negative: release precedes producer metadata read.
    bool read = false, violated = false;
    auto release = [&] { if (!read) violated = true; };
    release(); read = true;
    assert(violated);
    // Exercise the very helper called by production release_to; flush opens ring capacity.
    for (size_t k = 0; k != 64; ++k) {
        bool flushed = false, waited = false, released = false;
        skipped_entry([&] { flushed = true; },
                      [&] { assert(flushed); waited = true; },
                      [&] { assert(waited); released = true; });
        assert(released); // includes repeated logical wraps of rings 2, 4, and 8
    }
    // Submission/event/wait failure cutpoints: no later metadata or progress publication.
    for (int cut = 0; cut != 3; ++cut) {
        int progress = 0;
        try {
            skipped_entry([&] { require(cut != 0, "submit", error_text); },
                          [&] { require(cut != 1, "wait", error_text); },
                          [&] { require(cut != 2, "event", error_text); ++progress; });
            assert(false);
        } catch (const std::runtime_error&) { assert(progress == 0); }
    }
    // Producer fault wakes a blocked consumer. Always join before error escapes scope.
    std::atomic<size_t> issued{0}; std::atomic<bool> stop{false}; Failure failure;
    std::thread producer([&] {
        try { require(false, "copy_async_after", error_text); }
        catch (...) { failure.capture(); stop.store(true, std::memory_order_release); }
    });
    bool caught = false;
    try { wait_issued(issued, stop, failure, 0); }
    catch (const std::runtime_error& e) {
        caught = true; assert(std::string(e.what()).find("copy_async_after") != std::string::npos);
    }
    bool cancelled = false, drained = false;
    stop_join(stop, producer, [&] { cancelled = true; }, [&] {
        assert(cancelled && !producer.joinable()); drained = true;
    });
    assert(caught && drained && !producer.joinable());
    // Early consumer unwind must stop a producer blocked on ring capacity and then drain.
    stop.store(false);
    std::atomic<bool> exited{false};
    std::thread capacity_waiter([&] {
        while (!stop.load(std::memory_order_acquire)) std::this_thread::yield();
        exited.store(true, std::memory_order_release);
    });
    stop_join(stop, capacity_waiter, [] {}, [&] { assert(exited.load()); });
    assert(!capacity_waiter.joinable());
    // Abort without an exception also wakes waiters, while publication permits normal completion.
    Failure empty;
    caught = false;
    try { wait_issued(issued, stop, empty, 0); } catch (...) { caught = true; }
    assert(caught);
    stop.store(false); issued.store(1); wait_issued(issued, stop, empty, 0);
    // Group flush releases enough capacity to allow producer publication before skipped-entry wait.
    std::atomic<bool> capacity{false}; issued.store(0);
    std::thread blocked([&] { while (!capacity.load()) std::this_thread::yield(); issued.store(1); });
    skipped_entry([&] { capacity.store(true); },
                  [&] { wait_issued(issued, stop, empty, 0); }, [] {});
    blocked.join();
    // Old store can overwrite cancellation back to a waiter's sleeping value.
    std::atomic<int> staged{0};
    const int sleeping_value = staged.load();
    staged.store(1 << 30);
    staged.store(sleeping_value); // old expected negative
    assert(staged.load() == sleeping_value);
    staged.store(1 << 30);
    publish_staged(staged, sleeping_value);
    assert(staged.load() == (1 << 30));
    staged.store(0);
    publish_staged(staged, 4);
    publish_staged(staged, 2);
    assert(staged.load() == 4);
    for (int repeat = 0; repeat < 64; ++repeat) {
        staged.store(0);
        std::thread publish([&] { for (int i = 1; i <= 64; ++i) publish_staged(staged, i); });
        staged.store(1 << 30);
        publish.join();
        assert(staged.load() == (1 << 30));
    }
    std::cout << "production publication helper qualification passed\n";
}
