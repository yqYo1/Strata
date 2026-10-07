#include "event_completion.hpp"
#include <array>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <thread>
#include <vector>

struct Dma { std::atomic<bool> done{false}; };
struct Event { std::shared_ptr<Dma> dma; };
static bool query(const Event& event) { return event.dma->done.load(std::memory_order_acquire); }
int main() {
    auto state = strata::make_event_completion<Event>();
    assert(strata::event_completed(state, 0, query));
    assert(!strata::event_completed(state, 1, query));
    auto first = std::make_shared<Dma>();
    std::weak_ptr<Dma> lifetime = first;
    strata::record_event_completion(state, 1, Event{first});
    first.reset();
    assert(!lifetime.expired());
    assert(!strata::event_completed(state, 1, query));
    try {
        strata::event_completed(state, 1, [](const Event&) -> bool { throw std::runtime_error("query failed"); });
        assert(false);
    } catch (const std::runtime_error&) {}
    assert(!strata::event_completed(state, 1, query));
    lifetime.lock()->done.store(true, std::memory_order_release);
    assert(strata::event_completed(state, 1, query));
    auto second = std::make_shared<Dma>();
    strata::record_event_completion(state, 2, Event{second});
    assert(lifetime.expired());
    assert(strata::event_completed(state, 1, query));
    assert(!strata::event_completed(state, 2, query));
    assert(!strata::event_completed(state, 3, query));
    second->done.store(true, std::memory_order_release);
    std::vector<std::thread> readers;
    for (int i = 0; i < 8; ++i) readers.emplace_back([&] {
        for (int j = 0; j < 10000; ++j) assert(strata::event_completed(state, 2, query));
    });
    for (auto& reader : readers) reader.join();
    try { strata::record_event_completion(state, 2, Event{second}); assert(false); }
    catch (const std::logic_error&) {}

    // Model DMA reading host rings on a separate thread. Overwriting a source
    // before its real completion corrupts the checked generation's bytes.
    constexpr int ring = 8, count = 4096, words = 64;
    std::array<std::array<uint64_t, words>, ring> buffers{};
    std::array<strata::EventCompletion<Event>, ring> acknowledgements;
    std::array<uint64_t, ring> wanted{};
    for (auto& ack : acknowledgements) ack = strata::make_event_completion<Event>();
    struct Job { int slot; uint64_t generation; std::shared_ptr<Dma> dma; };
    std::deque<Job> jobs;
    std::mutex mutex;
    std::condition_variable changed;
    bool finished = false;
    std::thread dma_thread([&] {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return finished || !jobs.empty(); });
                if (jobs.empty()) return;
                job = jobs.front(); jobs.pop_front();
            }
            std::this_thread::yield();
            for (auto word : buffers[job.slot]) assert(word == job.generation);
            job.dma->done.store(true, std::memory_order_release);
        }
    });
    for (uint64_t generation = 1; generation <= count; ++generation) {
        int slot = (generation - 1) % ring;
        while (!strata::event_completed(acknowledgements[slot], wanted[slot], query)) std::this_thread::yield();
        buffers[slot].fill(generation);
        auto dma = std::make_shared<Dma>();
        strata::record_event_completion(acknowledgements[slot], generation, Event{dma});
        wanted[slot] = generation;
        { std::lock_guard<std::mutex> lock(mutex); jobs.push_back({slot, generation, dma}); }
        changed.notify_one();
    }
    for (int slot = 0; slot < ring; ++slot)
        while (!strata::event_completed(acknowledgements[slot], wanted[slot], query)) std::this_thread::yield();
    { std::lock_guard<std::mutex> lock(mutex); finished = true; }
    changed.notify_one(); dma_thread.join();
    std::cout << "PASS: real-completion ring reuse, event lifetime, query failure, monotonic generations and concurrent readers\n";
}
