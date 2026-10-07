// Test the actual extracted Stager with CPU allocation and deferred DMA stubs.
// No SYCL runtime, GPU or model is used by this executable.
#include "strata/host_completion.hpp"
#include "strata/host_wait.hpp"
#include "strata/sycl_allocation.hpp"
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
using Clock = std::chrono::steady_clock;

struct Queue {
    struct Handler {
        std::function<void()> task;
        template<class F> void host_task(F f) { task = std::move(f); }
    };
    std::vector<std::function<void()>> tasks;
    template<class F> void submit(F f) { Handler h; f(h); tasks.push_back(std::move(h.task)); }
    void drain() { for (auto& task : tasks) task(); tasks.clear(); }
};
namespace dpct {
using queue_ptr = Queue*;
inline Queue& get_in_order_queue() { static Queue q; return q; }
inline int get_current_device_id() { return 0; }
inline void select_device(int) {}
}
namespace sycl {
inline void* malloc_host(size_t size, Queue&) { return std::malloc(size); }
inline void free(void* p, Queue&) { std::free(p); }
}
namespace core {
struct ExpertSource { bool copy_blob(int32_t, int32_t, uint8_t*) { return false; } };
struct GgufExpertSource { bool read_into(int32_t, int32_t, uint8_t*, size_t) const { return false; } };
}
inline bool force_pageable() { return false; }
#define DPCT_CHECK_ERROR(...) ([&] { __VA_ARGS__; return 0; }())
#include "stager-under-test.hpp"

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    if (mode == "timeout") {
        strata::wait_host_ready([] { return false; }, "test missing DMA completion", std::chrono::milliseconds(10));
        return 99;
    }
    setenv("STRATA_STAGER_RING", "2", 1);
    constexpr size_t bytes = 128;
    std::vector<std::vector<uint8_t>> source(8, std::vector<uint8_t>(bytes));
    for (size_t i = 0; i != source.size(); ++i) std::fill(source[i].begin(), source[i].end(), uint8_t(i + 1));
    Queue queue;
    Stager stager;
    assert(stager.init(bytes, 3));
    auto jobs = [&](int count) {
        std::vector<Stager::Job> result;
        for (int j = 0; j != count; ++j) result.push_back({source[j].data(), bytes});
        return result;
    };
    if (mode == "finish-before-dma") {
        stager.start(jobs(4));
        for (int j = 0; j != 2; ++j) {
            const auto* p = stager.wait(j);
            queue.tasks.push_back([p, j] { for (size_t b = 0; b != bytes; ++b) assert(p[b] == j + 1); });
            stager.issued_one(j, &queue);
        }
        std::puts("first two DMAs queued; stopping before acknowledgement"); std::fflush(stdout);
        stager.finish();
        // The worker must not overwrite either slot even after finish returns.
        queue.drain();
        assert(!stager.ready[2].load() && !stager.ready[3].load());
    } else if (mode == "cancel-issuer") {
#if STRATA_STOP_WAIT
        stager.start(jobs(3));
        (void) stager.wait(0); (void) stager.wait(1);
        std::atomic<bool> stop{false};
        std::thread issuer([&] { assert(stager.wait(2, &stop) == nullptr); });
        stop.store(true, std::memory_order_release);
        issuer.join();
        stager.finish();
#else
        return 98;
#endif
    } else if (mode == "reuse") {
        uint64_t copied = 0;
        for (int generation = 0; generation != 256; ++generation) {
            stager.start(jobs(8));
            if (generation) queue.drain(); // Release the previous generation's final slots.
            for (int j = 0; j != 8; ++j) {
                if (j >= 2) queue.drain();
                const auto* p = stager.wait(j);
                queue.tasks.push_back([&, p, j] {
                    for (size_t b = 0; b != bytes; ++b) assert(p[b] == j + 1);
                    ++copied;
                });
                stager.issued_one(j, &queue);
            }
            stager.finish();
            // Leave the last two DMA reads pending until the next generation.
        }
        queue.drain();
        assert(copied == 2048);
    } else return 97;
    std::printf("PASS: %s\n", mode.c_str());
}
