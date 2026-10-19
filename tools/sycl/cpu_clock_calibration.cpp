// Standalone CPU-only measurement of the production CpuTimeRegion wrapper.
// No inference kernels or runtime libraries are linked.
#include "strata/core/cpu_clock.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

using namespace strata::core;
constexpr int jobs = 500000, body_steps = 256;

struct Sample {
    uint64_t cpu_ns = 0, checksum = 0;
    CpuTimeCounter region;
    int errors = 0;
};

struct Affinity {
    cpu_set_t previous{};
    bool saved = false;
    int errors = 0;
    explicit Affinity(int core) {
        saved = pthread_getaffinity_np(pthread_self(), sizeof previous, &previous) == 0;
        cpu_set_t wanted{}, actual{};
        CPU_ZERO(&wanted); CPU_SET(core, &wanted);
        if (!saved || !CPU_ISSET(core, &previous) ||
            pthread_setaffinity_np(pthread_self(), sizeof wanted, &wanted) != 0 ||
            pthread_getaffinity_np(pthread_self(), sizeof actual, &actual) != 0 ||
            CPU_COUNT(&actual) != 1 || !CPU_ISSET(core, &actual)) ++errors;
    }
    void restore() {
        if (saved && pthread_setaffinity_np(pthread_self(), sizeof previous, &previous) != 0) ++errors;
        saved = false;
    }
    ~Affinity() { restore(); }
};

void work(int id, bool enabled, Sample& s) {
    volatile uint64_t value = static_cast<uint64_t>(id + 19);
    const auto begin = thread_cpu_time();
    for (int j = 0; j < jobs; ++j) {
        CpuTimeRegion region(enabled ? &s.region : nullptr);
        for (int k = 0; k < body_steps; ++k)
            value = value * 6364136223846793005ULL + 1;
        region.close();
    }
    const auto delta = cpu_time_delta(begin, thread_cpu_time());
    s.cpu_ns = delta.ns;
    s.checksum = value;
    if (!delta.valid || delta.ns == 0 || s.region.calls != static_cast<uint64_t>(enabled ? jobs : 0) ||
        s.region.errors || s.region.unwound || s.region.ns > delta.ns) ++s.errors;
}

bool run_case(int sequence, int count, bool enabled) {
    std::vector<Sample> samples(static_cast<size_t>(count));
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    // Create before pinning the parent, so each worker can select its own core.
    for (int id = 1; id < count; ++id) workers.emplace_back([&, id] {
        Affinity affinity(id);
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        work(id, enabled, samples[static_cast<size_t>(id)]);
        affinity.restore();
        samples[static_cast<size_t>(id)].errors += affinity.errors;
    });
    Affinity affinity(0);
    while (ready.load(std::memory_order_acquire) != count - 1) std::this_thread::yield();
    const auto begin = std::chrono::steady_clock::now();
    start.store(true, std::memory_order_release);
    work(0, enabled, samples[0]);
    for (auto& worker : workers) worker.join();
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count();
    affinity.restore();
    samples[0].errors += affinity.errors;
    int errors = 0;
    for (const auto& s : samples) errors += s.errors;
    std::printf("{\"sequence\":%d,\"enabled\":%s,\"threads\":%d,\"jobs_per_thread\":%d,"
                "\"body_steps\":%d,\"wall_ns\":%lld,\"errors\":%d",
                sequence, enabled ? "true" : "false", count, jobs, body_steps,
                static_cast<long long>(wall), errors);
    const auto array = [&](const char* key, auto get) {
        std::printf(",\"%s\":[", key);
        for (int id = 0; id < count; ++id)
            std::printf("%s%llu", id ? "," : "", static_cast<unsigned long long>(get(samples[static_cast<size_t>(id)])));
        std::printf("]");
    };
    array("thread_cpu_ns", [](const Sample& s) { return s.cpu_ns; });
    array("region_cpu_ns", [](const Sample& s) { return s.region.ns; });
    array("region_calls", [](const Sample& s) { return s.region.calls; });
    array("checksum", [](const Sample& s) { return s.checksum; });
    std::printf("}\n");
    return errors == 0;
}

int main(int argc, char** argv) {
    if (argc != 3 || std::strlen(argv[1]) != 1 || argv[1][0] < '0' || argv[1][0] > '5' ||
        (std::strcmp(argv[2], "on") && std::strcmp(argv[2], "off"))) return 2;
    const int sequence = argv[1][0] - '0';
    const bool enabled = !std::strcmp(argv[2], "on");
    if (decode_cpu_profile_enabled() != enabled) return 2;
    bool passed = true;
    for (const int count : (sequence % 2 ? std::vector<int>{6, 1} : std::vector<int>{1, 6}))
        passed = run_case(sequence, count, enabled) && passed;
    return passed ? 0 : 1;
}
