// Repository LICENSE. CPU-only operation proxy; no runtime/device/model load.
// Sizes match the recorded 48-layer expert blobs. Ordinary aligned anonymous
// RAM is deliberately distinguished from the live source/pinned upload buffers.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <time.h>
#include <vector>

static uint64_t thread_ns() {
    timespec t{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) != 0) std::abort();
    return uint64_t(t.tv_sec) * 1000000000ULL + uint64_t(t.tv_nsec);
}

int main(int argc, char** argv) {
    if (argc != 7) return 2;
    std::array<int, 6> cpus{};
    for (int i = 0; i < 6; ++i) {
        char* end = nullptr;
        const long n = std::strtol(argv[i + 1], &end, 10);
        if (!end || *end || n < 0 || n >= CPU_SETSIZE) return 2;
        cpus[i] = int(n);
        if (std::find(cpus.begin(), cpus.begin() + i, cpus[i]) != cpus.begin() + i) return 2;
    }
    constexpr size_t experts = 512;
    constexpr std::array<size_t, 7> blobs{1510400, 1715200, 1868800, 1971200, 2176000, 2329600, 2662400};
    constexpr size_t maximum = experts * blobs.back();
    void *sp = nullptr, *dp = nullptr;
    if (posix_memalign(&sp, 64, maximum) || posix_memalign(&dp, 64, maximum)) {
        std::free(sp); std::free(dp); return 3;
    }
    std::unique_ptr<uint8_t, decltype(&std::free)> source(static_cast<uint8_t*>(sp), &std::free);
    std::unique_ptr<uint8_t, decltype(&std::free)> dest(static_cast<uint8_t*>(dp), &std::free);
    for (size_t i = 0; i < maximum; ++i) source.get()[i] = uint8_t((i * 131u + (i >> 12)) & 255u);
    std::memset(dest.get(), 0, maximum);
    bool all_ok = true;
    for (const size_t blob : blobs) {
        const size_t bytes = experts * blob;
        for (const int count : {1, 3, 6}) {
            std::barrier ready(count + 1);
            std::array<std::array<uint64_t, 4>, 6> cpu_time{};
            std::array<int, 6> affinity_status{};
            std::array<int, 6> actual_cpu{};
            std::vector<std::thread> workers;
            for (int worker = 0; worker < count; ++worker) {
                workers.emplace_back([&, worker] {
                    cpu_set_t mask;
                    CPU_ZERO(&mask); CPU_SET(cpus[worker], &mask);
                    affinity_status[worker] = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
                    for (int rep = 0; rep < 4; ++rep) {
                        ready.arrive_and_wait();
                        const uint64_t begin = thread_ns();
                        for (size_t e = size_t(worker); e < experts; e += size_t(count))
                            std::memcpy(dest.get() + e * blob, source.get() + e * blob, blob);
                        cpu_time[worker][rep] = thread_ns() - begin;
                        actual_cpu[worker] = sched_getcpu();
                        ready.arrive_and_wait();
                    }
                });
            }
            for (int rep = 0; rep < 4; ++rep) {
                const auto begin = std::chrono::steady_clock::now();
                ready.arrive_and_wait();
                ready.arrive_and_wait();
                const auto end = std::chrono::steady_clock::now();
                const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                // Whole payload validation is outside the copy timing.
                const bool equal = std::memcmp(dest.get(), source.get(), bytes) == 0;
                bool affinity = true;
                uint64_t cpu = 0;
                for (int worker = 0; worker < count; ++worker) {
                    affinity &= affinity_status[worker] == 0 && actual_cpu[worker] == cpus[worker];
                    cpu += cpu_time[worker][rep];
                }
                const bool ok = equal && affinity && elapsed > 0;
                all_ok &= ok;
                std::printf("{\"schema\":1,\"blob_bytes\":%zu,\"experts\":%zu,\"threads\":%d,\"rep\":%d,\"warmup\":%s,\"payload_bytes\":%zu,\"wall_ns\":%lld,\"worker_cpu_ns\":%llu,\"payload_GBps\":%.12g,\"logical_read_plus_write_GBps\":%.12g,\"full_compare\":%s,\"affinity\":%s,\"passed\":%s}\n",
                    blob, experts, count, rep, rep == 0 ? "true" : "false", bytes,
                    static_cast<long long>(elapsed), static_cast<unsigned long long>(cpu),
                    double(bytes) / double(elapsed), 2.0 * double(bytes) / double(elapsed),
                    equal ? "true" : "false", affinity ? "true" : "false", ok ? "true" : "false");
                std::fflush(stdout);
            }
            for (auto& worker : workers) worker.join();
        }
    }
    return all_ok ? 0 : 1;
}
