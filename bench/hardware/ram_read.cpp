// Main-authored read-only RAM envelope fixture; repository LICENSE applies.
// Logical load bytes are not a physical DRAM-traffic measurement.
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <latch>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;
[[gnu::cold, noreturn]] static void fail(const char* message) {
    throw std::runtime_error(message);
}

// CPUID: AVX2. Four independent integer accumulators; unsigned XOR is exact.
// The memory clobber and noinline boundary prevent hoisting repeated passes.
[[gnu::noinline]] static uint64_t read_pass(const uint64_t* p, size_t n) {
    asm volatile("" : : "r"(p) : "memory");
    __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a = _mm256_xor_si256(a, _mm256_loadu_si256((const __m256i*)(p + i)));
        b = _mm256_xor_si256(b, _mm256_loadu_si256((const __m256i*)(p + i + 4)));
        c = _mm256_xor_si256(c, _mm256_loadu_si256((const __m256i*)(p + i + 8)));
        d = _mm256_xor_si256(d, _mm256_loadu_si256((const __m256i*)(p + i + 12)));
    }
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256((__m256i*)lanes,
        _mm256_xor_si256(_mm256_xor_si256(a, b), _mm256_xor_si256(c, d)));
    uint64_t value = lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3];
    for (; i < n; ++i) value ^= p[i];
    _mm256_zeroupper();
    return value;
}

static double thread_seconds() {
    timespec t{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t)) fail("thread CPU clock");
    return double(t.tv_sec) + double(t.tv_nsec) * 1e-9;
}

struct alignas(64) Worker {
    double cpu = 0;
    uint64_t checksum = 0;
    int error = 0;
};
struct Sample {
    double wall, cpu;
    uint64_t bytes;
    long minor, major;
};

static Sample batch(const uint64_t* data, size_t count, const std::vector<int>& cpus,
                    const std::vector<uint64_t>& expected, uint64_t passes) {
    std::vector<Worker> workers(cpus.size());
    std::latch ready(cpus.size()), start(1), done(cpus.size());
    std::vector<std::thread> threads;
    for (size_t t = 0; t < cpus.size(); ++t) threads.emplace_back([&, t] {
        cpu_set_t set;
        CPU_ZERO(&set); CPU_SET(cpus[t], &set);
        workers[t].error = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        const size_t lo = count * t / cpus.size(), hi = count * (t + 1) / cpus.size();
        ready.count_down(); start.wait();
        if (!workers[t].error) {
            const double begin = thread_seconds();
            uint64_t checksum = 0;
            for (uint64_t p = 0; p < passes; ++p) checksum += read_pass(data + lo, hi - lo);
            workers[t].cpu = thread_seconds() - begin;
            workers[t].checksum = checksum;
        }
        done.count_down();
    });
    ready.wait();
    rusage before{}, after{};
    if (getrusage(RUSAGE_SELF, &before)) fail("rusage before");
    const auto begin = Clock::now();
    start.count_down(); done.wait();
    const double wall = std::chrono::duration<double>(Clock::now() - begin).count();
    if (getrusage(RUSAGE_SELF, &after)) fail("rusage after");
    for (auto& t : threads) t.join();
    double cpu = 0;
    for (size_t t = 0; t < workers.size(); ++t) {
        if (workers[t].error || workers[t].checksum != expected[t] * passes)
            fail("affinity or exact checksum failed");
        cpu += workers[t].cpu;
    }
    return {wall, cpu, uint64_t(count) * sizeof(uint64_t) * passes,
        after.ru_minflt - before.ru_minflt, after.ru_majflt - before.ru_majflt};
}

int main(int argc, char** argv) try {
    if (argc != 2 || !__builtin_cpu_supports("avx2")) fail("AVX2 CPU list required");
    std::vector<int> cpus;
    std::string list = argv[1];
    for (size_t pos = 0; pos < list.size();) {
        size_t end = list.find(',', pos); if (end == std::string::npos) end = list.size();
        std::string token = list.substr(pos, end - pos); size_t used = 0;
        int cpu = std::stoi(token, &used);
        if (used != token.size() || cpu < 0 || cpu >= CPU_SETSIZE ||
            std::find(cpus.begin(), cpus.end(), cpu) != cpus.end()) fail("invalid CPU");
        cpus.push_back(cpu); pos = end + 1;
    }
    if (cpus.empty() || cpus.size() > 6) fail("one through six CPUs required");
    const auto setup_begin = Clock::now();
    constexpr size_t bytes = size_t(1) << 30, count = bytes / sizeof(uint64_t);
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    void* mapping = mmap(nullptr, bytes + 2 * page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) fail("mmap");
    auto* data = reinterpret_cast<uint64_t*>(static_cast<char*>(mapping) + page);
    if (mprotect(data, bytes, PROT_READ | PROT_WRITE)) fail("mprotect RW");
    uint64_t state = 0x6578616374726561ULL;
    for (size_t i = 0; i < count; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17; data[i] = state;
    }
    if (mprotect(data, bytes, PROT_READ)) fail("mprotect read-only");
    std::vector<uint64_t> expected(cpus.size());
    for (size_t t = 0; t < cpus.size(); ++t) {
        const size_t lo = count * t / cpus.size(), hi = count * (t + 1) / cpus.size();
        uint64_t value = 0;
        for (size_t i = lo; i < hi; ++i) value ^= data[i];
        expected[t] = value;
        if (!value || read_pass(data + lo, hi - lo) != value) fail("full slice checksum");
    }
    for (size_t n : {size_t(0), size_t(1), size_t(9), size_t(16), size_t(35), size_t(255)}) {
        uint64_t expected_tail = 0;
        for (size_t i = 0; i < n; ++i) expected_tail ^= data[count - n + i];
        if (read_pass(data + count - n, n) != expected_tail) fail("guard-page tail");
    }
    const double setup = std::chrono::duration<double>(Clock::now() - setup_begin).count();
    constexpr int timer_iterations = 10000;
    const auto timer_begin = Clock::now();
    for (int i = 0; i < timer_iterations; ++i) {
        const auto value = Clock::now().time_since_epoch().count();
        const double cpu = thread_seconds();
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage)) fail("timer calibration rusage");
        asm volatile("" : : "g"(value), "g"(cpu), "g"(usage.ru_minflt) : "memory");
    }
    // Charge a complete clock+rusage triplet for every individual observer
    // call, including both worker CPU clocks. This is a conservative estimate,
    // measured outside all timed read batches rather than assumed negligible.
    const double timer_estimate = (4 + 2 * cpus.size()) *
        std::chrono::duration<double>(Clock::now() - timer_begin).count() / timer_iterations;
    if (!(timer_estimate > 0)) fail("timer calibration");
    uint64_t passes = 1;
    for (;;) {
        Sample s = batch(data, count, cpus, expected, passes);
        if (s.wall >= .75) break;
        if (passes >= 1024) fail("calibration too short");
        passes *= 2;
    }
    std::array<Sample, 3> samples;
    for (auto& s : samples) {
        s = batch(data, count, cpus, expected, passes);
        if (s.wall < .5 || s.major || !(timer_estimate / s.wall < .001))
            fail("ineligible duration, observer overhead or major fault");
    }
    std::printf("{\"schema\":\"ram-read-v1\",\"buffer_bytes\":%zu,\"threads\":%zu,"
        "\"passes\":%llu,\"setup_seconds\":%.9f,\"timer_iterations\":%d,"
        "\"timer_estimate_seconds\":%.12f,\"checksum_valid\":true,\"samples\":[",
        bytes, cpus.size(), (unsigned long long)passes, setup, timer_iterations, timer_estimate);
    for (size_t i = 0; i < samples.size(); ++i) {
        auto s = samples[i];
        std::printf("%s{\"wall_seconds\":%.9f,\"worker_cpu_seconds\":%.9f,\"logical_bytes\":%llu,"
            "\"logical_GBps\":%.9f,\"minor_faults\":%ld,\"major_faults\":%ld}", i ? "," : "",
            s.wall, s.cpu, (unsigned long long)s.bytes, double(s.bytes) / s.wall / 1e9, s.minor, s.major);
    }
    std::puts("]}");
    if (munmap(mapping, bytes + 2 * page)) fail("munmap");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "RAM_READ_FAILED: %s\n", e.what()); return 1;
}
