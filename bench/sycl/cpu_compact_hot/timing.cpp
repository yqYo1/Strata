// Main-authored GU/cache component timing; repository LICENSE applies.
// Isolates actual native GU versus cache lookup + compact GU, not ExpertPool.
#include "cache.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <time.h>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int H = 2560, FF = 640, experts = 32;
constexpr float guard = 123.f;
[[gnu::cold]] void need(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
uint64_t number(const char* value) {
    const std::string s(value);
    need(!s.empty() && s.size() <= 20 && std::all_of(s.begin(), s.end(),
         [](char c) { return c >= '0' && c <= '9'; }), "unsigned decimal required");
    return std::stoull(s);
}
double seconds(Clock::time_point first) {
    return std::chrono::duration<double>(Clock::now() - first).count();
}
double cpu_seconds() {
    timespec value{};
    need(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value), "process CPU clock");
    return value.tv_sec + value.tv_nsec * 1e-9;
}
long peak_kib() {
    rusage value{}; need(!getrusage(RUSAGE_SELF, &value), "peak RSS"); return value.ru_maxrss;
}
struct File {
    int fd;
    explicit File(const char* name) : fd(open(name, O_RDONLY | O_CLOEXEC)) {
        need(fd >= 0, "model open");
    }
    ~File() { close(fd); }
    void read(uint8_t* out, size_t bytes, uint64_t offset) {
        struct stat value{};
        need(!fstat(fd, &value) && value.st_size >= 0 && offset <= uint64_t(value.st_size) &&
             bytes <= uint64_t(value.st_size) - offset, "model extent");
        while (bytes) {
            const auto n = pread(fd, out, bytes, offset);
            need(n > 0, "model read"); out += n; bytes -= n; offset += n;
        }
    }
};
struct alignas(64) Output {
    std::array<float, FF + 2> data{};
    Output() { data.front() = data.back() = guard; }
    std::span<float> values() { return {data.data() + 1, FF}; }
};
void pin(unsigned core) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    need(!pthread_setaffinity_np(pthread_self(), sizeof set, &set), "thread pin");
    cpu_set_t actual; CPU_ZERO(&actual);
    need(!pthread_getaffinity_np(pthread_self(), sizeof actual, &actual) &&
         CPU_COUNT(&actual) == 1 && CPU_ISSET(core, &actual), "actual affinity");
}
struct Sample { double wall, cpu; uint64_t jobs; };
struct Fixture {
    cpu::NativeFmt format{};
    std::vector<std::vector<uint8_t>> blobs;
    std::array<std::vector<uint8_t>, 4> acts;
    std::unique_ptr<compact_gu_cache::Cache> cache;
    std::array<std::array<Output, experts>, 4> expected;
    double read_s = 0, encode_s = 0;
    long raw_peak = 0, encoded_peak = 0;

    Fixture(char** argv, bool encode) {
        std::string error;
        const auto down = number(argv[2]);
        need((down == 20 || down == 42) && compact_gu_cache::available() &&
             !cpu::cpu_avx512_ok() &&
             cpu::native_fmt(GGML_TYPE_IQ3_XXS, int(down), H, FF, format, error), "native geometry/CPU");
        need(compact_gu_cache::supported(format) && number(argv[6]) == format.bytes,
             "supported geometry required");
        const std::array<uint64_t, 3> offsets{number(argv[3]), number(argv[4]), number(argv[5])};
        const std::array<size_t, 3> sizes{format.up_off, format.up_off, format.bytes - format.down_off};
        const auto load = Clock::now();
        File file(argv[1]); blobs.resize(experts);
        for (int e = 0; e < experts; ++e) {
            blobs[e].resize(format.bytes); auto* out = blobs[e].data();
            for (size_t i = 0; i < sizes.size(); ++i) {
                need(offsets[i] <= UINT64_MAX - experts * sizes[i], "offset overflow");
                file.read(out, sizes[i], offsets[i] + e * sizes[i]); out += sizes[i];
            }
        }
        read_s = seconds(load); raw_peak = peak_kib();
        if (encode) {
            const auto start = Clock::now();
            cache = std::make_unique<compact_gu_cache::Cache>(1, experts, 64u << 20);
            for (int e = 0; e < experts; ++e)
                need(cache->prepare({0, size_t(e)}, format,
                    std::span<const uint8_t>(blobs[e]).first(format.down_off)), "prepare");
            cache->freeze(); encode_s = seconds(start);
        }
        encoded_peak = peak_kib();
        uint32_t seed = 20261112;
        for (size_t t = 0; t < acts.size(); ++t) {
            std::array<float, H> input{};
            for (int i = 0; i < H; ++i) {
                seed = 1664525u * seed + 1013904223u;
                input[i] = t == 1 ? 0.f : (int(seed >> 8) - 8388608) * 0x1p-23f;
                if (t == 2) input[i] *= i % 37 == 0 ? 32.f : .002f;
                if (t == 3 && i % 256 == 0) input[i] = -16.f;
            }
            acts[t].resize(format.act_bytes);
            cpu::native_quant_act(format, input.data(), acts[t].data());
            for (int e = 0; e < experts; ++e) {
                raw(e, t, expected[t][e]);
                for (float x : expected[t][e].values()) need(std::isfinite(x), "finite oracle");
            }
        }
    }
    void raw(int expert, size_t token, Output& out) const {
        const void* activation[]{acts[token].data()}; float* output[]{out.data.data() + 1};
        cpu::native_gu_rows(format, blobs[expert].data(), activation, 1, output, 0, FF,
                            cpu::NativeGuPhase::Decode);
    }
    void compact(int expert, size_t token, Output& out) const {
        auto lease = cache->lookup({0, size_t(expert)}, format, cpu::NativeGuPhase::Decode, 1);
        need(lease.gu_rows(cpu::NativeGuPhase::Decode, 1, acts[token], out.values(), 0, FF), "GU lookup/call");
    }
    void check(const Output& out, int expert, size_t token) const {
        need(out.data.front() == guard && out.data.back() == guard &&
             std::memcmp(out.data.data(), expected[token][expert].data.data(), sizeof out.data) == 0,
             "GU output/canary mismatch");
    }
    template<bool Compact> Sample batch(int threads, bool stream, uint64_t rounds) {
        need(threads == 1 || threads == 6, "thread geometry");
        pin(0);
        // Thread creation/affinity occur before the clocks. This deliberately
        // isolates synchronous per-worker GU work, not production pool launch.
        std::array<Output, 6> outputs;
        std::array<std::exception_ptr, 6> errors{};
        std::atomic<unsigned> ready{0}, done{0}; std::atomic<bool> begin{false};
        auto work = [&](int t) {
            for (uint64_t i = 0; i < rounds; ++i) {
                const int e = stream ? int((i * threads + t) % experts) : t;
                if constexpr (Compact) compact(e, i % 4, outputs[t]);
                else raw(e, i % 4, outputs[t]);
            }
        };
        std::vector<std::jthread> workers;
        struct Unblock {
            std::atomic<bool>& begin;
            ~Unblock() { begin.store(true, std::memory_order_release); }
        } unblock{begin}; // Also release already-created workers if construction fails.
        for (int t = 1; t < threads; ++t) workers.emplace_back([&, t] {
            try { pin(unsigned(t)); }
            catch (...) { errors[t] = std::current_exception(); }
            ready.fetch_add(1, std::memory_order_release);
            while (!begin.load(std::memory_order_acquire)) _mm_pause();
            if (!errors[t]) try { work(t); } catch (...) { errors[t] = std::current_exception(); }
            done.fetch_add(1, std::memory_order_release);
        });
        while (ready.load(std::memory_order_acquire) != unsigned(threads - 1)) _mm_pause();
        const double cpu_start = cpu_seconds(); const auto start = Clock::now();
        begin.store(true, std::memory_order_release);
        try { work(0); } catch (...) { errors[0] = std::current_exception(); }
        while (done.load(std::memory_order_acquire) != unsigned(threads - 1)) _mm_pause();
        const double elapsed = seconds(start), used = cpu_seconds() - cpu_start;
        workers.clear();
        for (int t = 0; t < threads; ++t) {
            if (errors[t]) std::rethrow_exception(errors[t]);
            const int e = stream ? int(((rounds - 1) * threads + t) % experts) : t;
            check(outputs[t], e, (rounds - 1) % 4);
        }
        return {elapsed, used, rounds * unsigned(threads)};
    }
};

int main(int argc, char** argv) {
    try {
        need(argc == 8, "timing SHARD DOWN GATE UP DOWN_OFFSET BLOB_BYTES check|reference|compact");
        const std::string arm(argv[7]);
        need(arm == "check" || arm == "reference" || arm == "compact", "arm");
        Fixture f(argv, arm != "reference");
        std::cout << std::setprecision(17);
        if (arm == "check") {
            for (size_t t = 0; t < 4; ++t) for (int e = 0; e < experts; ++e) {
                Output output; f.compact(e, t, output); f.check(output, e, t);
                need(f.cache->lookup({0, size_t(e)}, f.format, cpu::NativeGuPhase::Decode, 1).guards_intact(), "view guard");
            }
            const auto audit = cpu::iq3_gu_audit_snapshot();
            need(audit.enabled && !audit.overflow && audit.calls == 128 && audit.nt1_rows == 81920 &&
                 audit.pair_rows == 81920 && audit.threads == 1, "actual pair admission");
            std::cout << "{\"passed\":true,\"cases\":128,\"raw_pair_rows\":81920}\n"; return 0;
        }
        need(!cpu::iq3_gu_audit_snapshot().enabled, "clean audit setting");
        const auto probe = Clock::now();
        for (int i = 0; i < 10000; ++i) {
            const auto value = Clock::now().time_since_epoch().count();
            asm volatile("" : : "g"(value) : "memory");
        }
        const double timer_pair_s = 2 * seconds(probe) / 10000;
        need(timer_pair_s < 1e-4, "timer overhead");
        std::cout << "{\"arm\":\"" << arm << "\",\"down\":" << f.format.d_type
            << ",\"read_s\":" << f.read_s << ",\"encode_publish_s\":" << f.encode_s
            << ",\"raw_peak_kib\":" << f.raw_peak << ",\"encoded_peak_kib\":" << f.encoded_peak
            << ",\"known_view_bytes\":" << (f.cache ? f.cache->known_bytes() : 0)
            << ",\"raw_gu_bytes_per_job\":" << f.format.down_off
            << ",\"compact_gu_bytes_per_job\":" << 2 * compact_gu_cache::projection_blocks * sizeof(compact_gu_cache::Block)
            << ",\"timer_pair_s\":" << timer_pair_s << ",\"cells\":[";
        bool comma = false;
        for (int threads : {1, 6}) for (bool stream : {false, true}) {
            if (comma) std::cout << ','; comma = true;
            std::cout << "{\"threads\":" << threads << ",\"pattern\":\"" << (stream ? "stream32" : "reuse")
                << "\",\"calibration\":[";
            uint64_t rounds = 32; bool calibration_comma = false;
            for (;;) {
                const auto s = arm == "compact" ? f.batch<true>(threads, stream, rounds) : f.batch<false>(threads, stream, rounds);
                if (calibration_comma) std::cout << ','; calibration_comma = true;
                std::cout << "{\"rounds\":" << rounds << ",\"wall_s\":" << s.wall << '}';
                if (s.wall >= .2) break;
                need(rounds < 65536, "calibration duration floor not reached"); rounds *= 2;
            }
            std::cout << "],\"rounds_per_thread\":" << rounds << ",\"samples\":[";
            for (int sample = 0; sample < 3; ++sample) {
                const auto s = arm == "compact" ? f.batch<true>(threads, stream, rounds) : f.batch<false>(threads, stream, rounds);
                need(s.wall >= .1 && timer_pair_s / s.wall < .001, "ineligible timing batch");
                if (sample) std::cout << ',';
                std::cout << "{\"wall_s\":" << s.wall << ",\"process_cpu_s\":" << s.cpu << ",\"jobs\":" << s.jobs << '}';
            }
            std::cout << "]}";
        }
        std::cout << "],\"passed\":true,\"model_qualified\":false}\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
