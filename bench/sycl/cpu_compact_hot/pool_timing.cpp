// Main-authored production-pool comparison; repository LICENSE applies.
// CPU-only. V2focus:32jobs,5workers+host,2s calibration/1s eligibility floor.
// Includes coordinate/lease handoff, GU, hidden quantization and Down.
#include "cache.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
namespace gu = cpu::compact_gu;
using Clock = std::chrono::steady_clock;
constexpr int H = 2560, FF = 640, E = 32;
constexpr float guard = 123.f;
[[gnu::cold]] void need(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
uint64_t number(const char* value) {
    const std::string s(value);
    need(!s.empty() && s.size() <= 20 && std::all_of(s.begin(), s.end(),
        [](char c) { return c >= '0' && c <= '9'; }), "unsigned decimal required");
    return std::stoull(s);
}
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
double cpu_seconds() {
    timespec t{}; need(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t), "CPU clock");
    return t.tv_sec + t.tv_nsec * 1e-9;
}
long peak_kib() {
    rusage value{}; need(!getrusage(RUSAGE_SELF, &value), "peak RSS"); return value.ru_maxrss;
}
struct File {
    int fd;
    explicit File(const char* name) : fd(open(name, O_RDONLY | O_CLOEXEC)) { need(fd >= 0, "model open"); }
    ~File() { close(fd); }
    void read(uint8_t* dst, size_t bytes, uint64_t offset) {
        struct stat s{};
        need(!fstat(fd, &s) && s.st_size >= 0 && offset <= uint64_t(s.st_size) &&
             bytes <= uint64_t(s.st_size) - offset, "model extent");
        while (bytes) {
            const auto n = pread(fd, dst, bytes, offset);
            need(n > 0, "short model read"); dst += n; bytes -= n; offset += n;
        }
    }
};
struct alignas(64) Output {
    std::array<float, H + 2> data{};
    Output() { data.fill(guard); }
    float* values() { return data.data() + 1; }
    void check() const {
        need(data.front() == guard && data.back() == guard, "output bounds");
        for (int i = 1; i <= H; ++i) need(std::isfinite(data[i]), "finite output");
    }
};
bool equal(const Output& a, const Output& b) {
    return std::memcmp(a.data.data(), b.data.data(), sizeof a.data) == 0;
}
std::set<int> tids() {
    std::set<int> result;
    for (const auto& p : std::filesystem::directory_iterator("/proc/self/task"))
        result.insert(std::stoi(p.path().filename().string()));
    return result;
}
int single_cpu(int tid) {
    cpu_set_t mask; CPU_ZERO(&mask);
    if (sched_getaffinity(tid, sizeof mask, &mask) || CPU_COUNT(&mask) != 1) return -1;
    for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &mask)) return c;
    return -1;
}
struct RestoreAffinity {
    cpu_set_t saved{};
    RestoreAffinity() {
        need(!sched_getaffinity(0, sizeof saved, &saved) && CPU_COUNT(&saved) == 12,
             "full initial twelve-CPU affinity required");
    }
    ~RestoreAffinity() { if (sched_setaffinity(0, sizeof saved, &saved)) std::terminate(); }
};
struct Pool {
    // Capture broad affinity before the pool detects topology. Pinning the
    // caller first would make new workers inherit/detect only CPU0.
    RestoreAffinity restore;
    std::set<int> before = tids();
    cpu::ExpertPool native;
    std::map<int, int> workers;
    explicit Pool(int n) : native(n, true, n == 5) {
        need(n == 1 || n == 5, "worker geometry");
        cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(0, &mask);
        need(!sched_setaffinity(0, sizeof mask, &mask) && single_cpu(getpid()) == 0, "host CPU0 pin");
        const auto deadline = Clock::now() + std::chrono::seconds(2);
        do {
            workers.clear(); std::set<int> cores;
            for (int tid : tids()) if (!before.contains(tid)) {
                const int c = single_cpu(tid); workers[tid] = c; cores.insert(c);
            }
            std::set<int> expected; for (int c = 1; c <= n; ++c) expected.insert(c);
            if (int(workers.size()) == n && cores == expected) { verify(); return; }
            std::this_thread::yield();
        } while (Clock::now() < deadline);
        need(false, "actual worker affinity admission");
    }
    void verify() const {
        need(single_cpu(getpid()) == 0 && native.workers() == int(workers.size()) &&
             native.host_works() == (workers.size() == 5), "host/pool geometry changed");
        auto expected = before;
        for (const auto& [tid, core] : workers) {
            expected.insert(tid); need(single_cpu(tid) == core, "worker affinity changed");
        }
        need(tids() == expected, "unexpected or retired native thread");
    }
};
struct Fixture {
    cpu::NativeFmt format{};
    std::vector<uint8_t> raw;
    std::shared_ptr<const gu::Cache> cache;
    std::array<std::vector<uint8_t>, 4> acts;
    std::array<std::array<Output, E>, 4> expected;
    double read_s = 0, encode_s = 0;
    long raw_peak = 0, encoded_peak = 0;
    Fixture(char** argv, bool encode) {
        std::string error;
        const auto down = number(argv[2]);
        need((down == 20 || down == 42) && gu::available() &&
             cpu::native_fmt(18, int(down), H, FF, format, error) && gu::supported(format) &&
             format.bytes == number(argv[6]), "native CPU/format");
        const std::array<uint64_t, 3> offsets{number(argv[3]), number(argv[4]), number(argv[5])};
        const std::array<size_t, 3> sizes{format.up_off, format.up_off, format.bytes - format.down_off};
        const auto start = Clock::now(); File file(argv[1]); raw.resize(E * format.bytes);
        for (int e = 0; e < E; ++e) {
            auto* dst = raw.data() + size_t(e) * format.bytes;
            for (size_t r = 0; r < 3; ++r) {
                need(offsets[r] <= UINT64_MAX - E * sizes[r], "offset overflow");
                file.read(dst, sizes[r], offsets[r] + e * sizes[r]); dst += sizes[r];
            }
        }
        read_s = seconds(start); raw_peak = peak_kib();
        if (encode) {
            const auto begin = Clock::now();
            const gu::LayerView view{0, E, format.bytes, raw.size(), raw.data(), format};
            cache = gu::prepare(1, E, 64u << 20, {&view, 1}, error);
            need(cache && cache->prepared_entries() == E, "complete cache preparation"); encode_s = seconds(begin);
        }
        encoded_peak = peak_kib();
        uint32_t seed = 20261113;
        for (size_t t = 0; t < acts.size(); ++t) {
            std::array<float, H> input{};
            for (int i = 0; i < H; ++i) {
                seed = 1664525u * seed + 1013904223u;
                input[i] = t == 1 ? 0.f : (int(seed >> 8) - 8388608) * 0x1p-23f;
                if (t == 2) input[i] *= i % 37 == 0 ? 32.f : .002f;
                if (t == 3 && i % 256 == 0) input[i] = -16.f;
            }
            acts[t].resize(format.act_bytes); cpu::native_quant_act(format, input.data(), acts[t].data());
        }
        // Full production-pool oracle, excluded from setup/speed clocks.
        Pool oracle(1); std::array<cpu::ExpertJobMulti, E> jobs{};
        for (size_t t = 0; t < acts.size(); ++t) {
            for (int e = 0; e < E; ++e) {
                jobs[e].blob = raw.data() + size_t(e) * format.bytes; jobs[e].nt = 1;
                jobs[e].nact[0] = acts[t].data(); jobs[e].out[0] = expected[t][e].values();
            }
            oracle.native.run_split_multi_native(format, jobs.data(), E, cpu::NativeGuPhase::Decode);
            for (const auto& out : expected[t]) out.check();
        }
        oracle.verify();
    }
    void guards() const {
        if (cache) for (int e = 0; e < E; ++e)
            need(cache->lookup({0, size_t(e)}, format, cpu::NativeGuPhase::Decode, 1).guards_intact(), "GU guards");
    }
};
// The +7 per32 jobs makes key(j+96) differ by21, unlike the old period32 data.
int expert(int j, uint64_t round) { return int((13u * j + 7u * (j / E) + 13u * round + 3u) % E); }
size_t token(int j, uint64_t round, int t = 0) { return size_t((3u * j + j / 5 + round + t) % 4); }
struct Case {
    Fixture& f;
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<gu::Lease> views;
    std::vector<Output> out;
    explicit Case(Fixture& fixture, int n) : f(fixture), jobs(n), views(n), out(size_t(n) * 2) {}
    void bind(uint64_t round, bool stream, bool compact, cpu::NativeGuPhase phase,
              int policy = 0, bool misses = false, bool wrong_offset = false) {
        for (int j = 0; j < int(jobs.size()); ++j) {
            auto& job = jobs[j]; const int e = expert(j, stream ? round : 0);
            job.blob = f.raw.data() + size_t(e) * f.format.bytes;
            job.nt = policy == 0 ? 1 : policy == 1 ? 2 : 1 + (j & 1);
            for (int t = 0; t < job.nt; ++t) {
                job.nact[t] = f.acts[token(j, round, t)].data(); job.out[t] = out[size_t(j) * 2 + t].values();
            }
            if (compact) {
                const int key = wrong_offset ? expert(j % 96, stream ? round : 0) : e;
                views[j] = misses && (j + j / 17 + round) % 5 == 0 ? gu::Lease{} :
                    f.cache->lookup({0, size_t(key)}, f.format, phase, job.nt);
            }
        }
    }
    void run(Pool& pool, bool compact, cpu::NativeGuPhase phase) {
        pool.native.run_split_multi_native(f.format, jobs.data(), int(jobs.size()), phase,
            compact ? std::span<const gu::Lease>(views) : std::span<const gu::Lease>{});
    }
    void check(uint64_t round, bool stream) const {
        for (int j = 0; j < int(jobs.size()); ++j) {
            out[size_t(j) * 2].check();
            need(equal(out[size_t(j) * 2], f.expected[token(j, round)][expert(j, stream ? round : 0)]),
                 "timed final whole-pool bits");
        }
    }
};
void numerical(Fixture& f) {
    uint64_t cases = 0, nt1_rows = 0, cached_rows = 0;
    for (int workers : {1, 5}) {
        Pool pool(workers);
        for (int n : {1, 6, 32, 97, 193}) for (int policy : {0, 1, 2})
            for (auto phase : {cpu::NativeGuPhase::Decode, cpu::NativeGuPhase::Prefill}) for (bool misses : {false, true}) {
                Case ref(f, n), candidate(f, n); const uint64_t round = cases + 11;
                ref.bind(round, true, false, phase, policy, misses);
                candidate.bind(round, true, true, phase, policy, misses);
                uint64_t rows = 0, selected = 0;
                for (int j = 0; j < n; ++j) if (ref.jobs[j].nt == 1) {
                    rows += FF; if (candidate.views[j]) selected += FF;
                }
                const auto a = cpu::iq3_gu_audit_snapshot();
                need(a.enabled && !a.overflow, "audit must be enabled"); ref.run(pool, false, phase);
                const auto b = cpu::iq3_gu_audit_snapshot(); candidate.run(pool, true, phase);
                const auto c = cpu::iq3_gu_audit_snapshot();
                need(!c.overflow && b.nt1_rows - a.nt1_rows == rows && c.nt1_rows - b.nt1_rows == rows - selected &&
                     b.pair_rows - a.pair_rows == (phase == cpu::NativeGuPhase::Decode ? rows : 0) &&
                     c.pair_rows - b.pair_rows == (phase == cpu::NativeGuPhase::Decode ? rows - selected : 0),
                     "actual numeric dispatch accounting");
                for (int j = 0; j < n; ++j) for (int t = 0; t < ref.jobs[j].nt; ++t) {
                    const size_t i = size_t(j) * 2 + t; ref.out[i].check(); candidate.out[i].check();
                    need(equal(ref.out[i], candidate.out[i]), "nonperiodic pool parity");
                }
                nt1_rows += rows; cached_rows += selected; ++cases;
            }
        Case wrong(f, 193); wrong.bind(9, true, true, cpu::NativeGuPhase::Decode, 0, false, true);
        wrong.run(pool, true, cpu::NativeGuPhase::Decode); bool detected = false;
        for (int j = 96; j < 193; ++j) {
            wrong.out[size_t(j) * 2].check();
            if (!equal(wrong.out[size_t(j) * 2], f.expected[token(j, 9)][expert(j, 9)])) detected = true;
        }
        need(detected, "wrong-offset negative control must differ"); pool.verify();
    }
    f.guards();
    std::cout << "{\"passed\":true,\"cases\":" << cases << ",\"nt1_rows\":" << nt1_rows
              << ",\"compact_rows\":" << cached_rows << ",\"offset_control_detected\":true}\n";
}
struct Sample { double wall, cpu, gu, quant, down; uint64_t calls, jobs; };
Sample batch(Case& work, Pool& pool, bool compact, bool stream, uint64_t rounds) {
    pool.verify();
    const auto gu0 = pool.native.ms_multi_gu, q0 = pool.native.ms_multi_q, down0 = pool.native.ms_multi_down;
    const double cpu0 = cpu_seconds(); const auto start = Clock::now();
    for (uint64_t i = 0; i < rounds; ++i) {
        work.bind(i, stream, compact, cpu::NativeGuPhase::Decode); work.run(pool, compact, cpu::NativeGuPhase::Decode);
    }
    Sample s{seconds(start), cpu_seconds() - cpu0, (pool.native.ms_multi_gu - gu0) * .001,
        (pool.native.ms_multi_q - q0) * .001, (pool.native.ms_multi_down - down0) * .001,
        rounds, rounds * work.jobs.size()};
    need(s.gu >= 0 && s.quant >= 0 && s.down >= 0 && s.gu + s.quant + s.down <= s.wall + 1e-6,
         "pool stage conservation");
    work.check(rounds - 1, stream); pool.verify(); return s;
}
int main(int argc, char** argv) {
    try {
        need(argc == 8, "pool_timing SHARD DOWN GATE UP DOWN_OFFSET BLOB_BYTES check|reference|compact");
        const std::string arm(argv[7]); need(arm == "check" || arm == "reference" || arm == "compact", "arm");
        Fixture f(argv, arm != "reference"); std::cout << std::setprecision(17);
        if (arm == "check") { numerical(f); return 0; }
        need(!cpu::iq3_gu_audit_snapshot().enabled, "clean audit setting");
        const auto start = Clock::now();
        for (int i = 0; i < 10000; ++i) {
            const auto value = Clock::now().time_since_epoch().count(); asm volatile("" : : "g"(value) : "memory");
        }
        const double timer_pair = 2 * seconds(start) / 10000; need(timer_pair < 1e-4, "timer overhead");
        std::cout << "{\"timing_envelope\":\"focus32_2s_1s\",\"arm\":\"" << arm << "\",\"down\":" << f.format.d_type << ",\"read_s\":" << f.read_s
                  << ",\"encode_publish_s\":" << f.encode_s << ",\"raw_peak_kib\":" << f.raw_peak
                  << ",\"encoded_peak_kib\":" << f.encoded_peak << ",\"known_view_bytes\":" << (f.cache ? f.cache->known_bytes() : 0)
                  << ",\"raw_bytes_per_job\":" << f.format.bytes << ",\"timer_pair_s\":" << timer_pair << ",\"cells\":[";
        bool comma = false;
        for (int workers : {5}) {
            Pool pool(workers);
            for (int n : {32}) for (bool stream : {false, true}) {
                if (comma) std::cout << ','; comma = true; Case work(f, n);
                std::cout << "{\"workers\":" << workers << ",\"host_works\":" << (workers == 5 ? "true" : "false")
                          << ",\"host_cpu\":0,\"worker_cpus\":[";
                for (int i = 1; i <= workers; ++i) { if (i != 1) std::cout << ','; std::cout << i; }
                std::cout << "],\"jobs_per_call\":" << n << ",\"pattern\":\"" << (stream ? "stream32" : "reuse")
                          << "\",\"calibration\":[";
                uint64_t rounds = 1; bool calibrating = false;
                for (;;) {
                    const auto s = batch(work, pool, arm == "compact", stream, rounds);
                    if (calibrating) std::cout << ','; calibrating = true;
                    std::cout << "{\"rounds\":" << rounds << ",\"wall_s\":" << s.wall << '}';
                    if (s.wall >= 2.) break;
                    need(rounds < 65536, "calibration duration floor"); rounds *= 2;
                }
                std::cout << "],\"rounds\":" << rounds << ",\"samples\":[";
                for (int i = 0; i < 3; ++i) {
                    const auto s = batch(work, pool, arm == "compact", stream, rounds);
                    need(s.wall >= 1. && timer_pair / s.wall < .001, "ineligible batch");
                    if (i) std::cout << ',';
                    std::cout << "{\"wall_s\":" << s.wall << ",\"process_cpu_s\":" << s.cpu
                              << ",\"calls\":" << s.calls << ",\"jobs\":" << s.jobs << ",\"pool_gu_s\":" << s.gu
                              << ",\"pool_quant_s\":" << s.quant << ",\"pool_down_s\":" << s.down << '}';
                }
                std::cout << "]}";
            }
        }
        f.guards();
        std::cout << "],\"final_peak_kib\":" << peak_kib() << ",\"passed\":true,\"model_qualified\":false}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
