// Main-authored IQ3_XXS NT1 component comparison; repository LICENSE.
// Inference arithmetic comes from the linked Strata/ggml implementations.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int H = 2560, FF = 640, experts = 252, group = 6, repetitions = 5;
constexpr float sentinel = 123.0f;
void need(bool b, const char* why) { if (!b) throw std::runtime_error(why); }
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
double cpu_seconds() {
    rusage r{}; need(!getrusage(RUSAGE_SELF, &r), "rusage");
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec +
        (r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6;
}
uint64_t number(const char* value) {
    const std::string s(value);
    need(!s.empty() && s.size() <= 20 && std::all_of(s.begin(), s.end(),
        [](char c) { return c >= '0' && c <= '9'; }), "unsigned decimal required");
    return std::stoull(s);
}
uint64_t hash(const void* input, size_t n) {
    const auto* p = static_cast<const uint8_t*>(input);
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
struct Buffer {
    std::vector<uint8_t> raw;
    size_t n;
    explicit Buffer(size_t bytes) : raw(bytes + 128, 0xa7), n(bytes) {
        need(bytes && bytes <= (4u << 20), "buffer extent");
    }
    uint8_t* data() { return raw.data() + 64; }
    void check() const {
        for (int i = 0; i < 64; ++i)
            need(raw[i] == 0xa7 && raw[64 + n + i] == 0xa7, "buffer canary");
    }
};
struct File {
    int fd;
    explicit File(const char* name) : fd(open(name, O_RDONLY | O_CLOEXEC)) {
        need(fd >= 0, "model open");
    }
    ~File() { close(fd); }
    void read(uint8_t* p, size_t n, uint64_t off) const {
        struct stat st{};
        need(!fstat(fd, &st) && st.st_size >= 0 && off <= uint64_t(st.st_size) &&
            n <= uint64_t(st.st_size) - off, "model slice extent");
        while (n) {
            const auto k = pread(fd, p, n, off);
            need(k > 0, "model slice read"); p += k; n -= k; off += k;
        }
    }
};
struct Inputs {
    cpu::NativeFmt fmt;
    std::vector<Buffer> weights;
    std::vector<uint64_t> weight_hashes;
    std::vector<std::array<float, FF + 2>> outputs{experts};
    std::array<Buffer, 4> acts{Buffer(4096), Buffer(4096), Buffer(4096), Buffer(4096)};
    std::array<std::array<float, H>, 4> x{};
    std::array<uint64_t, 4> act_hashes{};
    Buffer scratch{4096};
    explicit Inputs(char** a) {
        std::string error;
        const bool valid = cpu::native_fmt(18, 20, H, FF, fmt, error);
        need(valid, error.c_str());
        need(fmt.bytes == number(a[5]) && number(a[6]) == 512, "model geometry");
        const uint64_t gate = number(a[2]), up = number(a[3]), down = number(a[4]);
        const size_t down_bytes = fmt.bytes - fmt.down_off;
        const auto max = std::numeric_limits<uint64_t>::max();
        need(gate <= max - experts * fmt.up_off && up <= max - experts * fmt.up_off &&
            down <= max - experts * down_bytes, "model offset overflow");
        File f(a[1]); weights.reserve(experts);
        for (int e = 0; e < experts; ++e) {
            weights.emplace_back(fmt.bytes); auto* w = weights.back().data();
            f.read(w, fmt.up_off, gate + e * fmt.up_off);
            f.read(w + fmt.up_off, fmt.up_off, up + e * fmt.up_off);
            f.read(w + fmt.down_off, down_bytes, down + e * down_bytes);
            weight_hashes.push_back(hash(w, fmt.bytes)); outputs[e].fill(sentinel);
        }
        uint32_t seed = 20261105;
        for (int t = 0; t < 4; ++t) {
            for (auto& v : x[t]) { seed = 1664525u * seed + 1013904223u;
                v = (int(seed >> 8) - 8388608) * 0x1p-23f; }
            cpu::native_quant_act(fmt, x[t].data(), acts[t].data());
            act_hashes[t] = hash(acts[t].data(), acts[t].n);
        }
    }
    void check() {
        for (int e = 0; e < experts; ++e) {
            weights[e].check();
            need(hash(weights[e].data(), fmt.bytes) == weight_hashes[e], "weight mutation");
            need(outputs[e].front() == sentinel && outputs[e].back() == sentinel,
                "output canary");
        }
        for (int t = 0; t < 4; ++t) {
            acts[t].check(); need(hash(acts[t].data(), acts[t].n) == act_hashes[t],
                "activation mutation");
        }
        scratch.check();
    }
    void call(bool pair, int e, const void* act, float* out, int begin, int end) {
        if (pair) need(cpu::iq3xxs_pair_gu_rows(fmt, weights[e].data(), act,
            out, begin, end), "pair producer refused");
        else { const void* ap[]{act}; float* fp[]{out};
            cpu::native_gu_rows(fmt, weights[e].data(), ap, 1, fp, begin, end,
                cpu::NativeGuPhase::Decode); }
    }
    void validate() {
        uint64_t checked = 0;
        for (int e = 0; e < experts; ++e) for (int t = 0; t < 4; ++t)
            for (auto range : {std::pair{0, FF}, std::pair{128, 256}}) {
                std::array<float, FF + 2> ref, got; ref.fill(sentinel); got = ref;
                call(false, e, acts[t].data(), ref.data() + 1, range.first, range.second);
                call(true, e, acts[t].data(), got.data() + 1, range.first, range.second);
                need(!std::memcmp(ref.data(), got.data(), sizeof ref), "real-model GU bits");
                for (int r = range.first; r < range.second; ++r)
                    need(std::isfinite(ref[r + 1]), "nonfinite real-model GU");
                ++checked;
            }
        check();
        std::cout << "{\"kind\":\"validation\",\"cases\":" << checked <<
            ",\"bitwise\":true,\"finite\":true,\"inputs_unchanged\":true}" << std::endl;
    }
    void run(bool pair, bool stream, bool pack, int rows) {
        const int count = stream ? experts : group, begin = rows == FF ? 0 : 128;
        for (int b = 0; b < count; b += group) {
            const int t = (b / group) % 4;
            const void* act = acts[t].data();
            if (pack) { cpu::native_quant_act(fmt, x[t].data(), scratch.data()); act = scratch.data(); }
            for (int e = b; e < b + group; ++e)
                call(pair, e, act, outputs[e].data() + 1, begin, begin + rows);
        }
    }
};
int main(int argc, char** argv) try {
    need(argc == 9, "shard gate up down blob_bytes total_experts arm process");
    const auto arm = number(argv[7]), process = number(argv[8]);
    need(arm <= 1 && process < 6, "arm/process");
    const char* single = std::getenv("STRATA_IQ_SINGLE_EXACT");
    need(single && std::string(single) == "0", "reference requires SINGLE_EXACT=0");
    need(!std::getenv("STRATA_IQ_MT_MIN") && !std::getenv("STRATA_IQ2_PAIR_AUDIT"),
        "no alternate NT rule or GU audit");
    const auto cores = cpu::physical_cores(false);
    need(cores.size() == 6 && cpu::pin_current_thread(cores[0]) > 0, "pin one physical core");
    cpu_set_t mask; CPU_ZERO(&mask);
    need(!sched_getaffinity(0, sizeof mask, &mask) && CPU_COUNT(&mask) == 1 &&
        CPU_ISSET(cores[0], &mask), "actual affinity");
    std::cout << std::setprecision(12);
    const auto setup = Clock::now(); Inputs in(argv); const double setup_s = elapsed(setup);
    std::cout << "{\"kind\":\"input\",\"arm\":" << arm << ",\"process\":" << process <<
        ",\"CPU\":" << cores[0] << ",\"experts\":" << experts << ",\"group\":" << group <<
        ",\"blob_bytes\":" << in.fmt.bytes << ",\"setup_s_excluded\":" << setup_s <<
        ",\"reference\":\"native_gu_rows/decode NT1\",\"candidate\":\"direct pair entry\"}" << std::endl;
    in.validate();
    for (int rows : {FF, 128}) for (bool pack : {false, true}) for (bool stream : {false, true}) {
        for (auto& out : in.outputs) out.fill(sentinel);
        in.run(false, stream, pack, rows);
        const auto expected = in.outputs;
        auto pass = [&] { in.run(arm, stream, pack, rows); };
        pass(); double fastest = std::numeric_limits<double>::max();
        for (int i = 0; i < 3; ++i) { const auto t = Clock::now(); pass();
            fastest = std::min(fastest, elapsed(t)); }
        const auto timer = Clock::now();
        for (int i = 0; i < 10000; ++i) { const auto t = Clock::now();
            asm volatile("" : : "r"(&t) : "memory"); }
        const double cost = elapsed(timer) / 10000;
        need(cost < fastest * .001, "timer overhead");
        const uint64_t jobs = stream ? experts : group;
        for (int rep = 0; rep < repetitions; ++rep) {
            const double cpu0 = cpu_seconds(); const auto start = Clock::now();
            uint64_t passes = 0; double wall;
            do { pass(); ++passes; wall = elapsed(start); } while (wall < .35 && passes < 1000000);
            const double scheduled = cpu_seconds() - cpu0;
            need(wall >= .25 && passes < 1000000, "timing duration/pass budget");
            need(!std::memcmp(in.outputs.data(), expected.data(),
                expected.size() * sizeof(expected[0])), "timed output or inactive rows differ");
            std::cout << "{\"kind\":\"sample\",\"arm\":" << arm << ",\"process\":" << process <<
                ",\"rows\":" << rows << ",\"pack\":" << pack << ",\"stream\":" << stream <<
                ",\"rep\":" << rep << ",\"passes\":" << passes << ",\"expert_calls\":" << passes * jobs <<
                ",\"row_ops\":" << passes * jobs * rows << ",\"quant_calls\":" << (pack ? passes * jobs / group : 0) <<
                ",\"wall_s\":" << wall << ",\"cpu_s\":" << scheduled << ",\"timer_s\":" << cost <<
                ",\"fastest_warm_pass_s\":" << fastest <<
                ",\"unique_GU_bytes\":" << jobs * 2 * rows * in.fmt.gu_row <<
                ",\"outputs_match_untimed_reference\":true}" << std::endl;
        }
        in.check();
    }
    in.validate();
    std::cout << "{\"kind\":\"closed\",\"passed\":true}" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "IQ3_XXS pair comparison failed: " << e.what() << std::endl; return 1;
}
