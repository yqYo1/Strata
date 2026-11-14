// Main-authored IQ3 NT1 comparisons; repository LICENSE.
// Inference arithmetic comes from the linked Strata/ggml implementations.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
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
#ifdef STRATA_BENCH_IQ3S_PAIR
constexpr bool direct_iq3s = true;
#else
constexpr bool direct_iq3s = false;
#endif
constexpr int gu_type = direct_iq3s ? 21 : 18;
constexpr float sentinel = 123.0f;
enum class Entry { Oracle, Production };
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
    unsigned arm;
    uint64_t direct_calls = 0, direct_rows = 0, direct_pair_rows = 0;
    std::vector<Buffer> weights;
    std::vector<uint64_t> weight_hashes;
    std::vector<std::array<float, FF + 2>> outputs{experts};
    std::array<Buffer, 4> acts{Buffer(4096), Buffer(4096), Buffer(4096), Buffer(4096)};
    std::array<std::array<float, H>, 4> x{};
    std::array<uint64_t, 4> act_hashes{};
    Buffer scratch{4096};
    explicit Inputs(char** a, unsigned selected_arm) : arm(selected_arm) {
        std::string error;
        const bool valid = cpu::native_fmt(gu_type, 20, H, FF, fmt, error);
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
    void call(Entry entry, int e, const void* act, float* out, int begin, int end) {
        if (entry == Entry::Oracle || (direct_iq3s && arm == 0)) {
            const auto dot = ggml_get_type_traits_cpu(ggml_type(gu_type))->vec_dot;
            for (int r = begin; r < end; ++r) {
                float g, u;
                dot(H, &g, 0, weights[e].data() + r * fmt.gu_row, 0, act, 0, 1);
                dot(H, &u, 0, weights[e].data() + fmt.up_off + r * fmt.gu_row, 0, act, 0, 1);
                out[r] = (g / (1.f + std::exp(-g))) * u;
            }
        } else if constexpr (direct_iq3s) {
            need(cpu::iq3s_pair_gu_rows(fmt, weights[e].data(), act, out, begin, end),
                "direct IQ3_S pair refused the admitted layout");
        } else { const void* ap[]{act}; float* fp[]{out};
            cpu::native_gu_rows(fmt, weights[e].data(), ap, 1, fp, begin, end,
                cpu::NativeGuPhase::Decode); }
        if constexpr (direct_iq3s) if (entry == Entry::Production) {
            ++direct_calls; direct_rows += end - begin;
            direct_pair_rows += arm ? end - begin : 0;
        }
    }
    void validate() {
        uint64_t checked = 0;
        for (int e = 0; e < experts; ++e) for (int t = 0; t < 4; ++t)
            for (auto range : {std::pair{0, FF}, std::pair{128, 256}}) {
                std::array<float, FF + 2> ref, got; ref.fill(sentinel); got = ref;
                call(Entry::Oracle, e, acts[t].data(), ref.data() + 1, range.first, range.second);
                call(Entry::Production, e, acts[t].data(), got.data() + 1, range.first, range.second);
                need(!std::memcmp(ref.data(), got.data(), sizeof ref), "real-model GU bits");
                for (int r = range.first; r < range.second; ++r)
                    need(std::isfinite(ref[r + 1]), "nonfinite real-model GU");
                ++checked;
            }
        check();
        std::cout << "{\"kind\":\"validation\",\"cases\":" << checked <<
            ",\"bitwise\":true,\"finite\":true,\"inputs_unchanged\":true}" << std::endl;
    }
    void run(Entry entry, bool stream, bool pack, int rows) {
        const int count = stream ? experts : group, begin = rows == FF ? 0 : 128;
        for (int b = 0; b < count; b += group) {
            const int t = (b / group) % 4;
            const void* act = acts[t].data();
            if (pack) { cpu::native_quant_act(fmt, x[t].data(), scratch.data()); act = scratch.data(); }
            for (int e = b; e < b + group; ++e)
                call(entry, e, act, outputs[e].data() + 1, begin, begin + rows);
        }
    }
};
int main(int argc, char** argv) try {
    need(argc == 9, "shard gate up down blob_bytes total_experts arm process");
    const auto arm = number(argv[7]), process = number(argv[8]);
    need(arm <= 1 && process < 6, "arm/process");
    const char* single = std::getenv("STRATA_IQ_SINGLE_EXACT");
    need(single && std::string(single) == "0", "reference requires SINGLE_EXACT=0");
    const char* pair = std::getenv("STRATA_IQ3_PAIR");
    const char* audit = std::getenv("STRATA_IQ3_PAIR_AUDIT");
    need(pair && std::string(pair) == (direct_iq3s ? "0" : std::to_string(arm)) && audit &&
        std::string(audit) == (direct_iq3s ? "0" : "1"),
        "comparison selection/audit must be fixed before exec");
    need(!std::getenv("STRATA_IQ_MT_MIN") && !std::getenv("STRATA_IQ2_PAIR_AUDIT"),
        "no alternate NT rule or GU audit");
    const auto cores = cpu::physical_cores(false);
    need(cores.size() == 6 && cpu::pin_current_thread(cores[0]) > 0, "pin one physical core");
    cpu_set_t mask; CPU_ZERO(&mask);
    need(!sched_getaffinity(0, sizeof mask, &mask) && CPU_COUNT(&mask) == 1 &&
        CPU_ISSET(cores[0], &mask), "actual affinity");
    std::cout << std::setprecision(12);
    const auto setup = Clock::now(); Inputs in(argv, arm); const double setup_s = elapsed(setup);
    std::cout << "{\"kind\":\"input\",\"arm\":" << arm << ",\"process\":" << process <<
        ",\"CPU\":" << cores[0] << ",\"experts\":" << experts << ",\"group\":" << group <<
        ",\"blob_bytes\":" << in.fmt.bytes << ",\"gu_type\":" << gu_type << ",\"setup_s_excluded\":" << setup_s <<
        ",\"weight_input_fingerprint_fnv1a64\":\"" << hash(in.weight_hashes.data(),
            in.weight_hashes.size() * sizeof(in.weight_hashes[0])) <<
        "\",\"activation_input_fingerprint_fnv1a64\":\"" << hash(in.act_hashes.data(),
            sizeof(in.act_hashes)) << "\",\"activation_bytes\":" << in.fmt.act_bytes <<
        ",\"reference\":\"" << (direct_iq3s ? "GGML IQ3_S two-dot GU" : "native_gu_rows/decode IQ3PAIR0") <<
        "\",\"candidate\":\"" << (direct_iq3s ? "iq3s_pair_gu_rows/direct" : "native_gu_rows/decode IQ3PAIR1") <<
        "\",\"production_entry\":" << !direct_iq3s << "}" << std::endl;
    in.validate();
    for (int rows : {FF, 128}) for (bool pack : {false, true}) for (bool stream : {false, true}) {
        for (auto& out : in.outputs) out.fill(sentinel);
        in.run(Entry::Oracle, stream, pack, rows);
        const auto expected = in.outputs;
        auto pass = [&] { in.run(Entry::Production, stream, pack, rows); };
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
            const auto before = cpu::iq3_gu_audit_snapshot();
            const auto direct_calls = in.direct_calls, direct_rows = in.direct_rows,
                direct_pair_rows = in.direct_pair_rows;
            const double cpu0 = cpu_seconds(); const auto start = Clock::now();
            uint64_t passes = 0; double wall;
            do { pass(); ++passes; wall = elapsed(start); } while (wall < .35 && passes < 1000000);
            const double scheduled = cpu_seconds() - cpu0;
            const auto after = cpu::iq3_gu_audit_snapshot();
            const uint64_t calls = passes * jobs, active = calls * rows;
            if constexpr (direct_iq3s) {
                need(!before.enabled && !after.enabled &&
                    in.direct_calls - direct_calls == calls && in.direct_rows - direct_rows == active &&
                    in.direct_pair_rows - direct_pair_rows == (arm ? active : 0),
                    "actual timed direct helper calls");
            } else {
                need(before.enabled && after.enabled && !before.overflow && !after.overflow &&
                    before.threads == 1 && after.threads == 1 && after.calls - before.calls == calls &&
                    after.nt1_rows - before.nt1_rows == active &&
                    after.pair_rows - before.pair_rows == (arm ? active : 0), "actual timed production dispatch");
            }
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
                (direct_iq3s ? ",\"timed_direct_calls\":" : ",\"timed_dispatch_calls\":") <<
                    (direct_iq3s ? in.direct_calls - direct_calls : after.calls - before.calls) <<
                ",\"timed_nt1_rows\":" << (direct_iq3s ? in.direct_rows - direct_rows : after.nt1_rows - before.nt1_rows) <<
                ",\"timed_pair_rows\":" << (direct_iq3s ? in.direct_pair_rows - direct_pair_rows : after.pair_rows - before.pair_rows) <<
                ",\"outputs_match_untimed_reference\":true}" << std::endl;
        }
        in.check();
    }
    in.validate();
    std::cout << "{\"kind\":\"closed\",\"passed\":true}" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << (direct_iq3s ? "IQ3_S direct" : "IQ3_XXS entry") <<
        " pair comparison failed: " << e.what() << std::endl; return 1;
}
