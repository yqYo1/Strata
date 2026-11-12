// Main-authored bounded GU layout experiment; repository LICENSE applies.
// Uses the owned134-byte codec and pinned MIT GGML tables as the reference.
// Candidate138-byte blocks store the eight odd integer scales in advance.
// This is a single-core kernel experiment, not a production-pool/model route.
#include "strata/kernels/cpu/detail/compact_iq3xxs.hpp"
#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace old = compact_iq3xxs;
using Clock = std::chrono::steady_clock;
constexpr size_t PB = old::projection_blocks, FF = old::rows_per_expert, E = 32;
constexpr float sentinel = 123.f;
[[gnu::cold]] void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
uint64_t number(const char* text) {
    std::string s(text);
    need(!s.empty() && s.size() <= 20 && std::all_of(s.begin(), s.end(),
        [](char c) { return c >= '0' && c <= '9'; }), "unsigned decimal");
    return std::stoull(s);
}
struct Expanded { uint16_t d; uint8_t scales[8]; uint8_t codes[128]; };
static_assert(sizeof(Expanded) == 138 && offsetof(Expanded, codes) == 10);
Expanded expand(const old::Block& b) {
    Expanded out{}; out.d = b.d; std::memcpy(out.codes, b.codes, sizeof out.codes);
    for (unsigned p = 0; p < 8; ++p) out.scales[p] = uint8_t(2 * ((b.scales[p / 2] >> (4 * (p & 1))) & 15) + 1);
    return out;
}
inline __m256i chunk(const Expanded& block, unsigned p, __m256i activation) {
    const auto table = _mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    const auto bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.codes + 16 * p));
    const auto mask = _mm_set1_epi8(15);
    const auto lo = _mm_shuffle_epi8(table, _mm_and_si128(bits, mask));
    const auto hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(bits, 4), mask));
    const auto weights = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const auto dot = _mm256_maddubs_epi16(_mm256_abs_epi8(weights), _mm256_sign_epi8(activation, weights));
    return _mm256_madd_epi16(dot, _mm256_set1_epi16(short(block.scales[p])));
}
[[gnu::noinline]] void expanded_pair(const Expanded* gate, const Expanded* up,
                                     const uint8_t* act, float& gout, float& uout) {
    __m256 ga = _mm256_setzero_ps(), ua = _mm256_setzero_ps();
    for (unsigned b = 0; b < old::blocks_per_row; ++b) {
        const auto* activation = act + b * sizeof(block_q8_K);
        __m256i gi = _mm256_setzero_si256(), ui = _mm256_setzero_si256();
        #pragma clang loop unroll(full)
        for (unsigned p = 0; p < 8; ++p) {
            const auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                activation + offsetof(block_q8_K, qs) + 32 * p));
            gi = _mm256_add_epi32(gi, chunk(gate[b], p, a));
            ui = _mm256_add_epi32(ui, chunk(up[b], p, a));
        }
        float ad; std::memcpy(&ad, activation + offsetof(block_q8_K, d), sizeof ad);
        const auto half = [](uint16_t bits) {
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(int(bits))));
        };
        // Integer terms and bounds are unchanged. Preserve the native FP FMA
        // sequence, horizontal reduction,0.25factor and scalar std::exp.
        ga = _mm256_fmadd_ps(_mm256_set1_ps(half(gate[b].d) * ad), _mm256_cvtepi32_ps(gi), ga);
        ua = _mm256_fmadd_ps(_mm256_set1_ps(half(up[b].d) * ad), _mm256_cvtepi32_ps(ui), ua);
    }
    gout = .25f * old::horizontal(ga); uout = .25f * old::horizontal(ua); _mm256_zeroupper();
}
using Output = std::array<float, FF + 2>;
void expanded_rows(std::span<const Expanded> gate, std::span<const Expanded> up,
                   std::span<const uint8_t> act, std::span<float> out, int first, int last) {
    need(gate.size() == PB && up.size() == PB && act.size() == 10 * sizeof(block_q8_K) &&
         out.size() == FF && first >= 0 && first <= last && last <= int(FF), "candidate extents");
    need(!old::overlap(gate.data(), gate.size_bytes(), up.data(), up.size_bytes()) &&
         !old::overlap(act.data(), act.size_bytes(), gate.data(), gate.size_bytes()) &&
         !old::overlap(act.data(), act.size_bytes(), up.data(), up.size_bytes()) &&
         !old::overlap(out.data(), out.size_bytes(), gate.data(), gate.size_bytes()) &&
         !old::overlap(out.data(), out.size_bytes(), up.data(), up.size_bytes()) &&
         !old::overlap(out.data(), out.size_bytes(), act.data(), act.size_bytes()), "candidate aliasing");
    for (int r = first; r < last; ++r) {
        float g, u; expanded_pair(gate.data() + size_t(r) * 10, up.data() + size_t(r) * 10, act.data(), g, u);
        out[size_t(r)] = (g / (1.f + std::exp(-g))) * u;
    }
}
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
double cpu_seconds() {
    timespec t{}; need(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t), "process CPU clock");
    return double(t.tv_sec) + t.tv_nsec * 1e-9;
}
long peak_kib() { rusage r{}; need(!getrusage(RUSAGE_SELF, &r), "RSS"); return r.ru_maxrss; }
void core_check() {
    cpu_set_t mask; CPU_ZERO(&mask);
    need(!sched_getaffinity(0, sizeof mask, &mask) && CPU_COUNT(&mask) == 1 && CPU_ISSET(0, &mask), "actual CPU0 affinity");
}
struct File {
    int fd;
    explicit File(const char* name) : fd(open(name, O_RDONLY | O_CLOEXEC)) { need(fd >= 0, "model open"); }
    ~File() { close(fd); }
    void read(std::span<uint8_t> out, uint64_t offset) {
        struct stat s{}; need(!fstat(fd, &s) && s.st_size >= 0 && offset <= uint64_t(s.st_size) &&
            out.size_bytes() <= uint64_t(s.st_size) - offset, "model extent");
        while (!out.empty()) { const auto n = pread(fd, out.data(), out.size(), offset);
            need(n > 0, "model read"); out = out.subspan(size_t(n)); offset += uint64_t(n); }
    }
};
struct Fixture {
    std::vector<old::Block> reference;
    std::vector<Expanded> candidate;
    std::array<std::array<block_q8_K, 10>, 4> acts{};
    std::array<Output, E> output{};
    double setup_s = 0;
    explicit Fixture(char** argv, bool ref, bool exp) {
        const auto start = Clock::now();
        if (ref) reference.resize(E * 2 * PB + 2);
        if (exp) candidate.resize(E * 2 * PB + 2);
        if (ref) { std::memset(&reference.front(), 0xa7, sizeof(old::Block));
                   std::memset(&reference.back(), 0xa7, sizeof(old::Block)); }
        if (exp) { std::memset(&candidate.front(), 0xa7, sizeof(Expanded));
                   std::memset(&candidate.back(), 0xa7, sizeof(Expanded)); }
        File file(argv[1]); const uint64_t gate = number(argv[2]), up = number(argv[3]);
        need(gate <= std::numeric_limits<uint64_t>::max() - E * PB * 98 &&
             up <= std::numeric_limits<uint64_t>::max() - E * PB * 98, "source offsets");
        std::vector<uint8_t> raw(PB * 98);
        for (size_t e = 0; e < E; ++e) for (size_t p = 0; p < 2; ++p) {
            file.read(raw, (p ? up : gate) + e * raw.size());
            for (size_t b = 0; b < PB; ++b) {
                old::Block block; need(old::encode_block({raw.data() + b * 98, 98}, block), "raw codec");
                const size_t i = 1 + (2 * e + p) * PB + b;
                if (ref) reference[i] = block;
                if (exp) candidate[i] = expand(block);
            }
        }
        for (size_t t = 0; t < acts.size(); ++t) for (size_t b = 0; b < 10; ++b) {
            auto& a = acts[t][b]; a.d = t == 0 ? 0.f : std::ldexp(1.f + float(b) / 16, -int(t + 4));
            for (size_t i = 0; i < 256; ++i) a.qs[i] = int8_t(int((i * 13 + b * 31 + t * 7) & 255) - 128);
        }
        for (auto& o : output) o.fill(sentinel);
        setup_s = elapsed(start);
    }
    void run(bool expanded, size_t expert, size_t token, Output& out, int first = 0, int last = int(FF)) {
        const size_t offset = 1 + 2 * expert * PB;
        const auto act = std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(acts[token].data()), sizeof acts[token]);
        if (expanded) expanded_rows({candidate.data() + offset, PB}, {candidate.data() + offset + PB, PB}, act,
                                   {out.data() + 1, FF}, first, last);
        else need(old::gu_rows({reference.data() + offset, PB}, {reference.data() + offset + PB, PB}, act,
                               {out.data() + 1, FF}, first, last), "owned reference refused");
    }
    void guards() const {
        for (const auto* b : {reference.empty() ? nullptr : &reference.front(), reference.empty() ? nullptr : &reference.back()})
            if (b) for (size_t i = 0; i < sizeof *b; ++i) need(reinterpret_cast<const uint8_t*>(b)[i] == 0xa7, "reference guard");
        for (const auto* b : {candidate.empty() ? nullptr : &candidate.front(), candidate.empty() ? nullptr : &candidate.back()})
            if (b) for (size_t i = 0; i < sizeof *b; ++i) need(reinterpret_cast<const uint8_t*>(b)[i] == 0xa7, "candidate guard");
        for (const auto& o : output) { need(o.front() == sentinel && o.back() == sentinel, "output guard");
            for (size_t i = 1; i <= FF; ++i) need(std::isfinite(o[i]), "finite output"); }
    }
};
int main(int argc, char** argv) {
    try {
        need(argc == 5, "scale_layout SHARD GATE UP check|reference|expanded");
        const std::string arm(argv[4]); need(arm == "check" || arm == "reference" || arm == "expanded", "arm");
        need(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c"), "AVX2/FMA/F16C");
        cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(0, &mask); need(!sched_setaffinity(0, sizeof mask, &mask), "CPU0 pin");
        Fixture f(argv, arm != "expanded", arm != "reference"); std::cout << std::setprecision(17);
        if (arm == "check") {
            unsigned cases = 0;
            for (size_t e = 0; e < E; ++e) for (size_t t = 0; t < 4; ++t)
                for (const auto range : {std::array<int,2>{0,640}, {1,639}, {0,0}}) {
                    Output r, c; r.fill(sentinel); c.fill(sentinel);
                    f.run(false, e, t, r, range[0], range[1]); f.run(true, e, t, c, range[0], range[1]);
                    need(!std::memcmp(r.data(), c.data(), sizeof r), "FP bits/range equality");
                    for (size_t i = 0; i < FF; ++i) need(i >= size_t(range[0]) && i < size_t(range[1]) ?
                        std::isfinite(c[i+1]) : c[i+1] == sentinel, "partial row bounds"); ++cases;
                }
            // A zeroed candidate gate must differ from real reference output.
            // This proves the candidate scale bytes, rather than a hidden
            // reference/fallback, actually contribute to the checked result.
            for (size_t b = 1; b <= PB; ++b) std::memset(f.candidate[b].scales, 0, 8);
            Output r, c; r.fill(sentinel); c.fill(sentinel);
            f.run(false, 0, 1, r); f.run(true, 0, 1, c);
            need(std::memcmp(r.data(), c.data(), sizeof r), "wrong-scale negative control");
            f.guards(); core_check();
            std::cout << "{\"passed\":true,\"cases\":" << cases
                      << ",\"bit_exact\":true,\"wrong_scale_control_detected\":true,\"single_core_kernel_only\":true}\n";
            return 0;
        }
        const auto start = Clock::now();
        for (unsigned i = 0; i < 10000; ++i) { const auto v = Clock::now().time_since_epoch().count();
            asm volatile("" : : "g"(v) : "memory"); }
        const double timer_pair = 2 * elapsed(start) / 10000;
        std::cout << "{\"passed\":true,\"arm\":\"" << arm << "\",\"cpu\":0,\"block_bytes\":"
                  << (arm == "expanded" ? sizeof(Expanded) : sizeof(old::Block)) << ",\"setup_s\":" << f.setup_s
                  << ",\"peak_kib\":" << peak_kib() << ",\"timer_pair_s\":" << timer_pair << ",\"cells\":[";
        bool comma = false;
        for (size_t pool : {size_t(1), size_t(6), size_t(32)}) {
            auto batch = [&](uint64_t rounds) {
                core_check();
                const double c0 = cpu_seconds(); const auto begin = Clock::now();
                for (uint64_t i = 0; i < rounds; ++i) {
                    const size_t e = size_t((13 * i + 3) % pool);
                    f.run(arm == "expanded", e, size_t(i % 4), f.output[e]);
                }
                const std::array<double,2> value{elapsed(begin), cpu_seconds() - c0};
                f.guards(); core_check(); return value;
            };
            uint64_t rounds = 1; std::array<double,2> calibration{};
            for (;;) { calibration = batch(rounds); if (calibration[0] >= 2) break;
                need(rounds < 65536, "calibration floor"); rounds *= 2; }
            if (comma) std::cout << ','; comma = true;
            std::cout << "{\"working_set_experts\":" << pool << ",\"rounds\":" << rounds
                      << ",\"calibration_s\":" << calibration[0] << ",\"samples\":[";
            for (int j = 0; j < 3; ++j) { const auto s = batch(rounds);
                need(s[0] >= 1 && timer_pair / s[0] < .001, "timing eligibility");
                if (j) std::cout << ','; std::cout << '[' << s[0] << ',' << s[1] << ']'; }
            std::cout << "]}";
        }
        std::cout << "]}\n"; return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
