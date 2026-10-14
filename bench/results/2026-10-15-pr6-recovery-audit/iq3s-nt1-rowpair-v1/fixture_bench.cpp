// Proposal only. No GPU, model loading, activation-format changes, or timing assertions.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

namespace strata::kernels::cpu {
bool iq256_iq3s_rowpair_gu_rows(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int);
void iq3s_probe_raw(const uint8_t*, size_t, const void*, float*, int);
}
namespace {
using namespace strata::kernels::cpu;
constexpr int N = 2560, FF = 640, NB = N/256;
constexpr size_t ROW = NB*sizeof(block_iq3_s), UP = FF*ROW, BYTES = 2*UP;
static_assert(ROW == 1100 && BYTES == 1408000 && sizeof(block_q8_K)*NB == 2920);
using Output = std::array<float, FF+2>;
ggml_vec_dot_t dot;
uint32_t next(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
uint64_t fnv64(const void* p, size_t n, uint64_t h = 14695981039346656037ull) {
    const auto* b = static_cast<const uint8_t*>(p);
    while (n--) { h ^= *b++; h *= 1099511628211ull; }
    return h;
}
uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
uint32_t ordered(float f) { const auto u = bits(f); return u & 0x80000000u ? ~u : u | 0x80000000u; }
struct Delta {
    uint64_t mismatch = 0, nonfinite = 0, max_ulp = 0;
    double max_abs = 0, max_rel = 0;
    void add(float ref, float got) {
        if (bits(ref) == bits(got)) return;
        ++mismatch;
        if (!std::isfinite(ref) || !std::isfinite(got)) { ++nonfinite; return; }
        const double a = std::abs(double(ref)-double(got));
        max_abs = std::max(max_abs, a);
        max_rel = std::max(max_rel, a/std::max(std::abs(double(ref)), 1e-30));
        const uint32_t x = ordered(ref), y = ordered(got);
        max_ulp = std::max(max_ulp, uint64_t(x > y ? x-y : y-x));
    }
};
struct Bank {
    std::vector<uint8_t> storage;
    uint8_t* w;
    std::array<block_q8_K, NB> a{};
    explicit Bank(uint32_t seed, int pattern = 0) : storage(BYTES+64) {
        w = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(storage.data())+31)&~uintptr_t(31));
        uint32_t state = seed;
        for (size_t j = 0; j < BYTES; ++j) w[j] = uint8_t(next(state)>>24);
        for (int h = 0; h < 2; ++h) for (int r = 0; r < FF; ++r) for (int b = 0; b < NB; ++b) {
            auto* p = w + size_t(h)*UP + size_t(r)*ROW + size_t(b)*sizeof(block_iq3_s);
            if (pattern == 3 || pattern == 4) std::memset(p+2, pattern == 3 ? 0 : 255, 108);
            const uint16_t d = pattern == 5 ? 0x7bff : pattern == 6 ? uint16_t(1 + (b%31)) :
                               pattern == 7 ? 0 : uint16_t(0x1400+(next(state)&1023));
            std::memcpy(p, &d, 2);
        }
        std::array<float, N> x{};
        for (int j = 0; j < N; ++j)
            x[j] = pattern == 1 ? 0.f : pattern == 2 ? (j%2 ? -1.f : 1.f) :
                   float(int(next(state)%65535)-32767)/32768.f;
        // Exact production CPU trait quantizer; initialized bsums also make the
        // all-zero quantizer's early-return bytes deterministic.
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), a.data(), N);
    }
    Bank(const Bank&) = delete;
    Bank& operator=(const Bank&) = delete;
    Bank(Bank&&) = default; // vector move retains the allocation w points to
};
void baseline(const uint8_t* w, const void* a, float* f, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        float g, u;
        dot(N, &g, 0, w+size_t(r)*ROW, 0, a, 0, 1);
        dot(N, &u, 0, w+UP+size_t(r)*ROW, 0, a, 0, 1);
        f[r] = (g/(1.f+std::exp(-g)))*u;
    }
}
void run(int arm, const uint8_t* w, const void* a, float* f, int r0, int r1) {
    if (arm == 0) baseline(w, a, f, r0, r1);
    else if (arm == 1) {
        if (!iq256_single_gu_rows(21, w, ROW, UP, N, a, f, r0, r1)) throw std::runtime_error("exact unavailable");
    } else if (!iq256_iq3s_rowpair_gu_rows(21, w, ROW, UP, N, a, f, r0, r1))
        throw std::runtime_error("candidate unavailable/rejected");
}
bool canaries(const Output& f, int r0, int r1) {
    for (int i = 0; i < FF+2; ++i)
        if ((i == 0 || i == FF+1 || i-1 < r0 || i-1 >= r1) && bits(f[i]) != bits(123.f)) return false;
    return true;
}
void print_delta(const char* name, const Delta& d) {
    std::printf("\"%s\":{\"mismatch\":%llu,\"nonfinite_diff\":%llu,\"max_abs\":%.17g,"
                "\"max_rel\":%.17g,\"max_ulp\":%llu}", name,
                (unsigned long long)d.mismatch, (unsigned long long)d.nonfinite,
                d.max_abs, d.max_rel, (unsigned long long)d.max_ulp);
}
bool compare_case(const Bank& b, const uint8_t* cw, const void* ca, int r0, int r1,
                  int pattern, int weight_offset, int act_offset) {
    Output ref, exact, cand;
    ref.fill(123.f); exact.fill(123.f); cand.fill(123.f);
    run(0, b.w, b.a.data(), ref.data()+1, r0, r1);
    run(1, b.w, b.a.data(), exact.data()+1, r0, r1);
    run(2, cw, ca, cand.data()+1, r0, r1);
    Delta de, dc, raw;
    for (int r = r0; r < r1; ++r) {
        de.add(ref[r+1], exact[r+1]); dc.add(ref[r+1], cand[r+1]);
    }
    // Inspect raw dot results too: SwiGLU can hide gate differences at saturation.
    for (int r = r0; r < r1; r += 2) {
        const int count = std::min(2, r1-r);
        for (size_t h : {size_t(0), UP}) {
            float got[2];
            iq3s_probe_raw(cw+h+size_t(r)*ROW, ROW, ca, got, count);
            for (int k = 0; k < count; ++k) {
                float want;
                dot(N, &want, 0, b.w+h+size_t(r+k)*ROW, 0, b.a.data(), 0, 1);
                raw.add(want, got[k]);
            }
        }
    }
    const bool guards = canaries(ref,r0,r1) && canaries(exact,r0,r1) && canaries(cand,r0,r1);
    const bool pass = !de.mismatch && !dc.mismatch && !raw.mismatch && guards;
    std::printf("{\"kind\":\"correctness_case\",\"pattern\":%d,\"weight_offset_mod32\":%d,"
                "\"act_offset_mod32\":%d,\"requested_weight_offset\":%d,\"requested_act_offset\":%d,"
                "\"r0\":%d,\"r1\":%d,\"guards\":%s,",
                pattern, int(uintptr_t(cw)%32), int(uintptr_t(ca)%32), weight_offset, act_offset,
                r0, r1, guards ? "true" : "false");
    print_delta("existing_exact",de); std::printf(","); print_delta("candidate",dc);
    std::printf(","); print_delta("raw_candidate",raw);
    std::printf(",\"bitwise_pass\":%s}\n",pass ? "true" : "false");
    return pass;
}
struct Guarded {
    void* mapping;
    size_t extent;
    uint8_t* data;
    explicit Guarded(size_t n) {
        const auto page = size_t(sysconf(_SC_PAGESIZE));
        extent = ((n+page-1)/page+2)*page;
        mapping = mmap(nullptr, extent, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("mmap failed");
        auto* p = static_cast<uint8_t*>(mapping);
        if (mprotect(p,page,PROT_NONE) || mprotect(p+extent-page,page,PROT_NONE))
            throw std::runtime_error("mprotect failed");
        data = p+extent-page-n; // exact input end touches a protected page
    }
    ~Guarded() { munmap(mapping,extent); }
};
int correctness() {
    bool pass = true;
    uint64_t cases = 0;
    const std::array<std::pair<int,int>,9> ranges{{{0,640},{1,640},{0,639},{1,638},
        {2,5},{638,640},{639,640},{0,0},{640,640}}};
    for (int pattern = 0; pattern < 8; ++pattern) {
        Bank b(0x71c3e6a5u+uint32_t(pattern),pattern);
        std::printf("{\"kind\":\"input\",\"pattern\":%d,\"seed\":%u,\"fnv64_bytes\":\"%016llx\","
                    "\"weight_bytes\":%zu,\"activation_bytes\":%zu}\n", pattern,
                    0x71c3e6a5u+uint32_t(pattern),
                    (unsigned long long)fnv64(b.a.data(),sizeof(b.a),fnv64(b.w,BYTES)),BYTES,sizeof(b.a));
        for (auto range : ranges) { pass &= compare_case(b,b.w,b.a.data(),range.first,range.second,pattern,0,0); ++cases; }
        // Candidate permits byte alignment; controls use identical aligned copies.
        // Natural GGML metadata alignment is never violated in the control arms.
        std::vector<uint8_t> scratch(BYTES+96), act(sizeof(b.a)+96);
        auto* w0 = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(scratch.data())+31)&~uintptr_t(31));
        auto* a0 = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(act.data())+31)&~uintptr_t(31));
        for (int offset = 0; offset < 32; ++offset) {
            const int ao = (offset*7)%32;
            std::memcpy(w0+offset,b.w,BYTES); std::memcpy(a0+ao,b.a.data(),sizeof(b.a));
            pass &= compare_case(b,w0+offset,a0+ao,1,640,pattern,offset,ao); ++cases;
        }
        Guarded gw(BYTES), ga(sizeof(b.a));
        std::memcpy(gw.data,b.w,BYTES); std::memcpy(ga.data,b.a.data(),sizeof(b.a));
        pass &= compare_case(b,gw.data,ga.data,638,640,pattern,0,int(uintptr_t(ga.data)%32)); ++cases;
        pass &= compare_case(b,gw.data,ga.data,639,640,pattern,0,int(uintptr_t(ga.data)%32)); ++cases;
    }
    float guard = 123.f;
    auto reject = [&](int ty,size_t row,size_t up,int n,int r0,int r1) {
        const bool rejected = !iq256_iq3s_rowpair_gu_rows(ty,nullptr,row,up,n,nullptr,&guard,r0,r1);
        pass &= rejected && bits(guard)==bits(123.f); ++cases;
    };
    reject(18,ROW,UP,N,0,1); reject(21,ROW-1,UP,N,0,1); reject(21,ROW,UP-1,N,0,1);
    reject(21,ROW,UP,2559,0,1); reject(21,ROW,UP,N,-1,1); reject(21,ROW,UP,N,2,1);
    reject(21,ROW,UP,N,0,641); reject(21,ROW,UP,N,0,1);
    pass &= iq256_iq3s_rowpair_gu_rows(21,nullptr,ROW,UP,N,nullptr,nullptr,640,640);
    ++cases;
    std::printf("{\"kind\":\"correctness_close\",\"cases\":%llu,\"pass\":%s,\"criterion\":\"bitwise\"}\n",
                (unsigned long long)cases,pass ? "true" : "false");
    return pass ? 0 : 1;
}
const char* names[] = {"ggml","existing_exact","rowpair"};
volatile uint32_t sink;
int number(const char* s, int lo, int hi) {
    char* end = nullptr; const long n = std::strtol(s,&end,10);
    if (!*s || *end || n < lo || n > hi) throw std::runtime_error("numeric argument outside bounds");
    return int(n);
}
int timing(bool streaming, int banks, int repeats, int inner) {
    std::vector<Bank> data;
    data.reserve(size_t(banks));
    uint64_t identity = 14695981039346656037ull;
    for (int i = 0; i < banks; ++i) {
        data.emplace_back(0xa1020001u+uint32_t(i));
        identity = fnv64(data.back().w,BYTES,identity);
        identity = fnv64(data.back().a.data(),sizeof(data.back().a),identity);
    }
    Output output{};
    std::printf("{\"kind\":\"timing_open\",\"N\":2560,\"FF\":640,\"nt\":1,"
                "\"regime\":\"%s\",\"banks\":%d,\"repeats\":%d,\"inner\":%d,"
                "\"weight_working_set_bytes\":%llu,\"input_fnv64\":\"%016llx\","
                "\"cycles_kind\":\"invariant_TSC_ticks_not_core_cycles\",\"correctness_in_this_process\":false}\n",
                streaming ? "beyond_LLC_sequential" : "one_expert_cache_hot",banks,repeats,inner,
                (unsigned long long)(banks*BYTES),(unsigned long long)identity);
    auto sweep = [&](int arm,int times) {
        for (int i = 0; i < times; ++i) for (const auto& b : data)
            run(arm,b.w,b.a.data(),output.data()+1,0,FF);
    };
    // All three comparisons have repeated ABBA order. Rotate comparison order
    // and reverse the pair orientation in alternate rounds. Every arm receives
    // the same prewarm and measured logical work, independent of speed.
    const int pairs[3][2] = {{0,2},{1,2},{0,1}};
    uint64_t sequence = 0;
    for (int round = 0; round < repeats; ++round) for (int p = 0; p < 3; ++p) {
        const int pair = (p+round)%3;
        const int left = pairs[pair][round%2], right = pairs[pair][1-round%2];
        const int abba[] = {left,right,right,left};
        for (int slot = 0; slot < 4; ++slot) {
            const int arm = abba[slot];
            sweep(arm,1);
            unsigned aux0,aux1;
            const int cpu0 = sched_getcpu();
            const auto start = std::chrono::steady_clock::now();
            _mm_lfence(); const uint64_t ticks0 = __rdtscp(&aux0); _mm_lfence();
            sweep(arm,inner);
            _mm_lfence(); const uint64_t ticks1 = __rdtscp(&aux1); _mm_lfence();
            const auto end = std::chrono::steady_clock::now();
            const int cpu1 = sched_getcpu();
            sink = bits(output[FF]);
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
            const uint64_t jobs = uint64_t(banks)*uint64_t(inner), rows = jobs*FF;
            std::printf("{\"kind\":\"timing_sample\",\"sequence\":%llu,\"round\":%d,\"pair\":%d,"
                        "\"slot\":%d,\"arm\":\"%s\",\"ns\":%lld,\"tsc_ticks\":%llu,"
                        "\"cpu_start\":%d,\"cpu_end\":%d,\"tsc_aux_start\":%u,\"tsc_aux_end\":%u,"
                        "\"migration_observed\":%s,\"jobs\":%llu,\"output_rows\":%llu,\"dots\":%llu,"
                        "\"superblocks\":%llu,\"grid_lookups\":%llu,\"logical_weight_bytes\":%llu,"
                        "\"ns_per_output_row\":%.9g,\"sink_bits\":%u}\n",
                        (unsigned long long)sequence++,round,pair,slot,names[arm],(long long)ns,
                        (unsigned long long)(ticks1-ticks0),cpu0,cpu1,aux0,aux1,
                        (cpu0!=cpu1 || aux0!=aux1) ? "true" : "false",(unsigned long long)jobs,
                        (unsigned long long)rows,(unsigned long long)(2*rows),
                        (unsigned long long)(2*NB*rows),(unsigned long long)(2*NB*64*rows),
                        (unsigned long long)(2*ROW*rows),double(ns)/double(rows),bits(output[FF]));
            std::fflush(stdout);
        }
    }
    std::printf("{\"kind\":\"timing_close\",\"samples\":%llu,\"complete\":true}\n",(unsigned long long)sequence);
    return 0;
}
}
int main(int argc,char** argv) {
    try {
        ggml_cpu_init();
        const auto* t = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S);
        if (!t || !t->vec_dot || t->vec_dot_type != GGML_TYPE_Q8_K ||
            !ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float)
            throw std::runtime_error("GGML IQ3_S/Q8_K trait contract unavailable");
        dot = t->vec_dot;
        if (!iq256_iq3s_rowpair_gu_rows(21,nullptr,ROW,UP,N,nullptr,nullptr,0,0) ||
            !iq256_single_gu_rows(21,nullptr,ROW,UP,N,nullptr,nullptr,0,0)) {
            std::fprintf(stderr,"Clang/znver3 candidate or exact helper unavailable\n"); return 77;
        }
        if (argc == 2 && std::string(argv[1]) == "--correctness") return correctness();
        if (argc == 7 && std::string(argv[1]) == "--timing") {
            const bool stream = std::string(argv[2]) == "stream";
            if (!stream && std::string(argv[2]) != "hot") throw std::runtime_error("regime must be hot/stream");
            const int banks = number(argv[3],stream ? 32 : 1,stream ? 128 : 1);
            const int repeats = number(argv[4],4,100), inner = number(argv[5],1,100);
            if (std::string(argv[6]) != "--correctness-approved-by-main")
                throw std::runtime_error("separate correctness receipt required by Main");
            return timing(stream,banks,repeats,inner);
        }
        std::fprintf(stderr,"usage: fixture --correctness | --timing hot|stream banks repeats inner --correctness-approved-by-main\n");
        return 2;
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 2; }
}
