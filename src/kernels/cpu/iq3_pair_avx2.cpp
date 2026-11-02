// Main-authored paired IQ3_S producer; repository LICENSE.
// Arithmetic/format reference: existing iq_single_avx2.cpp and ggml's IQ3_S
// AVX2 dot product. Copyright(c)2023-2026 the ggml authors;MIT,third_party/ggml/LICENSE.
// Interleave gate/up with one activation load;retain integer lanes,FMA order
// and horizontal reduction. No prefetch,storage or precision change.
#include "strata/kernels/cpu/iq3_pair_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace strata::kernels::cpu {
namespace {
std::atomic<uint64_t> dispatch_calls{0}, dispatch_rows{0};
bool audit_enabled() noexcept {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_IQ3_PAIR_AUDIT");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}
}
Iq3PairAudit iq3_pair_audit_snapshot() noexcept {
    return {audit_enabled(), dispatch_calls.load(std::memory_order_relaxed),
            dispatch_rows.load(std::memory_order_relaxed)};
}
void iq3_pair_note_dispatch(uint64_t rows) noexcept {
    if (rows && audit_enabled()) {
        dispatch_calls.fetch_add(1, std::memory_order_relaxed);
        dispatch_rows.fetch_add(rows, std::memory_order_relaxed);
    }
}
} // namespace strata::kernels::cpu

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
namespace strata::kernels::cpu {
namespace {
static_assert(sizeof(block_iq3_s) == 110 && offsetof(block_iq3_s, qs) == 2 &&
              offsetof(block_iq3_s, qh) == 66 && offsetof(block_iq3_s, signs) == 74 &&
              offsetof(block_iq3_s, scales) == 106);
template<class T> T read(const uint8_t* p) {
    T v; std::memcpy(&v, p, sizeof(v)); return v;
}
inline __m256i half(const uint8_t* b, int p, __m256i activation) {
    const uint8_t* q = b + 2 + 8*p;
    const unsigned h = b[66+p];
#define G(k) static_cast<int>(iq3s_grid[q[k] | (((h >> k) & 1u) << 8)])
    const auto grid = _mm256_set_epi32(G(7),G(6),G(5),G(4),G(3),G(2),G(1),G(0));
#undef G
    const auto sign = _mm256_set1_epi32(std::bit_cast<int32_t>(read<uint32_t>(b+74+4*p)));
    const auto selectors = _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,
                                           2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
    const auto bits = _mm256_setr_epi8(1,2,4,8,16,32,64,-128,1,2,4,8,16,32,64,-128,
                                       1,2,4,8,16,32,64,-128,1,2,4,8,16,32,64,-128);
    const auto neg = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(sign,selectors),bits),bits);
    const auto signed_activation = _mm256_sub_epi8(_mm256_xor_si256(activation,neg),neg);
    const unsigned scales = b[106+p/2];
    const int scale = 2*int((scales >> (4*(p&1))) & 15)+1;
    return _mm256_madd_epi16(_mm256_maddubs_epi16(grid,signed_activation),_mm256_set1_epi16(short(scale)));
}
inline float reduce(__m256 a) {
    auto v = _mm_add_ps(_mm256_extractf128_ps(a,1),_mm256_castps256_ps128(a));
    v = _mm_add_ps(v,_mm_movehl_ps(v,v));
    return _mm_cvtss_f32(_mm_add_ss(v,_mm_movehdup_ps(v)));
}
void pair_dot(const uint8_t* gate, const uint8_t* up, const block_q8_K* act,
              float& g, float& u) {
    auto ga = _mm256_setzero_ps(), ua = _mm256_setzero_ps();
    for (int b=0;b<10;++b) {
        const auto* gb = gate+size_t(b)*sizeof(block_iq3_s);
        const auto* ub = up+size_t(b)*sizeof(block_iq3_s);
        __m256i gi[2] = {_mm256_setzero_si256(),_mm256_setzero_si256()};
        __m256i ui[2] = {_mm256_setzero_si256(),_mm256_setzero_si256()};
        for (int p=0;p<8;++p) {
            const auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(act[b].qs+32*p));
            gi[p&1] = _mm256_add_epi32(gi[p&1],half(gb,p,a));
            ui[p&1] = _mm256_add_epi32(ui[p&1],half(ub,p,a));
        }
        const float gd = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(read<uint16_t>(gb))))*act[b].d;
        const float ud = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(read<uint16_t>(ub))))*act[b].d;
        ga = _mm256_fmadd_ps(_mm256_set1_ps(gd),_mm256_cvtepi32_ps(_mm256_add_epi32(gi[0],gi[1])),ga);
        ua = _mm256_fmadd_ps(_mm256_set1_ps(ud),_mm256_cvtepi32_ps(_mm256_add_epi32(ui[0],ui[1])),ua);
    }
    g = reduce(ga); u = reduce(ua);
}
}
bool iq3_pair_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* act,
                     float* ff, int r0, int r1) {
    constexpr size_t rows=640, blocks=10, output_bytes=rows*sizeof(float);
    if (f.gu_type!=21 || f.gu_act!=15 || f.n_embd!=2560 || f.n_ff!=640 ||
        !blob || !act || !ff || r0<0 || r1<r0 || r1>640 ||
        f.gu_row<blocks*sizeof(block_iq3_s) || f.act_bytes!=blocks*sizeof(block_q8_K) ||
        f.gu_row%alignof(block_iq3_s) || f.up_off%alignof(block_iq3_s) ||
        reinterpret_cast<uintptr_t>(blob)%alignof(block_iq3_s) ||
        reinterpret_cast<uintptr_t>(act)%alignof(block_q8_K) ||
        reinterpret_cast<uintptr_t>(ff)%alignof(float) || f.gu_row>f.bytes/rows)
        return false;
    const size_t gate_bytes=rows*f.gu_row;
    if (f.up_off<gate_bytes || f.up_off>f.bytes || gate_bytes>f.bytes-f.up_off ||
        f.down_off<f.up_off+gate_bytes || f.down_off>f.bytes)
        return false;
    const auto w=reinterpret_cast<uintptr_t>(blob),a=reinterpret_cast<uintptr_t>(act),o=reinterpret_cast<uintptr_t>(ff);
    const auto max=std::numeric_limits<uintptr_t>::max();
    if (f.bytes>max-w || f.act_bytes>max-a || output_bytes>max-o) return false;
    if ((w<a+f.act_bytes && a<w+f.bytes) || (w<o+output_bytes && o<w+f.bytes) ||
        (a<o+output_bytes && o<a+f.act_bytes)) return false;
    const auto* y=static_cast<const block_q8_K*>(act);
    for (int r=r0;r<r1;++r) {
        const auto* g=blob+size_t(r)*f.gu_row;
        float gv,uv; pair_dot(g,g+f.up_off,y,gv,uv);
        ff[r]=(gv/(1.f+std::exp(-gv)))*uv;
    }
    return true;
}
} // namespace strata::kernels::cpu
#else
namespace strata::kernels::cpu {
bool iq3_pair_gu_rows(const NativeFmt&,const uint8_t*,const void*,float*,int,int) { return false; }
}
#endif
