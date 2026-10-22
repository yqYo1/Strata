// Private Main-authored gather discriminator from original6e40 iq2_pair_avx2.cpp.
// Grid loads, constrained p-loop unrolling and private symbols differ; no production selection.
// Main-authored integration of the exact direct-pair component ad4f29ac9.
// Repository MIT license. Component source/provenance and bit/performance gates
// are preserved in cpu-iq2-current.json; this producer is not yet qualified.
#include "strata/kernels/cpu/iq2_pair_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#if defined(__clang__) && (defined(__znver3__) || defined(__znver3))
#include <immintrin.h>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
namespace strata::kernels::cpu {
namespace {
// Direct decode adapted from existing iq_single_avx2.cpp.
// Copyright (c) 2023-2026 The ggml authors; MIT, third_party/ggml/LICENSE.
__m256i scales(const block_iq2_s& x) {
    uint64_t a;std::memcpy(&a,x.scales,8);
    const auto v=_mm_set_epi64x(a>>4,a);
    return _mm256_cvtepi8_epi16(_mm_add_epi8(_mm_slli_epi16(_mm_and_si128(v,_mm_set1_epi8(15)),1),_mm_set1_epi8(1)));
}
__m256i half(const block_iq2_s& x,int p,__m256i q8,__m256i scale) {
    const uint8_t* q=x.qs+p*4;const unsigned h=x.qh[p];
    uint32_t packed;std::memcpy(&packed,q,4);
    const auto low=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(packed));
    const auto high=_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(h),_mm_setr_epi32(0,2,4,6)),_mm_set1_epi32(3));
    const auto indices=_mm_or_si128(low,_mm_slli_epi32(high,8));
    const auto grid=_mm256_i32gather_epi64(reinterpret_cast<const long long*>(iq2s_grid),indices,8);
    alignas(32) static const uint8_t sm[32]={0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3};
    alignas(32) static const uint8_t bm[32]={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    uint32_t sign;std::memcpy(&sign,x.qs+32+p*4,4);
    const auto bits=_mm256_load_si256(reinterpret_cast<const __m256i*>(bm));
    const auto expanded=_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sign),_mm256_load_si256(reinterpret_cast<const __m256i*>(sm))),bits);
    const auto neg=_mm256_cmpeq_epi8(expanded,bits);
    const auto signed_q8=_mm256_sub_epi8(_mm256_xor_si256(neg,q8),neg);
    const auto values=_mm256_maddubs_epi16(grid,signed_q8);
    const auto sc=_mm256_shuffle_epi8(scale,_mm256_set1_epi16(0x0100+0x0202*p));
    return _mm256_madd_epi16(values,sc);
}
float reduce(__m256 a) {
    __m128 v=_mm_add_ps(_mm256_extractf128_ps(a,1),_mm256_castps256_ps128(a));
    v=_mm_add_ps(v,_mm_movehl_ps(v,v));v=_mm_add_ss(v,_mm_movehdup_ps(v));return _mm_cvtss_f32(v);
}
__attribute__((noinline)) void gather_pair_dot(const block_iq2_s* g,const block_iq2_s* u,
    const block_q8_K* y,int n,float& gout,float& uout) {
    __m256 ga=_mm256_setzero_ps(),ua=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m256i gi=_mm256_setzero_si256(),ui=_mm256_setzero_si256();
        const auto gs=scales(g[b]),us=scales(u[b]);
        const auto part=[&](int p){
            const auto act=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].qs+32*p));
            gi=_mm256_add_epi32(gi,half(g[b],p,act,gs));ui=_mm256_add_epi32(ui,half(u[b],p,act,us));
        };
        #pragma clang loop unroll(disable)
        for(int p=0;p<8;++p)part(p);
        const float gd=_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(g[b].d)))*y[b].d;
        const float ud=_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(u[b].d)))*y[b].d;
        ga=_mm256_fmadd_ps(_mm256_set1_ps(gd),_mm256_cvtepi32_ps(gi),ga);
        ua=_mm256_fmadd_ps(_mm256_set1_ps(ud),_mm256_cvtepi32_ps(ui),ua);
    }
    gout=.125f*reduce(ga);uout=.125f*reduce(ua);
}
} // namespace
bool iq2_gather_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* act,
                       float* ff, int r0, int r1) {
    constexpr size_t rows=640, blocks=10;
    if(f.gu_type!=22 || f.gu_act!=15 || f.n_embd!=2560 || f.n_ff!=640 ||
       !blob || !act || !ff || r0<0 || r1<r0 || r1>640 ||
       f.gu_row<blocks*sizeof(block_iq2_s) || f.act_bytes!=blocks*sizeof(block_q8_K) ||
       f.gu_row%alignof(block_iq2_s) || f.up_off%alignof(block_iq2_s) ||
       reinterpret_cast<uintptr_t>(blob)%alignof(block_iq2_s) ||
       reinterpret_cast<uintptr_t>(act)%alignof(block_q8_K) ||
       reinterpret_cast<uintptr_t>(ff)%alignof(float) || f.gu_row>f.bytes/rows)
        return false;
    const size_t gate_bytes=rows*f.gu_row;
    if(f.up_off<gate_bytes || f.up_off>f.bytes || gate_bytes>f.bytes-f.up_off ||
       f.down_off<f.up_off+gate_bytes || f.down_off>f.bytes)
        return false;
    const auto w=reinterpret_cast<uintptr_t>(blob),a0=reinterpret_cast<uintptr_t>(act),o=reinterpret_cast<uintptr_t>(ff);
    constexpr size_t output_bytes=rows*sizeof(float);
    const auto max=std::numeric_limits<uintptr_t>::max();
    if(f.bytes>max-w || f.act_bytes>max-a0 || output_bytes>max-o ||
       (w<a0+f.act_bytes&&a0<w+f.bytes) ||
       (w<o+output_bytes&&o<w+f.bytes) ||
       (a0<o+output_bytes&&o<a0+f.act_bytes))return false;
    const auto* a=static_cast<const block_q8_K*>(act);
    for(int r=r0;r<r1;++r) {
        const auto* g=reinterpret_cast<const block_iq2_s*>(blob+size_t(r)*f.gu_row);
        const auto* u=reinterpret_cast<const block_iq2_s*>(blob+f.up_off+size_t(r)*f.gu_row);
        float gv,uv;gather_pair_dot(g,u,a,2560,gv,uv);
        ff[r]=(gv/(1.f+std::exp(-gv)))*uv;
    }
    return true;
}
} // namespace strata::kernels::cpu
#else
namespace strata::kernels::cpu {
bool iq2_gather_gu_rows(const NativeFmt&,const uint8_t*,const void*,float*,int,int) {return false;}
} // namespace strata::kernels::cpu
#endif
