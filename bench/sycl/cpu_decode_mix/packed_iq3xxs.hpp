// Main-authored packed98B candidate; repository LICENSE applies.
// Uses the pinned MIT GGML table and the existing native pair arithmetic contract.
// Benchmark-only: no production dispatch,format or ownership changes.
#pragma once
#include "compact_iq3xxs.hpp"
#include <bit>

namespace packed_iq3xxs {
constexpr size_t row_bytes=10*sizeof(block_iq3_xxs), projection_bytes=640*row_bytes;
// The native even-parity sign table,generated without a second copied table.
constexpr std::array<int64_t,128> signs=[] {
    std::array<int64_t,128> table{};
    for(unsigned code=0;code<128;++code) {
        const unsigned bits=code|((std::popcount(code)&1u)<<7);uint64_t word=0;
        for(unsigned j=0;j<8;++j)word|=uint64_t((bits&(1u<<j))?255:1)<<(8*j);
        table[code]=std::bit_cast<int64_t>(word);
    }
    return table;
}();

inline __m256i chunk(const uint8_t* block,unsigned p,__m256i activation) {
    // Load exactly8 indices;all gather indices are0..255 into the1024B table.
    const auto bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block+2+8*p));
    const auto indices=_mm256_cvtepu8_epi32(bytes);
    const auto grid=_mm256_i32gather_epi32(reinterpret_cast<const int*>(iq3xxs_grid),indices,4);
    uint32_t gas;std::memcpy(&gas,block+66+4*p,sizeof gas);
    const auto sign=_mm256_set_epi64x(signs[(gas>>21)&127],signs[(gas>>14)&127],
        signs[(gas>>7)&127],signs[gas&127]);
    const auto dot=_mm256_maddubs_epi16(grid,_mm256_sign_epi8(activation,sign));
    return _mm256_madd_epi16(dot,_mm256_set1_epi16(short(2*(gas>>28)+1)));
}

[[gnu::noinline]] inline void pair_dot(const uint8_t* gate,const uint8_t* up,
    const uint8_t* act,float& gout,float& uout) {
    __m256 ga=_mm256_setzero_ps(),ua=_mm256_setzero_ps();
    for(unsigned b=0;b<10;++b) {
        const auto* g=gate+98*b;const auto* u=up+98*b;const auto* a=act+sizeof(block_q8_K)*b;
        __m256i gi=_mm256_setzero_si256(),ui=_mm256_setzero_si256();
        #pragma clang loop unroll(full)
        for(unsigned p=0;p<8;++p) {
            const auto q=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+offsetof(block_q8_K,qs)+32*p));
            gi=_mm256_add_epi32(gi,chunk(g,p,q));ui=_mm256_add_epi32(ui,chunk(u,p,q));
        }
        uint16_t gd,ud;float ad;std::memcpy(&gd,g,2);std::memcpy(&ud,u,2);
        std::memcpy(&ad,a+offsetof(block_q8_K,d),4);
        const auto half=[](uint16_t bits) {return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(int(bits))));};
        // Native integer lanes,block-FMA order,PSIGN(-128) and reduction tree.
        ga=_mm256_fmadd_ps(_mm256_set1_ps(half(gd)*ad),_mm256_cvtepi32_ps(gi),ga);
        ua=_mm256_fmadd_ps(_mm256_set1_ps(half(ud)*ad),_mm256_cvtepi32_ps(ui),ua);
    }
    gout=.25f*compact_iq3xxs::horizontal(ga);uout=.25f*compact_iq3xxs::horizontal(ua);
    _mm256_zeroupper();
}

// NT1/H2560/FF640,original contiguous98B blocks only. No conversion/allocation.
// Unsupported extents,ranges and overlapping storage write nothing.
inline bool gu_rows(std::span<const uint8_t> raw,size_t up_offset,
    std::span<const uint8_t> act,std::span<float> out,int r0,int r1) {
    if(up_offset<projection_bytes || up_offset>raw.size() || projection_bytes>raw.size()-up_offset ||
        act.size()!=10*sizeof(block_q8_K) || out.size()<640 || r0<0 || r1<r0 || r1>640)return false;
    using compact_iq3xxs::overlap;
    if(overlap(raw.data(),raw.size_bytes(),act.data(),act.size_bytes()) ||
        overlap(raw.data(),raw.size_bytes(),out.data(),out.size_bytes()) ||
        overlap(act.data(),act.size_bytes(),out.data(),out.size_bytes()))return false;
    for(int r=r0;r<r1;++r) {
        float g,u;pair_dot(raw.data()+size_t(r)*row_bytes,raw.data()+up_offset+size_t(r)*row_bytes,
            act.data(),g,u);out[r]=(g/(1.f+std::exp(-g)))*u;
    }
    return true;
}
} // namespace packed_iq3xxs
