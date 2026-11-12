// Main-authored compact-view candidate; repository LICENSE applies.
// Format constants/tables come from the pinned MIT-licensed GGML dependency.
// Shared arithmetic for the owned CPU decode cache and its numerical fixtures.
#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <span>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

namespace compact_iq3xxs {
// Magnitude index plus sign, keeping the original half and eight 4-bit scales.
// Each 16-byte chunk stores the first16 weights in low nibbles,next16 in high.
struct Block { uint16_t d; uint8_t scales[4]; uint8_t codes[128]; };
static_assert(sizeof(Block)==134 && offsetof(Block,codes)==6);
static_assert(sizeof(block_iq3_xxs)==98 && sizeof(block_q8_K)==292);
constexpr size_t blocks_per_row=10, rows_per_expert=640;
constexpr size_t projection_blocks=blocks_per_row*rows_per_expert;
constexpr std::array<uint8_t,256> magnitude_code=[] {
    std::array<uint8_t,256> result{};result.fill(255);
    for(unsigned i=0;i<7;++i)result[4+8*i]=uint8_t(i);
    result[62]=7;return result;
}();

inline bool overlap(const void* p,size_t pn,const void* q,size_t qn) {
    const auto a=reinterpret_cast<uintptr_t>(p),b=reinterpret_cast<uintptr_t>(q);
    return a<=b ? b-a<pn : a-b<qn;
}

inline bool encode_block(std::span<const uint8_t> raw,Block& out) {
    if(raw.size()!=98 || overlap(raw.data(),raw.size_bytes(),&out,sizeof out))return false;
    out={};std::memcpy(&out.d,raw.data(),sizeof out.d);
    for(unsigned p=0;p<8;++p) {
        uint32_t gas;std::memcpy(&gas,raw.data()+66+4*p,sizeof gas);
        out.scales[p/2]|=uint8_t((gas>>28)<<(4*(p&1)));
        for(unsigned j=0;j<32;++j) {
            const uint32_t grid=iq3xxs_grid[raw[2+8*p+j/4]];
            const uint8_t magnitude=uint8_t(grid>>(8*(j&3)));
            const uint8_t code=magnitude_code[magnitude];
            if(code==255)return false;
            const unsigned sign=(ksigns_iq2xs[(gas>>(7*(j/8)))&127]>>(j&7))&1;
            out.codes[16*p+(j&15)]|=uint8_t((code|(sign<<3))<<(4*(j/16)));
        }
    }
    return true;
}

inline __m256i chunk(const Block& block,unsigned p,__m256i activation) {
    const auto table=_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    const auto bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(block.codes+16*p));
    const auto mask=_mm_set1_epi8(15);
    const auto lo=_mm_shuffle_epi8(table,_mm_and_si128(bits,mask));
    const auto hi=_mm_shuffle_epi8(table,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
    const auto weights=_mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1);
    const auto dot=_mm256_maddubs_epi16(_mm256_abs_epi8(weights),_mm256_sign_epi8(activation,weights));
    const unsigned scale=(block.scales[p/2]>>(4*(p&1)))&15;
    return _mm256_madd_epi16(dot,_mm256_set1_epi16(short(2*scale+1)));
}

inline float horizontal(__m256 value) {
    // Match the native paired kernel's float reduction tree exactly.
    auto sum=_mm_add_ps(_mm256_extractf128_ps(value,1),_mm256_castps256_ps128(value));
    sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));
    sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));return _mm_cvtss_f32(sum);
}

[[gnu::noinline]] inline void pair_dot(const Block* gate,const Block* up,const uint8_t* act,
                                      float& gout,float& uout) {
    __m256 ga=_mm256_setzero_ps(),ua=_mm256_setzero_ps();
    for(unsigned b=0;b<blocks_per_row;++b) {
        const auto* activation=act+b*sizeof(block_q8_K);
        __m256i gi=_mm256_setzero_si256(),ui=_mm256_setzero_si256();
        #pragma clang loop unroll(full)
        for(unsigned p=0;p<8;++p) {
            const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                activation+offsetof(block_q8_K,qs)+32*p));
            gi=_mm256_add_epi32(gi,chunk(gate[b],p,a));
            ui=_mm256_add_epi32(ui,chunk(up[b],p,a));
        }
        float ad;std::memcpy(&ad,activation+offsetof(block_q8_K,d),sizeof ad);
        const auto half=[](uint16_t bits) {
            return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(int(bits))));
        };
        // Preserve native block-FMA order and the final0.25 factor. Integer
        // lane bound8*4*62*128*31=7872512;pair maddubs bound2*62*128=15872.
        // PSIGN's -128 behavior is the same as the native paired kernel.
        ga=_mm256_fmadd_ps(_mm256_set1_ps(half(gate[b].d)*ad),_mm256_cvtepi32_ps(gi),ga);
        ua=_mm256_fmadd_ps(_mm256_set1_ps(half(up[b].d)*ad),_mm256_cvtepi32_ps(ui),ua);
    }
    gout=.25f*horizontal(ga);uout=.25f*horizontal(ua);_mm256_zeroupper();
}

// AVX2/FMA/F16C,NT1,H2560/FF640 only. The caller owns immutable,fully encoded
// gate/up views and the activation for the entire call. No allocation or cache
// management occurs here. Unsupported shapes/ranges/aliasing write nothing.
inline bool gu_rows(std::span<const Block> gate,std::span<const Block> up,
                    std::span<const uint8_t> act,std::span<float> out,int r0,int r1) {
    if(gate.size()!=projection_blocks || up.size()!=projection_blocks ||
       act.size()!=blocks_per_row*sizeof(block_q8_K) || out.size()<rows_per_expert ||
       r0<0 || r1<r0 || r1>int(rows_per_expert))return false;
    if(overlap(gate.data(),gate.size_bytes(),up.data(),up.size_bytes()) ||
       overlap(act.data(),act.size_bytes(),gate.data(),gate.size_bytes()) ||
       overlap(act.data(),act.size_bytes(),up.data(),up.size_bytes()) ||
       overlap(out.data(),out.size_bytes(),act.data(),act.size_bytes()) ||
       overlap(out.data(),out.size_bytes(),gate.data(),gate.size_bytes()) ||
       overlap(out.data(),out.size_bytes(),up.data(),up.size_bytes()))return false;
    for(int r=r0;r<r1;++r) {
        float g,u;pair_dot(gate.data()+size_t(r)*blocks_per_row,
            up.data()+size_t(r)*blocks_per_row,act.data(),g,u);
        out[r]=(g/(1.f+std::exp(-g)))*u;
    }
    return true;
}
} // namespace compact_iq3xxs
