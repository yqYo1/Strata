// Main-authored benchmark prototype; repository LICENSE applies.
// IQ constants/tables: pinned MIT-licensed GGML. No external implementation copy.
// AVX2/FMA/F16C, H2560/FF640/NT1, at most ten same-activation experts.
#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <span>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

namespace activation_lut {
constexpr size_t blocks=10, rows=640, row_bytes=980, projection_bytes=627200;
static_assert(sizeof(block_iq3_xxs)==98 && sizeof(block_q8_K)==292);
struct alignas(64) Slab { std::array<std::array<int16_t,4096>,64> values; };
struct alignas(32) Accumulator { float gate[8],up[8]; };
static_assert(sizeof(Slab)==524288 && sizeof(Accumulator)==64);
struct Projection { std::span<const uint8_t> raw; size_t up_offset; std::span<float> out; };
struct Counts { uint64_t slabs=0,block_rows=0,output_rows=0; };

inline bool extent(const void* p,size_t n) {
    return p && n<=std::numeric_limits<uintptr_t>::max()-reinterpret_cast<uintptr_t>(p);
}
inline bool overlap(const void* p,size_t pn,const void* q,size_t qn) {
    const auto a=reinterpret_cast<uintptr_t>(p),b=reinterpret_cast<uintptr_t>(q);
    return a<=b ? b-a<pn : a-b<qn;
}
inline int signed_activation(int8_t a,bool negative) {
    // PSIGNB wraps -(-128) to -128, rather than yielding +128.
    return negative && a!=-128 ? -int(a) : int(a);
}

inline bool build(std::span<const uint8_t> activation_block,Slab& slab) {
    if(activation_block.size()!=sizeof(block_q8_K) ||
       !extent(activation_block.data(),activation_block.size_bytes()) ||
       overlap(activation_block.data(),activation_block.size_bytes(),&slab,sizeof slab))return false;
    const auto* qs=activation_block.data()+offsetof(block_q8_K,qs);
    for(unsigned group=0;group<64;++group) {
        __m256i products[4][8];
        for(unsigned lane=0;lane<4;++lane)for(unsigned magnitude=0;magnitude<8;++magnitude) {
            int8_t a;std::memcpy(&a,qs+4*group+lane,1);
            const int m=magnitude==7?62:4+8*magnitude;
            std::array<int16_t,16> values{};
            for(unsigned signs=0;signs<16;++signs)
                values[signs]=int16_t(m*signed_activation(a,(signs>>lane)&1));
            products[lane][magnitude]=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data()));
        }
        for(unsigned code=0;code<256;++code) {
            const uint32_t grid=iq3xxs_grid[code];
            const auto a=_mm256_add_epi16(products[0][(grid&255)>>3],products[1][((grid>>8)&255)>>3]);
            const auto b=_mm256_add_epi16(products[2][((grid>>16)&255)>>3],products[3][(grid>>24)>>3]);
            // Each four-product partial is bounded by31744; no signed overflow.
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(slab.values[group].data()+16*code),_mm256_add_epi16(a,b));
        }
    }
    _mm256_zeroupper();return true;
}

inline __m256i chunk(const Slab& slab,const uint8_t* block,unsigned p) {
    uint32_t gas;std::memcpy(&gas,block+66+4*p,sizeof gas);
    std::array<int32_t,8> values{};
    for(unsigned lane=0;lane<8;++lane) {
        const unsigned signs=ksigns_iq2xs[(gas>>(7*(lane/2)))&127];
        const unsigned nibble=(signs>>(4*(lane&1)))&15;
        values[lane]=slab.values[8*p+lane][16*block[2+8*p+lane]+nibble];
    }
    return _mm256_mullo_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(values.data())),
        _mm256_set1_epi32(2*(gas>>28)+1));
}
inline float horizontal(__m256 value) {
    auto sum=_mm_add_ps(_mm256_extractf128_ps(value,1),_mm256_castps256_ps128(value));
    sum=_mm_add_ps(sum,_mm_movehl_ps(sum,sum));
    sum=_mm_add_ss(sum,_mm_movehdup_ps(sum));return _mm_cvtss_f32(sum);
}
inline void update(const Slab& slab,const uint8_t* g,const uint8_t* u,float ad,Accumulator& acc) {
    __m256i gi=_mm256_setzero_si256(),ui=_mm256_setzero_si256();
    for(unsigned p=0;p<8;++p) {
        gi=_mm256_add_epi32(gi,chunk(slab,g,p));ui=_mm256_add_epi32(ui,chunk(slab,u,p));
    }
    uint16_t gd,ud;std::memcpy(&gd,g,2);std::memcpy(&ud,u,2);
    const auto half=[](uint16_t v){return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(v)));};
    // Preserve each row's native block-FMA order; only row traversal changes.
    _mm256_storeu_ps(acc.gate,_mm256_fmadd_ps(_mm256_set1_ps(half(gd)*ad),_mm256_cvtepi32_ps(gi),_mm256_loadu_ps(acc.gate)));
    _mm256_storeu_ps(acc.up,_mm256_fmadd_ps(_mm256_set1_ps(half(ud)*ad),_mm256_cvtepi32_ps(ui),_mm256_loadu_ps(acc.up)));
}

// Caller owns reusable scratch. Every call still rebuilds all ten slabs and
// clears its row accumulators. Refuse all unsupported/overlapping extents before
// writing scratch or outputs. Read-only weight spans may overlap one another.
[[gnu::noinline]] inline bool gu(std::span<const Projection> experts,std::span<const uint8_t> act,
                                Slab& slab,std::span<Accumulator> accum,Counts& counts) {
    if(experts.empty() || experts.size()>10 || act.size()!=blocks*sizeof(block_q8_K) ||
       accum.size()!=experts.size()*rows || !extent(act.data(),act.size_bytes()) ||
       !extent(accum.data(),accum.size_bytes()) ||
       overlap(act.data(),act.size_bytes(),&slab,sizeof slab) ||
       overlap(act.data(),act.size_bytes(),accum.data(),accum.size_bytes()) ||
       overlap(&slab,sizeof slab,accum.data(),accum.size_bytes()))return false;
    for(const auto& e:experts) {
        if(e.up_offset<projection_bytes || e.up_offset>e.raw.size() ||
           projection_bytes>e.raw.size()-e.up_offset || e.out.size()!=rows ||
           !extent(e.raw.data(),e.raw.size_bytes()) || !extent(e.out.data(),e.out.size_bytes()) ||
           overlap(e.raw.data(),e.raw.size_bytes(),act.data(),act.size_bytes()) ||
           overlap(e.raw.data(),e.raw.size_bytes(),&slab,sizeof slab) ||
           overlap(e.raw.data(),e.raw.size_bytes(),accum.data(),accum.size_bytes()) ||
           overlap(e.out.data(),e.out.size_bytes(),act.data(),act.size_bytes()) ||
           overlap(e.out.data(),e.out.size_bytes(),&slab,sizeof slab) ||
           overlap(e.out.data(),e.out.size_bytes(),accum.data(),accum.size_bytes()))return false;
        for(const auto& other:experts)
            if(overlap(e.out.data(),e.out.size_bytes(),other.raw.data(),other.raw.size_bytes()) ||
               (&e!=&other && overlap(e.out.data(),e.out.size_bytes(),other.out.data(),other.out.size_bytes())))return false;
    }
    std::memset(accum.data(),0,accum.size_bytes());
    for(unsigned b=0;b<blocks;++b) {
        const auto block=act.subspan(b*sizeof(block_q8_K),sizeof(block_q8_K));
        if(!build(block,slab))return false;
        ++counts.slabs;float ad;std::memcpy(&ad,block.data()+offsetof(block_q8_K,d),sizeof ad);
        for(size_t e=0;e<experts.size();++e)for(unsigned r=0;r<rows;++r) {
            const auto* g=experts[e].raw.data()+r*row_bytes+b*sizeof(block_iq3_xxs);
            update(slab,g,g+experts[e].up_offset,ad,accum[e*rows+r]);++counts.block_rows;
        }
    }
    for(size_t e=0;e<experts.size();++e)for(unsigned r=0;r<rows;++r) {
        const auto& a=accum[e*rows+r];
        const float g=.25f*horizontal(_mm256_loadu_ps(a.gate));
        const float u=.25f*horizontal(_mm256_loadu_ps(a.up));
        _mm256_zeroupper();experts[e].out[r]=(g/(1.f+std::exp(-g)))*u;++counts.output_rows;
    }
    _mm256_zeroupper();return true;
}
} // namespace activation_lut
