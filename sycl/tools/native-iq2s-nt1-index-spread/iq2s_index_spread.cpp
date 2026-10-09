/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Arithmetic copied from pinned GGML x86 quants.c AVX2 IQ2_S dot; see README.
#include <immintrin.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"
#include "quants.h"
#include "simd-mappings.h"
namespace isolated_iq2s {
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
constexpr auto high = [] { std::array<uint64_t,256> a{}; for(unsigned h=0;h<256;++h) for(unsigned k=0;k<4;++k) a[h] |= uint64_t((h>>(2*k))&3)<<(8*k); return a; }();
inline void grid_indices(const uint8_t* q, uint64_t hi, uint16_t* sp) {
    uint64_t low; memcpy(&low,q,8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(sp), _mm_unpacklo_epi8(_mm_cvtsi64_si128(static_cast<long long>(low)),_mm_cvtsi64_si128(static_cast<long long>(hi))));
}
static inline float hsum_float_8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x,1);
    res = _mm_add_ps(res,_mm256_castps256_ps128(x));
    res = _mm_add_ps(res,_mm_movehl_ps(res,res));
    res = _mm_add_ss(res,_mm_movehdup_ps(res)); return _mm_cvtss_f32(res);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
__attribute__((noinline)) float direct_control(int n, const block_iq2_s* x, const block_q8_K* y) {
 if(n<256 || n>8192 || n%256) throw std::runtime_error("width");
 const int nb=n/256; float result; float* s=&result;
   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        uint16_t sign_words[16];
        memcpy(sign_words, x[i].qs + QK_K/8, sizeof(sign_words));
        const uint16_t * signs = sign_words;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            __m256i q2_1, q2_2;
            q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

 return result;
}
__attribute__((noinline)) float index_candidate(int n, const block_iq2_s* x, const block_q8_K* y) {
 if(n<256 || n>8192 || n%256) throw std::runtime_error("width");
 const int nb=n/256; float result; float* s=&result;
   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        uint16_t sign_words[16];
        memcpy(sign_words, x[i].qs + QK_K/8, sizeof(sign_words));
        const uint16_t * signs = sign_words;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            __m256i q2_1, q2_2;
                uint16_t sp1[8], sp2[8];
                grid_indices(qs, high[qh[ib32]], sp1);
                grid_indices(qs+4, high[qh[ib32+1]], sp2);
                q2_1 = _mm256_set_epi64x(iq2s_grid[sp1[3]], iq2s_grid[sp1[2]], iq2s_grid[sp1[1]], iq2s_grid[sp1[0]]);
                q2_2 = _mm256_set_epi64x(iq2s_grid[sp2[3]], iq2s_grid[sp2[2]], iq2s_grid[sp2[1]], iq2s_grid[sp2[0]]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

 return result;
}
void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
void exact(float a,float b,const char* why) { require(std::isfinite(a)&&std::isfinite(b)&&std::bit_cast<uint32_t>(a)==std::bit_cast<uint32_t>(b),why); }
uint32_t next(uint32_t& s) { s=s*1664525u+1013904223u;return s; }
uint64_t hash_bytes(const void* ptr,size_t n,uint64_t h=14695981039346656037ull) { const auto* p=static_cast<const uint8_t*>(ptr);for(size_t i=0;i<n;++i) {h^=p[i];h*=1099511628211ull;}return h; }
float finish(float g,float u) { return (g/(1.f+std::exp(-g)))*u; }
void indices() {
 std::array<bool,1024> seen{}; uint64_t count=0;
 for(unsigned h=0;h<256;++h) for(unsigned lo=0;lo<256;++lo) {
  uint8_t q[8]; memset(q,lo,8); uint16_t sp[8]; grid_indices(q,high[h],sp);
  uint64_t expected[4];
  for(unsigned k=0;k<4;++k) { unsigned ix=lo|((h<<(8-2*k))&0x300); require(sp[k]==ix,"index"); seen[ix]=true; expected[k]=iq2s_grid[ix]; ++count; }
  __m256i v=_mm256_set_epi64x(iq2s_grid[sp[3]],iq2s_grid[sp[2]],iq2s_grid[sp[1]],iq2s_grid[sp[0]]);
  uint64_t got[4]; _mm256_storeu_si256(reinterpret_cast<__m256i*>(got),v); require(memcmp(got,expected,32)==0,"grid lanes");
 }
 for(bool x:seen) require(x,"index coverage"); require(count==262144,"index count");
 printf("indices,%llu,1024,pass\n",static_cast<unsigned long long>(count));
}
void rows() {
 auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_IQ2_S); require(sizeof(block_iq2_s)==82 && QK_K==256,"layout"); require(traits&&traits->vec_dot&&traits->vec_dot_type==GGML_TYPE_Q8_K&&traits->nrows==1,"traits");
 const auto baseline=traits->vec_dot;
 for(int n: {256,512,2560,8192}) for(uint32_t seed: {1u,0x12345678u,0xdeadbeefu}) for(int mode=0;mode<8;++mode) {
  std::vector<float> input(n); std::vector<block_q8_K> act(n/256); memset(act.data(),0,act.size()*sizeof(block_q8_K));
  uint32_t rng=seed;
  for(int j=0;j<n;++j) {
   const int r=int(next(rng)%255)-127;
   float x=float(r)/128.f;
   if(mode==0) x=0; if(mode==1) x=1; if(mode==2) x=(j&1)?-1:1;
   if(mode==3 || mode>=6) { x=(j%256==0)?127.f:float((j%253)-126)+0.5f; if(j%256!=0 && mode==6) x=std::nextafter(x,0.f); if(j%256!=0 && mode==7) x=std::nextafter(x,x>0?256.f:-256.f); }
   if(mode==4) x*=0.0009765625f; if(mode==5) x*=16.f;
   input[j]=x;
  }
  for(int b=0;b<n/256;++b) { float amax=0,max=0; for(int j=0;j<256;++j) {float x=input[256*b+j];require(std::isfinite(x),"input finite");if(std::fabs(x)>amax){amax=std::fabs(x);max=x;}} if(amax) {float scale=-127.f/max;require(std::isfinite(scale),"scale finite");for(int j=0;j<256;++j) require(std::isfinite(scale*input[256*b+j])&&std::fabs(scale*input[256*b+j])<128,"scaled admitted");} }
  quantize_row_q8_K(input.data(),act.data(),n);
  for(const auto& b:act) { require(std::isfinite(b.d),"quant scale");for(int8_t q:b.qs) require(q>=-127,"Q8 admitted"); }
  uint64_t checksum=hash_bytes(input.data(),input.size()*sizeof(float)); checksum=hash_bytes(act.data(),act.size()*sizeof(block_q8_K),checksum);
  for(int row=0;row<16;++row) {
   std::vector<block_iq2_s> gate(n/256),up(n/256);
   for(auto* weights: {&gate,&up}) for(auto& b:*weights) {
    b.d=ggml_fp32_to_fp16((row%4==0)?0.f:((row%4==1)?0.0009765625f:((row%4==2)?0.125f:1.f)));
    for(auto& q:b.qs) q=uint8_t(next(rng)>>24);
    for(auto& q:b.qh) q=uint8_t(next(rng)>>24);
    for(auto& q:b.scales) q=(row%4==0)?0:((row%4==1)?255:((row%4==2)?0xf0:uint8_t(next(rng)>>24)));
    if(row==6) memset(b.qs+32,0,32); if(row==5) memset(b.qs+32,255,32);
   }
   checksum=hash_bytes(gate.data(),gate.size()*sizeof(block_iq2_s),checksum);checksum=hash_bytes(up.data(),up.size()*sizeof(block_iq2_s),checksum);
   float g[3],u[3]; baseline(n,&g[0],0,gate.data(),0,act.data(),0,1); baseline(n,&u[0],0,up.data(),0,act.data(),0,1);
   g[1]=direct_control(n,gate.data(),act.data());u[1]=direct_control(n,up.data(),act.data());
   g[2]=index_candidate(n,gate.data(),act.data());u[2]=index_candidate(n,up.data(),act.data());
   checksum=hash_bytes(g,sizeof(g),checksum);checksum=hash_bytes(u,sizeof(u),checksum);
   for(int arm=1;arm<3;++arm) {exact(g[0],g[arm],"Gate bits");exact(u[0],u[arm],"Up bits");exact(finish(g[0],u[0]),finish(g[arm],u[arm]),"GU bits");}
  }
  printf("rows,%d,%u,%d,16,traits_direct_index,%016llx,pass\n",n,seed,mode,static_cast<unsigned long long>(checksum));
 }
}
}
int main(int argc,char**) {
 try { if(argc!=1) throw std::runtime_error("no arguments admitted"); ggml_cpu_init(); isolated_iq2s::indices(); isolated_iq2s::rows(); printf("summary,synthetic_only,performance_false,adopted_false,pass\n"); if(fflush(stdout)!=0||ferror(stdout)) throw std::runtime_error("stdout flush"); return 0; }
 catch(const std::exception& e) {fprintf(stderr,"FAIL,%s\n",e.what());fflush(stderr);return 1;}
}
