// Temporary exact IQ4_NL row blocking experiment; GGML (MIT) arithmetic.
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
extern "C" float ggml_table_f32_f16[65536];
static inline float hsum(__m256 x) {
 __m128 v=_mm256_extractf128_ps(x,1);v=_mm_add_ps(v,_mm256_castps256_ps128(x));
 v=_mm_add_ps(v,_mm_movehl_ps(v,v));return _mm_cvtss_f32(_mm_add_ss(v,_mm_movehdup_ps(v)));
}
static inline __m256i dot4(const block_iq4_nl& w,__m256i a,__m128i table,__m128i mask,__m256i ones){
 const __m128i bits=_mm_loadu_si128((const __m128i*)w.qs);
 const __m128i lo=_mm_shuffle_epi8(table,_mm_and_si128(bits,mask));
 const __m128i hi=_mm_shuffle_epi8(table,_mm_and_si128(_mm_srli_epi16(bits,4),mask));
 const __m256i q=_mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1);
 return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(q,q),_mm256_sign_epi8(a,q)),ones);
}
template<int RR,bool Prepare>
static void rows(int n,float*out,const void*weights,size_t row,const void*activation,int r0,int r1){
 const auto*y=(const block_q8_0*)activation;
 const int nb=n/32;
 const auto*base=(const uint8_t*)weights;
 const __m128i table=_mm_loadu_si128((const __m128i*)kvalues_iq4nl),mask=_mm_set1_epi8(15);
 const __m256i ones=_mm256_set1_epi16(1);
 // Current expert down widths fit; the wrapper checks the bound before dispatch.
 alignas(32) float scales[128];
 if constexpr(Prepare)for(int b=0;b<nb;++b)scales[b]=ggml_table_f32_f16[y[b].d];
 int r=r0;
 for(;r+RR<=r1;r+=RR){
  __m256 a[RR],b[RR];const block_iq4_nl*x[RR];
  #pragma unroll
  for(int j=0;j<RR;++j){a[j]=b[j]=_mm256_setzero_ps();x[j]=(const block_iq4_nl*)(base+size_t(r+j)*row);}
  int ib=0;
  #pragma clang loop unroll_count(1)
  for(;ib+1<nb;ib+=2){
   const __m256i q0=_mm256_loadu_si256((const __m256i*)y[ib].qs),q1=_mm256_loadu_si256((const __m256i*)y[ib+1].qs);
   const float d0=Prepare?scales[ib]:ggml_table_f32_f16[y[ib].d],d1=Prepare?scales[ib+1]:ggml_table_f32_f16[y[ib+1].d];
   #pragma unroll
   for(int j=0;j<RR;++j){
    a[j]=_mm256_fmadd_ps(_mm256_set1_ps(d0*ggml_table_f32_f16[x[j][ib].d]),_mm256_cvtepi32_ps(dot4(x[j][ib],q0,table,mask,ones)),a[j]);
    b[j]=_mm256_fmadd_ps(_mm256_set1_ps(d1*ggml_table_f32_f16[x[j][ib+1].d]),_mm256_cvtepi32_ps(dot4(x[j][ib+1],q1,table,mask,ones)),b[j]);
   }
  }
  #pragma unroll
  for(int j=0;j<RR;++j){
   float value=hsum(_mm256_add_ps(a[j],b[j]));
   if(ib<nb){int s0=0,s1=0;for(int k=0;k<16;++k){s0+=y[ib].qs[k]*kvalues_iq4nl[x[j][ib].qs[k]&15];s1+=y[ib].qs[k+16]*kvalues_iq4nl[x[j][ib].qs[k]>>4];}const float d=ggml_table_f32_f16[y[ib].d]*ggml_table_f32_f16[x[j][ib].d];value+=d*(s0+s1);}
   out[r+j]=value;
  }
 }
 if constexpr(RR>1)if(r<r1)rows<1,Prepare>(n,out,weights,row,activation,r,r1);
}
#define WRAP(R,P,V) extern "C" void rows20_##V(int n,float*out,const void*w,size_t row,const void*a,int r0,int r1){rows<R,P>(n,out,w,row,a,r0,r1);}
WRAP(1,false,1) WRAP(2,false,2) WRAP(4,false,4)
WRAP(1,true,11) WRAP(2,true,12) WRAP(4,true,14)
