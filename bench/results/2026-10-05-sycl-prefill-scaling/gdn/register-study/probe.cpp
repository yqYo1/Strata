#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/kernels.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
namespace probe {
constexpr int S=128,HK=16,HV=48,C=10240,RG=4,RPG=32,CB=32,NCB=4;
__dpct_inline__ float warp_sum(float v) {
#pragma unroll
    /*
    DPCT1108: '__shfl_xor_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    for (int o = 16; o > 0; o >>= 1) v +=
        dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
    return v;
}
__dpct_inline__ float sigm(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}
__dpct_inline__ uint16_t hf(float f) {
    return sycl::bit_cast<unsigned short, sycl::half>(
               sycl::vec<float, 1>(f)
                   .convert<sycl::half, sycl::rounding_mode::rte>()[0]);
}
__dpct_inline__ void local_rec(float *__restrict__ state,
                                              const float *__restrict__ h,
                                              const float *__restrict__ gate,
                                              const float *__restrict__ beta,
                                              float *__restrict__ oc_out,
                                              int64_t T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int NT = CB * RG,
                  LPT = S / NT; // threads, q/k rows loaded per thread
    auto &sk = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sq = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[RG][CB]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int head = item_ct1.get_group(2) / NCB,
              cb = item_ct1.get_group(2) % NCB;
    const int c = item_ct1.get_local_id(2), rg = item_ct1.get_local_id(1),
              tid = rg * CB + c, col = cb * CB + c;
    const int qh = head % HK;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
    float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
    auto fetch = [&](int64_t t) {
        const float* ht = h + t * C;
#pragma unroll
        for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * S + tid + u * NT]; nk[u] = ht[HK * S + qh * S + tid + u * NT]; }
        nv = ht[2 * HK * S + head * S + col];
        ng = gate[t * HV + head];
        nb = beta[t * HV + head];
    };
    if (T > 0) fetch(0);
    for (int64_t t = 0; t < T; ++t) {
        float cq[LPT], ck[LPT];
#pragma unroll
        for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
        const float cv = nv, cg = ng, cbt = nb;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int u = 0; u < LPT; ++u) { sq[tid + u * NT] = cq[u]; sk[tid + u * NT] = ck[u]; }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (t + 1 < T) fetch(t + 1);
        const float g = sycl::native::exp(cg);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r)
            kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
        red[rg][c] = kv;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        const float kv_col = red[0][c] + red[1][c] + red[2][c] + red[3][c];
        const float delta = (cv - g * kv_col) * cbt;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = sycl::fma((float)g, s[r], sk[rg * RPG + r] * delta);
            o = sycl::fma(s[r], sq[rg * RPG + r], o);
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        red[rg][c] = o;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (rg == 0) oc_out[t * HV * S + head * S + col] =
            (red[0][c] + red[1][c] + red[2][c] + red[3][c]) *
            sycl::rsqrt((float)S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
}
__dpct_inline__ void norm_kernel(const float *__restrict__ z,
                                         const float *__restrict__ gamma,
                                         float eps, const float *__restrict__ y,
                                         uint16_t *__restrict__ y16) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &wsum = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[4]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int64_t t = item_ct1.get_group(2);
    const int head = item_ct1.get_group(1), col = item_ct1.get_local_id(2);
    const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
    const float oc = y[at];
    float sp = warp_sum(oc * oc);
    if ((col & 31) == 0) wsum[col >> 5] = sp;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float v = oc * sycl::rsqrt(ss / (float)S + eps) * gamma[col] *
                    sigm(z[t * HV * S + head * S + col]);
    y16[at] = hf(v);
}

__dpct_inline__ void shuffle_rec(float* state,const float* h,const float* gate,const float* beta,float* oc_out,int64_t T) {
 auto item=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
 auto &sq=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(item.get_group());
 auto &sk=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(item.get_group());
 const int tid=item.get_local_linear_id(),rg=tid%4,c=tid/4;
 const int head=item.get_group(2)/NCB,cb=item.get_group(2)%NCB,col=cb*CB+c,qh=head%HK;
 auto sg=item.get_sub_group();const int first=int(sg.get_local_linear_id())&~3;
 auto sum4=[&](float v){return sycl::select_from_group(sg,v,first)+sycl::select_from_group(sg,v,first+1)+sycl::select_from_group(sg,v,first+2)+sycl::select_from_group(sg,v,first+3);};
 float s[RPG];float* base=state+((size_t)(rg*RPG)*HV+head)*S+col;const size_t rs=(size_t)HV*S;
 #pragma unroll
 for(int r=0;r<RPG;r++)s[r]=base[r*rs];
 float nq=0,nk=0,nv=0,ng=0,nb=0;
 auto fetch=[&](int64_t t){const float* ht=h+t*C;nq=ht[qh*S+tid];nk=ht[HK*S+qh*S+tid];nv=ht[2*HK*S+head*S+col];ng=gate[t*HV+head];nb=beta[t*HV+head];};
 if(T>0)fetch(0);
 for(int64_t t=0;t<T;t++){
  const float cq=nq,ck=nk,cv=nv,cg=ng,cbt=nb;
  item.barrier(sycl::access::fence_space::local_space);
  sq[(tid/RPG)+(tid%RPG)*RG]=cq;sk[(tid/RPG)+(tid%RPG)*RG]=ck;
  item.barrier(sycl::access::fence_space::local_space);
  if(t+1<T)fetch(t+1);
  const float g=sycl::native::exp(cg);float kv=0;
  #pragma unroll
  for(int r=0;r<RPG;r++)kv=sycl::fma(s[r],sk[rg+r*RG],kv);
  const float kv_col=sum4(kv),delta=(cv-g*kv_col)*cbt;float o=0;
  #pragma unroll
  for(int r=0;r<RPG;r++){s[r]=sycl::fma(g,s[r],sk[rg+r*RG]*delta);o=sycl::fma(s[r],sq[rg+r*RG],o);}
  const float output=sum4(o);
  if(rg==0)oc_out[t*HV*S+head*S+col]=output*sycl::rsqrt((float)S);
 }
 #pragma unroll
 for(int r=0;r<RPG;r++)base[r*rs]=s[r];
}
__dpct_inline__ void pair_rec(float* state,const float* h,const float* gate,const float* beta,float* oc_out,int64_t T) {
 auto item=sycl::ext::oneapi::this_work_item::get_nd_item<3>();
 auto &sq=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(item.get_group());
 auto &sk=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[S]>(item.get_group());
 auto &rkv=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[4][CB]>(item.get_group());
 auto &ro=*sycl::ext::oneapi::group_local_memory_for_overwrite<float[4][CB]>(item.get_group());
 const int c=item.get_local_id(2),rg=item.get_local_id(1),tid=rg*CB+c;
 const int head=item.get_group(2)/NCB,cb=item.get_group(2)%NCB,col=cb*CB+c,qh=head%HK;
 const size_t rs=(size_t)HV*S;float s[2][RPG];
 #pragma unroll
 for(int j=0;j<2;j++){
  const float* base=state+((size_t)((rg+j*2)*RPG)*HV+head)*S+col;
  #pragma unroll
  for(int r=0;r<RPG;r++)s[j][r]=base[r*rs];
 }
 float nq[2],nk[2],nv=0,ng=0,nb=0;
 auto fetch=[&](int64_t t){const float* ht=h+t*C;
  #pragma unroll
  for(int u=0;u<2;u++){nq[u]=ht[qh*S+tid+u*64];nk[u]=ht[HK*S+qh*S+tid+u*64];}
  nv=ht[2*HK*S+head*S+col];ng=gate[t*HV+head];nb=beta[t*HV+head];
 };
 if(T>0)fetch(0);
 for(int64_t t=0;t<T;t++){
  float cq[2],ck[2];
  #pragma unroll
  for(int u=0;u<2;u++){cq[u]=nq[u];ck[u]=nk[u];}
  const float cv=nv,cg=ng,cbt=nb;
  // The preceding output barrier has completed every read of the prior q/k.
  // The next stage barrier also precedes this token's writes of rkv and ro.
  #pragma unroll
  for(int u=0;u<2;u++){sq[tid+u*64]=cq[u];sk[tid+u*64]=ck[u];}
  item.barrier(sycl::access::fence_space::local_space);
  if(t+1<T)fetch(t+1);
  const float g=sycl::native::exp(cg);float kv[2]={0,0},out[2]={0,0};
  #pragma unroll
  for(int r=0;r<RPG;r++){
   #pragma unroll
   for(int j=0;j<2;j++)kv[j]=sycl::fma(s[j][r],sk[(rg+j*2)*RPG+r],kv[j]);
  }
  #pragma unroll
  for(int j=0;j<2;j++)rkv[rg+j*2][c]=kv[j];
  item.barrier(sycl::access::fence_space::local_space);
  const float kv_col=rkv[0][c]+rkv[1][c]+rkv[2][c]+rkv[3][c];
  const float delta=(cv-g*kv_col)*cbt;
  #pragma unroll
  for(int r=0;r<RPG;r++){
   #pragma unroll
   for(int j=0;j<2;j++){
    s[j][r]=sycl::fma(g,s[j][r],sk[(rg+j*2)*RPG+r]*delta);
    out[j]=sycl::fma(s[j][r],sq[(rg+j*2)*RPG+r],out[j]);
   }
  }
  #pragma unroll
  for(int j=0;j<2;j++)ro[rg+j*2][c]=out[j];
  item.barrier(sycl::access::fence_space::local_space);
  if(rg==0)oc_out[t*HV*S+head*S+col]=(ro[0][c]+ro[1][c]+ro[2][c]+ro[3][c])*sycl::rsqrt((float)S);
 }
 #pragma unroll
 for(int j=0;j<2;j++){
  float* base=state+((size_t)((rg+j*2)*RPG)*HV+head)*S+col;
  #pragma unroll
  for(int r=0;r<RPG;r++)base[r*rs]=s[j][r];
 }
}

constexpr int GDN_TB=8,VPK=HV/HK;
__dpct_inline__ void gdn_cp4(float* s,const float* g){*s=*g;}
__dpct_inline__ void gdn_cp16(float* s,const float* g){*reinterpret_cast<sycl::float4*>(s)=*reinterpret_cast<const sycl::float4*>(g);}
__dpct_inline__ void gdn_cp_commit(){}
__dpct_inline__ void gdn_cp_wait_prev(){}
__dpct_inline__ void keyhead_rec(float *__restrict__ state,
                                       const float *__restrict__ h,
                                       const float *__restrict__ gate,
                                       const float *__restrict__ beta,
                                       float *__restrict__ oc_out, int64_t T) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int TB = GDN_TB, NT = CB * RG, QKP = S / 4,
                  VP = CB / 4; // threads, 16-byte pieces of a q/k row, of v
    auto &sq =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][TB][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sk =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][TB][S]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sv = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2][TB][VPK][CB]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sg =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][TB][VPK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &sb =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][TB][VPK]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &rkv = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[VPK][RG][CB]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &ro = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[VPK][RG][CB]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int qh = item_ct1.get_group(2) / NCB,
              cb = item_ct1.get_group(2) % NCB;
    const int c = item_ct1.get_local_id(2), rg = item_ct1.get_local_id(1),
              tid = rg * CB + c, col = cb * CB + c;
    float s[VPK][RPG];
    const size_t rs = (size_t) HV * S;
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        const float* base = state + ((size_t) (rg * RPG) * HV + qh + j * HK) * S + col;
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[j][r] = base[r * rs];
    }
    const int64_t nblk = (T + TB - 1) / TB;
    auto stage = [&](int64_t k) {   // tokens [k * TB, k * TB + TB) into buffer k & 1
        const int bb = (int) (k & 1);
        const int64_t t0 = k * TB;
        for (int p = tid; p < TB * 2 * QKP; p += NT) {
            const int i = p / (2 * QKP), w = p % (2 * QKP), isk = w / QKP, jj = (w % QKP) * 4;
            if (t0 + i < T)
                gdn_cp16(isk ? &sk[bb][i][jj] : &sq[bb][i][jj], h + (t0 + i) * C + (isk ? HK * S : 0) + qh * S + jj);
        }
        for (int p = tid; p < TB * VPK * VP; p += NT) {
            const int i = p / (VPK * VP), w = p % (VPK * VP), j = w / VP, jj = (w % VP) * 4;
            if (t0 + i < T)
                gdn_cp16(&sv[bb][i][j][jj], h + (t0 + i) * C + 2 * HK * S + (qh + j * HK) * S + cb * CB + jj);
        }
        for (int p = tid; p < 2 * TB * VPK; p += NT) {
            const int isb = p / (TB * VPK), w = p % (TB * VPK), i = w / VPK, j = w % VPK;
            if (t0 + i < T)
                gdn_cp4(isb ? &sb[bb][i][j] : &sg[bb][i][j], (isb ? beta : gate) + (t0 + i) * HV + qh + j * HK);
        }
    };
    if (nblk > 0) stage(0);
    gdn_cp_commit();
    for (int64_t k = 0; k < nblk; ++k) {
        // buffer (k + 1) & 1 was read by block k - 1, whose last token's second __syncthreads every thread has passed
        if (k + 1 < nblk) stage(k + 1);
        gdn_cp_commit();
        gdn_cp_wait_prev();
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        const int bb = (int) (k & 1);
        const int n = (int) ((T - k * TB) < TB ? (T - k * TB) : TB);
        for (int i = 0; i < n; ++i) {
            const int64_t t = k * TB + i;
            float kc[RPG];
#pragma unroll
            for (int r = 0; r < RPG; ++r) kc[r] = sk[bb][i][rg * RPG + r];
            float g[VPK], kv[VPK], delta[VPK], o[VPK];
#pragma unroll
            for (int j = 0; j < VPK; ++j) {
                g[j] = sycl::native::exp(sg[bb][i][j]); kv[j] = 0.0f;
                o[j] = 0.0f;
            }
#pragma unroll
            for (int r = 0; r < RPG; ++r)
#pragma unroll
                for (int j = 0; j < VPK; ++j)
                    kv[j] = sycl::fma(s[j][r], kc[r], kv[j]);
#pragma unroll
            for (int j = 0; j < VPK; ++j) rkv[j][rg][c] = kv[j];
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
#pragma unroll
            for (int j = 0; j < VPK; ++j) {
                const float kv_col = rkv[j][0][c] + rkv[j][1][c] + rkv[j][2][c] + rkv[j][3][c];
                delta[j] = (sv[bb][i][j][c] - g[j] * kv_col) * sb[bb][i][j];
            }
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                const float qr = sq[bb][i][rg * RPG + r];
#pragma unroll
                for (int j = 0; j < VPK; ++j) {
                    s[j][r] = sycl::fma(g[j], s[j][r], kc[r] * delta[j]);
                    o[j] = sycl::fma(s[j][r], (float)qr, o[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < VPK; ++j) ro[j][rg][c] = o[j];
            /*
            DPCT1118: SYCL group functions and algorithms must be
            encountered in converged control flow. You may need to adjust the
            code.
            */
            /*
            DPCT1065: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            if (rg < VPK)   // row group j writes head j's output
                oc_out[t * HV * S + (qh + rg * HK) * S + col] =
                    (ro[rg][0][c] + ro[rg][1][c] + ro[rg][2][c] +
                     ro[rg][3][c]) *
                    sycl::rsqrt((float)S);
        }
    }
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        float* base = state + ((size_t) (rg * RPG) * HV + qh + j * HK) * S + col;
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * rs] = s[j][r];
    }
}
}
class GdnKH32Large;class GdnKH16;class GdnKH16Large;class GdnPair;class GdnPairLarge;class GdnLocal;class GdnShuffle;class GdnNorm;
void candidate(sycl::queue &q,int variant,float* state,const float* h,const float* gate,const float* beta,const float* z,const float* gamma,float* y,uint16_t* y16,int64_t T){
 auto shape=sycl::nd_range<3>(sycl::range<3>(1,4,48*4*32),sycl::range<3>(1,4,32));
 if(variant==1)q.parallel_for<GdnLocal>(shape,[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::local_rec(state,h,gate,beta,y,T);});
 else if(variant==2)q.parallel_for<GdnShuffle>(shape,[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::shuffle_rec(state,h,gate,beta,y,T);});
 auto paired=sycl::nd_range<3>(sycl::range<3>(1,2,48*4*32),sycl::range<3>(1,2,32));
 if(variant==3)q.parallel_for<GdnPair>(paired,[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::pair_rec(state,h,gate,beta,y,T);});
 if(variant==4)q.parallel_for<GdnPairLarge>(paired,sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::pair_rec(state,h,gate,beta,y,T);});
 auto kh=sycl::nd_range<3>(sycl::range<3>(1,4,16*4*32),sycl::range<3>(1,4,32));
 if(variant==5)q.parallel_for<GdnKH32Large>(kh,sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::keyhead_rec(state,h,gate,beta,y,T);});
 if(variant==6)q.parallel_for<GdnKH16>(kh,[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {probe::keyhead_rec(state,h,gate,beta,y,T);});
 if(variant==7)q.parallel_for<GdnKH16Large>(kh,sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {probe::keyhead_rec(state,h,gate,beta,y,T);});
 q.parallel_for<GdnNorm>(sycl::nd_range<3>(sycl::range<3>(1,48,T*128),sycl::range<3>(1,1,128)),[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::norm_kernel(z,gamma,1e-6f,y,y16);});
}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
bool run(sycl::queue &q,int64_t T,int variant){
 const size_t ns=128*48*128,nh=T*10240,ny=T*48*128,ng=T*48;
 std::mt19937 rng(5927+T);std::uniform_real_distribution<float>d(-1,1);
 std::vector<float> hs(ns),hh(nh),hg(ng),hb(ng),hz(ny),hgamma(128);
 for(auto &x:hs)x=d(rng)*.01f;for(auto &x:hh)x=d(rng)*.06f;
 for(auto &x:hg)x=-.1f-std::abs(d(rng))*.1f;for(auto &x:hb)x=.1f+std::abs(d(rng))*.5f;
 for(auto &x:hz)x=d(rng);for(auto &x:hgamma)x=1+d(rng)*.1f;
 auto oldstate=sycl::malloc_device<float>(ns,q),newstate=sycl::malloc_device<float>(ns,q),h=sycl::malloc_device<float>(nh,q);
 auto gate=sycl::malloc_device<float>(ng,q),beta=sycl::malloc_device<float>(ng,q),z=sycl::malloc_device<float>(ny,q),gamma=sycl::malloc_device<float>(128,q);
 auto oldy=sycl::malloc_device<float>(ny,q),newy=sycl::malloc_device<float>(ny,q);
 auto old16=sycl::malloc_device<uint16_t>(ny,q),new16=sycl::malloc_device<uint16_t>(ny,q);
 q.memcpy(oldstate,hs.data(),ns*4);q.memcpy(newstate,hs.data(),ns*4);q.memcpy(h,hh.data(),nh*4);q.memcpy(gate,hg.data(),ng*4);q.memcpy(beta,hb.data(),ng*4);q.memcpy(z,hz.data(),ny*4);q.memcpy(gamma,hgamma.data(),128*4);q.wait_and_throw();
 auto old=[&]{strata::prefill::gdn_recurrence(oldstate,h,gate,beta,z,gamma,1e-6f,oldy,old16,T,&q);};
 auto fresh=[&]{candidate(q,variant,newstate,h,gate,beta,z,gamma,newy,new16,T);};
 old();fresh();q.wait_and_throw();
 auto id=variant==3?sycl::get_kernel_id<GdnPair>():variant==4?sycl::get_kernel_id<GdnPairLarge>():variant==5?sycl::get_kernel_id<GdnKH32Large>():variant==6?sycl::get_kernel_id<GdnKH16>():sycl::get_kernel_id<GdnKH16Large>();
 auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(),{q.get_device()},{id});
 auto kernel=bundle.get_kernel(id);
 printf("RESOURCE T=%lld variant=%d private=%llu spill=%llu\n",(long long)T,variant,
 (unsigned long long)kernel.get_info<sycl::info::kernel_device_specific::private_mem_size>(q.get_device()),
 (unsigned long long)kernel.get_info<sycl::ext::intel::info::kernel_device_specific::spill_memory_size>(q.get_device()));

 std::vector<float>a(std::max(ns,ny)),b(a.size());std::vector<uint16_t>a16(ny),b16(ny);
 size_t ds=0,dy=0,dh=0;bool finite=true;
 q.memcpy(a.data(),oldstate,ns*4);q.memcpy(b.data(),newstate,ns*4);q.wait_and_throw();
 for(size_t i=0;i<ns;i++){ds+=std::bit_cast<uint32_t>(a[i])!=std::bit_cast<uint32_t>(b[i]);finite=finite&&std::isfinite(a[i])&&std::isfinite(b[i]);}
 q.memcpy(a.data(),oldy,ny*4);q.memcpy(b.data(),newy,ny*4);q.memcpy(a16.data(),old16,ny*2);q.memcpy(b16.data(),new16,ny*2);q.wait_and_throw();
 for(size_t i=0;i<ny;i++){dy+=std::bit_cast<uint32_t>(a[i])!=std::bit_cast<uint32_t>(b[i]);dh+=a16[i]!=b16[i];finite=finite&&std::isfinite(a[i])&&std::isfinite(b[i])&&((a16[i]&0x7c00)!=0x7c00)&&((b16[i]&0x7c00)!=0x7c00);}
 auto time=[&](auto fn){q.wait_and_throw();auto begin=std::chrono::steady_clock::now();for(int r=0;r<5;r++)fn();q.wait_and_throw();return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/5;};
 std::vector<double> ot,nt;for(int r=0;r<3;r++){if(r%2){nt.push_back(time(fresh));ot.push_back(time(old));}else{ot.push_back(time(old));nt.push_back(time(fresh));}}
 const bool ok=finite&&ds==0&&dy==0&&dh==0;
 printf("%s T=%lld variant=%d state_unequal=%zu output_unequal=%zu half_unequal=%zu finite=%d original=%.6fms new=%.6fms\n",ok?"PASS":"FAIL",(long long)T,variant,ds,dy,dh,finite,median(ot),median(nt));fflush(stdout);
 for(void*p:std::vector<void*>{oldstate,newstate,h,gate,beta,z,gamma,oldy,newy,old16,new16})sycl::free(p,q);return ok;
}
int main(){auto &q=*strata::q_of(nullptr);bool ok=true;for(int64_t t:{1,17,257,1024,4096,8192})for(int v:{3,4,5,6,7})ok=run(q,t,v)&&ok;return ok?0:2;}
