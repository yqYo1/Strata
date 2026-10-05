#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
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
}
class GdnLocal;class GdnShuffle;class GdnNorm;
void candidate(sycl::queue &q,int variant,float* state,const float* h,const float* gate,const float* beta,const float* z,const float* gamma,float* y,uint16_t* y16,int64_t T){
 auto shape=sycl::nd_range<3>(sycl::range<3>(1,4,48*4*32),sycl::range<3>(1,4,32));
 if(variant==1)q.parallel_for<GdnLocal>(shape,[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::local_rec(state,h,gate,beta,y,T);});
 else q.parallel_for<GdnShuffle>(shape,[=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {probe::shuffle_rec(state,h,gate,beta,y,T);});
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
int main(){auto &q=*strata::q_of(nullptr);bool ok=true;for(int64_t t:{1,17,257,1024,4096,8192})for(int v:{1,2})ok=run(q,t,v)&&ok;return ok?0:2;}
