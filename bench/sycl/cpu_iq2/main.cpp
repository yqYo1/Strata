// Main-authored component, repository MIT license. No engine dispatch changes.
// Format/codebook and reference: pinned ggml 3cf03257f219afbe7334045ff7c6a06ac68c627d
// (MIT, third_party/ggml/LICENSE). The existing direct-codebook helper is linked.
// Fused direct gate/up comparison; no XeStrata implementation input.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr int H=2560, FF=640, count=96, guard=64;
using Clock=std::chrono::steady_clock;
void need(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
double cpu_time() { rusage r{};need(!getrusage(RUSAGE_SELF,&r),"rusage");return r.ru_utime.tv_sec+r.ru_utime.tv_usec*1e-6+r.ru_stime.tv_sec+r.ru_stime.tv_usec*1e-6; }
struct Guarded {
    std::vector<uint8_t> raw;size_t n;
    explicit Guarded(size_t bytes):raw(bytes+2*guard,0xa7),n(bytes){}
    uint8_t* data(){return raw.data()+guard;}
    void check()const {for(int i=0;i<guard;++i)need(raw[i]==0xa7&&raw[guard+n+i]==0xa7,"canary");}
};
// Direct decode adapted from existing iq_single_avx2.cpp.
// Copyright (c) 2023-2026 The ggml authors; MIT, third_party/ggml/LICENSE.
__m256i scales(const block_iq2_s& x) {
    uint64_t a;std::memcpy(&a,x.scales,8);
    const auto v=_mm_set_epi64x(a>>4,a);
    return _mm256_cvtepi8_epi16(_mm_add_epi8(_mm_slli_epi16(_mm_and_si128(v,_mm_set1_epi8(15)),1),_mm_set1_epi8(1)));
}
__m256i half(const block_iq2_s& x,int p,__m256i q8,__m256i scale) {
    const uint8_t* q=x.qs+p*4;const unsigned h=x.qh[p];
    const auto grid=_mm256_set_epi64x(iq2s_grid[q[3]|((h<<2)&0x300)],iq2s_grid[q[2]|((h<<4)&0x300)],
        iq2s_grid[q[1]|((h<<6)&0x300)],iq2s_grid[q[0]|((h<<8)&0x300)]);
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
template<bool Expanded> __attribute__((noinline)) void pair_dot(const block_iq2_s* g,const block_iq2_s* u,
    const block_q8_K* y,int n,float& gout,float& uout) {
    __m256 ga=_mm256_setzero_ps(),ua=_mm256_setzero_ps();
    for(int b=0;b<n/256;++b) {
        __m256i gi=_mm256_setzero_si256(),ui=_mm256_setzero_si256();
        const auto gs=scales(g[b]),us=scales(u[b]);
        const auto part=[&](int p){
            const auto act=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[b].qs+32*p));
            gi=_mm256_add_epi32(gi,half(g[b],p,act,gs));ui=_mm256_add_epi32(ui,half(u[b],p,act,us));
        };
        if constexpr(Expanded) {
            #pragma clang loop unroll(full)
            for(int p=0;p<8;++p)part(p);
        } else {
            #pragma clang loop unroll(disable)
            for(int p=0;p<8;++p)part(p);
        }
        const float gd=_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(g[b].d)))*y[b].d;
        const float ud=_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(u[b].d)))*y[b].d;
        ga=_mm256_fmadd_ps(_mm256_set1_ps(gd),_mm256_cvtepi32_ps(gi),ga);
        ua=_mm256_fmadd_ps(_mm256_set1_ps(ud),_mm256_cvtepi32_ps(ui),ua);
    }
    gout=.125f*reduce(ga);uout=.125f*reduce(ua);
}
template<bool Expanded> __attribute__((noinline)) void gu(const uint8_t* blob,size_t stride,size_t up,
    const block_q8_K* y,float* out,int r0,int r1) {
    need(r0>=0&&r1>=r0&&r1<=FF&&stride>=size_t(H/256)*sizeof(block_iq2_s),"GU span");
    for(int r=r0;r<r1;++r) {
        float g,u;pair_dot<Expanded>(reinterpret_cast<const block_iq2_s*>(blob+r*stride),
            reinterpret_cast<const block_iq2_s*>(blob+up+r*stride),y,H,g,u);
        out[r]=(g/(1.f+std::exp(-g)))*u;
    }
}
bool bits(float a,float b){return std::memcmp(&a,&b,sizeof a)==0;}
uint64_t checksum(const uint8_t* p,size_t n) {
    uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;
}
float oracle_dot(int n,const void* x,const void* y) {
    float v=0;ggml_get_type_traits_cpu(GGML_TYPE_IQ2_S)->vec_dot(n,&v,0,x,0,y,0,1);return v;
}
void contract() {
    block_q8_K y{};y.d=.0009765625f;block_iq2_s x{};uint64_t cases=0;
    for(int pattern=0;pattern<4;++pattern) {
        for(int j=0;j<256;++j)y.qs[j]=pattern==0?127:pattern==1?-127:pattern==2?(j%2?127:-127):int((j*73)%255)-127;
        for(unsigned id=0;id<1024;++id)for(unsigned sign=0;sign<256;++sign) {
            x.qh[0]=uint8_t((id>>8)*85);x.scales[0]=0;
            for(int j=0;j<4;++j){x.qs[j]=id&255;x.qs[32+j]=sign;}
            int actual[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(actual),half(x,0,
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y.qs)),scales(x)));
            for(int j=0;j<8;++j) {
                int want=0;for(int k=0;k<4;++k)want+=int((iq2s_grid[id]>>(8*((j%2)*4+k)))&255)*
                    ((sign&(1<<((j%2)*4+k)))?-1:1)*int(y.qs[j*4+k]);
                need(actual[j]==want,"exact signed four-lane integer dot");
            }
            ++cases;
        }
    }
    constexpr size_t stride=H/256*sizeof(block_iq2_s)+4,up=stride*FF+16;
    Guarded weights(up+stride*FF),out(FF*sizeof(float));
    std::array<block_q8_K,H/256> a{};std::array<float,FF> ref{};
    uint32_t state=20261021;auto rnd=[&](){state=1664525u*state+1013904223u;return state;};
    for(int b=0;b<H/256;++b){a[b].d=(b%2?1:-1)*.0009765625f;for(auto& v:a[b].qs)v=int(rnd()%255)-127;}
    for(int side=0;side<2;++side)for(int r=0;r<FF;++r)for(int b=0;b<H/256;++b) {
        auto* w=reinterpret_cast<block_iq2_s*>(weights.data()+side*up+r*stride)+b;
        auto* raw=reinterpret_cast<uint8_t*>(w);for(size_t k=0;k<sizeof(*w);++k)raw[k]=rnd()>>24;
        w->d=ggml_fp32_to_fp16((r%5==0?0.f:r%5==1?-.0625f:r%5==2?0x1p-24f:.0625f));
    }
    for(int r=0;r<FF;++r) {
        const auto* g=reinterpret_cast<const block_iq2_s*>(weights.data()+r*stride);
        const auto* u=reinterpret_cast<const block_iq2_s*>(weights.data()+up+r*stride);
        const float wg=oracle_dot(H,g,a.data()),wu=oracle_dot(H,u,a.data());float ag,au;
        pair_dot<false>(g,u,a.data(),H,ag,au);need(bits(ag,wg)&&bits(au,wu),"loop scale/FMA/tree bits");
        pair_dot<true>(g,u,a.data(),H,ag,au);need(bits(ag,wg)&&bits(au,wu),"expanded scale/FMA/tree bits");
    }
    need(strata::kernels::cpu::iq256_single_gu_rows(22,weights.data(),stride,up,H,a.data(),ref.data(),3,FF-3),"direct helper unavailable");
    for(int method=1;method<=2;++method) {
        std::fill(out.raw.begin(),out.raw.end(),0xa7);float* dst=reinterpret_cast<float*>(out.data());
        if(method==1)gu<false>(weights.data(),stride,up,a.data(),dst,3,FF-3);else gu<true>(weights.data(),stride,up,a.data(),dst,3,FF-3);
        for(int r=3;r<FF-3;++r)need(bits(dst[r],ref[r])&&std::isfinite(dst[r]),"GU/SwiGLU bits");
        for(int r=0;r<3*int(sizeof(float));++r)need(out.data()[r]==0xa7&&out.data()[(FF-3)*sizeof(float)+r]==0xa7,"outside row span");out.check();
    }
    weights.check();std::cout<<"{\"kind\":\"contract\",\"integer_cases\":"<<cases<<",\"row_dots\":1280,\"GU_rows_per_method\":634,\"valid_extreme_Q8_checked\":true,\"passed\":true}"<<std::endl;
}
struct Inputs {
    std::vector<Guarded> weights;std::array<block_q8_K,H/256> act{};
    std::array<std::array<float,FF>,count> expected{},out{};
    std::array<uint64_t,count> input_hash{};std::array<block_q8_K,H/256> original_act{};
    Inputs(const char* path,uint64_t gate,uint64_t up,size_t bytes) {
        need(bytes==size_t(H/256)*sizeof(block_iq2_s)*FF,"real GU geometry");
        const int fd=open(path,O_RDONLY|O_CLOEXEC);need(fd>=0,"open shard");struct stat st{};need(!fstat(fd,&st),"shard stat");
        for(int e=0;e<count;++e) {
            weights.emplace_back(2*bytes);
            for(int side=0;side<2;++side) {
                const uint64_t off=(side?up:gate)+uint64_t(e)*bytes;need(off<=uint64_t(st.st_size)&&bytes<=uint64_t(st.st_size)-off,"shard slice");
                size_t done=0;while(done<bytes){auto n=pread(fd,weights.back().data()+side*bytes+done,bytes-done,off+done);need(n>0,"shard short read");done+=n;}
            }
        }
        close(fd);std::array<float,H> f{};uint32_t state=20261021;
        for(auto& x:f){state=1664525u*state+1013904223u;x=(int(state>>8)-8388608)*0x1p-23f;}
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(f.data(),act.data(),H);
        original_act=act;
        for(int e=0;e<count;++e){input_hash[e]=checksum(weights[e].data(),2*bytes);call(0,e);expected[e]=out[e];}
    }
    __attribute__((noinline)) void call(int method,int e) {
        const auto* p=weights[e].data();constexpr size_t row=H/256*sizeof(block_iq2_s),up=row*FF;
        if(method==0)need(strata::kernels::cpu::iq256_single_gu_rows(22,p,row,up,H,act.data(),out[e].data(),0,FF),"direct helper unavailable");
        else if(method==1)gu<false>(p,row,up,act.data(),out[e].data(),0,FF);
        else gu<true>(p,row,up,act.data(),out[e].data(),0,FF);
        asm volatile(""::"r"(out[e].data()):"memory");
    }
    void validate() {
        for(int m=0;m<3;++m)for(int e=0;e<count;++e){call(m,e);need(!std::memcmp(out[e].data(),expected[e].data(),sizeof(out[e])),"real GU bits");for(float x:out[e])need(std::isfinite(x),"real output finite");weights[e].check();}
        for(int e=0;e<count;++e)need(checksum(weights[e].data(),weights[e].n)==input_hash[e],"weight immutability");
        need(!std::memcmp(act.data(),original_act.data(),sizeof act),"activation immutability");
    }
};
void perf(Inputs& in,int process) {
    double fastest=1e6;
    for(int m=0;m<3;++m){auto t=Clock::now();for(int e=0;e<6;++e)in.call(m,e);fastest=std::min(fastest,elapsed(t));}
    auto t=Clock::now();for(int j=0;j<10000;++j){auto now=Clock::now();asm volatile(""::"r"(&now):"memory");}
    const double clock_s=elapsed(t)/10000;need(clock_s<fastest*.001,"per-pass clock overhead");
    std::cout<<"{\"kind\":\"clock\",\"per_call_s\":"<<clock_s<<",\"fastest_warm_pass_s\":"<<fastest<<"}"<<std::endl;
    std::mt19937 rng(20261021+process);std::array<int,count> experts{};std::iota(experts.begin(),experts.end(),0);
    std::shuffle(experts.begin(),experts.end(),rng);
    volatile double checksum=0;
    for(int round=0;round<6;++round)for(int mode=0;mode<2;++mode) {
        std::array<int,3> order{0,1,2};std::rotate(order.begin(),order.begin()+(round+process)%3,order.end());
        for(int m:order) {
            const int n=mode?count:6;uint64_t passes=0;const double before=cpu_time();auto start=Clock::now();double wall=0;
            do {for(int j=0;j<n;++j)in.call(m,mode?experts[j]:j);++passes;wall=elapsed(start);}while(wall<.35&&passes<1000000);
            const double cpu=cpu_time()-before;need(wall>=.25&&passes<1000000,"sample floor/bound");
            checksum=checksum+in.out[mode?experts[0]:0][0];
            std::cout<<"{\"kind\":\"sample\",\"process\":"<<process<<",\"round\":"<<round<<",\"mode\":"<<mode<<",\"method\":"<<m
                <<",\"passes\":"<<passes<<",\"experts\":"<<n<<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<cpu<<"}"<<std::endl;
        }
    }
    need(std::isfinite(checksum),"checksum");
}
} // namespace
int main(int argc,char** argv) {
    try {
        std::cout<<std::setprecision(17);need(argc==8,"mode shard gate up gu_bytes process cpu");
        const std::string mode=argv[1];need(mode=="check"||mode=="perf","mode");
        cpu_set_t cpus;CPU_ZERO(&cpus);const int core=std::stoi(argv[7]);need(core>=0&&core<CPU_SETSIZE,"cpu id");CPU_SET(core,&cpus);need(!sched_setaffinity(0,sizeof cpus,&cpus),"affinity");
        ggml_cpu_init();if(mode=="check")contract();
        Inputs input(argv[2],std::stoull(argv[3]),std::stoull(argv[4]),std::stoull(argv[5]));input.validate();
        std::cout<<"{\"kind\":\"real_validation\",\"experts\":96,\"methods\":3,\"rows_per_method\":61440,\"bitwise\":true}"<<std::endl;
        if(mode=="perf")perf(input,std::stoi(argv[6]));input.validate();
        std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
