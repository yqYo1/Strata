// Main-authored component, repository MIT license. No engine dispatch changes.
// Format/codebook and reference: pinned ggml 3cf03257f219afbe7334045ff7c6a06ac68c627d
// (MIT, third_party/ggml/LICENSE). The existing direct-codebook helper is linked.
// Independent exact level-mask algebra; no XeStrata implementation input.
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
struct Masks {
    std::array<uint16_t,1024> word{};
    Masks() {
        for(int i=0;i<1024;++i) for(int k=0;k<8;++k) {
            const unsigned v=(iq2s_grid[i]>>(8*k))&255;
            need(v==8||v==25||v==43,"IQ2 codebook levels");
            word[i]|=uint16_t((v==25?1:0)<<(k%4+(k/4)*8));
            word[i]|=uint16_t((v==43?1:0)<<(k%4+4+(k/4)*8));
        }
    }
};
const Masks masks;
template<class T> struct alignas(64) Tables {
    // T[m] = 2*P[m] - P[15], in [-508,508] for valid Q8_K codes.
    // H2560: exactly20KiB int16 or40KiB int32. No padded gather/alias trick.
    std::array<T,H/4*16> value{};int n=0;
    __attribute__((noinline)) void prepare(const block_q8_K* y,int length) {
        need(length>0&&length<=H&&length%256==0,"activation shape");n=length;
        for(int b=0;b<n/256;++b) {
            for(int k=0;k<256;++k) need(y[b].qs[k]!=-128,"invalid production Q8_K code -128");
            for(int j=0;j<64;++j) for(int m=0;m<16;++m) {
                int sum=0;for(int k=0;k<4;++k)sum+=(m&(1<<k)?1:-1)*int(y[b].qs[4*j+k]);
                value[(b*64+j)*16+m]=T(sum);
            }
        }
    }
};
int four(const int16_t* t,unsigned map,unsigned sign) {
    // g=8+17*M25+35*M43. D_s(m)=(T[m xor s]-T[s])/2.
    // Numerator is even; intermediates fit int32, including +/-127.
    return (17*int(t[(map&15)^sign])+35*int(t[((map>>4)&15)^sign])-68*int(t[sign]))/2;
}
template<class T> __m256i chunk(const block_iq2_s& x,const Tables<T>& t,int b,int part) {
    int maps[8],signs[8];
    for(int j=0;j<4;++j) {
        const unsigned id=x.qs[part*4+j]|((unsigned(x.qh[part])>>(2*j)&3)<<8);
        const unsigned m=masks.word[id],s=x.qs[32+part*4+j];
        maps[2*j]=m&255;maps[2*j+1]=m>>8;signs[2*j]=s&15;signs[2*j+1]=s>>4;
    }
    const T* base=t.value.data()+(b*64+part*8)*16;
    if constexpr(sizeof(T)==2) {
        int out[8];for(int j=0;j<8;++j)out[j]=four(base+j*16,maps[j],signs[j]);
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(out));
    } else {
        const __m256i m=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(maps));
        const __m256i s=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(signs));
        const __m256i off=_mm256_setr_epi32(0,16,32,48,64,80,96,112),lo=_mm256_set1_epi32(15);
        const auto a=_mm256_i32gather_epi32(base,_mm256_add_epi32(off,_mm256_xor_si256(s,_mm256_and_si256(m,lo))),4);
        const auto c=_mm256_i32gather_epi32(base,_mm256_add_epi32(off,_mm256_xor_si256(s,_mm256_and_si256(_mm256_srli_epi32(m,4),lo))),4);
        const auto z=_mm256_i32gather_epi32(base,_mm256_add_epi32(off,s),4);
        return _mm256_srai_epi32(_mm256_sub_epi32(_mm256_add_epi32(_mm256_mullo_epi32(a,_mm256_set1_epi32(17)),
            _mm256_mullo_epi32(c,_mm256_set1_epi32(35))),_mm256_mullo_epi32(z,_mm256_set1_epi32(68))),1);
    }
}
float reduce(__m256 a) {
    __m128 v=_mm_add_ps(_mm256_extractf128_ps(a,1),_mm256_castps256_ps128(a));
    v=_mm_add_ps(v,_mm_movehl_ps(v,v));v=_mm_add_ss(v,_mm_movehdup_ps(v));return _mm_cvtss_f32(v);
}
template<class T> float dot(const block_iq2_s* x,const block_q8_K* y,const Tables<T>& t) {
    __m256 acc=_mm256_setzero_ps();
    for(int b=0;b<t.n/256;++b) {
        __m256i isum=_mm256_setzero_si256();
        for(int p=0;p<8;++p) {
            const unsigned sc=x[b].scales[p];
            const auto scales=_mm256_setr_epi32(2*(sc&15)+1,2*(sc&15)+1,2*(sc&15)+1,2*(sc&15)+1,
                2*(sc>>4)+1,2*(sc>>4)+1,2*(sc>>4)+1,2*(sc>>4)+1);
            isum=_mm256_add_epi32(isum,_mm256_mullo_epi32(chunk(x[b],t,b,p),scales));
        }
        // Same eight integer lanes, per-block scale/FMA and final float tree as direct.
        const float d=_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x[b].d)))*y[b].d;
        acc=_mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(isum),acc);
    }
    return .125f*reduce(acc);
}
template<class T> __attribute__((noinline)) void gu(const uint8_t* blob,size_t stride,size_t up,
    const block_q8_K* y,const Tables<T>& table,float* out,int r0,int r1) {
    need(r0>=0&&r1>=r0&&r1<=FF&&stride>=size_t(table.n/256)*sizeof(block_iq2_s),"GU span");
    for(int r=r0;r<r1;++r) {
        const float g=dot(reinterpret_cast<const block_iq2_s*>(blob+r*stride),y,table);
        const float u=dot(reinterpret_cast<const block_iq2_s*>(blob+up+r*stride),y,table);
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
    block_q8_K y{};y.d=.0009765625f;
    Tables<int16_t> t16;Tables<int32_t> t32;block_iq2_s x{};
    uint64_t cases=0;
    for(int pattern=0;pattern<4;++pattern) {
        for(int j=0;j<256;++j)y.qs[j]=pattern==0?127:pattern==1?-127:pattern==2?(j%2?127:-127):int((j*73)%255)-127;
        t16.prepare(&y,256);t32.prepare(&y,256);
        for(unsigned id=0;id<1024;++id)for(unsigned s=0;s<256;++s) {
            x.qh[0]=uint8_t((id>>8)*85);
            for(int j=0;j<4;++j){x.qs[j]=id&255;x.qs[32+j]=s;}
            int actual16[8],actual32[8];_mm256_storeu_si256(reinterpret_cast<__m256i*>(actual16),chunk(x,t16,0,0));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(actual32),chunk(x,t32,0,0));
            for(int j=0;j<8;++j) {
                int want=0;for(int k=0;k<4;++k)want+=int((iq2s_grid[id]>>(8*((j%2)*4+k)))&255)*
                    ((s&(1<<((j%2)*4+k)))?-1:1)*int(y.qs[j*4+k]);
                need(actual16[j]==want&&actual32[j]==want,"exact signed four-lane integer dot");
            }
            ++cases;
        }
    }
    y.qs[255]=-128;bool rejected=false;try{t16.prepare(&y,256);}catch(const std::runtime_error&){rejected=true;}need(rejected,"invalid activation rejection");
    rejected=false;try{t32.prepare(&y,256);}catch(const std::runtime_error&){rejected=true;}need(rejected,"invalid gather activation rejection");
    for(int length:{0,255,H+256}){rejected=false;try{t16.prepare(&y,length);}catch(const std::runtime_error&){rejected=true;}need(rejected,"invalid activation shape rejection");}
    // Padded rows, all codebook indices/signs/scales, ten blocks and finite half scales.
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
    t16.prepare(a.data(),H);t32.prepare(a.data(),H);
    for(int side=0;side<2;++side)for(int r=0;r<FF;++r) {
        const auto* w=reinterpret_cast<const block_iq2_s*>(weights.data()+side*up+r*stride);
        const float want=oracle_dot(H,w,a.data());need(bits(dot(w,a.data(),t16),want)&&bits(dot(w,a.data(),t32),want),"per-row float scale/FMA/tree bits");
    }
    need(strata::kernels::cpu::iq256_single_gu_rows(22,weights.data(),stride,up,H,a.data(),ref.data(),3,FF-3),"direct helper unavailable");
    for(int method=1;method<=2;++method) {
        std::fill(out.raw.begin(),out.raw.end(),0xa7);float* dst=reinterpret_cast<float*>(out.data());
        if(method==1)gu(weights.data(),stride,up,a.data(),t16,dst,3,FF-3);else gu(weights.data(),stride,up,a.data(),t32,dst,3,FF-3);
        for(int r=3;r<FF-3;++r)need(bits(dst[r],ref[r])&&std::isfinite(dst[r]),"GU/SwiGLU bits");
        for(int r=0;r<3*int(sizeof(float));++r)need(out.data()[r]==0xa7&&out.data()[(FF-3)*sizeof(float)+r]==0xa7,"outside row span");out.check();
    }
    weights.check();std::cout<<"{\"kind\":\"contract\",\"integer_cases\":"<<cases<<",\"row_dots\":1280,\"GU_rows_per_method\":634,\"invalid_q8_rejected\":true,\"passed\":true}"<<std::endl;
}
struct Inputs {
    std::vector<Guarded> weights;std::array<block_q8_K,H/256> act{};
    std::array<std::array<float,FF>,count> expected{},out{};
    Tables<int16_t> t16;Tables<int32_t> t32;
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
        else if(method==1){t16.prepare(act.data(),H);gu(p,row,up,act.data(),t16,out[e].data(),0,FF);}
        else {t32.prepare(act.data(),H);gu(p,row,up,act.data(),t32,out[e].data(),0,FF);}
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
