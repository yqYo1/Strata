// CPU-only isolated proposal. Product source is included unchanged to expose its
// internal decoder and raw dot oracles; no product dispatch or cache integration.
#include "src/kernels/cpu/iq_avx2.cpp"
#include "src/kernels/cpu/iq_single_avx2.cpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <x86intrin.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unordered_map>
#include <vector>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

namespace probe {
using namespace strata::kernels::cpu;
constexpr int H=2560, FF=640, NB=10;
uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }
std::string context="startup";
void require(bool b,const char* s) { if (!b) throw std::runtime_error(context+": "+s); }
uint32_t random32(uint32_t& s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
uint64_t hash(const void* p,size_t n,uint64_t h=14695981039346656037ull) {
    auto b=static_cast<const uint8_t*>(p); while(n--) {h^=*b++;h*=1099511628211ull;} return h;
}
size_t pb(int ty) { require(ty==18||ty==21||ty==22,"unsupported type"); return ty==18?98:ty==21?110:82; }
// Exact byte layout: D[2], signed codebook magnitudes[256], raw subscales[8/4].
// No float/int16 expansion, padding, source-pointer, packed witness or requantization.
size_t eb(int ty) { return ty==21?262:266; }
size_t packed_bytes(int ty,int rows=FF,int nb=NB) { return 2*size_t(rows)*nb*pb(ty); }
size_t expanded_bytes(int ty,int rows=FF,int nb=NB) { return 2*size_t(rows)*nb*eb(ty); }
size_t pages(size_t n) {size_t p=size_t(sysconf(_SC_PAGESIZE));return (n+p-1)/p*p;}
struct Bytes {
    size_t n,mapped;uint8_t* p;
    explicit Bytes(size_t count):n(count),mapped(pages(count)) {
        p=static_cast<uint8_t*>(mmap(nullptr,mapped,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
        require(p!=MAP_FAILED,"payload mmap");
        require(!madvise(p,mapped,MADV_NOHUGEPAGE),"payload normal-page policy");
    }
    ~Bytes() {munmap(p,mapped);}
    Bytes(const Bytes&)=delete;Bytes& operator=(const Bytes&)=delete;
    uint8_t* data() { return p; } const uint8_t* data() const { return p; }
};
struct Guarded {
    void* base; size_t extent; uint8_t* p;
    Guarded(size_t n,bool at_end=true) {
        size_t page=size_t(sysconf(_SC_PAGESIZE)); extent=((n+page-1)/page+2)*page;
        base=mmap(nullptr,extent,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        require(base!=MAP_FAILED,"mmap"); auto b=static_cast<uint8_t*>(base);
        require(!mprotect(b,page,PROT_NONE)&&!mprotect(b+extent-page,page,PROT_NONE),"mprotect");
        p=at_end?b+extent-page-n:b+page;
    }
    ~Guarded() {munmap(base,extent);}
};
template<int TY> void prepare_block(const uint8_t* b,uint8_t* e) {
    std::memcpy(e,b,2);
    for(int half=0;half<8;++half) {
        __m256i g,s,sc; Fmt32<TY>::decode(b,half/2,half%2,g,s,sc);
        _mm256_storeu_si256((__m256i*)(e+2+32*half),_mm256_sign_epi8(g,s));
    }
    if constexpr(TY==18) for(int h=0;h<8;++h) e[258+h]=uint8_t(u32(b+66+4*h)>>28);
    if constexpr(TY==21) std::memcpy(e+258,b+106,4);
    if constexpr(TY==22) std::memcpy(e+258,b+74,8);
}
void prepare(int ty,const uint8_t* p,uint8_t* e,size_t blocks) {
    for(size_t i=0;i<blocks;++i,p+=pb(ty),e+=eb(ty)) {
        if(ty==18) prepare_block<18>(p,e); else if(ty==21) prepare_block<21>(p,e); else prepare_block<22>(p,e);
    }
}
// Independent scalar inverse: reconstruct the original packed bytes from expanded
// bytes alone. Uniqueness of each table is required rather than silently choosing
// a canonical code. This is correctness-only and never a bandwidth benchmark.
template<class T,size_t N> unsigned inverse(T x,const T (&table)[N]) {
    static const auto reverse=[&table] {
        std::unordered_map<T,unsigned> m;m.reserve(N);
        for(unsigned i=0;i<N;++i) require(m.emplace(table[i],i).second,"nonunique codebook inverse");
        return m;
    }();
    auto it=reverse.find(x);require(it!=reverse.end(),"missing codebook inverse");return it->second;
}
void reconstruct_block(int ty,const uint8_t* e,uint8_t* p) {
    std::memset(p,0,pb(ty));std::memcpy(p,e,2);
    for(int h=0;h<8;++h) {
        const auto* w=reinterpret_cast<const int8_t*>(e+2+32*h);
        uint32_t sign=0;
        for(int k=0;k<32;++k) {require(w[k]!=0&&w[k]!=-128,"invalid expanded lane"); if(w[k]<0) sign|=uint32_t(1)<<k;}
        if(ty==22) {
            for(int k=0;k<4;++k) {
                uint64_t v=0;for(int l=0;l<8;++l) v|=uint64_t(std::abs(int(w[8*k+l])))<<(8*l);
                unsigned idx=inverse(v,iq2s_grid);p[2+4*h+k]=uint8_t(idx);p[66+h]|=uint8_t((idx>>8)<<(2*k));
            }
            std::memcpy(p+34+4*h,&sign,4);p[74+h]=e[258+h];
        } else {
            for(int k=0;k<8;++k) {
                uint32_t v=0;for(int l=0;l<4;++l) v|=uint32_t(std::abs(int(w[4*k+l])))<<(8*l);
                unsigned idx=ty==18?inverse(v,iq3xxs_grid):inverse(v,iq3s_grid);
                p[2+8*h+k]=uint8_t(idx);if(ty==21) p[66+h]|=uint8_t((idx>>8)<<k);
            }
            if(ty==21) {std::memcpy(p+74+4*h,&sign,4);std::memcpy(p+106,e+258,4);}
            else {
                uint32_t aux=uint32_t(e[258+h])<<28;
                for(int k=0;k<4;++k) aux|=inverse(uint8_t(sign>>(8*k)),ksigns_iq2xs)<<(7*k);
                std::memcpy(p+66+4*h,&aux,4);
            }
        }
    }
}
template<class T,size_t N> void validate_table(const T (&table)[N]) {
    for(const T word:table) {
        (void)inverse(word,table);
        for(unsigned j=0;j<sizeof(T);++j) {unsigned m=unsigned((word>>(8*j))&255);require(m>0&&m<128,"signed-byte representation invalid magnitude");}
    }
}
void reconstruction(int ty,const uint8_t* p,const uint8_t* e,size_t blocks) {
    uint8_t recovered[110];
    for(size_t b=0;b<blocks;++b) {reconstruct_block(ty,e+b*eb(ty),recovered);require(!std::memcmp(recovered,p+b*pb(ty),pb(ty)),"packed reconstruction mismatch");}
}
inline __m256i expanded_scale(int ty,const uint8_t* b,int h) {
    const uint8_t s=b[258+(ty==21?h/2:h)];
    if(ty==22) return sc16(2*(s&15)+1,2*(s>>4)+1);
    return sc32(2*(ty==21?((h%2)?s>>4:s&15):s)+1);
}
inline float exact_hsum(__m256 x) {
    __m128 v=_mm_add_ps(_mm256_extractf128_ps(x,1),_mm256_castps256_ps128(x));
    v=_mm_add_ps(v,_mm_movehl_ps(v,v));v=_mm_add_ss(v,_mm_movehdup_ps(v));return _mm_cvtss_f32(v);
}
template<int TY,int NT,bool Exact> __attribute__((noinline))
void expanded_dot(const uint8_t* row,int nb,const block_q8_K* const* y,float* out) {
    __m256 af[NT];for(int t=0;t<NT;++t) af[t]=_mm256_setzero_ps();
    for(int b=0;b<nb;++b) {
        const uint8_t* e=row+size_t(b)*eb(TY);
        __m256i ai[NT][2];for(int t=0;t<NT;++t) ai[t][0]=ai[t][1]=_mm256_setzero_si256();
        for(int h=0;h<8;++h) {
            const __m256i w=_mm256_loadu_si256((const __m256i*)(e+2+32*h));
            const __m256i mag=_mm256_abs_epi8(w),sc=expanded_scale(TY,e,h);
            for(int t=0;t<NT;++t) {
                const __m256i a=_mm256_loadu_si256((const __m256i*)(y[t][b].qs+32*h));
                // Preserve PSIGN wrap for activation -128. Do not sign the weights
                // using activations: that is a different integer dot at -128.
                const __m256i term=_mm256_madd_epi16(_mm256_maddubs_epi16(mag,_mm256_sign_epi8(a,w)),sc);
                constexpr bool split=Exact;const int k=split?(h&1):0;
                ai[t][k]=_mm256_add_epi32(ai[t][k],term);
            }
        }
        const float d=h2f(u16(e));
        for(int t=0;t<NT;++t) {
            __m256i sum=Exact?_mm256_add_epi32(ai[t][0],ai[t][1]):ai[t][0];
            // GGML/exact applies K after final horizontal sum; custom applies K
            // before activation scale per block. Preserve these separate arms.
            const float scale=Exact?d*y[t][b].d:(d*Fmt32<TY>::K)*y[t][b].d;
            af[t]=_mm256_fmadd_ps(_mm256_set1_ps(scale),_mm256_cvtepi32_ps(sum),af[t]);
        }
    }
    for(int t=0;t<NT;++t) out[t]=Exact?Fmt32<TY>::K*exact_hsum(af[t]):hsum8(af[t]);
}
template<int TY,int NT,bool Exact> void expanded_gu(const uint8_t* e,int rows,int nb,
    const void* const* a,float* const* f,int r0,int r1) {
    const block_q8_K* y[NT];for(int t=0;t<NT;++t)y[t]=static_cast<const block_q8_K*>(a[t]);
    size_t row=size_t(nb)*eb(TY),up=size_t(rows)*row;
    for(int r=r0;r<r1;++r) {
        float g[NT],u[NT];expanded_dot<TY,NT,Exact>(e+size_t(r)*row,nb,y,g);
        expanded_dot<TY,NT,Exact>(e+up+size_t(r)*row,nb,y,u);
        for(int t=0;t<NT;++t) f[t][r]=(g[t]/(1.f+std::exp(-g[t])))*u[t];
    }
}
template<int TY,bool Exact> void expanded_nt(int nt,const uint8_t* e,int rows,int nb,const void* const* a,float* const* f,int r0,int r1) {
#define ARM(N) case N:expanded_gu<TY,N,Exact>(e,rows,nb,a,f,r0,r1);break
    switch(nt) {ARM(1);ARM(2);ARM(3);ARM(4);default:throw std::runtime_error("nt");}
#undef ARM
}
void candidate(int ty,bool exact,int nt,const uint8_t* e,int rows,int nb,const void* const* a,float* const* f,int r0,int r1) {
#define FORMAT(T) if(ty==T) {if(exact)expanded_nt<T,true>(nt,e,rows,nb,a,f,r0,r1);else expanded_nt<T,false>(nt,e,rows,nb,a,f,r0,r1);return;}
    FORMAT(18);FORMAT(21);FORMAT(22);throw std::runtime_error("type");
#undef FORMAT
}
float packed_raw(int arm,int ty,const uint8_t* p,int nb,const block_q8_K* a) {
    float out=0;
    if(arm==0) ggml_get_type_traits_cpu(ggml_type(ty))->vec_dot(nb*256,&out,0,p,0,a,0,1);
    else if(arm==1) {auto dot=ty==18?iq3xxs_dot:ty==21?iq3s_dot:iq2s_dot;dot(nb*256,&out,0,p,0,a,0,1);}
    else {const block_q8_K* y[]={a};if(ty==18)row_dot<18,1>(p,nb,y,&out);if(ty==21)row_dot<21,1>(p,nb,y,&out);if(ty==22)row_dot<22,1>(p,nb,y,&out);}
    return out;
}
float expanded_raw(int ty,bool exact,const uint8_t* e,int nb,const block_q8_K* a) {
    float out=0;const block_q8_K* y[]={a};
#define RAW(T) if(ty==T) {if(exact)expanded_dot<T,1,true>(e,nb,y,&out);else expanded_dot<T,1,false>(e,nb,y,&out);}
    RAW(18);RAW(21);RAW(22);return out;
#undef RAW
}
void packed_gu(int arm,int ty,const uint8_t* p,int rows,int nb,const void* const* a,int nt,float* const* f,int r0,int r1) {
    const size_t row=pb(ty)*nb,up=row*rows;
    if(arm==2) {iq256_gu_rows(ty,p,row,up,nb*256,a,nt,f,r0,r1);return;}
    if(arm==1) {for(int t=0;t<nt;++t)require(iq256_single_gu_rows(ty,p,row,up,nb*256,a[t],f[t],r0,r1),"exact helper unavailable");return;}
    for(int r=r0;r<r1;++r)for(int t=0;t<nt;++t) {float g=packed_raw(0,ty,p+r*row,nb,(const block_q8_K*)a[t]);float u=packed_raw(0,ty,p+up+r*row,nb,(const block_q8_K*)a[t]);f[t][r]=(g/(1.f+std::exp(-g)))*u;}
}
// Finite outputs require all bits (including signed zero). NaNs require NaN
// class; infinities require class and sign. No tolerance-based success.
bool same(float x,float y) {return std::isnan(x)?std::isnan(y):bits(x)==bits(y);}
struct Inputs {
    int ty,rows,nb;Bytes packed;std::array<std::vector<block_q8_K>,4> a;
    Inputs(int t,int r,int b,uint32_t seed,int pattern=0):ty(t),rows(r),nb(b),packed(packed_bytes(t,r,b)) {
        for(size_t j=0;j<packed.n;++j)packed.data()[j]=uint8_t(random32(seed));
        const uint16_t ds[]={0x1400,0,0x8000,1,0x03ff,0x7bff,0x7c00,0xfc00,0x7e01};
        for(size_t j=0;j<packed.n;j+=pb(t)) {
            auto p=packed.data()+j;
            if(pattern==1||pattern==2)std::memset(p+2,pattern==1?0:255,pb(t)-2);
            uint16_t d=pattern>=12?uint16_t(0x1400):pattern>=3?ds[(pattern-3)%9]:uint16_t(0x1400+(random32(seed)&1023));std::memcpy(p,&d,2);
        }
        for(int k=0;k<4;++k) {
            a[k].resize(nb);
            if(pattern==0) {
                std::vector<float> x(size_t(nb)*256);for(auto& f:x)f=float(int(random32(seed)%65535)-32767)/32768.f;
                ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(),a[k].data(),nb*256);
            } else for(auto& q:a[k]) {
                q.d=pattern==14?0.f:pattern==15?std::numeric_limits<float>::infinity():
                    pattern==16?std::numeric_limits<float>::quiet_NaN():pattern==17?std::numeric_limits<float>::denorm_min():.001f*float(k+1);
                const int8_t extremes[]={-128,0,127,-127,1,-1};
                for(int j=0;j<256;++j)q.qs[j]=pattern==12?int8_t(0):pattern==13?int8_t(-128):extremes[(j+k)%6];
                for(int j=0;j<16;++j){q.bsums[j]=0;for(int l=0;l<16;++l)q.bsums[j]+=q.qs[16*j+l];}
            }
        }
    }
};
uint64_t ns(clockid_t c) {timespec t{};require(!clock_gettime(c,&t),"clock_gettime");return uint64_t(t.tv_sec)*1000000000ull+t.tv_nsec;}
struct Stamp {int cpu;unsigned aux;uint64_t wall,thread,tsc;};
Stamp stamp() {Stamp s{};s.cpu=sched_getcpu();s.wall=ns(CLOCK_MONOTONIC_RAW);s.thread=ns(CLOCK_THREAD_CPUTIME_ID);_mm_lfence();s.tsc=__rdtscp(&s.aux);_mm_lfence();return s;}
void allocation(int ty,int experts) {
    const size_t p=packed_bytes(ty),e=expanded_bytes(ty);
    std::printf("{\"kind\":\"allocation\",\"type\":%d,\"experts\":%d,\"packed_block_bytes\":%zu,\"expanded_block_bytes\":%zu,\"packed_requested_bytes\":%zu,\"expanded_requested_bytes\":%zu,\"signed_weight_bytes\":%zu,\"metadata_bytes\":%zu,\"expanded_over_packed\":%.9g,\"packed_mapped_bytes\":%zu,\"expanded_mapped_bytes\":%zu,\"page_bytes\":%zu,\"normal_4K_or_host_page_policy\":true,\"controller_and_activation_allocations_excluded\":true}\n",ty,experts,pb(ty),eb(ty),p*experts,e*experts,size_t(experts)*2*FF*H,(e-2*FF*H)*experts,double(e)/p,pages(p)*experts,pages(e)*experts,size_t(sysconf(_SC_PAGESIZE)));
}
uint64_t check_case(const Inputs& d,const uint8_t* e,int nt,int r0,int r1,const void* a0_override=nullptr) {
    context="ty="+std::to_string(d.ty)+" nb="+std::to_string(d.nb)+" nt="+std::to_string(nt)+" range="+std::to_string(r0)+":"+std::to_string(r1);
    const void* a[4];std::array<std::vector<float>,5*4> out;
    for(int t=0;t<4;++t)a[t]=d.a[t].data();
    if(a0_override)a[0]=a0_override;
    for(auto& o:out)o.assign(size_t(d.rows)+2,123.f);
    for(int arm=0;arm<5;++arm) {
        float* f[4];for(int t=0;t<nt;++t)f[t]=out[4*arm+t].data()+1;
        if(arm<3)packed_gu(arm,d.ty,d.packed.data(),d.rows,d.nb,a,nt,f,r0,r1);
        else candidate(d.ty,arm==3,nt,e,d.rows,d.nb,a,f,r0,r1);
    }
    uint64_t checks=0;
    for(int t=0;t<nt;++t) for(int r=-1;r<=d.rows;++r) {
        if(r<r0||r>=r1) {for(int arm=0;arm<5;++arm)require(bits(out[4*arm+t][r+1])==bits(123.f),"output guard");}
        else {
            require(same(out[t][r+1],out[4+t][r+1]),"existing exact vs ggml ff");
            require(same(out[t][r+1],out[12+t][r+1]),"expanded exact vs ggml ff");
            require(same(out[8+t][r+1],out[16+t][r+1]),"expanded custom vs packed custom ff");++checks;
            for(int half=0;half<2;++half) {
                size_t j=size_t(half*d.rows+r)*d.nb;
                const auto* p=d.packed.data()+j*pb(d.ty);const auto* x=e+j*eb(d.ty);
                float gg=packed_raw(0,d.ty,p,d.nb,static_cast<const block_q8_K*>(a[t]));
                require(same(gg,packed_raw(1,d.ty,p,d.nb,static_cast<const block_q8_K*>(a[t]))),"existing exact vs ggml raw");
                require(same(gg,expanded_raw(d.ty,true,x,d.nb,static_cast<const block_q8_K*>(a[t]))),"expanded exact vs ggml raw");
                require(same(packed_raw(2,d.ty,p,d.nb,static_cast<const block_q8_K*>(a[t])),expanded_raw(d.ty,false,x,d.nb,static_cast<const block_q8_K*>(a[t]))),"expanded custom vs packed custom raw");++checks;
            }
        }
    }
    return checks;
}
// Exhaust every grid index, every sign byte/sign index and every subscale nibble.
// Extra combinations are deterministic; this is exhaustive fields, not exhaustive
// Cartesian product of all possible blocks.
uint64_t fields(int ty) {
    const int count=ty==22?1024:ty==21?512:256;
    block_q8_K a{};a.d=.001f;
    for(int j=0;j<256;++j)a.qs[j]=int8_t((j%3)==0?-128:(j%3)==1?127:0);
    for(int i=0;i<count;++i) {
        context="field ty="+std::to_string(ty)+" code="+std::to_string(i);
        uint8_t p[110]{},e[266]{};uint16_t d=uint16_t(0x1400+(i&1023));std::memcpy(p,&d,2);
        for(int h=0;h<8;++h) {
            if(ty==22) {
                for(int k=0;k<4;++k)p[2+4*h+k]=uint8_t(i);
                p[66+h]=uint8_t((i>>8)*0x55);p[74+h]=uint8_t(i);
                for(int k=0;k<4;++k)p[34+4*h+k]=uint8_t(i+17*k+h);
            } else {
                for(int k=0;k<8;++k)p[2+8*h+k]=uint8_t(i);
                if(ty==21) {
                    p[66+h]=i>=256?255:0;
                    for(int k=0;k<4;++k)p[74+4*h+k]=uint8_t(i+17*k+h);
                    p[106+h/2]=uint8_t(i);
                } else {
                    uint32_t s=uint32_t(i&15)<<28;
                    for(int k=0;k<4;++k)s|=uint32_t((i+17*k+h)&127)<<(7*k);
                    std::memcpy(p+66+4*h,&s,4);
                }
            }
        }
        prepare(ty,p,e,1);reconstruction(ty,p,e,1);
        const float gg=packed_raw(0,ty,p,1,&a);
        require(same(gg,packed_raw(1,ty,p,1,&a)),"field exact oracle");
        require(same(gg,expanded_raw(ty,true,e,1,&a)),"field expanded exact");
        require(same(packed_raw(2,ty,p,1,&a),expanded_raw(ty,false,e,1,&a)),"field expanded custom");
    }
    return uint64_t(count);
}
int correctness() {
    uint64_t cases=0,checks=0,field_blocks=0;
    for(int ty:{18,22,21}) {
        allocation(ty,1);allocation(ty,16);allocation(ty,64);field_blocks+=fields(ty);
        for(int pattern=0;pattern<18;++pattern)for(int nb:{1,10}) {
            context="correctness type="+std::to_string(ty)+" pattern="+std::to_string(pattern)+" nb="+std::to_string(nb);
            std::fprintf(stderr,"begin %s\n",context.c_str());
            Inputs d(ty,9,nb,0x713aaa01u+uint32_t(pattern),pattern);
            Bytes e(expanded_bytes(ty,d.rows,nb));prepare(ty,d.packed.data(),e.data(),2*d.rows*nb);
            reconstruction(ty,d.packed.data(),e.data(),2*d.rows*nb);
            for(int nt=1;nt<=4;++nt)for(auto bounds:{std::pair{0,9},std::pair{1,8},std::pair{8,9},std::pair{9,9}}) {
                checks+=check_case(d,e.data(),nt,bounds.first,bounds.second);++cases;
            }
            // Candidate expanded weights may start at any byte. Activations keep
            // natural block_q8_K alignment; GGML controls use original copies.
            Bytes unaligned(e.n+32);
            for(int off=0;off<32;++off) {
                std::memcpy(unaligned.data()+off,e.data(),e.n);
                checks+=check_case(d,unaligned.data()+off,1,8,9);++cases;
            }
            Bytes act_offsets(size_t(nb)*sizeof(block_q8_K)+32);
            for(int off=0;off<32;off+=int(alignof(block_q8_K))) {
                std::memcpy(act_offsets.data()+off,d.a[0].data(),size_t(nb)*sizeof(block_q8_K));
                checks+=check_case(d,e.data(),1,8,9,act_offsets.data()+off);++cases;
            }
            for(bool end:{false,true}) {
                Guarded gp(d.packed.n,end),ge(e.n,end),ga(size_t(nb)*sizeof(block_q8_K),end);
                std::memcpy(gp.p,d.packed.data(),d.packed.n);std::memcpy(ga.p,d.a[0].data(),size_t(nb)*sizeof(block_q8_K));
                prepare(ty,gp.p,ge.p,2*d.rows*nb);reconstruction(ty,gp.p,ge.p,2*d.rows*nb);
                checks+=check_case(d,ge.p,4,0,9,ga.p);++cases;
                // Guard-page activation end/begin for the expanded raw kernel.
                for(int half=0;half<2;++half) {
                    size_t j=size_t(half*d.rows+8)*nb;
                    require(same(expanded_raw(ty,true,ge.p+j*eb(ty),nb,(block_q8_K*)ga.p),
                                 packed_raw(0,ty,gp.p+j*pb(ty),nb,(block_q8_K*)ga.p)),"guard raw");++checks;
                }
            }
            std::printf("{\"kind\":\"correctness_case\",\"type\":%d,\"pattern\":%d,\"nb\":%d,\"rows\":9,\"nt_max\":4,\"packed_fnv64\":\"%016llx\",\"expanded_fnv64\":\"%016llx\",\"reconstruction\":true,\"raw_gu_ff\":true,\"guards\":true,\"pass\":true}\n",ty,pattern,nb,(unsigned long long)hash(d.packed.data(),d.packed.n),(unsigned long long)hash(e.data(),e.n));
            std::fflush(stdout);
        }
        // Actual full gate+up geometry, same controls, all nt arms, raw and ff.
        Inputs d(ty,FF,NB,0x927ac103);Bytes e(expanded_bytes(ty));prepare(ty,d.packed.data(),e.data(),2*FF*NB);
        reconstruction(ty,d.packed.data(),e.data(),2*FF*NB);
        for(int nt=1;nt<=4;++nt) {checks+=check_case(d,e.data(),nt,0,FF);++cases;}
    }
    std::printf("{\"kind\":\"correctness_close\",\"cases\":%llu,\"value_checks\":%llu,\"field_blocks\":%llu,\"pass\":true,\"criterion\":\"finite bitwise including signed zero; NaN class; infinity sign/class\",\"full_geometry\":true,\"nt\":[1,2,3,4]}\n",(unsigned long long)cases,(unsigned long long)checks,(unsigned long long)field_blocks);
    return 0;
}
const char* armname[]={"ggml","existing_exact","packed_custom","expanded_exact","expanded_custom"};
volatile uint64_t sink=0;
struct Bank {Inputs d;Bytes e;Bank(int ty,uint32_t seed):d(ty,FF,NB,seed),e(expanded_bytes(ty)) {}};
void timing(int experts,int nt,int rounds,int inner) {
    context="timing";
    cpu_set_t cpus;CPU_ZERO(&cpus);require(!sched_getaffinity(0,sizeof(cpus),&cpus),"getaffinity");require(CPU_COUNT(&cpus)==1,"pin to exactly one CPU before timing");
    const int core=sched_getcpu();std::mt19937 rng(0x9bbac15);
    uint64_t sequence=0;
    for(int ty:{18,22,21}) {
        allocation(ty,experts);
        std::vector<std::unique_ptr<Bank>> banks;uint64_t identity=14695981039346656037ull;
        // Allocate both whole cohorts before any fill timing. Each expert owns a
        // distinct packed and expanded payload; no shared repeated weight pointer.
        for(int k=0;k<experts;++k) {
            banks.push_back(std::make_unique<Bank>(ty,0xcab10001u+uint32_t(k)));
            identity=hash(banks.back()->d.packed.data(),banks.back()->d.packed.n,identity);
            for(int t=0;t<nt;++t)identity=hash(banks.back()->d.a[t].data(),NB*sizeof(block_q8_K),identity);
        }
        auto fill=[&](){for(auto& b:banks)prepare(ty,b->d.packed.data(),b->e.data(),2*FF*NB);};
        // First-touch fill includes destination page faults. Refill is reported
        // separately; neither is a cache-hit kernel sample or memcpy bandwidth.
        for(int pass=0;pass<2;++pass) {
            auto s=stamp();fill();auto e=stamp();
            std::printf("{\"kind\":\"prepare_sample\",\"type\":%d,\"phase\":\"%s\",\"experts\":%d,\"wall_ns\":%llu,\"thread_cpu_ns\":%llu,\"tsc_ticks\":%llu,\"cpu_start\":%d,\"cpu_end\":%d,\"migration_observed\":%s,\"source_fnv64\":\"%016llx\"}\n",ty,pass?"overwrite_refill":"cold_first_touch_fill",experts,(unsigned long long)(e.wall-s.wall),(unsigned long long)(e.thread-s.thread),(unsigned long long)(e.tsc-s.tsc),s.cpu,e.cpu,(s.cpu!=e.cpu||s.aux!=e.aux)?"true":"false",(unsigned long long)identity);
            require(s.cpu==core&&e.cpu==core&&s.aux==e.aux,"fill migration");
        }
        for(auto& b:banks)reconstruction(ty,b->d.packed.data(),b->e.data(),2*FF*NB);
        std::array<std::array<float,FF>,4> outputs{};
        auto sweep=[&](int arm,int times) {
            for(int i=0;i<times;++i)for(auto& b:banks) {
                const void* a[4];float* f[4];for(int t=0;t<nt;++t){a[t]=b->d.a[t].data();f[t]=outputs[t].data();}
                if(arm<3)packed_gu(arm,ty,b->d.packed.data(),FF,NB,a,nt,f,0,FF);
                else candidate(ty,arm==3,nt,b->e.data(),FF,NB,a,f,0,FF);
            }
        };
        // Each contract gets its own three paired comparisons. nt1 is primary;
        // nt2..4 is the existing custom arithmetic arm (exact measured separately
        // only when explicitly requested by changing this experiment later).
        const int pairs1[3][2]={{0,3},{1,3},{0,1}};
        const int pairsM[3][2]={{2,4},{0,2},{0,4}};
        const auto& pairs=nt==1?pairs1:pairsM;
        std::printf("{\"kind\":\"timing_open\",\"type\":%d,\"experts\":%d,\"nt\":%d,\"rounds\":%d,\"inner\":%d,\"input_fnv64\":\"%016llx\",\"regime\":\"%s\",\"TSC_is_not_core_cycles\":true,\"core_cycles_available\":false,\"no_model_speed_claim\":true}\n",ty,experts,nt,rounds,inner,(unsigned long long)identity,experts==1?"one_expert_hot":"sequential_distinct_experts");
        for(int round=0;round<rounds;++round) {
            std::array<int,3> order{0,1,2};std::shuffle(order.begin(),order.end(),rng);
            for(int pair:order) {
                const bool reverse=bool(rng()&1);int left=pairs[pair][reverse],right=pairs[pair][!reverse];int abba[]={left,right,right,left};
                for(int slot=0;slot<4;++slot) {
                    int arm=abba[slot];sweep(arm,1);auto s=stamp();sweep(arm,inner);auto e=stamp();
                    const uint64_t jobs=uint64_t(experts)*inner,rows=jobs*FF*nt;
                    sink=uint64_t(bits(outputs[0][FF-1]));
                    bool migrated=s.cpu!=core||e.cpu!=core||s.aux!=e.aux;
                    std::printf("{\"kind\":\"timing_sample\",\"sequence\":%llu,\"type\":%d,\"experts\":%d,\"nt\":%d,\"round\":%d,\"pair\":%d,\"slot\":%d,\"arm\":\"%s\",\"wall_ns\":%llu,\"thread_cpu_ns\":%llu,\"tsc_ticks\":%llu,\"cpu_start\":%d,\"cpu_end\":%d,\"aux_start\":%u,\"aux_end\":%u,\"migration_observed\":%s,\"expert_jobs\":%llu,\"expert_tokens\":%llu,\"output_rows\":%llu,\"wall_ns_per_output_row\":%.9g,\"thread_cpu_ns_per_output_row\":%.9g,\"logical_weight_payload_bytes\":%llu,\"payload_is_not_DRAM_bytes\":true,\"sink\":%llu}\n",(unsigned long long)sequence++,ty,experts,nt,round,pair,slot,armname[arm],(unsigned long long)(e.wall-s.wall),(unsigned long long)(e.thread-s.thread),(unsigned long long)(e.tsc-s.tsc),s.cpu,e.cpu,s.aux,e.aux,migrated?"true":"false",(unsigned long long)jobs,(unsigned long long)(jobs*nt),(unsigned long long)rows,double(e.wall-s.wall)/rows,double(e.thread-s.thread)/rows,(unsigned long long)(jobs*(arm<3?packed_bytes(ty):expanded_bytes(ty))*(arm<2?nt:1)),(unsigned long long)sink);
                    std::fflush(stdout);require(!migrated,"sample migration");
                }
            }
        }
    }
    std::printf("{\"kind\":\"timing_close\",\"samples\":%llu,\"complete\":true}\n",(unsigned long long)sequence);
}
int number(const char* p,int lo,int hi) {char* end;long n=std::strtol(p,&end,10);require(*p&&!*end&&n>=lo&&n<=hi,"argument bounds");return int(n);}
} // namespace probe
int main(int argc,char** argv) {
    try {
        using namespace probe;ggml_cpu_init();
        validate_table(iq3xxs_grid);validate_table(iq3s_grid);validate_table(iq2s_grid);
        std::printf("{\"kind\":\"runtime_open\",\"mxcsr\":%u,\"page_bytes\":%ld,\"cpu\":%d,\"activation_alignment_bytes\":%zu,\"TSC_is_not_core_cycles\":true}\n",_mm_getcsr(),sysconf(_SC_PAGESIZE),sched_getcpu(),alignof(block_q8_K));
        require(iq256_single_gu_rows(18,nullptr,0,0,256,nullptr,nullptr,0,0),"clang znver3 exact helper unavailable");
        for(int t:{18,22,21})require(ggml_get_type_traits_cpu(ggml_type(t))->vec_dot_type==GGML_TYPE_Q8_K,"Q8_K contract");
        if(argc==2&&std::string(argv[1])=="--correctness")return correctness();
        if(argc==7&&std::string(argv[1])=="--timing") {
            int experts=number(argv[2],1,64);require(experts==1||experts==16||experts==64,"cohort must be 1/16/64");
            int nt=number(argv[3],1,4),rounds=number(argv[4],12,12),inner=number(argv[5],1,32);
            require(std::string(argv[6])=="--correctness-approved-by-main","Main correctness gate required");
            timing(experts,nt,rounds,inner);return 0;
        }
        std::fprintf(stderr,"usage: fixture --correctness | --timing 1|16|64 1..4 12 inner --correctness-approved-by-main\n");return 2;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
