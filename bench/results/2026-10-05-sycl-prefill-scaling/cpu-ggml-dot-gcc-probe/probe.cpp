#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sched.h>
#include <string>
#include <vector>

using Dot = void (*)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
#define DECL(NAME) extern "C" { \
    void reference_##NAME(int,float*,size_t,const void*,size_t,const void*,size_t,int); \
    void gcc_##NAME(int,float*,size_t,const void*,size_t,const void*,size_t,int); }
DECL(ggml_vec_dot_iq2_xxs_q8_K)
DECL(ggml_vec_dot_iq2_xs_q8_K)
DECL(ggml_vec_dot_iq2_s_q8_K)
DECL(ggml_vec_dot_iq3_xxs_q8_K)
DECL(ggml_vec_dot_iq3_s_q8_K)
DECL(ggml_vec_dot_iq1_s_q8_K)
DECL(ggml_vec_dot_iq1_m_q8_K)
DECL(ggml_vec_dot_iq4_xs_q8_K)
DECL(ggml_vec_dot_iq4_nl_q8_0)
struct Format {
    ggml_type type;
    Dot reference, alternate;
    uint64_t checked = 0, mismatched = 0, nonfinite = 0;
    std::vector<uint8_t> streaming;
    int n = 0, rows = 0, real_cases = 0;
    size_t stride = 0, expert_bytes = 0;
    std::string role;
};
#define FORMAT(TYPE, NAME) {TYPE, reference_##NAME, gcc_##NAME}
static Format formats[] = {
    FORMAT(GGML_TYPE_IQ2_XXS,ggml_vec_dot_iq2_xxs_q8_K),
    FORMAT(GGML_TYPE_IQ2_XS,ggml_vec_dot_iq2_xs_q8_K),
    FORMAT(GGML_TYPE_IQ2_S,ggml_vec_dot_iq2_s_q8_K),
    FORMAT(GGML_TYPE_IQ3_XXS,ggml_vec_dot_iq3_xxs_q8_K),
    FORMAT(GGML_TYPE_IQ3_S,ggml_vec_dot_iq3_s_q8_K),
    FORMAT(GGML_TYPE_IQ1_S,ggml_vec_dot_iq1_s_q8_K),
    FORMAT(GGML_TYPE_IQ1_M,ggml_vec_dot_iq1_m_q8_K),
    FORMAT(GGML_TYPE_IQ4_XS,ggml_vec_dot_iq4_xs_q8_K),
    FORMAT(GGML_TYPE_IQ4_NL,ggml_vec_dot_iq4_nl_q8_0),
};
static void require(bool ok, const char* why) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",why); std::exit(1); }
}
static double milliseconds() {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Activations {
    std::vector<uint8_t> bytes;
    const void* pointers[8];
    Activations(ggml_type type,int n,std::mt19937& rng,bool edges) {
        const auto* traits = ggml_get_type_traits_cpu(type);
        const auto* quant = ggml_get_type_traits_cpu(traits->vec_dot_type);
        const size_t size = ggml_row_size(traits->vec_dot_type,n);
        require(quant && quant->from_float,"missing original activation quantizer");
        bytes.resize(8*size);
        std::vector<float> x(n);
        std::normal_distribution<float> normal(0.f,.5f);
        for(int t=0;t<8;++t) {
            for(int i=0;i<n;++i)
                x[i] = edges && t==1 ? 0.f : edges && t==2 ? .125f :
                       edges && t==3 ? (i%2 ? -.25f : .25f) : normal(rng);
            pointers[t] = bytes.data()+t*size;
            quant->from_float(x.data(),bytes.data()+t*size,n);
        }
    }
};
static void verify(Format& f,const uint8_t* w,size_t stride,int n,int rows,const Activations& act) {
    Dot production = ggml_get_type_traits_cpu(f.type)->vec_dot;
    for(int t=0;t<8;++t) for(int r=0;r<rows;++r) {
        float a[3]={.125f,.125f,.125f},b[3]={.125f,.125f,.125f},p[3]={.125f,.125f,.125f};
        f.reference(n,a+1,0,w+r*stride,0,act.pointers[t],0,1);
        f.alternate(n,b+1,0,w+r*stride,0,act.pointers[t],0,1);
        production(n,p+1,0,w+r*stride,0,act.pointers[t],0,1);
        require(std::memcmp(a,p,sizeof(a))==0,"renamed object differs from actual CPU traits");
        require(a[0]==.125f && a[2]==.125f && b[0]==.125f && b[2]==.125f,"dot output guard overwritten");
        for(int k=0;k<3;++k) {
            require(std::isfinite(a[k]),"non-finite original dot");
            ++f.checked;
            if(!std::isfinite(b[k])) ++f.nonfinite;
            if(std::memcmp(a+k,b+k,sizeof(float))) {
                if(f.mismatched<8) {
                    uint32_t ab,bb;std::memcpy(&ab,a+k,4);std::memcpy(&bb,b+k,4);
                    std::printf("{\"kind\":\"mismatch\",\"type\":%d,\"n\":%d,\"row\":%d,\"activation\":%d,\"original_bits\":%u,\"gcc_bits\":%u}\n",f.type,n,r,t,ab,bb);
                }
                ++f.mismatched;
            }
        }
    }
}
static void set_scale(ggml_type type,uint8_t* block,uint16_t h) {
    if(type!=GGML_TYPE_IQ1_M) { std::memcpy(block,&h,2); return; }
    // The global IQ1_M half scale occupies the high nibble of four words.
    uint8_t* scales=block+offsetof(block_iq1_m,scales);
    for(int j=0;j<4;++j) {
        uint16_t word;std::memcpy(&word,scales+2*j,2);
        word=(word&0x0fff)|(((h>>(4*j))&15)<<12);
        std::memcpy(scales+2*j,&word,2);
    }
}
int main(int argc,char** argv) {
    require(argc==2,"provide real GGUF shard");
    std::setvbuf(stdout,nullptr,_IONBF,0);
    cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(2,&cpus);
    require(sched_setaffinity(0,sizeof(cpus),&cpus)==0,"cannot pin CPU 2");
    ggml_cpu_init();std::mt19937 rng(1913);
    for(auto& f:formats) {
        const auto widths=f.type==GGML_TYPE_IQ4_NL ? std::vector<int>{32,96,640,768,2560} : std::vector<int>{256,768,2560};
        const size_t block_bytes=ggml_type_size(f.type);
        for(int n:widths) for(int offset:{0,1,3,7,15}) for(int padding:{0,1,13}) {
            const int rows=19;
            const size_t payload=ggml_row_size(f.type,n),stride=payload+padding;
            std::vector<uint8_t> data(rows*stride+offset+512);
            for(auto& b:data)b=rng();
            uint8_t* w=data.data()+offset;
            for(int r=0;r<rows;++r) for(size_t k=0;k<payload;k+=block_bytes)
                set_scale(f.type,w+r*stride+k,rng()&0xfbff);
            Activations act(f.type,n,rng,true);
            verify(f,w,stride,n,rows,act);
        }
    }
    strata::GgufFile file(argv[1]);
    for(int layer=0;layer<48;++layer) for(const char* role:{"gate","up","down"}) {
        const std::string name="blk."+std::to_string(layer)+".ffn_"+role+"_exps.weight";
        const strata::TensorInfo* tensor=nullptr;
        for(const auto& t:file.tensors())if(t.name==name)tensor=&t;
        require(tensor,"missing native expert tensor");
        Format* f=nullptr;for(auto& candidate:formats)if(tensor->type==candidate.type)f=&candidate;
        if(!f) { std::printf("{\"kind\":\"skip\",\"layer\":%d,\"role\":\"%s\",\"type\":%u}\n",layer,role,tensor->type);continue; }
        const int n=tensor->shape[0],rows=tensor->shape[1];
        const size_t stride=ggml_row_size(f->type,n),bytes=stride*rows;
        Activations act(f->type,n,rng,true);
        for(int e:{0,7,31}) { verify(*f,file.tensor_data(*tensor)+e*bytes,stride,n,rows,act);++f->real_cases; }
        if(f->streaming.empty()) {
            f->n=n;f->rows=rows;f->stride=stride;f->expert_bytes=bytes;f->role=role;
            f->streaming.resize(96*bytes);
            std::memcpy(f->streaming.data(),file.tensor_data(*tensor),f->streaming.size());
        }
    }
    for(auto& f:formats) {
        if(f.streaming.empty())continue;
        Activations act(f.type,f.n,rng,false);
        // Every weight used by timings is checked separately, including experts
        // beyond the real 0/7/31 validation samples.
        verify(f,f.streaming.data(),f.stride,f.n,96*f.rows,act);
        std::vector<float> output(f.rows);
        auto run=[&](Dot dot,int experts,int repeats) {
            const double start=milliseconds();
            for(int z=0;z<repeats;++z)for(int e=0;e<experts;++e)for(int r=0;r<f.rows;++r)
                dot(f.n,output.data()+r,0,f.streaming.data()+e*f.expert_bytes+r*f.stride,0,act.pointers[0],0,1);
            return milliseconds()-start;
        };
        for(int experts:{1,96}) {
            const int repeats=experts==1?128:2;
            run(f.reference,experts,1);run(f.alternate,experts,1);
            for(int round=0;round<9;++round) {
                double a,b;
                if(round%2==0) { a=run(f.reference,experts,repeats);b=run(f.alternate,experts,repeats); }
                else { b=run(f.alternate,experts,repeats);a=run(f.reference,experts,repeats); }
                std::printf("{\"kind\":\"timing\",\"type\":%d,\"nt\":1,\"role\":\"%s\",\"n\":%d,\"rows\":%d,\"streaming_bytes\":%zu,\"experts\":%d,\"repeats\":%d,\"round\":%d,\"original_first\":%s,\"original_ms\":%.9f,\"gcc_ms\":%.9f,\"speed_ratio\":%.9f}\n",f.type,f.role.c_str(),f.n,f.rows,f.streaming.size(),experts,repeats,round,round%2==0?"true":"false",a,b,a/b);
            }
        }
    }
    for(const auto& f:formats)
        std::printf("{\"kind\":\"validation\",\"type\":%d,\"checked_floats\":%llu,\"real_expert_cases\":%d,\"mismatched_floats\":%llu,\"nonfinite_floats\":%llu,\"all_bits_equal\":%s}\n",f.type,(unsigned long long)f.checked,f.real_cases,(unsigned long long)f.mismatched,(unsigned long long)f.nonfinite,f.mismatched==0 && f.nonfinite==0?"true":"false");
    std::printf("{\"kind\":\"completed\",\"cpu\":2,\"nt\":1}\n");
}
