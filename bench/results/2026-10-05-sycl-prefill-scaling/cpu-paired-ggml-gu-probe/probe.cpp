#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sched.h>
#include <string>
#include <vector>

namespace cpu=strata::kernels::cpu;
using Pair=void(*)(int,float*,float*,const void*,const void*,const void*);
#define DECL(PREFIX, NAME) extern "C" void PREFIX##_paired_##NAME(int,float*,float*,const void*,const void*,const void*);
DECL(icx,iq3_xxs) DECL(icx,iq3_s) DECL(icx,iq2_s)
DECL(gcc,iq3_xxs) DECL(gcc,iq3_s) DECL(gcc,iq2_s)
void paired_icx_rows(const cpu::NativeFmt&,const uint8_t*,const void*,float*,int,int);
void paired_gcc_rows(const cpu::NativeFmt&,const uint8_t*,const void*,float*,int,int);
using Rows=void(*)(const cpu::NativeFmt&,const uint8_t*,const void*,float*,int,int);
struct Format {
    int type;Pair pairs[2];
    uint64_t raw_checked=0,ff_checked=0;
    int real_cases=0;
    cpu::NativeFmt fmt;
    std::vector<uint8_t> streaming;
};
static Format formats[]={
    {18,{icx_paired_iq3_xxs,gcc_paired_iq3_xxs}},
    {21,{icx_paired_iq3_s,gcc_paired_iq3_s}},
    {22,{icx_paired_iq2_s,gcc_paired_iq2_s}},
};
static Rows candidates[]={paired_icx_rows,paired_gcc_rows};
static const char* arms[]={"icx","gcc"};
static void require(bool ok,const char* why) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",why);std::exit(1);}
}
static void equal(const float* a,const float* b,size_t n,int type,const char* stage) {
    for(size_t i=0;i<n;++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]),"non-finite output");
        if(std::memcmp(a+i,b+i,4)) {
            uint32_t ab,bb;std::memcpy(&ab,a+i,4);std::memcpy(&bb,b+i,4);
            std::fprintf(stderr,"type=%d stage=%s index=%zu original=%08x candidate=%08x\n",type,stage,i,ab,bb);
            require(false,"float bits differ");
        }
    }
}
struct Activations {
    std::vector<uint8_t> bytes;
    const void* pointers[8];
    Activations(int type,int n,std::mt19937& rng,bool edges) {
        const auto* traits=ggml_get_type_traits_cpu((ggml_type)type);
        const auto* quant=ggml_get_type_traits_cpu(traits->vec_dot_type);
        const size_t size=ggml_row_size(traits->vec_dot_type,n);
        bytes.resize(8*size);std::vector<float> x(n);
        std::normal_distribution<float> normal(0.f,.5f);
        for(int t=0;t<8;++t) {
            for(int i=0;i<n;++i)x[i]=edges && t==1?0.f:edges && t==2?.125f:edges && t==3?(i%2?-.25f:.25f):normal(rng);
            pointers[t]=bytes.data()+t*size;
            quant->from_float(x.data(),bytes.data()+t*size,n);
        }
    }
};
static cpu::NativeFmt geometry(int type,int n,int rows,size_t stride) {
    cpu::NativeFmt f{};f.gu_type=type;f.n_embd=n;f.n_ff=rows;f.gu_row=stride;f.up_off=stride*rows;return f;
}
static void verify(Format& f,const cpu::NativeFmt& fmt,const uint8_t* blob,const Activations& act) {
    const int n=fmt.n_embd,rows=fmt.n_ff;
    auto dot=ggml_get_type_traits_cpu((ggml_type)f.type)->vec_dot;
    for(int t=0;t<8;++t) {
        for(int r=0;r<rows;++r) {
            float g[3]={.125f,.125f,.125f},u[3]={.125f,.125f,.125f};
            const auto* gr=blob+r*fmt.gu_row;const auto* ur=blob+fmt.up_off+r*fmt.gu_row;
            dot(n,g+1,0,gr,0,act.pointers[t],0,1);dot(n,u+1,0,ur,0,act.pointers[t],0,1);
            for(int arm=0;arm<2;++arm) {
                float cg[3]={.125f,.125f,.125f},cu[3]={.125f,.125f,.125f};
                f.pairs[arm](n,cg+1,cu+1,gr,ur,act.pointers[t]);
                equal(g,cg,3,f.type,arms[arm]);equal(u,cu,3,f.type,arms[arm]);
                require(g[0]==.125f && g[2]==.125f && u[0]==.125f && u[2]==.125f,"original raw guard overwritten");
                f.raw_checked+=6;
            }
        }
        const void* acts[]={act.pointers[t]};
        std::vector<float> a(rows+2,.125f),b(a);float* out[]={a.data()+1};
        cpu::native_gu_rows(fmt,blob,acts,1,out,2,rows-1);
        for(int arm=0;arm<2;++arm) {
            b.assign(rows+2,.125f);candidates[arm](fmt,blob,act.pointers[t],b.data()+1,2,rows-1);
            equal(a.data(),b.data(),a.size(),f.type,arms[arm]);f.ff_checked+=a.size();
        }
        for(int r=0;r<rows+2;++r)if(r<3 || r>=rows)require(a[r]==.125f,"original partial GU guard overwritten");
    }
}
static double milliseconds() {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
int main(int argc,char** argv) {
    require(argc==2,"provide real GGUF shard");std::setvbuf(stdout,nullptr,_IONBF,0);
    cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(2,&cpus);require(sched_setaffinity(0,sizeof(cpus),&cpus)==0,"pin CPU 2");
    ggml_cpu_init();std::mt19937 rng(2019);
    for(auto& f:formats)for(int n:{256,768,2560})for(int offset:{0,1,3,7,15})for(int padding:{0,1,13}) {
        const int rows=19;const size_t block=ggml_type_size((ggml_type)f.type),payload=ggml_row_size((ggml_type)f.type,n),stride=payload+padding;
        auto fmt=geometry(f.type,n,rows,stride);
        std::vector<uint8_t> data(2*fmt.up_off+offset+512);
        for(auto& b:data)b=rng();uint8_t* blob=data.data()+offset;
        for(int s=0;s<2;++s)for(int r=0;r<rows;++r)for(size_t k=0;k<payload;k+=block) {
            uint16_t half=rng()&0xfbff;std::memcpy(blob+s*fmt.up_off+r*stride+k,&half,2);
        }
        Activations act(f.type,n,rng,true);verify(f,fmt,blob,act);
    }
    strata::GgufFile file(argv[1]);
    for(int layer=0;layer<48;++layer) {
        const strata::TensorInfo *gate=nullptr,*up=nullptr;
        for(const auto& t:file.tensors()) {
            if(t.name=="blk."+std::to_string(layer)+".ffn_gate_exps.weight")gate=&t;
            if(t.name=="blk."+std::to_string(layer)+".ffn_up_exps.weight")up=&t;
        }
        require(gate && up && gate->type==up->type,"missing gate/up tensor");
        Format* f=nullptr;for(auto& candidate:formats)if((uint32_t)candidate.type==gate->type)f=&candidate;
        if(!f){std::printf("{\"kind\":\"skip\",\"layer\":%d,\"type\":%u}\n",layer,gate->type);continue;}
        const int n=gate->shape[0],rows=gate->shape[1];
        auto fmt=geometry(f->type,n,rows,ggml_row_size((ggml_type)f->type,n));
        Activations act(f->type,n,rng,true);
        for(int e:{0,7,31}) {
            std::vector<uint8_t> data(2*fmt.up_off);
            std::memcpy(data.data(),file.tensor_data(*gate)+e*fmt.up_off,fmt.up_off);
            std::memcpy(data.data()+fmt.up_off,file.tensor_data(*up)+e*fmt.up_off,fmt.up_off);
            verify(*f,fmt,data.data(),act);++f->real_cases;
        }
        if(f->streaming.empty()) {
            f->fmt=fmt;f->streaming.resize(96*2*fmt.up_off);
            for(int e=0;e<96;++e) {
                std::memcpy(f->streaming.data()+e*2*fmt.up_off,file.tensor_data(*gate)+e*fmt.up_off,fmt.up_off);
                std::memcpy(f->streaming.data()+e*2*fmt.up_off+fmt.up_off,file.tensor_data(*up)+e*fmt.up_off,fmt.up_off);
            }
        }
    }
    for(auto& f:formats) {
        const auto& fmt=f.fmt;require(!f.streaming.empty(),"no timed dataset");
        Activations act(f.type,fmt.n_embd,rng,false);
        for(int e=0;e<96;++e)verify(f,fmt,f.streaming.data()+e*2*fmt.up_off,act);
        std::vector<float> output(fmt.n_ff);const void* acts[]={act.pointers[0]};float* outs[]={output.data()};
        auto run=[&](int arm,int experts,int repeats) {
            const double start=milliseconds();
            for(int z=0;z<repeats;++z)for(int e=0;e<experts;++e) {
                const uint8_t* blob=f.streaming.data()+e*2*fmt.up_off;
                if(arm<0)cpu::native_gu_rows(fmt,blob,acts,1,outs,0,fmt.n_ff);
                else candidates[arm](fmt,blob,act.pointers[0],output.data(),0,fmt.n_ff);
            }
            return milliseconds()-start;
        };
        for(int arm=0;arm<2;++arm)for(int experts:{1,96}) {
            const int repeats=experts==1?128:2;run(-1,experts,1);run(arm,experts,1);
            for(int round=0;round<9;++round) {
                double a,b;
                if(round%2==0){a=run(-1,experts,repeats);b=run(arm,experts,repeats);}
                else{b=run(arm,experts,repeats);a=run(-1,experts,repeats);}
                std::printf("{\"kind\":\"timing\",\"type\":%d,\"arm\":\"%s\",\"nt\":1,\"experts\":%d,\"repeats\":%d,\"round\":%d,\"original_first\":%s,\"original_ms\":%.9f,\"candidate_ms\":%.9f,\"speed_ratio\":%.9f,\"streaming_bytes\":%zu}\n",f.type,arms[arm],experts,repeats,round,round%2==0?"true":"false",a,b,a/b,f.streaming.size());
            }
        }
        std::printf("{\"kind\":\"validation\",\"type\":%d,\"real_expert_cases\":%d,\"raw_checked_floats\":%llu,\"ff_checked_floats\":%llu,\"all_bits_equal\":true}\n",f.type,f.real_cases,(unsigned long long)f.raw_checked,(unsigned long long)f.ff_checked);
    }
    std::printf("{\"kind\":\"completed\",\"cpu\":2,\"nt\":1,\"all_bits_equal\":true}\n");
}
