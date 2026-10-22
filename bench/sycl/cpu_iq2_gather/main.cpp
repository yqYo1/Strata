// Main-authored, CPU-only IQ2_S NT1 GU instruction discriminator; repository LICENSE.
// Input/canary pattern follows the maintained cpu_iq2 fixture. No XeStrata code.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/iq2_pair_avx2.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
namespace cpu=strata::kernels::cpu;
namespace strata::kernels::cpu {bool iq2_gather_gu_rows(const NativeFmt&,const uint8_t*,const void*,float*,int,int);}
using Clock=std::chrono::steady_clock;
constexpr int H=2560,FF=640,experts=168,guard=64;
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
double cpu_time(){rusage r{};need(!getrusage(RUSAGE_SELF,&r),"rusage");return r.ru_utime.tv_sec+r.ru_utime.tv_usec*1e-6+r.ru_stime.tv_sec+r.ru_stime.tv_usec*1e-6;}
uint64_t hash(const uint8_t* p,size_t n){uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;}
struct Buffer {
    std::vector<uint8_t> raw;size_t n;
    explicit Buffer(size_t bytes):raw(bytes+2*guard,0xa7),n(bytes){}
    uint8_t* data(){return raw.data()+guard;}
    void check()const{for(int i=0;i<guard;++i)need(raw[i]==0xa7&&raw[guard+n+i]==0xa7,"canary");}
};
using Producer=bool(*)(const cpu::NativeFmt&,const uint8_t*,const void*,float*,int,int);
struct Inputs {
    cpu::NativeFmt f;std::vector<Buffer> weights,outputs;std::vector<uint64_t> hashes;
    std::array<Buffer,4> acts{Buffer(4096),Buffer(4096),Buffer(4096),Buffer(4096)};
    std::array<uint64_t,4> act_hash{};std::vector<int> order;
    Inputs(const std::string& shard,int down,uint64_t go,uint64_t up,uint64_t dn,size_t bytes){
        std::string err;need(cpu::native_fmt(22,down,H,FF,f,err),err.c_str());need(f.bytes==bytes,"format");
        std::ifstream file(shard,std::ios::binary|std::ios::ate);need(bool(file),"model open");const auto size=file.tellg();need(size>0,"model size");
        auto read=[&](uint8_t* dst,uint64_t offset,size_t n){need(offset<=uint64_t(size)&&n<=uint64_t(size)-offset,"model bounds");file.seekg(offset);file.read(reinterpret_cast<char*>(dst),n);need(bool(file),"model read");};
        for(int e=0;e<experts;++e){weights.emplace_back(f.bytes);outputs.emplace_back(FF*sizeof(float));auto* p=weights.back().data();read(p,go+e*f.up_off,f.up_off);read(p+f.up_off,up+e*f.up_off,f.up_off);read(p+f.down_off,dn+e*(f.bytes-f.down_off),f.bytes-f.down_off);hashes.push_back(hash(p,f.bytes));}
        uint32_t seed=20261022;for(int t=0;t<4;++t){std::array<float,H> x;for(auto& v:x){seed=seed*1664525u+1013904223u;v=(int(seed>>8)-8388608)*0x1p-23f;}cpu::native_quant_act(f,x.data(),acts[t].data());act_hash[t]=hash(acts[t].data(),acts[t].n);}
        order.resize(experts);std::iota(order.begin(),order.end(),0);std::mt19937 random(20261022);std::shuffle(order.begin(),order.end(),random);
    }
    void guards(){for(int e=0;e<experts;++e){weights[e].check();outputs[e].check();need(hash(weights[e].data(),f.bytes)==hashes[e],"weights changed");}for(int t=0;t<4;++t){acts[t].check();need(hash(acts[t].data(),acts[t].n)==act_hash[t],"activations changed");}}
    void run(Producer producer,int access){const int n=access?experts:6;for(int i=0;i<n;++i){const int e=order[i];need(producer(f,weights[e].data(),acts[e%4].data(),reinterpret_cast<float*>(outputs[e].data()),0,FF),"producer declined");}}
    void validate(Producer producer,std::ofstream* capture){
        for(int e=0;e<experts;++e){std::array<float,FF> reference;need(cpu::iq256_single_gu_rows(22,weights[e].data(),f.gu_row,f.up_off,H,acts[e%4].data(),reference.data(),0,FF),"independent reference");need(producer(f,weights[e].data(),acts[e%4].data(),reinterpret_cast<float*>(outputs[e].data()),0,FF),"validation producer");need(!std::memcmp(reference.data(),outputs[e].data(),sizeof reference),"GU output bits");for(float v:reference)need(std::isfinite(v),"finite output");if(capture)capture->write(reinterpret_cast<char*>(outputs[e].data()),outputs[e].n);}
        guards();need(!capture||bool(*capture),"capture write");
    }
    void contracts(Producer producer){
        for(auto bounds:{std::pair{0,FF},std::pair{3,FF-3},std::pair{FF-1,FF},std::pair{FF/2,FF/2}}){std::array<float,FF+2> ref,out;ref.fill(123);out=ref;need(cpu::iq256_single_gu_rows(22,weights[0].data(),f.gu_row,f.up_off,H,acts[0].data(),ref.data()+1,bounds.first,bounds.second),"row reference");need(producer(f,weights[0].data(),acts[0].data(),out.data()+1,bounds.first,bounds.second)&&!std::memcmp(ref.data(),out.data(),sizeof out),"row/guard contract");}
        std::array<float,FF> out;out.fill(123);const auto original=out;auto bad=f;bad.gu_type=18;
        need(!producer(bad,weights[0].data(),acts[0].data(),out.data(),0,1)&&out==original,"bad type refusal");
        need(!producer(f,nullptr,acts[0].data(),out.data(),0,1)&&out==original,"null refusal");
        need(!producer(f,weights[0].data(),acts[0].data(),out.data(),0,641)&&out==original,"row extent refusal");
        need(!producer(f,weights[0].data()+1,acts[0].data(),out.data(),0,1)&&out==original,"alignment refusal");
    }
};
void timing(Inputs& in,Producer producer,int arm){
    for(int access=0;access<2;++access){
        for(int i=0;i<8;++i)in.run(producer,access);
        const auto warm_start=Clock::now();in.run(producer,access);const double warm=elapsed(warm_start);auto start=Clock::now();
        for(int i=0;i<10000;++i){auto now=Clock::now();asm volatile(""::"r"(&now):"memory");}const double timer=elapsed(start)/10000;need(timer<warm*.001,"timer overhead");
        for(int rep=0;rep<5;++rep){const double cpu_start=cpu_time();start=Clock::now();unsigned passes=0;double wall=0;
            do{in.run(producer,access);++passes;wall=elapsed(start);}while(wall<1.2&&passes<1000000);
            const double scheduled=cpu_time()-cpu_start;need(wall>=1&&passes<1000000,"sample floor");const uint64_t jobs=uint64_t(passes)*(access?experts:6);
            std::cout<<"{\"kind\":\"sample\",\"arm\":"<<arm<<",\"down\":"<<in.f.d_type<<",\"access\":"<<access<<",\"rep\":"<<rep<<",\"passes\":"<<passes<<",\"jobs\":"<<jobs<<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<scheduled<<",\"timer_s\":"<<timer<<",\"warm_s\":"<<warm<<"}"<<std::endl;
        }
    }
}
int main(int argc,char** argv){try{
    std::cout<<std::setprecision(17);need(argc==5,"mode fixture capture arm");const std::string mode=argv[1];need(mode=="check"||mode=="measure","mode");const int arm=std::stoi(argv[4]);need(arm==0||arm==1,"arm");Producer selected=arm?cpu::iq2_gather_gu_rows:cpu::iq2_pair_gu_rows;
    const auto cores=cpu::physical_cores(false);need(cores.size()==6,"physical cores");need(cpu::pin_current_thread(cores[0])>0,"pin compute thread");need(sched_getcpu()==cores[0],"actual CPU");
    std::cout<<"{\"kind\":\"placement\",\"cpu\":"<<cores[0]<<",\"computing_threads\":1,\"arm\":"<<arm<<"}"<<std::endl;
    std::ifstream spec(argv[2]);need(bool(spec),"fixture");std::string shard;int down,count=0;uint64_t go,up,dn,bytes;std::ofstream capture(argv[3],std::ios::binary);need(bool(capture),"capture open");
    while(spec>>shard>>down>>go>>up>>dn>>bytes){need(down==(count==0?20:42)&&count<2,"format sequence");Inputs in(shard,down,go,up,dn,bytes);in.contracts(selected);in.validate(selected,nullptr);if(mode=="measure")timing(in,selected,arm);in.validate(selected,&capture);need(sched_getcpu()==cores[0],"CPU changed");++count;}
    need(spec.eof()&&count==2,"fixture complete");capture.close();need(bool(capture),"capture close");std::cout<<"{\"kind\":\"closed\",\"arm\":"<<arm<<",\"formats\":2,\"experts_per_format\":168,\"bitwise\":true,\"all_guards\":true,\"inputs_unchanged\":true,\"passed\":true}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
