// Main-authored IQ3_S production GU component check; repository LICENSE.
// References: existing native_expert/iq_single producers and linked ggml.
// No XeStrata implementation, helper or test input.
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/iq3_pair_avx2.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sched.h>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace cpu=strata::kernels::cpu;
using Clock=std::chrono::steady_clock;
constexpr int H=2560,FF=640,experts=64;
void need(bool b,const char* message){if(!b)throw std::runtime_error(message);}
uint64_t hash(const void* data,size_t n){auto* p=static_cast<const uint8_t*>(data);uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;}
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double cpu_seconds(){rusage r{};need(!getrusage(RUSAGE_SELF,&r),"rusage");return r.ru_utime.tv_sec+r.ru_utime.tv_usec*1e-6+r.ru_stime.tv_sec+r.ru_stime.tv_usec*1e-6;}
struct Bytes{
    std::vector<uint8_t> raw;size_t n;
    explicit Bytes(size_t bytes):raw(bytes+128,0xa7),n(bytes){}
    uint8_t* data(){return raw.data()+64;}
    void check()const{for(int i=0;i<64;++i)need(raw[i]==0xa7&&raw[64+n+i]==0xa7,"weight canary");}
};
struct Activation{
    std::array<uint8_t,64> before{};
    alignas(64) std::array<block_q8_K,10> blocks{};
    std::array<uint8_t,64> after{};
    Activation(){before.fill(0xa7);after.fill(0xa7);}
    void check()const{for(int i=0;i<64;++i)need(before[i]==0xa7&&after[i]==0xa7,"activation canary");}
};
using Output=std::array<float,FF+2>;
struct Fd{int fd;explicit Fd(const std::string& p):fd(open(p.c_str(),O_RDONLY|O_CLOEXEC)){need(fd>=0,"model open");}~Fd(){close(fd);}Fd(const Fd&)=delete;};
void read_exact(int fd,uint8_t* p,size_t n,uint64_t offset){
    struct stat s{};need(!fstat(fd,&s)&&s.st_size>=0&&offset<=uint64_t(s.st_size)&&n<=uint64_t(s.st_size)-offset,"model bounds");
    while(n){auto got=pread(fd,p,n,offset);need(got>0,"model read");p+=got;n-=got;offset+=got;}
}
void audit_delta(cpu::Iq3PairAudit before,uint64_t calls,uint64_t rows){auto now=cpu::iq3_pair_audit_snapshot();need(before.enabled&&now.enabled&&now.calls-before.calls==calls&&now.rows-before.rows==rows,"actual selected dispatch");}
void reference(const cpu::NativeFmt& f,const uint8_t* blob,const void* act,Output& out,int a=0,int b=FF){
    const void* ap[]={act};float* fp[]={out.data()+1};cpu::native_gu_rows(f,blob,ap,1,fp,a,b,cpu::NativeGuPhase::Prefill);
}
void same(const Output& a,const Output& b){need(!std::memcmp(a.data(),b.data(),sizeof a),"GU output bits or row guard");for(float v:b)need(std::isfinite(v),"nonfinite GU output");}
struct Inputs{
    cpu::NativeFmt f;std::vector<Bytes> blobs;std::vector<Output> out;std::vector<uint64_t> weight_hash;
    std::array<Activation,4> act;std::array<uint64_t,4> act_hash{};
    Inputs(const std::string& path,int down,uint64_t gate,uint64_t up,uint64_t dn,size_t bytes){
        std::string error;need(cpu::native_fmt(21,down,H,FF,f,error),error.c_str());need(f.bytes==bytes&&f.gu_row==1100&&f.act_bytes==sizeof act[0].blocks,"format");Fd file(path);
        for(int e=0;e<experts;++e){blobs.emplace_back(f.bytes);out.emplace_back();auto* p=blobs.back().data();read_exact(file.fd,p,f.up_off,gate+e*f.up_off);read_exact(file.fd,p+f.up_off,f.up_off,up+e*f.up_off);const size_t d=f.bytes-f.down_off;read_exact(file.fd,p+f.down_off,d,dn+e*d);weight_hash.push_back(hash(p,f.bytes));}
        uint32_t seed=20261103;std::array<float,H> x{};
        for(int t=0;t<4;++t){for(int j=0;j<H;++j){seed=1664525u*seed+1013904223u;x[j]=t==0?0.f:t==2?(j%2?1.f:-1.f):t==3?(j%2?64.f:-.03125f):(int(seed>>8)-8388608)*0x1p-23f;}cpu::native_quant_act(f,x.data(),act[t].blocks.data());act_hash[t]=hash(act[t].blocks.data(),sizeof act[t].blocks);}
    }
    void check(){for(int e=0;e<experts;++e){blobs[e].check();need(hash(blobs[e].data(),f.bytes)==weight_hash[e],"weight mutation");}for(int t=0;t<4;++t){act[t].check();need(hash(act[t].blocks.data(),sizeof act[t].blocks)==act_hash[t],"activation mutation");}}
};
unsigned guards(Inputs& in){
    const auto f=in.f;auto* w=in.blobs[0].data();auto* a=in.act[1].blocks.data();Output expected{},out{};unsigned cases=0;
    for(auto bounds:{std::pair{0,FF},std::pair{3,FF-3},std::pair{FF-1,FF},std::pair{FF/2,FF/2}}){expected.fill(123);out=expected;reference(f,w,a,expected,bounds.first,bounds.second);auto before=cpu::iq3_pair_audit_snapshot();need(cpu::iq3_pair_gu_rows(f,w,a,out.data()+1,bounds.first,bounds.second),"producer unavailable");audit_delta(before,0,0);same(expected,out);++cases;}
    auto reject=[&](const cpu::NativeFmt& fmt,const uint8_t* weight,const void* activation,float* result,int first=0,int last=1){out.fill(123);const auto saved=out;need(!cpu::iq3_pair_gu_rows(fmt,weight,activation,result,first,last),"unsupported producer accepted");same(saved,out);++cases;};
    auto bad=[&](auto change){auto v=f;change(v);reject(v,w,a,out.data()+1);};
    bad([](auto& v){v.gu_type=22;});bad([](auto& v){v.gu_act=8;});bad([](auto& v){v.n_embd=2304;});bad([](auto& v){v.n_ff=639;});bad([](auto& v){--v.act_bytes;});bad([](auto& v){v.gu_row=1099;});bad([](auto& v){++v.gu_row;});bad([](auto& v){v.gu_row=std::numeric_limits<size_t>::max();});bad([](auto& v){v.up_off=0;});bad([](auto& v){++v.up_off;});bad([](auto& v){v.up_off=v.bytes+2;});bad([](auto& v){v.bytes=v.down_off-1;});bad([](auto& v){--v.down_off;});
    for(auto bounds:{std::pair{-1,1},std::pair{0,FF+1},std::pair{2,1}})reject(f,w,a,out.data()+1,bounds.first,bounds.second);
    reject(f,nullptr,a,out.data()+1);reject(f,w,nullptr,out.data()+1);reject(f,w,a,nullptr);reject(f,w+1,a,out.data()+1);reject(f,w,reinterpret_cast<uint8_t*>(a)+1,out.data()+1);reject(f,w,a,reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(out.data()+1)+1));
    reject(f,w,a,reinterpret_cast<float*>(w));reject(f,w,a,reinterpret_cast<float*>(a));reject(f,w,w,out.data()+1);
    const auto end=std::numeric_limits<uintptr_t>::max()&~uintptr_t(63);reject(f,reinterpret_cast<const uint8_t*>(end),a,out.data()+1);reject(f,w,reinterpret_cast<const void*>(end),out.data()+1);reject(f,w,a,reinterpret_cast<float*>(end));
    auto padded=f;padded.gu_row+=4;padded.up_off=padded.gu_row*FF+16;padded.down_off=padded.up_off+padded.gu_row*FF+16;padded.bytes=padded.down_off+(f.bytes-f.down_off);Bytes pw(padded.bytes);
    for(int r=0;r<FF;++r){std::memcpy(pw.data()+r*padded.gu_row,w+r*f.gu_row,f.gu_row);std::memcpy(pw.data()+padded.up_off+r*padded.gu_row,w+f.up_off+r*f.gu_row,f.gu_row);}
    expected.fill(123);out=expected;reference(padded,pw.data(),a,expected);need(cpu::iq3_pair_gu_rows(padded,pw.data(),a,out.data()+1,0,FF),"padded producer");same(expected,out);pw.check();++cases;
    // All grid/high-bit/sign/scale patterns, including finite F16 boundaries.
    Bytes synthetic(f.bytes);uint32_t seed=20261103;const uint16_t scales[]={0,1,0x0400,0x1800,0x3000,0x3c00,0x7bff};
    for(int role=0;role<2;++role)for(int r=0;r<FF;++r)for(int b=0;b<10;++b){block_iq3_s block{};const auto d=scales[(r+b+role)%7];std::memcpy(&block.d,&d,2);for(auto& q:block.qs){seed=1664525u*seed+1013904223u;q=seed>>24;}for(auto& q:block.qh){seed=1664525u*seed+1013904223u;q=seed>>24;}for(auto& q:block.signs)q=r%4==0?0:r%4==1?255:r%4==2?0xaa:0x55;for(auto& q:block.scales)q=r%4==0?0:r%4==1?255:r%4==2?0xf0:0x0f;std::memcpy(synthetic.data()+role*f.up_off+r*f.gu_row+b*sizeof block,&block,sizeof block);}
    for(int t=0;t<4;++t){expected.fill(123);out=expected;reference(f,synthetic.data(),in.act[t].blocks.data(),expected);need(cpu::iq3_pair_gu_rows(f,synthetic.data(),in.act[t].blocks.data(),out.data()+1,0,FF),"synthetic producer");same(expected,out);++cases;}
    synthetic.check();in.check();return cases;
}
void validate(Inputs& in,bool selected){
    auto before=cpu::iq3_pair_audit_snapshot();
    for(int e=0;e<experts;++e)for(int t=0;t<4;++t){Output expected;expected.fill(123);in.out[e]=expected;const void* ap[]={in.act[t].blocks.data()};float* fp[]={in.out[e].data()+1};reference(in.f,in.blobs[e].data(),ap[0],expected);cpu::native_gu_rows(in.f,in.blobs[e].data(),ap,1,fp,0,FF,cpu::NativeGuPhase::Decode);same(expected,in.out[e]);}
    audit_delta(before,selected?experts*4:0,selected?experts*4*FF:0);
    // Both explicit phases and NT2/NT4 must leave the new selected counter alone.
    before=cpu::iq3_pair_audit_snapshot();
    for(int nt:{2,4}){std::array<Output,4> ref{},out{};const void* ap[4];float* rp[4];float* op[4];for(int t=0;t<4;++t){ref[t].fill(123);out[t]=ref[t];ap[t]=in.act[t].blocks.data();rp[t]=ref[t].data()+1;op[t]=out[t].data()+1;}cpu::native_gu_rows(in.f,in.blobs[0].data(),ap,nt,rp,0,FF,cpu::NativeGuPhase::Prefill);cpu::native_gu_rows(in.f,in.blobs[0].data(),ap,nt,op,0,FF,cpu::NativeGuPhase::Decode);for(int t=0;t<4;++t)same(ref[t],out[t]);}
    for(int t=0;t<4;++t){Output out;out.fill(123);reference(in.f,in.blobs[0].data(),in.act[t].blocks.data(),out);}
    audit_delta(before,0,0);in.check();
}
void timing(Inputs& in,bool selected,int arm,int process){
    auto pass=[&](bool stream){for(int e=0;e<(stream?experts:1);++e){const void* ap[]={in.act[1].blocks.data()};float* fp[]={in.out[e].data()+1};cpu::native_gu_rows(in.f,in.blobs[e].data(),ap,1,fp,0,FF,cpu::NativeGuPhase::Decode);}};
    for(int access=0;access<2;++access){auto start=Clock::now();pass(access);const double warm=elapsed(start);start=Clock::now();for(int i=0;i<10000;++i){auto now=Clock::now();asm volatile(""::"r"(&now):"memory");}const double timer=elapsed(start)/10000;need(timer<warm*.001,"clock overhead");
        for(int rep=0;rep<6;++rep){const auto before=cpu::iq3_pair_audit_snapshot();const double c0=cpu_seconds();start=Clock::now();uint64_t passes=0;double wall=0;do{pass(access);++passes;wall=elapsed(start);}while(wall<.35&&passes<1000000);const double c=cpu_seconds()-c0;need(wall>=.25&&passes<1000000,"timing floor or pass bound");const uint64_t calls=passes*(access?experts:1);audit_delta(before,selected?calls:0,selected?calls*FF:0);
            std::cout<<"{\"kind\":\"sample\",\"process\":"<<process<<",\"arm\":"<<arm<<",\"down\":"<<in.f.d_type<<",\"access\":"<<access<<",\"rep\":"<<rep<<",\"passes\":"<<passes<<",\"calls\":"<<calls<<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<c<<",\"timer_s\":"<<timer<<",\"warm_s\":"<<warm<<",\"selected_calls\":"<<(selected?calls:0)<<",\"selected_rows\":"<<(selected?calls*FF:0)<<"}"<<std::endl;
        }
    }
}
int main(int argc,char** argv){
    try{
        need(argc==7,"mode fixture arm process cpu blocked");const std::string mode=argv[1];need(mode=="check"||mode=="perf","mode");const int arm=std::stoi(argv[3]),process=std::stoi(argv[4]),core=std::stoi(argv[5]),blocked=std::stoi(argv[6]);need((arm==0||arm==1)&&(blocked==0||blocked==1),"arm");
        const char* flag=std::getenv("STRATA_IQ3_PAIR");need(flag&&std::string(flag)==std::to_string(arm),"startup selector");need((std::getenv("STRATA_NO_IQ256")!=nullptr)==bool(blocked),"startup fallback");need(std::getenv("STRATA_IQ_MT_MIN")&&std::string(std::getenv("STRATA_IQ_MT_MIN"))=="2","MT startup");need(std::getenv("STRATA_IQ_SINGLE_EXACT")&&std::string(std::getenv("STRATA_IQ_SINGLE_EXACT"))=="0","single startup");
        need(core>=0&&core<CPU_SETSIZE,"core bound");cpu_set_t affinity;CPU_ZERO(&affinity);CPU_SET(core,&affinity);need(!sched_setaffinity(0,sizeof affinity,&affinity),"pin core");CPU_ZERO(&affinity);need(!sched_getaffinity(0,sizeof affinity,&affinity)&&CPU_COUNT(&affinity)==1&&CPU_ISSET(core,&affinity)&&sched_getcpu()==core,"actual placement");need(cpu::iq3_pair_audit_snapshot().enabled,"audit startup");
        std::cout<<std::setprecision(17);std::ifstream spec(argv[2]);need(bool(spec),"fixture open");std::string shard;int down,n;uint64_t gate,up,dn,bytes;int forms=0;unsigned contracts=0;const bool selected=arm&&!blocked;
        while(spec>>shard>>down>>gate>>up>>dn>>bytes>>n){need(n==experts&&((forms==0&&down==20)||(forms==1&&down==42)),"fixture fields");Inputs in(shard,down,gate,up,dn,bytes);validate(in,selected);if(mode=="check")contracts+=guards(in);if(mode=="perf")timing(in,selected,arm,process);validate(in,selected);std::cout<<"{\"kind\":\"validation\",\"down\":"<<down<<",\"experts\":64,\"activations\":4,\"bitwise\":true,\"phase_isolated\":true,\"input_hash\":"<<hash(in.blobs[0].data(),in.f.bytes)<<",\"core\":"<<core<<",\"passed\":true}"<<std::endl;++forms;}
        need(spec.eof()&&forms==2,"fixture extent");std::cout<<"{\"kind\":\"closed\",\"passed\":true,\"contract_cases\":"<<contracts<<"}"<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
