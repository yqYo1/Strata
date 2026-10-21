// Main-owned production-pool IQ2_S NT1 comparison, repository MIT license.
// Math comes only from linked Strata/ggml producers. No copied XeStrata code.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq2_pair_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace cpu=strata::kernels::cpu;
using Clock=std::chrono::steady_clock;
constexpr int H=2560, FF=640, count=168, guard=64;
void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double cpu_seconds(){rusage r{};need(!getrusage(RUSAGE_SELF,&r),"rusage");return r.ru_utime.tv_sec+r.ru_utime.tv_usec*1e-6+r.ru_stime.tv_sec+r.ru_stime.tv_usec*1e-6;}
uint64_t hash(const uint8_t* p,size_t n){uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;}
struct Buffer{
    std::vector<uint8_t> raw;size_t n;
    explicit Buffer(size_t bytes):raw(bytes+2*guard,0xa7),n(bytes){}
    uint8_t* data(){return raw.data()+guard;}
    void check()const{for(int i=0;i<guard;++i)need(raw[i]==0xa7&&raw[guard+n+i]==0xa7,"canary");}
    void poison(){std::fill(raw.begin(),raw.end(),0xa7);}
};
void read_exact(int fd,uint8_t* p,size_t n,uint64_t off){
    struct stat s{};need(!fstat(fd,&s)&&off<=uint64_t(s.st_size)&&n<=uint64_t(s.st_size)-off,"model extent");
    while(n){auto k=pread(fd,p,n,off);need(k>0,"model read");p+=k;n-=k;off+=k;}
}
struct Inputs{
    cpu::NativeFmt f;std::vector<Buffer> blobs,output;std::vector<uint64_t> hashes;
    std::array<Buffer,4> acts{Buffer(4096),Buffer(4096),Buffer(4096),Buffer(4096)};
    std::array<std::array<float,H>,4> x{};std::array<uint64_t,4> act_hash{};
    Inputs(const std::string& shard,int down,uint64_t go,uint64_t up,uint64_t dn,size_t bytes){
        std::string err;need(cpu::native_fmt(22,down,H,FF,f,err),err.c_str());need(f.bytes==bytes,"model format");
        int fd=open(shard.c_str(),O_RDONLY|O_CLOEXEC);need(fd>=0,"model open");
        for(int e=0;e<count;++e){
            blobs.emplace_back(f.bytes);output.emplace_back(4*H*sizeof(float));auto* p=blobs.back().data();
            read_exact(fd,p,f.up_off,go+e*f.up_off);read_exact(fd,p+f.up_off,f.up_off,up+e*f.up_off);
            const size_t dbytes=f.bytes-f.down_off;read_exact(fd,p+f.down_off,dbytes,dn+e*dbytes);hashes.push_back(hash(p,f.bytes));
        }
        close(fd);uint32_t seed=20261021;
        for(int t=0;t<4;++t){for(float& v:x[t]){seed=1664525u*seed+1013904223u;v=(int(seed>>8)-8388608)*0x1p-23f;}cpu::native_quant_act(f,x[t].data(),acts[t].data());act_hash[t]=hash(acts[t].data(),acts[t].n);}
    }
    void check(){for(int e=0;e<count;++e){blobs[e].check();output[e].check();need(hash(blobs[e].data(),f.bytes)==hashes[e],"weight mutation");}for(int t=0;t<4;++t){acts[t].check();need(hash(acts[t].data(),acts[t].n)==act_hash[t],"activation mutation");}}
};
using Jobs=std::vector<cpu::ExpertJobMulti>;
struct Schedule{
    int n,entries,ones;std::vector<int> nt,order;std::vector<Jobs> stream;Jobs repeat;
    Schedule(Inputs& in,int mix):n(mix?24:28),entries(mix?36:35),ones(mix?16:22){
        nt.assign(ones,1);nt.insert(nt.end(),5,2);nt.insert(nt.end(),mix?2:1,3);if(mix)nt.push_back(4);
        need(int(nt.size())==n&&std::accumulate(nt.begin(),nt.end(),0)==entries,"NT mixture");
        std::mt19937 rng(20261021+mix);std::shuffle(nt.begin(),nt.end(),rng);order.resize(count);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
        auto make=[&](const std::vector<int>& ids,int at){Jobs v(n);for(int j=0;j<n;++j){const int e=ids[at+j];v[j].blob=in.blobs[e].data();v[j].nt=nt[j];for(int t=0;t<nt[j];++t){v[j].nact[t]=in.acts[(e+t)%4].data();v[j].out[t]=reinterpret_cast<float*>(in.output[e].data())+t*H;}}return v;};
        for(int at=0;at<count;at+=n)stream.push_back(make(order,at));std::vector<int> seq(count);std::iota(seq.begin(),seq.end(),0);repeat=make(seq,0);
    }
    void run(cpu::ExpertPool& pool,const cpu::NativeFmt& f,int access){if(access)for(auto& b:stream)pool.run_split_multi_native(f,b.data(),n);else pool.run_split_multi_native(f,repeat.data(),n);}
};
unsigned producer_contract(Inputs& in){
    auto f=in.f;const auto* blob=in.blobs[0].data();auto* act=in.acts[0].data();std::array<float,FF+2> ref{},out{};unsigned cases=0;
    // Independent direct helper reference, rather than the selected dispatcher.
    for(auto bounds:{std::pair{0,FF},std::pair{3,FF-3},std::pair{FF-1,FF},std::pair{FF/2,FF/2}}){
        ref.fill(123);out=ref;need(cpu::iq256_single_gu_rows(22,blob,f.gu_row,f.up_off,H,act,ref.data()+1,bounds.first,bounds.second),"direct reference");
        need(cpu::iq2_pair_gu_rows(f,blob,act,out.data()+1,bounds.first,bounds.second),"producer unavailable");need(!std::memcmp(ref.data(),out.data(),sizeof ref),"producer/row canary bits");++cases;
    }
    std::vector<cpu::NativeFmt> bad;auto add=[&](auto change){auto v=f;change(v);bad.push_back(v);};
    add([](auto& v){v.gu_type=18;});add([](auto& v){v.gu_act=8;});add([](auto& v){v.n_embd=256;});add([](auto& v){v.n_ff=639;});
    add([](auto& v){--v.act_bytes;});add([](auto& v){v.gu_row=819;});add([](auto& v){v.gu_row=821;});add([](auto& v){v.gu_row=std::numeric_limits<size_t>::max();});
    add([](auto& v){v.up_off=0;});add([](auto& v){++v.up_off;});add([](auto& v){v.up_off=v.bytes+2;});add([](auto& v){v.bytes=v.down_off-1;});add([](auto& v){--v.down_off;});
    for(auto& v:bad){out.fill(123);const auto before=out;need(!cpu::iq2_pair_gu_rows(v,blob,act,out.data()+1,0,1)&&out==before,"malformed format mutated output");++cases;}
    for(auto r:{std::pair{-1,1},std::pair{0,641},std::pair{2,1}}){out.fill(123);auto before=out;need(!cpu::iq2_pair_gu_rows(f,blob,act,out.data()+1,r.first,r.second)&&out==before,"invalid rows");++cases;}
    for(int variant=0;variant<6;++variant){
        auto* w=blob;const void* a=act;float* o=out.data()+1;
        if(variant==0)w=nullptr;if(variant==1)a=nullptr;if(variant==2)o=nullptr;if(variant==3)w=blob+1;if(variant==4)a=act+1;if(variant==5)o=reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(o)+1);
        out.fill(123);auto before=out;need(!cpu::iq2_pair_gu_rows(f,w,a,o,0,1)&&out==before,"null/alignment rejection");++cases;
    }
    need(!cpu::iq2_pair_gu_rows(f,blob,act,reinterpret_cast<float*>(const_cast<uint8_t*>(blob)),0,1),"output/weight overlap");++cases;
    need(!cpu::iq2_pair_gu_rows(f,blob,act,reinterpret_cast<float*>(act),0,1),"output/act overlap");++cases;
    need(!cpu::iq2_pair_gu_rows(f,blob,blob,out.data()+1,0,1),"act/weight overlap");++cases;
    // Positive padded-stride/blob-gap case, with the same complete GU reference.
    auto padded=f;padded.gu_row+=4;padded.up_off=padded.gu_row*FF+16;padded.down_off=padded.up_off+padded.gu_row*FF+16;padded.bytes=padded.down_off+(f.bytes-f.down_off);Buffer w(padded.bytes);
    for(int r=0;r<FF;++r){std::memcpy(w.data()+r*padded.gu_row,blob+r*f.gu_row,f.gu_row);std::memcpy(w.data()+padded.up_off+r*padded.gu_row,blob+f.up_off+r*f.gu_row,f.gu_row);}
    ref.fill(123);out=ref;need(cpu::iq256_single_gu_rows(22,w.data(),padded.gu_row,padded.up_off,H,act,ref.data()+1,0,FF),"padded reference");
    need(cpu::iq2_pair_gu_rows(padded,w.data(),act,out.data()+1,0,FF)&&!std::memcmp(ref.data(),out.data(),sizeof ref),"padded producer");++cases;w.check();in.check();return cases;
}
void serial(const cpu::NativeFmt& f,const cpu::ExpertJobMulti& job,float* expected){
    std::array<std::array<float,FF>,4> hidden{};std::array<Buffer,4> hq{Buffer(1024),Buffer(1024),Buffer(1024),Buffer(1024)};std::array<cpu::ActQ,4> a2{};
    float* fp[4];const void* hp[4];const cpu::ActQ* a2p[4];float* op[4];
    for(int t=0;t<job.nt;++t){fp[t]=hidden[t].data();hp[t]=hq[t].data();a2p[t]=&a2[t];op[t]=expected+t*H;}
    if(job.nt==1)need(cpu::iq256_single_gu_rows(22,job.blob,f.gu_row,f.up_off,H,job.nact[0],fp[0],0,FF),"serial direct helper");
    else cpu::native_gu_rows(f,job.blob,job.nact,job.nt,fp,0,FF);
    for(int t=0;t<job.nt;++t){if(f.d_type==42)cpu::act_quant_any(fp[t],FF,a2[t]);else cpu::native_quant_h(f,fp[t],hq[t].data());}
    if(f.d_type==42)cpu::q2_rows_any(job.blob+f.down_off,f.d_row,FF/64,a2p,job.nt,op,0,H);else cpu::native_down_rows(f,job.blob,hp,job.nt,op,0,H);
}
void validate(cpu::ExpertPool& pool,Inputs& in,Schedule& s,std::ofstream* capture){
    for(auto& out:in.output)out.poison();std::vector<std::array<float,4*H>> expected(count);
    // Reference/output identity is attached to each actual job pointer, not pool order.
    for(auto& batch:s.stream){
        for(auto& job:batch){int e=-1;for(int i=0;i<count;++i)if(job.out[0]==reinterpret_cast<float*>(in.output[i].data())){e=i;break;}need(e>=0,"job output identity");serial(in.f,job,expected[e].data());}
        pool.run_split_multi_native(in.f,batch.data(),s.n);
        for(auto& job:batch){int e=-1;for(int i=0;i<count;++i)if(job.out[0]==reinterpret_cast<float*>(in.output[i].data())){e=i;break;}
            need(!std::memcmp(job.out[0],expected[e].data(),job.nt*H*sizeof(float)),"complete pool output bits");
            for(int t=0;t<job.nt;++t)for(int r=0;r<H;++r)need(std::isfinite(job.out[t][r]),"pool finite output");
            for(size_t k=job.nt*H*sizeof(float);k<in.output[e].n;++k)need(in.output[e].data()[k]==0xa7,"inactive token output");
            if(capture)capture->write(reinterpret_cast<const char*>(job.out[0]),job.nt*H*sizeof(float));
        }
    }
    in.check();need(!capture||bool(*capture),"capture write");
}
struct Delta{uint64_t rows=0,nt1=0,pair=0,calls=0,min_rows=std::numeric_limits<uint64_t>::max();};
Delta delta(const cpu::Iq2GuAudit& before,const cpu::Iq2GuAudit& after,uint64_t calls,const Schedule& s,bool paired){
    need(before.enabled&&after.enabled&&!before.overflow&&!after.overflow&&before.size==6&&after.size==6,"audit six threads");Delta d;
    for(unsigned i=0;i<6;++i){const auto& a=before.threads[i];const auto& b=after.threads[i];need(a.tid==b.tid&&a.cpu==b.cpu&&b.pinned&&b.row_ops>=a.row_ops&&b.nt1_rows>=a.nt1_rows&&b.pair_rows>=a.pair_rows&&b.calls>=a.calls,"thread audit changed/overflow");
        uint64_t rows=b.row_ops-a.row_ops;need(rows>0,"thread performed no GU math");d.rows+=rows;d.nt1+=b.nt1_rows-a.nt1_rows;d.pair+=b.pair_rows-a.pair_rows;d.calls+=b.calls-a.calls;d.min_rows=std::min(d.min_rows,rows);}
    need(d.rows==calls*FF*s.entries&&d.nt1==calls*FF*s.ones&&d.pair==(paired?d.nt1:0),"actual route/work conservation");
    need(d.calls==calls*uint64_t(s.n+18-std::gcd(s.n,18)),"actual callback conservation");return d;
}
void timing(cpu::ExpertPool& pool,Inputs& in,Schedule& s,int mix,int arm,int process){
    for(int access=0;access<2;++access){
        auto start=Clock::now();s.run(pool,in.f,access);const double warm=seconds(start);start=Clock::now();for(int j=0;j<10000;++j){auto now=Clock::now();asm volatile(""::"r"(&now):"memory");}const double timer=seconds(start)/10000;need(timer<warm*.001,"per-pass timer overhead");
        for(int rep=0;rep<6;++rep){
            const auto before=cpu::iq2_gu_audit_snapshot();const double gu0=pool.ms_multi_gu,q0=pool.ms_multi_q,d0=pool.ms_multi_down,pcpu=cpu_seconds();start=Clock::now();uint64_t passes=0;double wall=0;
            do{s.run(pool,in.f,access);++passes;wall=seconds(start);}while(wall<.35&&passes<1000000);
            const double cpu=cpu_seconds()-pcpu;need(wall>=.25&&passes<1000000,"timing floor/bound");const uint64_t calls=passes*(access?s.stream.size():1);auto work=delta(before,cpu::iq2_gu_audit_snapshot(),calls,s,arm);
            std::cout<<"{\"kind\":\"sample\",\"process\":"<<process<<",\"arm\":"<<arm<<",\"down\":"<<in.f.d_type<<",\"mixture\":"<<mix<<",\"access\":"<<access<<",\"rep\":"<<rep<<",\"passes\":"<<passes<<",\"batch_calls\":"<<calls<<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<cpu<<",\"GU_s\":"<<(pool.ms_multi_gu-gu0)/1000<<",\"quant_s\":"<<(pool.ms_multi_q-q0)/1000<<",\"Down_s\":"<<(pool.ms_multi_down-d0)/1000<<",\"row_ops\":"<<work.rows<<",\"NT1_rows\":"<<work.nt1<<",\"paired_rows\":"<<work.pair<<",\"min_thread_rows\":"<<work.min_rows<<",\"timer_s\":"<<timer<<",\"warm_pass_s\":"<<warm<<"}"<<std::endl;
        }
    }
}
int main(int argc,char** argv){
    try{
        need(argc==6,"mode fixture capture arm process");const std::string mode=argv[1];need(mode=="check"||mode=="perf","mode");const int arm=std::stoi(argv[4]),process=std::stoi(argv[5]);need(arm==0||arm==1,"arm");
        const char* flag=std::getenv("STRATA_IQ2_PAIR");need(bool(flag&&std::atoi(flag))==bool(arm),"startup arm differs");
        auto cores=cpu::physical_cores(false);need(cores.size()==6,"six physical cores");cpu::ExpertPool pool(5,true,true);need(pool.workers()==5&&pool.host_works(),"pool placement");need(cpu::pin_current_thread(cores[0])>0,"pin host");
        std::ifstream spec(argv[2]);need(bool(spec),"fixture open");std::vector<std::unique_ptr<Inputs>> forms;
        std::string shard;int down,experts;uint64_t go,up,dn,bytes;
        while(spec>>shard>>down>>go>>up>>dn>>bytes>>experts){need(experts==count&&(down==20||down==42),"fixture fields");forms.push_back(std::make_unique<Inputs>(shard,down,go,up,dn,bytes));}
        need(spec.eof()&&forms.size()==2&&forms[0]->f.d_type==20&&forms[1]->f.d_type==42,"fixture complete");
        unsigned contracts=0;if(mode=="check")for(auto& in:forms)contracts+=producer_contract(*in);
        for(auto& in:forms)for(int mix=0;mix<2;++mix){Schedule s(*in,mix);validate(pool,*in,s,nullptr);}
        const auto initial=cpu::iq2_gu_audit_snapshot();need(initial.enabled&&!initial.overflow&&initial.size==6,"audit startup");uint64_t nt1=0,paired=0;for(const auto& t:initial.threads){nt1+=t.nt1_rows;paired+=t.pair_rows;}need(nt1>0&&paired==(arm?nt1:0),"initial actual route");std::set<int> actual_cores;std::set<uint64_t> tids;
        std::cout<<"{\"kind\":\"placement\",\"workers\":5,\"host_works\":true,\"host_tid\":"<<syscall(SYS_gettid)<<",\"threads\":[";
        for(unsigned i=0;i<6;++i){const auto& t=initial.threads[i];need(t.tid&&t.pinned&&t.row_ops>0,"thread math/pinning");actual_cores.insert(t.cpu);tids.insert(t.tid);if(i)std::cout<<",";std::cout<<"["<<t.tid<<","<<t.cpu<<"]";}std::cout<<"]}"<<std::endl;
        need(actual_cores==std::set<int>(cores.begin(),cores.end())&&tids.size()==6&&tids.count(syscall(SYS_gettid)),"actual six distinct cores/threads");
        if(mode=="perf")for(auto& in:forms)for(int mix=0;mix<2;++mix){Schedule s(*in,mix);timing(pool,*in,s,mix,arm,process);}
        std::ofstream capture(argv[3],std::ios::binary);need(bool(capture),"capture open");for(auto& in:forms)for(int mix=0;mix<2;++mix){Schedule s(*in,mix);validate(pool,*in,s,&capture);}capture.close();need(bool(capture),"capture close");
        std::cout<<"{\"kind\":\"validation\",\"contract_cases\":"<<contracts<<",\"experts_per_format\":168,\"formats\":2,\"mixtures\":2,\"bitwise\":true,\"passed\":true}"<<std::endl;
        std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
