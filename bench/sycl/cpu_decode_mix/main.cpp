// Main-authored workload calibration; all inference math links production code.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "compact_iq3xxs.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int batches = 100, repetitions = 5;
constexpr uint32_t unused = 0x7fc12345;
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
double process_cpu() {
    rusage r{}; need(getrusage(RUSAGE_SELF,&r)==0,"rusage");
    return r.ru_utime.tv_sec+r.ru_stime.tv_sec+(r.ru_utime.tv_usec+r.ru_stime.tv_usec)*1e-6;
}
uint64_t number(const char* value) {
    const std::string s(value); need(!s.empty() && s.size()<=18 &&
        std::all_of(s.begin(),s.end(),[](char c){return c>='0'&&c<='9';}),"unsigned decimal required");
    return std::stoull(s);
}
struct Buffer {
    uint8_t* p=nullptr; size_t bytes;
    explicit Buffer(size_t n):bytes(n) {
        need(n && n<=4ull<<20,"buffer extent");
        need(posix_memalign(reinterpret_cast<void**>(&p),64,n+128)==0,"allocation");
        std::memset(p,0xA7,n+128);
    }
    ~Buffer(){std::free(p);}
    Buffer(const Buffer&)=delete;
    uint8_t* data() const {return p+64;}
    void check() const {
        for(int i=0;i<64;++i) need(p[i]==0xA7 && p[64+bytes+i]==0xA7,"canary");
    }
};
void read_slice(int fd,uint8_t* out,size_t n,uint64_t offset) {
    struct stat st{}; need(fstat(fd,&st)==0 && offset<=uint64_t(st.st_size) &&
        n<=uint64_t(st.st_size)-offset,"model slice bounds");
    while(n){const auto k=pread(fd,out,n,offset);need(k>0,"model slice read");out+=k;n-=k;offset+=k;}
}
std::vector<int> schedule(int group,const std::array<uint64_t,4>& observed) {
    const uint64_t sum=std::accumulate(observed.begin(),observed.end(),uint64_t{0});
    need(sum>0 && sum<=4000000000ull,"histogram total");
    const int n=batches*group; std::array<int,4> count{};std::array<uint64_t,4> remainder{};
    int assigned=0;
    for(int i=0;i<4;++i){count[i]=n*observed[i]/sum;remainder[i]=n*observed[i]%sum;assigned+=count[i];}
    for(;assigned<n;++assigned){int at=std::max_element(remainder.begin(),remainder.end())-remainder.begin();++count[at];remainder[at]=0;}
    std::vector<int> result;
    for(int i=0;i<4;++i)result.insert(result.end(),count[i],i+1);
    std::mt19937 rng(20261022);std::shuffle(result.begin(),result.end(),rng);
    return result;
}
void verify_affinity(const cpu::CpuTopology& topology,int workers) {
    std::vector<int> expected{topology.host_core};
    expected.insert(expected.end(),topology.worker_cores.begin(),topology.worker_cores.begin()+workers);
    std::vector<int> found(expected.size());int tasks=0;
    for(const auto& entry:std::filesystem::directory_iterator("/proc/self/task")) {
        need(++tasks<=128,"unexpected task extent");
        cpu_set_t mask;CPU_ZERO(&mask);
        const auto tid=number(entry.path().filename().c_str());
        need(sched_getaffinity(tid,sizeof mask,&mask)==0,"read actual thread affinity");
        for(size_t i=0;i<expected.size();++i)
            if(CPU_COUNT(&mask)==1 && CPU_ISSET(expected[i],&mask))++found[i];
    }
    need(std::all_of(found.begin(),found.end(),[](int n){return n==1;}),"actual pool affinity differs");
}
struct Work {
    cpu::NativeFmt fmt;
    int group,count;
    std::vector<std::unique_ptr<Buffer>> weights,outputs;
    std::array<std::unique_ptr<Buffer>,4> activations;
    std::array<std::array<float,cpu::H>,4> input;
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<int> nt;
    Work(char** a,int g,const std::vector<int>& sequence):group(g),nt(sequence) {
        std::string err;const auto gu=number(a[2]),dn=number(a[3]);need(gu<=42 && dn<=42,"format ID");
        const bool valid=cpu::native_fmt(gu,dn,cpu::H,cpu::FF,fmt,err);need(valid,err.c_str());
        need(fmt.bytes==number(a[7]) && fmt.gu_type!=42,"native geometry");
        count=group*int(((128ull<<20)+group*fmt.bytes-1)/(group*fmt.bytes));
        need(count>=group && count<=256 && count*fmt.bytes>=128ull<<20,"stream extent");
        const int fd=open(a[1],O_RDONLY|O_CLOEXEC);need(fd>=0,"open model");
        for(int e=0;e<count;++e){
            auto w=std::make_unique<Buffer>(fmt.bytes);
            read_slice(fd,w->data(),fmt.up_off,number(a[4])+e*fmt.up_off);
            read_slice(fd,w->data()+fmt.up_off,fmt.up_off,number(a[5])+e*fmt.up_off);
            const auto dn=fmt.bytes-fmt.down_off;
            read_slice(fd,w->data()+fmt.down_off,dn,number(a[6])+e*dn);
            weights.push_back(std::move(w));outputs.push_back(std::make_unique<Buffer>(4*cpu::H*sizeof(float)));
        }
        close(fd);uint32_t seed=20261022;
        for(int t=0;t<4;++t){
            for(auto& x:input[t]){seed=1664525u*seed+1013904223u;x=(int(seed>>8)-8388608)*0x1p-23f;}
            activations[t]=std::make_unique<Buffer>(fmt.act_bytes);
            cpu::native_quant_act(fmt,input[t].data(),activations[t]->data());
        }
        jobs.resize(count);
        for(int e=0;e<count;++e){
            jobs[e].blob=weights[e]->data();
            for(int t=0;t<4;++t){jobs[e].nact[t]=activations[t]->data();jobs[e].out[t]=reinterpret_cast<float*>(outputs[e]->data())+t*cpu::H;}
        }
    }
    void run(cpu::ExpertPool& pool,bool stream) {
        for(int b=0;b<batches;++b){
            const int offset=stream?(b%(count/group))*group:0;
            for(int e=0;e<group;++e)jobs[offset+e].nt=nt[b*group+e];
            pool.run_split_multi_native(fmt,jobs.data()+offset,group,cpu::NativeGuPhase::Decode);
        }
    }
    void validate_compact() {
        namespace view=compact_iq3xxs;
        need(fmt.gu_type==18 && fmt.gu_row==980 && fmt.n_embd==2560 && fmt.n_ff==640,
            "compact IQ3_XXS geometry");
        constexpr size_t n=view::projection_blocks;
        std::vector<view::Block> storage(2*n+2);
        std::memset(storage.data(),0xA7,storage.size()*sizeof(view::Block));
        const auto guard=storage.front();
        auto gate=std::span(storage).subspan(1,n),up=std::span(storage).subspan(1+n,n);
        Buffer expected(cpu::FF*sizeof(float)),actual(cpu::FF*sizeof(float));
        auto* want=reinterpret_cast<float*>(expected.data());
        auto* got=reinterpret_cast<float*>(actual.data());
        for(int e=0;e<count;++e) {
            const auto* raw=weights[e]->data();
            for(size_t b=0;b<n;++b) {
                need(view::encode_block(std::span<const uint8_t>(raw+98*b,98),gate[b]),"encode gate");
                need(view::encode_block(std::span<const uint8_t>(raw+fmt.up_off+98*b,98),up[b]),"encode up");
            }
            for(int t=0;t<4;++t) {
                const auto* act=activations[t]->data();
                need(cpu::iq3xxs_pair_gu_rows(fmt,raw,act,want,0,cpu::FF),"native paired GU reference");
                need(view::gu_rows(gate,up,std::span<const uint8_t>(act,fmt.act_bytes),
                    std::span<float>(got,cpu::FF),0,cpu::FF),"compact paired GU");
                need(std::memcmp(want,got,cpu::FF*sizeof(float))==0,"compact/native GU bits");
                for(int r=0;r<cpu::FF;++r)need(std::isfinite(got[r]),"nonfinite compact GU");
                expected.check();actual.check();
            }
            need(std::memcmp(&storage.front(),&guard,sizeof guard)==0 &&
                std::memcmp(&storage.back(),&guard,sizeof guard)==0,"compact storage guards");
            weights[e]->check();
        }
        for(auto& a:activations)a->check();
        std::cout<<"{\"kind\":\"compact_validation\",\"format\":18,\"nt\":1,\"activation_cases\":4,\"experts\":"<<count
            <<",\"native_pair_bits_equal\":true,\"finite\":true,\"canaries\":true,\"compact_GU_bytes_per_expert\":"<<2*n*sizeof(view::Block)
            <<",\"raw_plus_compact_bytes_per_expert\":"<<fmt.bytes+2*n*sizeof(view::Block)<<"}"<<std::endl;
    }
    void serial(const cpu::ExpertJobMulti& j,float* out) {
        std::array<float,4*cpu::FF> ff{};
        std::array<std::unique_ptr<Buffer>,4> hq;
        std::array<cpu::ActQ,4> aq{};
        const void* ap[4];const void* hp[4];const cpu::ActQ* qp[4];float* fp[4];float* op[4];
        for(int t=0;t<j.nt;++t){ap[t]=j.nact[t];fp[t]=ff.data()+t*cpu::FF;op[t]=out+t*cpu::H;
            hq[t]=std::make_unique<Buffer>(fmt.h_bytes);hp[t]=hq[t]->data();qp[t]=&aq[t];}
        cpu::native_gu_rows(fmt,j.blob,ap,j.nt,fp,0,cpu::FF,cpu::NativeGuPhase::Decode);
        for(int t=0;t<j.nt;++t){
            if(fmt.d_type==42)cpu::act_quant_any(fp[t],cpu::FF,aq[t]);
            else cpu::native_quant_h(fmt,fp[t],hq[t]->data());
        }
        if(fmt.d_type==42)cpu::q2_rows_any(j.blob+fmt.down_off,fmt.d_row,cpu::FF/64,qp,j.nt,op,0,cpu::H);
        else cpu::native_down_rows(fmt,j.blob,hp,j.nt,op,0,cpu::H);
        for(int t=0;t<j.nt;++t)hq[t]->check();
    }
    void validate(cpu::ExpertPool& pool) {
        // Exercise every loaded expert with every NT, outside timed work. Unlike
        // identical blobs, distinct model experts detect job/index permutations.
        std::array<float,4*cpu::H> want{};
        auto reset=[&](int e){
            auto* words=reinterpret_cast<uint32_t*>(outputs[e]->data());
            std::fill(words,words+4*cpu::H,unused);
        };
        auto compare=[&](int e){
            const int n=jobs[e].nt;
            serial(jobs[e],want.data());
            need(std::memcmp(want.data(),outputs[e]->data(),n*cpu::H*sizeof(float))==0,"pool/serial bits");
            const auto* words=reinterpret_cast<const uint32_t*>(outputs[e]->data());
            for(int i=n*cpu::H;i<4*cpu::H;++i)need(words[i]==unused,"inactive NT output written");
            for(int i=0;i<n*cpu::H;++i)need(std::isfinite(want[i]),"nonfinite output");
        };
        for(int n=1;n<=4;++n){
            for(int e=0;e<count;++e){jobs[e].nt=n;reset(e);}
            pool.run_split_multi_native(fmt,jobs.data(),count,cpu::NativeGuPhase::Decode);
            for(int e=0;e<count;++e)compare(e);
        }
        // Validate the same group size as timing with heterogeneous NTs. Force
        // NT1..4 into each group so sparse marginal bins cannot omit coverage.
        for(int offset=0;offset<count;offset+=group){
            for(int e=0;e<group;++e){
                jobs[offset+e].nt=e<4?e+1:nt[(offset+e)%nt.size()];
                reset(offset+e);
            }
            pool.run_split_multi_native(fmt,jobs.data()+offset,group,cpu::NativeGuPhase::Decode);
            for(int e=0;e<group;++e)compare(offset+e);
        }
        for(auto& w:weights)w->check();for(auto& o:outputs)o->check();for(auto& a:activations)a->check();
        std::cout<<"{\"kind\":\"validation\",\"distinct_experts\":"<<count<<",\"NT_mask\":30,\"pool_serial_bits_equal\":true,\"mixed_NT_bits_equal\":true,\"inactive_outputs_untouched\":true,\"canaries\":true}"<<std::endl;
    }
};
int main(int argc,char** argv) try {
    std::cout<<std::setprecision(12);
    need(argc==16,"shard gu dn gate up down bytes workers group check|measure|compact-check n1 n2 n3 n4 version");
    need(std::string(argv[15])=="20261022","protocol version");
    const auto requested_workers=number(argv[8]),requested_group=number(argv[9]);
    const std::string mode=argv[10];
    need(requested_workers>=1 && requested_workers<=5 && requested_group>=6 && requested_group<=32 &&
        (mode=="check"||mode=="measure"||mode=="compact-check"),"mode/shape");
    need(mode!="compact-check" || number(argv[2])==18,"compact IQ3_XXS only");
    // Intermediate worker counts expose the scaling curve. Five workers also
    // enable the production host compute thread; smaller counts leave it waiting.
    const int workers=static_cast<int>(requested_workers),group=static_cast<int>(requested_group);
    std::array<uint64_t,4> observed{};for(int i=0;i<4;++i){observed[i]=number(argv[11+i]);need(observed[i]<=1000000000ull,"histogram count");}
    const auto nt=schedule(group,observed);std::array<int,4> actual{};for(int n:nt)++actual[n-1];
    const auto topology=cpu::detect_cpu_topology(true);
    need(topology.host_core>=0 && topology.worker_cores.size()>=5,"six physical cores required");
    cpu::ExpertPool pool(workers,true,workers==5);
    cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(topology.host_core,&mask);need(sched_setaffinity(0,sizeof mask,&mask)==0,"host affinity");
    Work work(argv,group,nt);
    if(mode=="compact-check") {
        work.validate_compact();verify_affinity(topology,workers);
        std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl;return 0;
    }
    std::cout<<"{\"kind\":\"work\",\"workers\":"<<workers<<",\"host_works\":"<<(workers==5?"true":"false")
        <<",\"group\":"<<group<<",\"batches\":"<<batches<<",\"unique_stream_experts\":"<<work.count<<",\"blob_bytes\":"<<work.fmt.bytes
        <<",\"jobs_NT\":["<<actual[0]<<","<<actual[1]<<","<<actual[2]<<","<<actual[3]<<"],\"phase\":\"decode\"}"<<std::endl;
    work.validate(pool);verify_affinity(topology,workers);
    std::cout<<"{\"kind\":\"placement\",\"workers\":"<<pool.workers()<<",\"host_cpu\":"<<topology.host_core<<",\"worker_cpus\":[";
    for(int i=0;i<workers;++i)std::cout<<(i?",":"")<<topology.worker_cores[i];
    std::cout<<"],\"actual_affinity_verified\":true}"<<std::endl;
    if(mode=="measure")for(bool stream:{false,true}){
        auto pass=[&]{work.run(pool,stream);};pass();
        double fastest=1e9;for(int i=0;i<3;++i){const auto t=Clock::now();pass();fastest=std::min(fastest,seconds(t));}
        const auto timer=Clock::now();for(int i=0;i<10000;++i){const auto t=Clock::now();(void)t;}
        const double cost=seconds(timer)/10000;need(cost/fastest<.001,"timer overhead");
        std::cout<<"{\"kind\":\"timer\",\"stream\":"<<(stream?"true":"false")<<",\"query_s\":"<<cost<<",\"fastest_pass_s\":"<<fastest<<"}"<<std::endl;
        for(int rep=0;rep<repetitions;++rep){
            const double gu=pool.ms_multi_gu,q=pool.ms_multi_q,dn=pool.ms_multi_down,cpu0=process_cpu();
            const auto begin=Clock::now();int passes=0;do{pass();++passes;}while(seconds(begin)<1.3 && passes<1000000);
            const double wall=seconds(begin),scheduled=process_cpu()-cpu0;need(wall>=1 && passes<1000000,"sample floor/budget");
            std::cout<<"{\"kind\":\"sample\",\"stream\":"<<(stream?"true":"false")<<",\"rep\":"<<rep<<",\"passes\":"<<passes
                <<",\"wall_s\":"<<wall<<",\"cpu_s\":"<<scheduled<<",\"gu_s\":"<<(pool.ms_multi_gu-gu)*.001
                <<",\"quant_s\":"<<(pool.ms_multi_q-q)*.001<<",\"down_s\":"<<(pool.ms_multi_down-dn)*.001<<"}"<<std::endl;
        }
        work.validate(pool);
    }
    verify_affinity(topology,workers);
    std::cout<<"{\"kind\":\"closed\",\"passed\":true}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"CPU decode-mix failed: "<<e.what()<<std::endl;return 1;}
