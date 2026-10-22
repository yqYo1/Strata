// Main-authored unchanged-policy CPU state diagnostic; repository LICENSE.
// Math comes only from linked Strata/ggml producers. No copied XeStrata code.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq2_pair_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <thread>
#include <optional>
#include "observer.hpp"
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
    int n,entries,ones;std::vector<int> nt,order;std::vector<Jobs> stream;
    Schedule(Inputs& in,int mix):n(mix?24:28),entries(mix?36:35),ones(mix?16:22){
        nt.assign(ones,1);nt.insert(nt.end(),5,2);nt.insert(nt.end(),mix?2:1,3);if(mix)nt.push_back(4);
        need(int(nt.size())==n&&std::accumulate(nt.begin(),nt.end(),0)==entries,"NT mixture");
        std::mt19937 rng(20261021+mix);std::shuffle(nt.begin(),nt.end(),rng);order.resize(count);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
        auto make=[&](const std::vector<int>& ids,int at){Jobs v(n);for(int j=0;j<n;++j){const int e=ids[at+j];v[j].blob=in.blobs[e].data();v[j].nt=nt[j];for(int t=0;t<nt[j];++t){v[j].nact[t]=in.acts[(e+t)%4].data();v[j].out[t]=reinterpret_cast<float*>(in.output[e].data())+t*H;}}return v;};
        for(int at=0;at<count;at+=n)stream.push_back(make(order,at));
    }

};
void serial(const cpu::NativeFmt& f,const cpu::ExpertJobMulti& job,float* expected){
    std::array<std::array<float,FF>,4> hidden{};std::array<Buffer,4> hq{Buffer(1024),Buffer(1024),Buffer(1024),Buffer(1024)};std::array<cpu::ActQ,4> a2{};
    float* fp[4];const void* hp[4];const cpu::ActQ* a2p[4];float* op[4];
    for(int t=0;t<job.nt;++t){fp[t]=hidden[t].data();hp[t]=hq[t].data();a2p[t]=&a2[t];op[t]=expected+t*H;}
    if(job.nt==1)need(cpu::iq256_single_gu_rows(22,job.blob,f.gu_row,f.up_off,H,job.nact[0],fp[0],0,FF),"serial direct helper");
    else cpu::native_gu_rows(f,job.blob,job.nact,job.nt,fp,0,FF);
    for(int t=0;t<job.nt;++t){if(f.d_type==42)cpu::act_quant_any(fp[t],FF,a2[t]);else cpu::native_quant_h(f,fp[t],hq[t].data());}
    if(f.d_type==42)cpu::q2_rows_any(job.blob+f.down_off,f.d_row,FF/64,a2p,job.nt,op,0,H);else cpu::native_down_rows(f,job.blob,hp,job.nt,op,0,H);
}
struct Delta{uint64_t rows=0,nt1=0,pair=0,calls=0,min_rows=std::numeric_limits<uint64_t>::max();};
Delta delta(const cpu::Iq2GuAudit& before,const cpu::Iq2GuAudit& after,uint64_t calls,const Schedule& s,bool paired){
    need(before.enabled&&after.enabled&&!before.overflow&&!after.overflow&&before.size==6&&after.size==6,"audit six threads");Delta d;
    for(unsigned i=0;i<6;++i){const auto& a=before.threads[i];const auto& b=after.threads[i];need(a.tid==b.tid&&a.cpu==b.cpu&&b.pinned&&b.row_ops>=a.row_ops&&b.nt1_rows>=a.nt1_rows&&b.pair_rows>=a.pair_rows&&b.calls>=a.calls,"thread audit changed/overflow");
        uint64_t rows=b.row_ops-a.row_ops;need(rows>0,"thread performed no GU math");d.rows+=rows;d.nt1+=b.nt1_rows-a.nt1_rows;d.pair+=b.pair_rows-a.pair_rows;d.calls+=b.calls-a.calls;d.min_rows=std::min(d.min_rows,rows);}
    need(d.rows==calls*FF*s.entries&&d.nt1==calls*FF*s.ones&&d.pair==(paired?d.nt1:0),"actual route/work conservation");
    need(d.calls==calls*uint64_t(s.n+18-std::gcd(s.n,18)),"actual callback conservation");return d;
}
struct Sample {uint64_t passes,calls;double wall,cpu,gu,quant,down;Delta work;};
void run_pass(cpu::ExpertPool& pool,Inputs& in,Schedule& schedule,cpu::NativeGuPhase phase,int gap_ms){
    for(auto& jobs:schedule.stream){pool.run_split_multi_native(in.f,jobs.data(),schedule.n,phase);if(gap_ms)std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));}
}
void emit_profile(uint64_t calls,cpu::NativeGuPhase phase,const std::vector<int>& physical){
#if POOL_OBSERVED
    using namespace pool_observer;std::set<uint64_t> tids;std::set<int> cores;
    std::array<uint64_t,Kinds> entries{};const auto selected=phase_of(phase);
    for(unsigned i=0;i<6;++i){const auto& s=slots[i];need(s.closed&&s.tid&&s.errors==0&&s.depth==0,"closed observer thread");tids.insert(s.tid);cores.insert(s.core);uint64_t wall=0,cpu_ns=0;
        for(int p=0;p<Phases;++p)for(int k=0;k<Kinds;++k){const auto& c=s.counters[p][k];wall+=c.wall;cpu_ns+=c.cpu;
            if(k==GuTask||k==DownTask||k==Quant||k==WaitDone||k==WaitParked)need(!c.intervals||p==selected,"mis-tagged known work");
            if(k==Spin||k==CvWaitScope)need(!c.intervals||p==Unknown,"idle phase must remain unassigned");
            need(k!=OtherTask||!c.intervals,"unexpected job kind");
            if(p==selected)entries[k]+=c.intervals;
        }
        need(wall==s.end_wall-s.start_wall&&cpu_ns==s.end_cpu-s.start_cpu&&cpu_ns<=wall+1000000,"disjoint full-lifetime conservation");need(s.voluntary_end>=s.voluntary_start&&s.involuntary_end>=s.involuntary_start,"context switch counters");
        std::cout<<"{\"kind\":\"thread\",\"slot\":"<<i<<",\"tid\":"<<s.tid<<",\"core\":"<<s.core<<",\"life_wall_ns\":"<<wall<<",\"life_cpu_ns\":"<<cpu_ns<<",\"voluntary_switches\":"<<s.voluntary_end-s.voluntary_start<<",\"involuntary_switches\":"<<s.involuntary_end-s.involuntary_start<<",\"states\":[";
        bool first=true;for(int p=0;p<Phases;++p)for(int k=0;k<Kinds;++k){const auto& c=s.counters[p][k];if(!c.intervals)continue;if(!first)std::cout<<",";first=false;std::cout<<"["<<p<<","<<k<<","<<c.intervals<<","<<c.wall<<","<<c.cpu<<"]";}std::cout<<"]}"<<std::endl;
    }
    need(tids.size()==6&&cores==std::set<int>(physical.begin(),physical.end()),"six physical observer threads");
    need(entries[GuTask]==18*calls&&entries[DownTask]==18*calls&&entries[Quant]==calls&&entries[WaitDone]==2*calls&&entries[WaitParked]==4*calls&&entries[OtherTask]==0,"observer task/phase entry conservation");
#else
    (void)calls;(void)phase;(void)physical;
#endif
}
int main(int argc,char** argv){try{
    need(argc==8,"mode fixture capture down mixture phase gap_ms");const std::string mode=argv[1];need(mode=="check"||mode=="measure","mode");const int selected_down=std::stoi(argv[4]),mix=std::stoi(argv[5]),selected_phase=std::stoi(argv[6]),gap=std::stoi(argv[7]);
    need((selected_down==20&&mix==0)||(selected_down==42&&mix==1),"bounded shape");need((selected_phase==0||selected_phase==1)&&(gap==0||gap==40),"bounded phase/gap");const auto phase=selected_phase?cpu::NativeGuPhase::Decode:cpu::NativeGuPhase::Prefill;
    std::cout<<std::setprecision(17);const auto physical=cpu::physical_cores(false);need(physical.size()==6,"six physical cores before pin");const auto inherited_affinity=cpu::pin_current_thread(physical[0]);need(inherited_affinity>0,"pin oracle host physical core");
    const char* pair=std::getenv("STRATA_IQ2_PAIR");need(pair&&std::strcmp(pair,"1")==0,"unchanged pair selection");
    std::ifstream spec(argv[2]);need(bool(spec),"fixture");std::string shard;int down,experts;uint64_t go,up,dn,bytes;std::unique_ptr<Inputs> in;unsigned formats=0;
    while(spec>>shard>>down>>go>>up>>dn>>bytes>>experts){need(experts==count&&(down==20||down==42),"format input");++formats;if(down==selected_down){need(!in,"duplicate format");in=std::make_unique<Inputs>(shard,down,go,up,dn,bytes);}}
    need(spec.eof()&&formats==2&&bool(in),"fixture completeness");Schedule schedule(*in,mix);
    // Oracle and model loading precede every observer/thread lifetime.
    std::vector<std::array<float,4*H>> expected(count);
    for(auto& batch:schedule.stream)for(auto& job:batch){int e=-1;for(int i=0;i<count;++i)if(job.out[0]==reinterpret_cast<float*>(in->output[i].data())){e=i;break;}need(e>=0,"reference job identity");serial(in->f,job,expected[e].data());}
    auto clock_begin=Clock::now();for(int i=0;i<10000;++i){auto value=pool_observer::stamp();asm volatile(""::"r"(&value):"memory");}const double clock_probe=seconds(clock_begin)/10000;
    std::vector<Sample> samples;samples.reserve(5);uint64_t total_calls=0;double warm_pass=0;
    {
#if POOL_OBSERVED
        std::optional<pool_observer::Root> host;
#endif
        { // Topology discovery and workers must inherit the original allowed CPU mask.
          cpu::restore_thread_affinity(inherited_affinity);need(cpu::physical_cores(false)==physical,"restore original topology before pool");
          cpu::ExpertPool pool(5,true,true);need(pool.workers()==5&&pool.host_works(),"unchanged pool configuration");
          need(cpu::pin_current_thread(physical[0])>0,"pin pool host physical core");
#if POOL_OBSERVED
          host.emplace(5);
#endif
          run_pass(pool,*in,schedule,phase,gap);total_calls+=schedule.stream.size();
          const auto initial=cpu::iq2_gu_audit_snapshot();need(initial.enabled&&!initial.overflow&&initial.size==6,"six native audit threads");
          std::set<uint64_t> tids;std::set<int> cores;for(unsigned i=0;i<6;++i){const auto& t=initial.threads[i];need(t.tid&&t.pinned,"native pinned thread");tids.insert(t.tid);cores.insert(t.cpu);}need(tids.size()==6&&cores==std::set<int>(physical.begin(),physical.end()),"native physical placement");
          auto warm=Clock::now();run_pass(pool,*in,schedule,phase,gap);warm_pass=seconds(warm);total_calls+=schedule.stream.size();need(clock_probe<warm_pass*.001,"clock probe budget");
          for(int rep=0;rep<(mode=="check"?0:5);++rep){const auto before=cpu::iq2_gu_audit_snapshot();const double gu=pool.ms_multi_gu,q=pool.ms_multi_q,d=pool.ms_multi_down,pcpu=cpu_seconds();const auto start=Clock::now();uint64_t passes=0;double wall=0;
              do{run_pass(pool,*in,schedule,phase,gap);++passes;wall=seconds(start);}while(wall<1.2&&passes<1000000);
              const double used=cpu_seconds()-pcpu;need(wall>=1&&passes<1000000,"timing floor");const uint64_t calls=passes*schedule.stream.size();const auto work=delta(before,cpu::iq2_gu_audit_snapshot(),calls,schedule,selected_phase==1);
              samples.push_back({passes,calls,wall,used,(pool.ms_multi_gu-gu)/1000,(pool.ms_multi_q-q)/1000,(pool.ms_multi_down-d)/1000,work});total_calls+=calls;
          }
        } // Stop/join all workers before any profile read.
    } // Close host ledger after the pool's stop/join protocol.
    emit_profile(total_calls,phase,physical);
    std::ofstream capture(argv[3],std::ios::binary);need(bool(capture),"capture");
    for(auto& batch:schedule.stream)for(auto& job:batch){int e=-1;for(int i=0;i<count;++i)if(job.out[0]==reinterpret_cast<float*>(in->output[i].data())){e=i;break;}need(e>=0,"output job identity");need(!std::memcmp(job.out[0],expected[e].data(),job.nt*H*sizeof(float)),"full pool output bits");for(int t=0;t<job.nt;++t)for(int r=0;r<H;++r)need(std::isfinite(job.out[t][r]),"finite pool output");for(size_t k=job.nt*H*sizeof(float);k<in->output[e].n;++k)need(in->output[e].data()[k]==0xa7,"inactive output");capture.write(reinterpret_cast<const char*>(job.out[0]),job.nt*H*sizeof(float));}
    capture.close();need(bool(capture),"capture close");in->check();
    for(unsigned i=0;i<samples.size();++i){const auto& s=samples[i];std::cout<<"{\"kind\":\"sample\",\"rep\":"<<i<<",\"passes\":"<<s.passes<<",\"batch_calls\":"<<s.calls<<",\"wall_s\":"<<s.wall<<",\"cpu_s\":"<<s.cpu<<",\"GU_s\":"<<s.gu<<",\"quant_s\":"<<s.quant<<",\"Down_s\":"<<s.down<<",\"row_ops\":"<<s.work.rows<<",\"NT1_rows\":"<<s.work.nt1<<",\"pair_rows\":"<<s.work.pair<<",\"min_thread_rows\":"<<s.work.min_rows<<"}"<<std::endl;}
    std::cout<<"{\"kind\":\"closed\",\"mode\":\""<<mode<<"\",\"observed\":"<<(POOL_OBSERVED?"true":"false")<<",\"down\":"<<selected_down<<",\"mixture\":"<<mix<<",\"phase\":"<<selected_phase<<",\"gap_ms\":"<<gap<<",\"calls_including_warmups\":"<<total_calls<<",\"clock_probe_s\":"<<clock_probe<<",\"warm_pass_s\":"<<warm_pass<<",\"bitwise\":true,\"guards\":true,\"inputs_unchanged\":true,\"passed\":true}"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
