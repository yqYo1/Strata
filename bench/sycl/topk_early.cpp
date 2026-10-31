// Main-authored component fixture; repository LICENSE. No model/decode hook.
#include "topk-early.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <time.h>
#include <vector>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace sb = strata::sycl_backend;
namespace sk = strata::kernels;
using Clock = std::chrono::steady_clock;
constexpr size_t Guard = 16;
constexpr int32_t Marker = -123456789;
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void gate(const char* stage){std::cout<<"GATE stage="<<stage<<std::endl;std::string s;
    need(bool(std::getline(std::cin,s))&&s=="continue","controller acknowledgement");}
struct Drain {
    sycl::queue queue;std::shared_ptr<sb::Runtime> runtime;
    ~Drain(){try{queue.wait_and_throw();runtime->check();}catch(...){std::_Exit(86);}}
};
template<class T> struct Buffer {
    sb::Allocation allocation;size_t size;T marker;std::vector<T> upload,download;
    Buffer(std::shared_ptr<sb::Runtime> rt,size_t n,T m)
        :allocation(rt,(n+2*Guard)*sizeof(T),sb::MemoryKind::Device),size(n),marker(m){}
    T* data(){return allocation.template as<T>()+Guard;}
    void put(sycl::queue& q,const std::vector<T>& values){
        need(values.size()==size,"fixture upload extent");q.wait_and_throw();
        upload.assign(size+2*Guard,marker);std::copy(values.begin(),values.end(),upload.begin()+Guard);
        q.memcpy(allocation.data(),upload.data(),allocation.size()).wait_and_throw();}
    const std::vector<T>& get(sycl::queue& q){q.wait_and_throw();download.resize(size+2*Guard);
        q.memcpy(download.data(),allocation.data(),allocation.size()).wait_and_throw();
        for(size_t i=0;i<Guard;++i)need(std::memcmp(&download[i],&marker,sizeof(T))==0&&
            std::memcmp(&download[download.size()-1-i],&marker,sizeof(T))==0,"allocation guard changed");
        return download;}
    void immutable(sycl::queue& q){get(q);need(download.size()==upload.size()&&
        std::memcmp(download.data(),upload.data(),download.size()*sizeof(T))==0,"fixture input modified");}
};
uint32_t random_word(uint32_t& x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
uint32_t ordered_key(float x){if(std::isnan(x))return 0;if(x==0.f)x=0.f;
    const auto b=std::bit_cast<uint32_t>(x);return std::signbit(x)?~b:b|0x80000000u;}
float score(int pattern,int b,int row,uint32_t& seed){
    const uint32_t r=random_word(seed);
    if(pattern==0)return float(1+r%1000003)/1024.f;
    if(pattern==1)return float(r%1024)/16.f;
    if(pattern==2)return std::bit_cast<float>(r);
    if(pattern==3)return 1.f;
    if(pattern==4)return std::bit_cast<float>(0x7fc00001u+uint32_t(b%8192));
    if(pattern==5)return std::bit_cast<float>(uint32_t(b&1)<<31);
    if(pattern==6){constexpr uint32_t special[]={0xff800000u,0x7f800000u,0x80000000u,0,
        0x7fc00001u,0xffc12345u,0xbf800000u,0x3f800000u,1,0x80000001u};
        return std::bit_cast<float>(special[(b+row)%10]);}
    return b<2?3.f:(b%13==0?-1.f:2.f);
}
struct Spec{int cells,capacity,queries,cap,pattern;bool invalid=false;};
Spec checked(Spec s){need(s.cells>=s.queries&&s.queries>=1&&s.queries<=256&&
    s.cells<=s.capacity&&s.capacity<=262144&&s.cap>=1&&s.cap<=2051&&
    s.pattern>=0&&s.pattern<=7,"bounded fixture geometry");return s;}
struct Fixture {
    std::shared_ptr<sb::Runtime> rt;sycl::queue queue;Spec spec;int blocks,active;
    std::vector<float> hs;std::vector<int32_t> ht,expected,expected_candidate_rounds,expected_original_rounds;
    Buffer<float> scores;Buffer<int32_t> steps,ids,rounds;Drain drain;
    Fixture(std::shared_ptr<sb::Runtime> runtime,const sycl::queue& q,Spec s)
        :rt(std::move(runtime)),queue(q),spec(checked(s)),blocks(spec.capacity/4+1),active(spec.cells/4+1),
         hs(size_t(spec.queries)*blocks,std::numeric_limits<float>::infinity()),
         ht(size_t(spec.queries)*sk::kStepCount,0),expected(size_t(spec.queries)*spec.cap,Marker),
         expected_candidate_rounds(spec.queries),expected_original_rounds(spec.queries),
         scores(rt,hs.size(),-12345.5f),steps(rt,ht.size(),Marker),ids(rt,expected.size(),Marker),
         rounds(rt,spec.queries,Marker),drain{queue,rt}{
        uint32_t seed=0x67149a3u;
        struct Cell{float value;int32_t id;};
        const auto better=[](Cell a,Cell b){const bool an=std::isnan(a.value),bn=std::isnan(b.value);
            if(an!=bn)return !an;if(!an&&a.value!=b.value)return a.value>b.value;return a.id<b.id;};
        for(int qi=0;qi<spec.queries;++qi){const int n=spec.cells-spec.queries+1+qi;
            const int width=std::min(n,spec.cap);auto* st=ht.data()+qi*sk::kStepCount;
            st[sk::kStepPos]=n-1;st[sk::kStepNKv]=n;st[sk::kStepNBid]=n/4;st[sk::kStepWidth]=width;
            float* sc=hs.data()+size_t(qi)*blocks;
            for(int b=0;b<=n/4;++b)sc[b]=score(spec.pattern,b,qi,seed);
            if(spec.pattern<=1&&n%4)sc[n/4]+=1e9f;
            if(n%4==0)sc[n/4]=std::numeric_limits<float>::infinity();
            if(spec.invalid&&qi%5==0){st[sk::kStepNBid]=-1;continue;}
            if(spec.invalid&&qi%5==1){st[sk::kStepWidth]=n+1;continue;}
            if(spec.invalid&&qi%5==2){st[sk::kStepWidth]=0;continue;}
            // Numeric per-cell rank is independent of the kernel's radix search.
            std::vector<Cell> cells;cells.reserve(n);
            for(int id=0;id<n;++id)cells.push_back({sc[id/4],id});
            if(width<n)std::nth_element(cells.begin(),cells.begin()+width,cells.end(),better);
            std::sort(cells.begin(),cells.begin()+width,better);
            auto* out=expected.data()+size_t(qi)*spec.cap;
            for(int i=0;i<width;++i)out[i]=cells[i].id;std::sort(out,out+width);
            if(width==n)continue;
            expected_original_rounds[qi]=8;expected_candidate_rounds[qi]=8;
            const uint32_t kth=ordered_key(cells[width-1].value);
            // Determine the first decisive prefix from the independently ranked
            // boundary key,not by reproducing digit histograms or rank updates.
            for(int pass=1;pass<=8;++pass){const int shift=32-4*pass;
                const uint32_t mask=0xffffffffu<<shift,prefix=kth&mask;
                int above=0,bucket=0;
                for(int b=0;b<=n/4;++b){const int weight=std::min(4,std::max(0,n-4*b));
                    const uint32_t p=ordered_key(sc[b])&mask;
                    if(p>prefix)above+=weight;if(p==prefix)bucket+=weight;}
                if(width-above==bucket||bucket<=4){expected_candidate_rounds[qi]=pass;break;}
            }
        }
        scores.put(queue,hs);steps.put(queue,ht);ids.put(queue,std::vector<int32_t>(expected.size(),Marker));
        rounds.put(queue,std::vector<int32_t>(spec.queries,Marker));
    }
    sycl::event call(bool candidate,bool capture=false){auto shape=sk::qsa_real_shapes();
        return candidate?sk::topk_early_candidate::launch(scores.data(),steps.data(),spec.queries,
            blocks,spec.cap,shape,ids.data(),queue,active,capture?rounds.data():nullptr,capture):
            sk::topk_early_original::launch(scores.data(),steps.data(),spec.queries,
            blocks,spec.cap,shape,ids.data(),queue,active,capture?rounds.data():nullptr,capture);}
    void finish(){queue.wait_and_throw();rt->check();}
    void output(){const auto& x=ids.get(queue);need(std::equal(expected.begin(),expected.end(),x.begin()+Guard),"numeric cell-rank oracle differs");}
    void immutable(){scores.immutable(queue);steps.immutable(queue);ids.get(queue);rounds.get(queue);}
    void qualify(){
        int64_t saved=0;int stopped=0;
        for(bool arm:{false,true}){ids.put(queue,std::vector<int32_t>(expected.size(),Marker));
            rounds.put(queue,std::vector<int32_t>(spec.queries,Marker));call(arm);finish();output();
            const auto plain=rounds.get(queue);
            need(std::all_of(plain.begin()+Guard,plain.end()-Guard,[](int v){return v==Marker;}),"normal variant wrote counters");
            call(arm,true);finish();output();const auto captured=rounds.get(queue);
            const auto& wanted=arm?expected_candidate_rounds:expected_original_rounds;
            need(std::equal(wanted.begin(),wanted.end(),captured.begin()+Guard),"actual radix pass count differs");
            call(arm);finish();output();need(rounds.get(queue)==captured,"normal repeat modified counters");
        }
        for(int i=0;i<spec.queries;++i){saved+=expected_original_rounds[i]-expected_candidate_rounds[i];
            stopped+=expected_original_rounds[i]>0&&expected_candidate_rounds[i]<8;}
        immutable();std::cout<<"CASE cells="<<spec.cells<<" capacity="<<spec.capacity<<" queries="<<spec.queries
            <<" cap="<<spec.cap<<" pattern="<<spec.pattern<<" invalid="<<spec.invalid<<" oracle=PASS counters=PASS"
            <<" saved_passes="<<saved<<" stopped_queries="<<stopped<<" inputs=PASS guards=PASS repeatability=PASS\n";
    }
};
void equal_ids(const std::vector<int>& a,const std::vector<int>& b){need(a==b,"ID comparator");}
void comparator_controls(){int refused=0;for(const auto& x:{std::vector<int>{2,1},std::vector<int>{1,1},std::vector<int>{1,-1}})
    try{equal_ids(x,{1,2});}catch(const std::runtime_error&){++refused;}
    need(refused==3,"comparator disabled");equal_ids({1,2},{1,2});
    std::cout<<"COMPARATOR refusals=3 identity=PASS\n";}
void admission(std::shared_ptr<sb::Runtime> rt,const sycl::queue& q){
    Fixture f(rt,q,{17,32,3,5,0});int refused=0;
    for(bool arm:{false,true}){auto launch=arm?sk::topk_early_candidate::launch:sk::topk_early_original::launch;
        auto attempt=[&](auto fn){try{fn();}catch(const std::invalid_argument&){++refused;return;}throw std::runtime_error("invalid admission accepted");};
        const auto s=sk::qsa_real_shapes();
        auto run=[&](int nq,int blocks,int cap,int active,const sk::QsaShapes& shape){
            launch(f.scores.data(),f.steps.data(),nq,blocks,cap,shape,f.ids.data(),f.queue,active,nullptr,false);};
        attempt([&]{run(0,f.blocks,5,f.active,s);});attempt([&]{run(65536,f.blocks,5,f.active,s);});
        attempt([&]{run(3,0,5,f.active,s);});attempt([&]{run(3,f.blocks,0,f.active,s);});
        attempt([&]{run(3,f.blocks,5,0,s);});attempt([&]{run(3,f.blocks,5,f.blocks+1,s);});
        for(int k=0;k<3;++k){auto bad=s;if(k==0)bad.idx_dim=127;if(k==1)bad.idx_n_head=3;if(k==2)bad.idx_block=3;
            attempt([&]{run(3,f.blocks,5,f.active,bad);});}
        attempt([&]{launch(nullptr,f.steps.data(),3,f.blocks,5,s,f.ids.data(),f.queue,f.active,nullptr,false);});
        attempt([&]{launch(f.scores.data(),nullptr,3,f.blocks,5,s,f.ids.data(),f.queue,f.active,nullptr,false);});
        attempt([&]{launch(f.scores.data(),f.steps.data(),3,f.blocks,5,s,f.ids.data(),f.queue,f.active,nullptr,true);});
        attempt([&]{launch(f.scores.data(),f.steps.data(),3,f.blocks,5,s,f.ids.data(),f.queue,f.active,f.ids.data(),true);});
    }
    need(refused==26,"admission controls count");f.finish();f.immutable();
    std::cout<<"ADMISSION refusals="<<refused<<" PASS=1\n";
}
void numerical(std::shared_ptr<sb::Runtime> rt,const sycl::queue& q,bool smoke){
    if(smoke){Fixture f(rt,q,{17,32,3,5,0});f.qualify();return;}
    admission(rt,q);
    for(int cells:{1,3,4,5,31,32,33,2047,2048,2051,2052,32768,65536,117488,262141,262142,262143,262144})
        for(int pattern=0;pattern<8;++pattern){Fixture f(rt,q,{cells,cells,std::min(3,cells),2051,pattern});f.qualify();}
    Fixture invalid(rt,q,{65,128,10,13,6,true});invalid.qualify();
}
void properties(std::shared_ptr<sb::Runtime> rt,const sycl::queue& q,const char* path){
    void* loader=dlopen(path,RTLD_LOCAL|RTLD_NOW|RTLD_NOLOAD);need(loader,"Level Zero loader absent");
    auto query=reinterpret_cast<decltype(&zeKernelGetProperties)>(dlsym(loader,"zeKernelGetProperties"));need(query,"kernel property API absent");
    int count=0;
    for(const auto& id:sycl::get_kernel_ids()){const std::string n=id.get_name();
        const bool candidate=n.find("topk_early_candidate")!=std::string::npos;
        if(!candidate&&n.find("topk_early_original")==std::string::npos)continue;
        auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(),{q.get_device()},{id});
        const auto kernel=bundle.get_kernel(id);ze_kernel_properties_t p{};p.stype=ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
        need(query(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kernel),&p)==ZE_RESULT_SUCCESS,"kernel property query failed");
        const auto maximum=kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
        need(maximum>=256&&(!p.requiredGroupSizeX||p.requiredGroupSizeX==256)&&
            (!p.requiredGroupSizeY||p.requiredGroupSizeY==1)&&(!p.requiredGroupSizeZ||p.requiredGroupSizeZ==1),"compiled group geometry");
        need(size_t(p.localMemSize)+4096<=rt->device().get_info<sycl::info::device::local_mem_size>(),"compiled local-memory headroom");
        std::cout<<"PROPERTY arm="<<candidate<<" maximum_group="<<maximum<<" subgroup="<<p.requiredSubgroupSize
            <<" static_local="<<p.localMemSize<<" spill="<<p.spillMemSize<<" PASS=1\n";++count;}
    need(count==4,"four normal/counter kernels required");dlclose(loader);
}
void timing(std::shared_ptr<sb::Runtime> rt,sycl::queue q,int process){
    static_assert(Clock::is_steady);timespec resolution{};need(clock_getres(CLOCK_MONOTONIC,&resolution)==0&&
        resolution.tv_sec==0&&resolution.tv_nsec<=1000,"monotonic clock resolution");
    std::vector<uint64_t> samples;auto previous=Clock::now();for(int i=0;i<1000;++i){const auto now=Clock::now();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(now-previous).count());previous=now;}
    std::sort(samples.begin(),samples.end());need(samples[990]<=10000,"timer p99 above10us");
    std::cout<<"TIMER samples=1000 p99_ns="<<samples[990]<<" resolution_ns="<<resolution.tv_nsec<<'\n';
    for(int i=-1;i<5;++i){const auto begin=Clock::now();q.ext_oneapi_submit_barrier().wait_and_throw();rt->check();
        std::cout<<"CONTROL repetition="<<i<<" wall_s="<<std::chrono::duration<double>(Clock::now()-begin).count()<<'\n';}
    constexpr int Batch=1024;int scenario=0;
    for(int cells:{32768,65536,117488})for(int pattern:{0,1}){
        Fixture f(rt,q,{cells,131072,256,2051,pattern});f.qualify();
        for(int rep=-1;rep<3;++rep)for(int slot=0;slot<2;++slot){const bool arm=((process+scenario+rep+1+slot)&1)!=0;
            std::vector<sycl::event> events(Batch);f.finish();const auto begin=Clock::now();
            for(int i=0;i<Batch;++i){events[i]=f.call(arm);if(i%64==63)f.finish();}
            f.finish();const double wall=std::chrono::duration<double>(Clock::now()-begin).count();double service=0;
            uint64_t last=0;
            for(const auto& event:events){const auto a=event.get_profiling_info<sycl::info::event_profiling::command_start>(),
                b=event.get_profiling_info<sycl::info::event_profiling::command_end>();
                need(b>a&&(!last||a>=last),"event clock/order failed");last=b;service+=double(b-a)/1e9;}
            need(wall>=.02&&service>=.02&&service<=wall*1.01,"ineligible batch wall/service");f.output();
            std::cout<<"SAMPLE process="<<process<<" scenario="<<scenario<<" cells="<<cells<<" pattern="<<pattern
                <<" queries=256 capacity=131072 cap=2051 batch="<<Batch<<" checkpoint=64 repetition="<<rep
                <<" arm="<<arm<<" wall_s="<<wall<<" service_s="<<service<<" output=PASS\n";}
        f.immutable();std::cout<<"SCENARIO index="<<scenario<<" inputs=PASS guards=PASS\n";++scenario;}
}
int main(int argc,char** argv){try{std::cout<<std::setprecision(12)<<std::unitbuf;
    need(argc>=2,"explicit bounded mode required");const std::string mode=argv[1];
    need((argc==2&&(mode=="--smoke"||mode=="--numerics"))||(argc==3&&mode=="--properties")||
        (argc==3&&mode=="--compare"&&(std::string(argv[2])=="0"||std::string(argv[2])=="1"||std::string(argv[2])=="2")),"bounded mode/arguments");
    comparator_controls();gate("before-runtime");auto rt=sb::runtime_for();
    need(rt->device().is_gpu()&&rt->device().get_info<sycl::info::device::vendor_id>()==0x8086,"Intel GPU required");
    std::cout<<"DEVICE name="<<rt->device().get_info<sycl::info::device::name>()<<'\n';gate("runtime-ready");
    auto q=rt->make_queue(true);Drain drain{q,rt};
    if(mode=="--properties")properties(rt,q,argv[2]);else if(mode=="--compare")timing(rt,q,argv[2][0]-'0');
    else numerical(rt,q,mode=="--smoke");q.wait_and_throw();rt->check();
    std::cout<<"DONE component_only=1 decode_changed=0\n";gate("before-exit");return 0;
}catch(const std::exception& e){std::cerr<<"FAIL topk="<<e.what()<<'\n';std::cerr.flush();std::_Exit(1);}}
