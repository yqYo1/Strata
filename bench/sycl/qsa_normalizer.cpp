// Authored for Strata; repository LICENSE applies.
// Compare only the32-term denominator fold. No production/decode entrypoint.
#include "qsa-head.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <time.h>
#include <utility>
#include <vector>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace sb = strata::sycl_backend;
namespace sk = strata::kernels;
using Clock = std::chrono::steady_clock;
constexpr size_t Guard = 16;
constexpr int Dim = 256, Heads = 24, KV = 2;
constexpr double Atol = .02, Rtol = .02;
void need(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
uint16_t bf16(float x) {
    const uint32_t b=std::bit_cast<uint32_t>(x);
    return std::isnan(x)?uint16_t((b>>16)|0x40):uint16_t((b+0x7fff+((b>>16)&1))>>16);
}
float bf(uint16_t x) {return std::bit_cast<float>(uint32_t(x)<<16);}
uint16_t half(float x) {return std::bit_cast<uint16_t>(_Float16(x));}
float hf(uint16_t x) {return float(std::bit_cast<_Float16>(x));}
void gate(const char* stage) {
    std::cout<<"GATE stage="<<stage<<std::endl;std::string s;
    need(bool(std::getline(std::cin,s))&&s=="continue","controller acknowledgement");
}
struct Drain {
    sycl::queue queue;std::shared_ptr<sb::Runtime> runtime;
    ~Drain() {try{queue.wait_and_throw();runtime->check();}catch(...){std::_Exit(86);}}
};
template<class T> struct Buffer {
    sb::Allocation allocation;size_t size,guard;T marker;
    std::vector<T> upload,download;
    Buffer(std::shared_ptr<sb::Runtime> rt,size_t n,T mark,size_t g=Guard)
        :allocation(rt,(n+2*g)*sizeof(T),sb::MemoryKind::Device),size(n),guard(g),marker(mark) {}
    T* data(){return allocation.template as<T>()+guard;}
    void put(sycl::queue& q,const std::vector<T>& values) {
        need(values.size()==size,"fixture upload extent");
        q.wait_and_throw();upload.assign(size+2*guard,marker);
        std::copy(values.begin(),values.end(),upload.begin()+guard);
        q.memcpy(allocation.data(),upload.data(),allocation.size()).wait_and_throw();
    }
    const std::vector<T>& get(sycl::queue& q) {
        q.wait_and_throw();download.resize(size+2*guard);
        q.memcpy(download.data(),allocation.data(),allocation.size()).wait_and_throw();
        for(size_t i=0;i<guard;++i)
            need(download[i]==marker&&download[download.size()-1-i]==marker,"fixture guard changed");
        return download;
    }
    void immutable(sycl::queue& q){need(get(q)==upload,"fixture input modified");}
};
uint32_t random_word(uint32_t& x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}

// Final Drain runs before every allocation and host-DMA destination on unwind.
struct Fixture {
    std::shared_ptr<sb::Runtime> rt;sycl::queue queue;
    int cells,cap,queries,page,pages,rows,kind;
    std::vector<int8_t> hk,hv;std::vector<uint16_t> hks,hvs,hq;
    std::vector<int32_t> ht,hi,hs;
    Buffer<int8_t> k,v;Buffer<uint16_t> ks,vs,q,pk,pv,out;
    Buffer<int32_t> table,ids,steps,bad;
    Buffer<float> state;Buffer<uint32_t> details;
    Drain drain;
    Fixture(std::shared_ptr<sb::Runtime> runtime,const sycl::queue& stream,
            int context,int width,int nq,int pg,int distribution,bool gapped=false,bool masked=false)
        :rt(std::move(runtime)),queue(stream),cells(context),cap(width),queries(nq),page(pg),
         pages((cells+page-1)/page),rows(pages*KV*page),kind(distribution),
         hk(size_t(rows)*Dim),hv(hk.size()),hks(size_t(rows)*4),hvs(hks.size()),hq(size_t(queries)*Heads*Dim),
         ht(pages),hi(size_t(queries)*cap),hs(size_t(queries)*sk::kStepCount),
         k(rt,hk.size(),int8_t(0x55)),v(rt,hv.size(),int8_t(0x55)),ks(rt,hks.size(),uint16_t(0x59a3)),
         vs(rt,hvs.size(),uint16_t(0x59a3)),q(rt,hq.size(),uint16_t(0x59a3)),
         pk(rt,size_t(rows)*Dim,uint16_t(0x59a3),17),pv(rt,size_t(rows)*Dim,uint16_t(0x59a3),17),
         out(rt,hq.size(),uint16_t(0x59a3)),table(rt,ht.size(),int32_t(0x12345678)),
         ids(rt,hi.size(),int32_t(0x12345678)),steps(rt,hs.size(),int32_t(0x12345678)),
         bad(rt,rows,int32_t(0x12345678)),state(rt,size_t((cap+31)/32)*4,-913.75f),
         details(rt,size_t((cap+31)/32)*64+512,uint32_t(0xdeadbeef)),drain{queue,rt} {
        need(cells>=1&&cells<=262144&&cap>=1&&cap<=2051&&queries>=1&&queries<=8192&&
             (page==4||page==16),"bounded fixture dimensions");
        uint32_t seed=0x5d87a331u+unsigned(context)*17+unsigned(distribution)*1237;
        for(int p=0;p<pages;++p)ht[p]=masked&&p%7==2?-1:pages-1-p;
        for(auto& x:hk)x=int8_t(int(random_word(seed)%15)-7);
        for(auto& x:hv)x=int8_t(int(random_word(seed)%31)-15);
        for(auto& x:hks)x=half(std::ldexp(1.f,-5-int(random_word(seed)%4)));
        for(auto& x:hvs)x=half(std::ldexp((random_word(seed)&1)?1.f:-1.f,-7-int(random_word(seed)%4)));
        for(auto& x:hq)x=bf16(float(int(random_word(seed)%17)-8)*.125f);
        if(kind>=1&&kind<=5) {
            std::fill(hq.begin(),hq.end(),bf16(kind==1?0.f:kind==3?16.f:1.f));
            std::fill(hk.begin(),hk.end(),int8_t(0));std::fill(hv.begin(),hv.end(),int8_t(1));
            std::fill(hks.begin(),hks.end(),half(1.f));std::fill(hvs.begin(),hvs.end(),half(1.f));
            if(kind==3)for(size_t i=0;i<hq.size();++i)if(i%Dim)hq[i]=0;
            for(int cell=0;cell<cells;++cell)for(int head=0;head<KV;++head) {
                const int physical=ht[cell/page];if(physical<0)continue;
                const size_t row=size_t((physical*KV+head)*page+cell%page);
                const int value=kind==2?(cell==cells-1?32:-1):
                    kind==3?(cell==0?0:-int(67+cell%61)):
                    kind==4?(cell<32?-32:0):kind==5?(cell<32?0:-32):0;
                for(int d=0;d<Dim;++d)hk[row*Dim+d]=int8_t(value);
            }
        }
        for(int query=0;query<queries;++query) {
            hs[size_t(query)*sk::kStepCount+sk::kStepPos]=cells-1;
            hs[size_t(query)*sk::kStepCount+sk::kStepNKv]=cells;
            hs[size_t(query)*sk::kStepCount+sk::kStepWidth]=cap;
            int stride=239;while(std::gcd(stride,cells)!=1)stride+=2;
            for(int c=0;c<cap;++c)hi[size_t(query)*cap+c]=gapped?int((int64_t(c)*stride+query*11)%cells):c%cells;
            if(kind==2)hi[size_t(query)*cap+cap-1]=cells-1;
            if(masked&&cap>5)hi[size_t(query)*cap+5]=-1;
        }
        k.put(queue,hk);v.put(queue,hv);ks.put(queue,hks);vs.put(queue,hvs);q.put(queue,hq);
        table.put(queue,ht);ids.put(queue,hi);steps.put(queue,hs);
        pk.put(queue,std::vector<uint16_t>(pk.size,0x7fc1));pv.put(queue,std::vector<uint16_t>(pv.size,0x7fc1));
        bad.put(queue,std::vector<int32_t>(bad.size,0x12345678));
        out.put(queue,std::vector<uint16_t>(out.size,0x7fc1));
    }
    bool call(bool candidate,bool capture=false,int head=0,int kv_head=0,
              sycl::event* attention=nullptr,sycl::event* pack=nullptr) {
        sk::QsaAttnPools pools{};pools.k_q=k.data();pools.v_q=v.data();pools.k_scale=ks.data();pools.v_scale=vs.data();pools.page_table=table.data();
        auto shape=sk::qsa_real_shapes();shape.page_size=page;
        const sk::QsaBf16PackedScratch scratch{pk.data(),pv.data(),bad.data()};
        if(capture) {
            const sk::QsaPrefixStateProbe probe{0,kv_head,head,state.data()};
            return candidate?sk::qsa_head_candidate::capture(q.data(),pools,ids.data(),steps.data(),cap,shape,out.data(),queries,{pages,pages},cells,scratch,&queue,probe,attention,pack,details.data()):
                             sk::qsa_head_original::capture(q.data(),pools,ids.data(),steps.data(),cap,shape,out.data(),queries,{pages,pages},cells,scratch,&queue,probe,attention,pack,details.data());
        }
        return candidate?sk::qsa_head_candidate::batch(q.data(),pools,ids.data(),steps.data(),cap,shape,out.data(),queries,{pages,pages},cells,scratch,&queue,attention,pack):
                         sk::qsa_head_original::batch(q.data(),pools,ids.data(),steps.data(),cap,shape,out.data(),queries,{pages,pages},cells,scratch,&queue,attention,pack);
    }
    void finish(){queue.wait_and_throw();rt->check();}
    std::vector<uint16_t> output(){const auto& x=out.get(queue);return {x.begin()+Guard,x.end()-Guard};}
    void immutable() {
        k.immutable(queue);v.immutable(queue);ks.immutable(queue);vs.immutable(queue);q.immutable(queue);
        table.immutable(queue);ids.immutable(queue);steps.immutable(queue);
        const auto& packed_k=pk.get(queue);const auto& packed_v=pv.get(queue);const auto& diagnostic=bad.get(queue);
        for(int row=0;row<rows;++row) {
            const int logical=row/(KV*page),head=row/page%KV,offset=row%page,cell=logical*page+offset;
            const int physical=ht[logical];const bool valid=cell<cells&&physical>=0&&physical<pages;
            const size_t source=valid?size_t((physical*KV+head)*page+offset):0;
            need(diagnostic[Guard+row]==0,"packed row status differs");
            for(int d=0;d<Dim;++d) {
                const uint16_t key=valid?bf16(float(hk[source*Dim+d])*hf(hks[source*4+d/64])):0;
                const uint16_t value=valid?bf16(float(hv[source*Dim+d])*hf(hvs[source*4+d/64])):0;
                need(packed_k[17+size_t(row)*Dim+d]==key&&packed_v[17+size_t(row)*Dim+d]==value,"complete packed K/V differs");
            }
        }
    }
    std::vector<uint16_t> invoke(bool arm) {need(call(arm),"matrix path refused");finish();return output();}
};
double compare(const std::vector<uint16_t>& actual,const std::vector<uint16_t>& expected) {
    need(actual.size()==expected.size(),"output extent mismatch");double maximum=0;
    for(size_t i=0;i<actual.size();++i) {
        const double a=bf(actual[i]),b=bf(expected[i]);need(std::isfinite(a)&&std::isfinite(b),"nonfinite output");
        const double ratio=std::abs(a-b)/(Atol+Rtol*std::abs(b));maximum=std::max(maximum,ratio);
        need(ratio<=1,"SGLang BF16 output comparison failed");
    }
    return maximum;
}
void comparator_controls() {
    unsigned rejected=0;
    for(const auto value:{bf16(2.f),uint16_t(0x7fc1),uint16_t(0x7f80)}) {
        bool caught=false;
        try{(void)compare({value},{bf16(1.f)});}catch(const std::runtime_error&){caught=true;}
        need(caught,"output comparator accepted corruption");++rejected;
    }
    need(compare({bf16(1.f)},{bf16(1.f)})==0,"output comparator rejected identity");
    std::cout<<"COMPARATOR refusals="<<rejected<<" identity=PASS\n";
}
void oracle(Fixture& f,const std::vector<uint16_t>& a,const std::vector<uint16_t>& b) {
    double maximum=0;size_t checked=0;
    for(int query=0;query<f.queries;++query)for(int head=0;head<Heads;++head) {
        std::vector<double> score(f.cap,-INFINITY),values(size_t(f.cap)*Dim);
        double largest=-INFINITY;
        for(int c=0;c<f.cap;++c) {
            const int cell=f.hi[size_t(query)*f.cap+c];if(cell<0||cell>=f.cells)continue;
            const int physical=f.ht[cell/f.page];if(physical<0||physical>=f.pages)continue;
            const size_t row=size_t((physical*KV+head/(Heads/KV))*f.page+cell%f.page);
            double dot=0;
            for(int d=0;d<Dim;++d) {
                const float q=bf(bf16((bf(f.hq[(size_t(query)*Heads+head)*Dim+d])*.0625f)*1.4426950408f));
                const float k=bf(bf16(float(f.hk[row*Dim+d])*hf(f.hks[row*4+d/64])));
                dot+=double(q)*k;
                values[size_t(c)*Dim+d]=bf(bf16(float(f.hv[row*Dim+d])*hf(f.hvs[row*4+d/64])));
            }
            score[c]=dot;largest=std::max(largest,dot);
        }
        std::vector<double> sum(Dim);double denominator=0;
        for(int c=0;c<f.cap;++c)if(std::isfinite(score[c])) {
            const double p=std::exp2(score[c]-largest);denominator+=p;
            for(int d=0;d<Dim;++d)sum[d]+=p*values[size_t(c)*Dim+d];
        }
        for(int d=0;d<Dim;++d) {
            const double expected=denominator>0?sum[d]/denominator:0;
            const size_t at=(size_t(query)*Heads+head)*Dim+d;
            for(const auto* output:{&a,&b}) {
                const double actual=bf((*output)[at]);need(std::isfinite(actual),"oracle output nonfinite");
                const double ratio=std::abs(actual-expected)/(Atol+Rtol*std::abs(expected));
                maximum=std::max(maximum,ratio);need(ratio<=1,"independent BF16-operand output oracle failed");++checked;
            }
        }
    }
    std::cout<<"ORACLE compared="<<checked<<" maximum_budget_ratio="<<maximum<<" rtol="<<Rtol<<" atol="<<Atol<<" PASS=1\n";
}
void captures(Fixture& f,const std::vector<uint16_t>& original,const std::vector<uint16_t>& candidate) {
    const size_t tiles=(f.cap+31)/32;double maximum_norm_delta=0;size_t compared=0;
    for(const auto coordinate:{std::pair{0,0},std::pair{0,11},std::pair{1,5},std::pair{1,15}}) {
        std::vector<float> first_state;std::vector<uint32_t> first_details;
        for(bool arm:{false,true}) {
            f.state.put(f.queue,std::vector<float>(f.state.size,-913.75f));
            f.details.put(f.queue,std::vector<uint32_t>(f.details.size,0xdeadbeef));
            need(f.call(arm,true,coordinate.second,coordinate.first),"capture refused");f.finish();
            need(f.output()==(arm?candidate:original),"capture changed corresponding normal output");
            const auto observed=f.state.get(f.queue);const auto details=f.details.get(f.queue);
            if(!arm){first_state=observed;first_details=details;continue;}
            for(size_t tile=0;tile<tiles;++tile) {
                for(size_t field:{size_t(0),size_t(1)})
                    need(std::bit_cast<uint32_t>(observed[Guard+tile*4+field])==std::bit_cast<uint32_t>(first_state[Guard+tile*4+field]),"maximum/alpha bits changed");
                for(size_t field:{size_t(2),size_t(3)}) {
                    const float x=observed[Guard+tile*4+field],y=first_state[Guard+tile*4+field];
                    need(std::isfinite(x)&&std::isfinite(y)&&x>=0&&y>=0,"normalizer nonfinite/negative");
                    maximum_norm_delta=std::max(maximum_norm_delta,double(std::abs(x-y)));
                }
                for(size_t i=0;i<64;++i)
                    need(details[Guard+tile*64+i]==first_details[Guard+tile*64+i],"probability/PV operand bits changed");
            }
            for(size_t d=0;d<Dim;++d)
                need(details[Guard+tiles*64+d]==first_details[Guard+tiles*64+d],"FP32 PV accumulator bits changed");
            ++compared;
        }
    }
    std::cout<<"STATE coordinates="<<compared<<" probability_bits=PASS PV_bits=PASS maximum_alpha_bits=PASS accumulator_bits=PASS normalizer_max_abs_delta="<<maximum_norm_delta<<" guards=PASS\n";
}
void numerical(std::shared_ptr<sb::Runtime> rt,const sycl::queue& queue,bool smoke) {
    auto run=[&](int cells,int cap,int queries,int page,int kind,bool masked=false) {
        Fixture f(rt,queue,cells,cap,queries,page,kind,false,masked);
        const auto original=f.invoke(false),candidate=f.invoke(true);
        const double error=compare(candidate,original);oracle(f,original,candidate);captures(f,original,candidate);
        need(f.invoke(false)==original&&f.invoke(true)==candidate,"per-arm repeatability failed");f.immutable();
        std::cout<<"NUMERICAL cells="<<cells<<" cap="<<cap<<" queries="<<queries<<" page="<<page<<" kind="<<kind<<" masked="<<masked
                 <<" output_budget_ratio="<<error<<" inputs=PASS guards=PASS repeatability=PASS\n";
    };
    if(smoke){run(17,17,1,4,0);return;}
    for(int page:{4,16}) {
        for(int cap:{1,31,32,33,65,2051})run(2113,cap,3,page,0,true);
        for(int kind:{1,2,3,4,5})run(97,65,1,page,kind);
    }
    run(262144,2051,1,16,0);
}
void properties(std::shared_ptr<sb::Runtime> rt,const sycl::queue& queue,const char* loader_path) {
    void* loader=dlopen(loader_path,RTLD_NOLOAD|RTLD_LOCAL|RTLD_NOW);need(loader,"Level Zero loader absent");
    auto query=reinterpret_cast<decltype(&zeKernelGetProperties)>(dlsym(loader,"zeKernelGetProperties"));need(query,"kernel property API absent");
    int count=0;
    for(const auto& id:sycl::get_kernel_ids()) {
        const std::string name=id.get_name();const bool candidate=name.find("qsa_head_candidate")!=std::string::npos;
        if(!candidate&&name.find("qsa_head_original")==std::string::npos)continue;
        auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(queue.get_context(),{queue.get_device()},{id});
        auto kernel=bundle.get_kernel(id);const auto sg=kernel.get_info<sycl::info::kernel_device_specific::compile_sub_group_size>(queue.get_device());
        if(sg!=16)continue;
        ze_kernel_properties_t p{};p.stype=ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
        need(query(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kernel),&p)==ZE_RESULT_SUCCESS,"kernel properties failed");
        need(kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(queue.get_device())>=128&&
             p.requiredSubgroupSize==16&&p.maxNumSubgroups>=8,"compiled group geometry");
        need((!p.requiredGroupSizeX||p.requiredGroupSizeX==128)&&(!p.requiredGroupSizeY||p.requiredGroupSizeY==1)&&
             (!p.requiredGroupSizeZ||p.requiredGroupSizeZ==1)&&(!p.requiredNumSubGroups||p.requiredNumSubGroups==8),"compiled required dimensions");
        need(size_t(p.localMemSize)+20036+4096<=rt->device().get_info<sycl::info::device::local_mem_size>(),"conservative local allocation");
        std::cout<<"PROPERTY arm="<<candidate<<" subgroup="<<sg<<" static_local="<<p.localMemSize<<" spill="<<p.spillMemSize<<" PASS=1\n";++count;
    }
    need(count==4,"four normal/state kernel properties required");dlclose(loader);
}
void timing(std::shared_ptr<sb::Runtime> rt,sycl::queue queue,int process) {
    static_assert(Clock::is_steady);timespec resolution{};
    need(clock_getres(CLOCK_MONOTONIC,&resolution)==0&&resolution.tv_sec==0&&resolution.tv_nsec<=1000,"monotonic clock resolution");
    std::vector<uint64_t> clock;auto prior=Clock::now();
    for(int i=0;i<1000;++i){const auto now=Clock::now();clock.push_back(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now-prior).count()));prior=now;}
    std::sort(clock.begin(),clock.end());need(clock[990]<=10000,"timer p99 above10us");
    std::cout<<"TIMER samples=1000 p99_ns="<<clock[990]<<" resolution_ns="<<resolution.tv_nsec<<'\n';
    for(int i=-1;i<5;++i){const auto begin=Clock::now();queue.ext_oneapi_submit_barrier().wait_and_throw();rt->check();
        std::cout<<"CONTROL repetition="<<i<<" wall_s="<<std::chrono::duration<double>(Clock::now()-begin).count()<<'\n';}
    constexpr int Batch=8;int scenario=0;
    for(int cells:{32768,117488})for(bool gapped:{false,true}) {
        Fixture f(rt,queue,cells,2051,8192,4,0,gapped);
        const auto original=f.invoke(false),candidate=f.invoke(true);compare(candidate,original);
        for(int repetition=-1;repetition<3;++repetition)for(int slot=0;slot<2;++slot) {
            const bool arm=((process+scenario+repetition+1+slot)&1)!=0;
            std::vector<sycl::event> attention(Batch),pack(Batch);f.finish();
            const auto begin=Clock::now();
            for(int i=0;i<Batch;++i)need(f.call(arm,false,0,0,&attention[i],&pack[i]),"timed path refused");
            f.finish();const double wall=std::chrono::duration<double>(Clock::now()-begin).count();double attention_s=0,pack_s=0;
            auto duration=[](const sycl::event& event){const auto a=event.get_profiling_info<sycl::info::event_profiling::command_start>(),b=event.get_profiling_info<sycl::info::event_profiling::command_end>();need(b>a,"event clock reversed");return double(b-a)/1e9;};
            for(int i=0;i<Batch;++i){attention_s+=duration(attention[i]);pack_s+=duration(pack[i]);}
            need(wall>=.1&&attention_s+pack_s<=wall*1.01,"ineligible batch duration/service");
            need(f.output()==(arm?candidate:original),"timed corresponding-arm output changed");
            std::cout<<"SAMPLE process="<<process<<" scenario="<<scenario<<" cells="<<cells<<" gapped="<<gapped<<" queries=8192 cap=2051 page=4 batch=8 repetition="<<repetition<<" arm="<<arm
                     <<" wall_s="<<wall<<" pack_s="<<pack_s<<" attention_s="<<attention_s<<" output=PASS\n";
        }
        f.immutable();std::cout<<"CASE scenario="<<scenario<<" inputs=PASS guards=PASS\n";++scenario;
    }
}
int main(int argc,char** argv) {
    try {
        std::cout<<std::setprecision(12)<<std::unitbuf;
        need(argc>=2,"explicit mode required");const std::string mode=argv[1];
        need((argc==2&&(mode=="--smoke"||mode=="--numerics"))||
             (argc==3&&mode=="--properties")||(argc==3&&mode=="--compare"&&
             (std::string(argv[2])=="0"||std::string(argv[2])=="1"||std::string(argv[2])=="2")),"bounded mode/arguments required");
        comparator_controls();
        gate("before-runtime");auto rt=sb::runtime_for();need(rt->device().is_gpu()&&rt->device().get_info<sycl::info::device::vendor_id>()==0x8086,"Intel GPU required");
        std::cout<<"DEVICE name="<<rt->device().get_info<sycl::info::device::name>()<<'\n';gate("runtime-ready");
        auto queue=rt->make_queue(true);Drain drain{queue,rt};
        if(mode=="--properties")properties(rt,queue,argv[2]);
        else if(mode=="--compare")timing(rt,queue,argv[2][0]-'0');
        else numerical(rt,queue,mode=="--smoke");
        queue.wait_and_throw();rt->check();std::cout<<"DONE component_only=1 decode_changed=0\n";gate("before-exit");return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL normalizer="<<e.what()<<'\n';std::cerr.flush();std::_Exit(1);}
}
