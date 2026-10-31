// Authored for Strata; repository LICENSE applies. Mechanism/ownership probe,
// NOT an LLM throughput benchmark. Isolate exact packed-copy batching with
// identical integer readers in one qualified in-order queue; no XMX claim.
#include "strata/prefill/expert_batch_upload.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

namespace sb = strata::sycl_backend;
namespace pd = strata::prefill::detail;
using Clock = std::chrono::steady_clock;
constexpr size_t Guard = 64;
constexpr uint32_t Sentinel = 0xa5a5a5a5;
void need(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
uint32_t word(size_t index, int expert, unsigned seed) {
    return (uint32_t(index) * 747796405u ^ uint32_t(expert + 1) * 2891336453u ^ seed * 277803737u) | 1u;
}
void gate(const char* stage) {
    if (!std::getenv("STRATA_TRANSFER_PROBE_GATES")) return;
    std::cout << "GATE stage=" << stage << std::endl;
    std::string input;
    need(bool(std::getline(std::cin, input)) && input == "continue", "controller acknowledgement");
}
struct Drain {
    sycl::queue queue;
    std::shared_ptr<sb::Runtime> runtime;
    ~Drain() {
        try { queue.wait_and_throw(); runtime->check(); } catch (...) { std::_Exit(86); }
    }
};
struct Interval { uint64_t begin, end; };
Interval interval(const sycl::event& e) {
    return {e.get_profiling_info<sycl::info::event_profiling::command_start>(),
            e.get_profiling_info<sycl::info::event_profiling::command_end>()};
}
uint64_t overlap(std::vector<Interval> a, std::vector<Interval> b) {
    auto merge = [](std::vector<Interval> rows) {
        std::sort(rows.begin(), rows.end(), [](auto x, auto y) { return x.begin < y.begin; });
        std::vector<Interval> out;
        for (auto r : rows) {
            need(r.end >= r.begin, "device event interval reversed");
            if (out.empty() || out.back().end < r.begin) out.push_back(r);
            else out.back().end = std::max(out.back().end, r.end);
        }
        return out;
    };
    a = merge(std::move(a)); b = merge(std::move(b));
    uint64_t total = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        const auto first = std::max(a[i].begin, b[j].begin), last = std::min(a[i].end, b[j].end);
        if (last > first) total += last - first;
        if (a[i].end < b[j].end) ++i; else ++j;
    }
    return total;
}
void timing_controls(sycl::queue& q) {
    std::vector<uint64_t> times;
    auto prior = Clock::now();
    for (int i = 0; i < 1000; ++i) {
        const auto now = Clock::now();
        times.push_back(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now - prior).count()));
        prior = now;
    }
    std::sort(times.begin(), times.end());
    std::cout << "TIMER samples=1000 p99_ns=" << times[990] << " max_ns=" << times.back() << '\n';
    auto runtime = sb::runtime_for();
    for (int i = -1; i < 5; ++i) {
        const auto begin = Clock::now();
        q.ext_oneapi_submit_barrier().wait_and_throw(); runtime->check();
        std::cout << "CONTROL repetition=" << i
            << " wall_s=" << std::chrono::duration<double>(Clock::now() - begin).count() << '\n';
    }
}

std::vector<int32_t> expert_order(const std::string& pattern, int count) {
    std::vector<int32_t> order(size_t(count), 0);
    for (int i=0;i<count;++i) {
        if(pattern=="contiguous")order[size_t(i)]=i+1;
        else if(pattern=="runs4")order[size_t(i)]=(i/4)*6+i%4+1;
        else if(pattern=="gapped")order[size_t(i)]=i*3+1;
        else if(pattern=="edge" && count==4)order={0,1,510,511};
        else throw std::invalid_argument("unknown bounded expert pattern");
    }
    return order;
}
size_t checked_geometry(size_t bytes,int count,size_t capacity) {
    need(bytes && bytes%256==0 && bytes<=2662400 && count>=1 && count<=160 &&
         capacity>=1 && capacity<=128,"bounded fixture geometry");return bytes;
}
void mark(uint32_t* flags,uint32_t bits) {
    sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
        sycl::access::address_space::global_space>(*flags).fetch_or(bits);
}

// Reuse the real queue and all addresses across the balanced arms. Source is
// immutable throughout a cohort; numerical checking and resets are outside the
// timed region. Drain is the final member, so it retires before EVERY owner or
// asynchronous host readback/metadata destination, including exception exits.
struct Workspace {
    std::shared_ptr<sb::Runtime> runtime;
    sycl::queue queue;
    size_t bytes,words,count,capacity;
    std::string pattern;
    unsigned seed;
    std::vector<uint32_t> host;
    std::vector<int32_t> order;
    std::array<int32_t,128> occupants{};
    uint32_t flags=~uint32_t(0),corrupt_word=0,restore_word=0;
    sb::Allocation bank,output,status,order_device,occupants_device;
    Drain drain;

    Workspace(std::shared_ptr<sb::Runtime> rt,const sycl::queue& q,size_t b,
              int n,size_t cap,const std::string& pat,unsigned sample_seed)
        :runtime(std::move(rt)),queue(q),bytes(checked_geometry(b,n,cap)),words(bytes/4),
         count(size_t(n)),capacity(cap),pattern(pat),seed(sample_seed),
         host(512*words+2*Guard,Sentinel),order(expert_order(pat,n)),
         bank(runtime,capacity*bytes+2*Guard*4,sb::MemoryKind::Device),
         output(runtime,count*bytes+2*Guard*4,sb::MemoryKind::Device),
         status(runtime,4,sb::MemoryKind::Device),order_device(runtime,count*4,sb::MemoryKind::Device),
         occupants_device(runtime,capacity*4,sb::MemoryKind::Device),drain{queue,runtime} {
        need(sycl::get_pointer_type(source(),queue.get_context())==sycl::usm::alloc::unknown,
             "ordinary CPU source required");
        for(auto e:order)for(size_t i=0;i<words;++i)host[Guard+size_t(e)*words+i]=word(i,e,seed);
        queue.memcpy(order_device.data(),order.data(),count*4);
        queue.wait_and_throw();runtime->check();
    }
    const uint8_t* source() const {return reinterpret_cast<const uint8_t*>(host.data()+Guard);}
    uint8_t* device() {return bank.as<uint8_t>()+Guard*4;}
    void prepare() {
        queue.memset(bank.data(),0xa5,bank.size());queue.memset(output.data(),0xa5,output.size());
        queue.memset(status.data(),0,4);queue.wait_and_throw();runtime->check();
    }
    uint32_t verify_gpu(size_t completed,size_t uploaded,uint32_t a,uint32_t b) {
        need(completed<=count && uploaded<=count,"verification extent");
        occupants.fill(-1);
        for(size_t i=0;i<uploaded;++i)occupants[i%capacity]=order[i];
        queue.memcpy(occupants_device.data(),occupants.data(),capacity*4);
        auto* out=output.as<uint32_t>();auto* storage=bank.as<uint32_t>();
        auto* bad=status.as<uint32_t>();auto* ids=order_device.as<int32_t>();
        auto* expected_occupants=occupants_device.as<int32_t>();
        const size_t width=words,n=count,slots=capacity;const unsigned sample_seed=seed;
        queue.parallel_for(sycl::range<1>(output.size()/4),[=](sycl::id<1> id) {
            const size_t i=id[0];
            if(i<Guard || i>=Guard+n*width) {if(out[i]!=Sentinel)mark(bad,8);}
            else {
                const size_t ordinal=(i-Guard)/width,column=(i-Guard)%width;
                const uint32_t expected=ordinal<completed?a*word(column,ids[ordinal],sample_seed)+b:Sentinel;
                if(out[i]!=expected)mark(bad,4);
            }
        });
        queue.parallel_for(sycl::range<1>(bank.size()/4),[=](sycl::id<1> id) {
            const size_t i=id[0];
            if(i<Guard || i>=Guard+slots*width) {if(storage[i]!=Sentinel)mark(bad,32);}
            else {
                const size_t slot=(i-Guard)/width,column=(i-Guard)%width;
                const int expert=expected_occupants[slot];
                if(storage[i]!=(expert<0?Sentinel:word(column,expert,sample_seed)))mark(bad,16);
            }
        });
        flags=~uint32_t(0);queue.memcpy(&flags,status.data(),4);
        queue.wait_and_throw();runtime->check();return flags;
    }
    unsigned verify_negative_controls(size_t completed,size_t uploaded,uint32_t a,uint32_t b) {
        auto* out=output.as<uint32_t>();auto* storage=bank.as<uint32_t>();
        const std::array<uint32_t*,3> targets={out+Guard,out,storage};
        const std::array<uint32_t,3> values={a*word(0,order[0],seed)+b,Sentinel,Sentinel};
        const std::array<uint32_t,3> expected_flags={4,8,32};
        for(size_t i=0;i<targets.size();++i) {
            queue.memset(status.data(),0,4);corrupt_word=values[i]^1u;
            queue.memcpy(targets[i],&corrupt_word,4);
            need(verify_gpu(completed,uploaded,a,b)==expected_flags[i],"GPU verifier accepted corruption");
            restore_word=values[i];queue.memcpy(targets[i],&restore_word,4);
            queue.wait_and_throw();runtime->check();
        }
        queue.memset(status.data(),0,4);
        need(verify_gpu(completed,uploaded,a,b)==0,"GPU verifier restoration differs");return 3;
    }
    void verify_cpu(size_t completed,uint32_t a,uint32_t b) {
        std::vector<uint32_t> out(output.size()/4),storage(bank.size()/4);
        Drain readback{queue,runtime}; // Before local DMA destinations unwind.
        queue.memcpy(out.data(),output.data(),output.size());queue.memcpy(storage.data(),bank.data(),bank.size());
        queue.wait_and_throw();runtime->check();
        for(size_t e=0;e<count;++e)for(size_t i=0;i<words;++i)
            need(out[Guard+e*words+i]==(e<completed?a*word(i,order[e],seed)+b:Sentinel),"CPU output oracle differs");
        for(size_t slot=0;slot<capacity;++slot)for(size_t i=0;i<words;++i)
            need(storage[Guard+slot*words+i]==(occupants[slot]<0?Sentinel:word(i,occupants[slot],seed)),"CPU bank oracle differs");
        for(size_t i=0;i<Guard;++i)
            need(out[i]==Sentinel && out[out.size()-1-i]==Sentinel && storage[i]==Sentinel &&
                 storage[storage.size()-1-i]==Sentinel,"CPU device guard oracle differs");
    }
    void finish_source() {
        queue.wait_and_throw();runtime->check();std::array<bool,512> used{};
        for(auto e:order)used[size_t(e)]=true;
        for(size_t e=0;e<512;++e)for(size_t i=0;i<words;++i)
            need(host[Guard+e*words+i]==(used[e]?word(i,int(e),seed):Sentinel),"immutable source differs");
        for(size_t i=0;i<Guard;++i)need(host[i]==Sentinel && host[host.size()-1-i]==Sentinel,"host guard differs");
        std::cout<<"SOURCE bytes="<<bytes<<" pattern="<<pattern<<" seed="<<seed<<" full=PASS\n";
    }
};

void batch(Workspace& w,size_t group,int loops,int repetition,bool cancel=false,
           bool reject_controls=false,bool compat=false,bool cpu_reference=false) {
    need(loops==1 || loops==64,"bounded integer workload");w.prepare();
    auto& queue=w.queue;const auto runtime=w.runtime;const size_t bytes=w.bytes,words=w.words,count=w.count,capacity=w.capacity;
    auto* device=w.device();const auto* source=w.source();const auto& order=w.order;
    auto* results=w.output.as<uint32_t>()+Guard;auto* bad=w.status.as<uint32_t>();const unsigned seed=w.seed;
    std::vector<sycl::event> copies,readers;copies.reserve(count);readers.reserve(count*2);
    size_t completed=0,uploads=0,calls=0,requested_bytes=0;unsigned rejected=0,verifier_controls=0;
    bool cancelled=false;double submission_s=0;
    uint32_t a=1,b=0;for(int i=0;i<loops;++i){a*=1664525u;b=b*1664525u+1013904223u;}
    auto reject=[&](auto action){bool caught=false;try{action();}catch(const std::invalid_argument&){caught=true;}
                               need(caught,"invalid batch admission/order accepted");++rejected;};
    const auto begin=Clock::now();
    try {
        if(reject_controls) {
            auto make=[&](void* bank,size_t bank_bytes,size_t dest_stride,const void* src,size_t source_bytes,
                          size_t source_stride,std::vector<int32_t> ids,void* stream,size_t g=4) {
                pd::ExpertUploadBatch test(bank,bank_bytes,dest_stride,src,source_bytes,source_stride,bytes,ids,stream,g);
            };
            reject([&]{make(nullptr,capacity*bytes,bytes,source,512*bytes,bytes,order,&queue);});
            reject([&]{make(device,bytes-1,bytes,source,512*bytes,bytes,order,&queue);});
            reject([&]{make(device,capacity*bytes,bytes+256,source,512*bytes,bytes,order,&queue);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes-1,bytes,order,&queue);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes,bytes+256,order,&queue);});
            reject([&]{make(device,capacity*bytes,bytes,nullptr,512*bytes,bytes,order,&queue);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes,bytes,order,nullptr);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes,bytes,{1,1},&queue);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes,bytes,{2,1},&queue);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes,bytes,{-1},&queue);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes,bytes,{512},&queue);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes,bytes,{},&queue);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes,bytes,order,&queue,0);});
            reject([&]{make(device,capacity*bytes,bytes,source,512*bytes,bytes,order,&queue,16);});
        }
        pd::ExpertUploadBatch schedule(device,capacity*bytes,bytes,source,512*bytes,bytes,bytes,order,&queue,group);
        if(reject_controls) {
            reject([&]{schedule.consumed(0);});reject([&]{schedule.acquire(1);});reject([&]{schedule.finish_submissions();});
        }
        for(size_t e=0;e<count;++e) {
            const int expert=order[e];
            const auto lease=compat?schedule.acquire(e):schedule.acquire(e,[&](void* dst,const void* src,size_t n,void* stream){
                need(stream==&queue,"batch enqueue stream differs");copies.push_back(queue.memcpy(dst,src,n));requested_bytes+=n;return true;});
            need(lease.expert==expert && lease.ordinal==e,"batch ticket differs");
            const auto* blob=reinterpret_cast<const uint32_t*>(lease.data);
            if(reject_controls && e==0){reject([&]{schedule.acquire(0);});reject([&]{schedule.consumed(1);});}
            readers.push_back(queue.parallel_for(sycl::range<1>(words),[=](sycl::id<1> id){
                const size_t i=id[0];uint32_t x=blob[i];if(x!=word(i,expert,seed))mark(bad,1);
                for(int step=0;step<loops;++step)x=x*1664525u+1013904223u;results[e*words+i]=x;
            }));
            readers.push_back(queue.parallel_for(sycl::range<1>(words),[=](sycl::id<1> id){
                if(blob[id[0]]!=word(id[0],expert,seed))mark(bad,2);
            }));
            ++completed;schedule.consumed(e);uploads=schedule.uploaded_experts();calls=schedule.copy_calls();
            if(cancel && e==std::min(count-1,size_t(4)))throw std::runtime_error("owned cancellation");
        }
        schedule.finish_submissions();
        if(reject_controls){reject([&]{schedule.acquire(count);});reject([&]{schedule.consumed(count);});reject([&]{schedule.finish_submissions();});}
        submission_s=std::chrono::duration<double>(Clock::now()-begin).count();queue.wait_and_throw();runtime->check();
    }catch(const std::runtime_error& error){
        need(cancel && std::string(error.what())=="owned cancellation",error.what());cancelled=true;
        submission_s=std::chrono::duration<double>(Clock::now()-begin).count();queue.wait_and_throw();runtime->check();
    }
    const auto end=Clock::now();need(cancelled==cancel,"cancellation route differs");
    size_t expected_uploads=0,expected_calls=0;
    while(expected_uploads<completed) {
        const size_t next=std::min({count,expected_uploads+group,expected_uploads+capacity-expected_uploads%capacity});
        for(size_t i=expected_uploads;i<next;++i)if(i==expected_uploads || order[i]!=order[i-1]+1)++expected_calls;
        expected_uploads=next;
    }
    need(uploads==expected_uploads && calls==expected_calls,"copy/expert count differs");
    need(compat || (copies.size()==calls && requested_bytes==uploads*bytes),"copy includes gaps/overfetch");
    need(w.verify_gpu(completed,uploads,a,b)==0,"GPU output/bank/guard oracle differs");
    if(reject_controls)verifier_controls=w.verify_negative_controls(completed,uploads,a,b);
    if(cpu_reference)w.verify_cpu(completed,a,b);
    std::vector<Interval> copy_times,reader_times;uint64_t copy_ns=0,reader_ns=0,first=UINT64_MAX,last=0;
    for(const auto& e:copies){auto x=interval(e);copy_ns+=x.end-x.begin;copy_times.push_back(x);}
    for(const auto& e:readers){auto x=interval(e);reader_ns+=x.end-x.begin;reader_times.push_back(x);}
    for(const auto& xs:{copy_times,reader_times})for(const auto& x:xs){first=std::min(first,x.begin);last=std::max(last,x.end);}
    std::cout<<"BATCH group="<<group<<" bytes="<<bytes<<" loops="<<loops<<" pattern="<<w.pattern
        <<" capacity="<<capacity<<" repetition="<<repetition<<" experts="<<count<<" completed="<<completed
        <<" uploaded_experts="<<uploads<<" copy_calls="<<calls<<" requested_upload_bytes="<<uploads*bytes
        <<" copy_events="<<copies.size()<<" readers="<<readers.size()<<" wall_s="<<std::chrono::duration<double>(end-begin).count()
        <<" submission_s="<<submission_s<<" copy_s="<<double(copy_ns)/1e9<<" reader_s="<<double(reader_ns)/1e9
        <<" overlap_s="<<double(overlap(copy_times,reader_times))/1e9<<" device_span_s="<<double(last-first)/1e9
        <<" cancelled="<<int(cancelled)<<" admission_rejections="<<rejected<<" compat="<<int(compat)
        <<" cpu_reference="<<int(cpu_reference)<<" verifier_negative_controls="<<verifier_controls
        <<" source_usm=unknown gpu_verify=PASS source=PASS guards=PASS output=PASS drain=PASS\n";
}

int main(int argc,char** argv) {
    try {
        std::cout<<std::setprecision(12)<<std::unitbuf;
        const bool smoke=argc==2 && std::string(argv[1])=="--smoke";
        const bool compare=argc==3 && std::string(argv[1])=="--compare" &&
            (std::string(argv[2])=="0" || std::string(argv[2])=="1" || std::string(argv[2])=="2");
        need(smoke||compare,"choose --smoke or --compare0|1|2");gate("before-runtime");auto rt=sb::runtime_for();
        need(rt->device().is_gpu() && rt->device().get_info<sycl::info::device::vendor_id>()==0x8086,"Intel GPU required");
        std::cout<<"DEVICE name="<<rt->device().get_info<sycl::info::device::name>()<<'\n';
        gate("runtime-ready");auto queue=rt->make_queue(true);timing_controls(queue);
        if(smoke) {
            auto small=[&](size_t group,int n,size_t cap,const std::string& pattern,int repetition,
                           bool cancel=false,bool reject=false,bool compat=false,int loops=1){
                Workspace w(rt,queue,4096,n,cap,pattern,7+unsigned(repetition));
                batch(w,group,loops,repetition,cancel,reject,compat,true);w.finish_source();
            };
            small(1,8,3,"contiguous",0);small(4,8,3,"contiguous",0,false,true);
            small(4,8,3,"runs4",0);small(8,8,3,"gapped",0);small(4,8,3,"contiguous",1,true);
            small(8,1,3,"contiguous",2);small(8,8,1,"contiguous",3);small(4,4,3,"edge",4);
            small(4,8,3,"contiguous",5,false,false,true);
            small(2,8,3,"contiguous",6,false,false,false,64);
        }else {
            const int process=argv[2][0]-'0';
            for(const size_t bytes:{size_t(1971200),size_t(2662400)})
                for(const std::string pattern:{"contiguous","runs4","gapped"}) {
                    Workspace w(rt,queue,bytes,160,128,pattern,7+unsigned(process));
                    for(const int loops:{1,64})for(int repetition=-1;repetition<5;++repetition)
                        for(int arm=0;arm<4;++arm) {
                            const size_t group=size_t(1)<<((arm+repetition+1+process)%4);
                            batch(w,group,loops,repetition);
                        }
                    w.finish_source();
                }
        }
        rt->wait();rt->check();std::cout<<"DONE mechanism_only=1 decode_changed=0\n";gate("before-exit");return 0;
    }catch(const std::exception& error){std::cerr<<"FAIL expert_batch="<<error.what()<<'\n';std::cerr.flush();std::_Exit(1);}
}
