// Authored for Strata; repository LICENSE applies. Mechanism/ownership probe,
// NOT an LLM throughput benchmark. Isolate exact packed-copy batching with
// identical integer readers in one qualified in-order queue; no XMX claim.
#include "strata/prefill/expert_batch_upload.hpp"
#include <chrono>
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
void timing_controls() {
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
    auto q = runtime->make_queue(true);
    for (int i = -1; i < 5; ++i) {
        const auto begin = Clock::now();
        q.ext_oneapi_submit_barrier().wait_and_throw(); runtime->check();
        std::cout << "CONTROL repetition=" << i
            << " wall_s=" << std::chrono::duration<double>(Clock::now() - begin).count() << '\n';
    }
}

std::vector<int32_t> expert_order(const std::string& pattern, int count) {
    std::vector<int32_t> order(size_t(count), 0);
    for (int i = 0; i < count; ++i) {
        if (pattern == "contiguous") order[size_t(i)] = i + 1;
        else if (pattern == "runs4") order[size_t(i)] = (i / 4) * 6 + i % 4 + 1;
        else if (pattern == "gapped") order[size_t(i)] = i * 3 + 1;
        else if (pattern == "edge" && count == 4) order = {0, 1, 510, 511};
        else throw std::invalid_argument("unknown bounded expert pattern");
    }
    return order;
}
void batch(bool grouped, size_t bytes, int loops, int count, size_t capacity,
           const std::string& pattern, int repetition, bool cancel = false,
           bool reject_controls = false, bool compat = false) {
    need(bytes && bytes % 256 == 0 && loops >= 1 && count >= 1 && count <= 160 &&
         capacity >= 1 && capacity <= 128, "bounded fixture geometry");
    auto runtime = sb::runtime_for();
    auto queue = runtime->make_queue(true);
    const size_t words = bytes / 4, source_words = 512 * words;
    const unsigned seed = unsigned(repetition + 7);
    std::vector<uint32_t> host(source_words + 2 * Guard, Sentinel);
    auto* source = reinterpret_cast<const uint8_t*>(host.data() + Guard);
    auto order = expert_order(pattern, count);
    need(sycl::get_pointer_type(source,queue.get_context()) == sycl::usm::alloc::unknown,
         "ordinary CPU source required");
    for (auto e : order) for (size_t i = 0; i < words; ++i)
        host[Guard + size_t(e) * words + i] = word(i, e, seed);
    sb::Allocation bank(runtime, capacity * bytes + 2 * Guard * 4, sb::MemoryKind::Device);
    sb::Allocation output(runtime, size_t(count) * bytes + 2 * Guard * 4, sb::MemoryKind::Device);
    sb::Allocation status(runtime, 4, sb::MemoryKind::Device);
    auto* device = bank.as<uint8_t>() + Guard * 4;
    auto* results = output.as<uint32_t>() + Guard;
    auto* bad = status.as<uint32_t>();
    std::vector<uint32_t> observed(output.size() / 4), bank_observed(bank.size() / 4);
    Drain cleanup{queue,runtime}; // Before every borrowed owner can unwind.
    queue.memset(bank.data(),0xa5,bank.size());
    queue.memset(output.data(),0xa5,output.size());
    queue.memset(status.data(),0,4);
    queue.wait_and_throw(); runtime->check();
    std::vector<sycl::event> copies, readers;
    copies.reserve(size_t(count)); readers.reserve(size_t(count)*2);
    uint32_t flags = ~uint32_t(0);
    size_t completed=0, uploads=0, calls=0, requested_bytes=0;
    unsigned rejected=0;
    bool cancelled=false;
    double submission_s=0;
    auto reject = [&](auto action) {
        bool caught=false;
        try {action();} catch (const std::invalid_argument&) {caught=true;}
        need(caught,"invalid batch admission/order accepted"); ++rejected;
    };
    const auto begin=Clock::now();
    try {
        if (reject_controls) {
            auto make = [&](void* b,size_t bank_bytes,size_t dest_stride,const void* s,
                            size_t source_bytes,size_t source_stride,
                            std::vector<int32_t> ids,void* q) {
                pd::ExpertUploadBatch test(b,bank_bytes,dest_stride,s,source_bytes,
                    source_stride,bytes,ids,q,true);
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
        }
        pd::ExpertUploadBatch schedule(device,capacity*bytes,bytes,source,512*bytes,
                                     bytes,bytes,order,&queue,grouped);
        if (reject_controls) {
            reject([&]{schedule.consumed(0);});
            reject([&]{schedule.acquire(1);});
            reject([&]{schedule.finish_submissions();});
        }
        for (int e=0;e<count;++e) {
            const int expert=order[size_t(e)];
            const auto lease=compat ? schedule.acquire(size_t(e)) :
                schedule.acquire(size_t(e),[&](void* dst,const void* src,size_t n,void* stream) {
                    need(stream==&queue,"batch enqueue stream differs");
                    copies.push_back(queue.memcpy(dst,src,n)); requested_bytes+=n; return true;
                });
            need(lease.expert==expert && lease.ordinal==size_t(e),"batch ticket differs");
            const auto* blob=reinterpret_cast<const uint32_t*>(lease.data);
            if (reject_controls && e==0) {
                reject([&]{schedule.acquire(0);});
                reject([&]{schedule.consumed(1);});
            }
            readers.push_back(queue.parallel_for(sycl::range<1>(words),[=](sycl::id<1> id) {
                const size_t i=id[0];uint32_t x=blob[i];
                if(x!=word(i,expert,seed)) sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,
                    sycl::memory_scope::device,sycl::access::address_space::global_space>(*bad).fetch_or(1);
                for(int step=0;step<loops;++step)x=x*1664525u+1013904223u;
                results[size_t(e)*words+i]=x;
            }));
            readers.push_back(queue.parallel_for(sycl::range<1>(words),[=](sycl::id<1> id) {
                const size_t i=id[0];
                if(blob[i]!=word(i,expert,seed))sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,
                    sycl::memory_scope::device,sycl::access::address_space::global_space>(*bad).fetch_or(2);
            }));
            ++completed;schedule.consumed(size_t(e));
            uploads=schedule.uploaded_experts();calls=schedule.copy_calls();
            if(cancel && e==std::min(count-1,4))throw std::runtime_error("owned cancellation");
        }
        schedule.finish_submissions();
        if(reject_controls) {
            reject([&]{schedule.acquire(size_t(count));});
            reject([&]{schedule.consumed(size_t(count));});
            reject([&]{schedule.finish_submissions();});
        }
        submission_s=std::chrono::duration<double>(Clock::now()-begin).count();
        queue.wait_and_throw();runtime->check();
    } catch(const std::runtime_error& error) {
        need(cancel && std::string(error.what())=="owned cancellation",error.what());
        cancelled=true;submission_s=std::chrono::duration<double>(Clock::now()-begin).count();
        queue.wait_and_throw();runtime->check();
    }
    const auto end=Clock::now();need(cancelled==cancel,"cancellation route differs");
    queue.memcpy(&flags,status.data(),4);
    queue.memcpy(observed.data(),output.data(),output.size());
    queue.memcpy(bank_observed.data(),bank.data(),bank.size());
    queue.wait_and_throw();runtime->check();need(flags==0,"GPU saw wrong/overwritten expert");
    uint32_t a=1,b=0;
    for(int i=0;i<loops;++i){a*=1664525u;b=b*1664525u+1013904223u;}
    for(size_t e=0;e<size_t(count);++e)for(size_t i=0;i<words;++i)
        need(observed[Guard+e*words+i]==(e<completed?a*word(i,order[e],seed)+b:Sentinel),"output differs");
    for(size_t i=0;i<Guard;++i)
        need(observed[i]==Sentinel && observed[observed.size()-1-i]==Sentinel &&
             bank_observed[i]==Sentinel && bank_observed[bank_observed.size()-1-i]==Sentinel &&
             host[i]==Sentinel && host[host.size()-1-i]==Sentinel,"outer guard changed");
    std::vector<int32_t> occupants(capacity,-1);
    for(size_t i=0;i<uploads;++i)occupants[i%capacity]=order[i];
    for(size_t slot=0;slot<capacity;++slot)for(size_t i=0;i<words;++i)
        need(bank_observed[Guard+slot*words+i]==(occupants[slot]<0?Sentinel:word(i,occupants[slot],seed)),"bank content/unused slot differs");
    std::array<bool,512> used{};for(auto e:order)used[size_t(e)]=true;
    for(int e=0;e<512;++e)for(size_t i=0;i<words;++i)
        need(host[Guard+size_t(e)*words+i]==(used[size_t(e)]?word(i,e,seed):Sentinel),"immutable source changed");
    const size_t expected_uploads=grouped?std::min(size_t(count),((completed+capacity-1)/capacity)*capacity):completed;
    size_t expected_calls=0;
    for(size_t i=0;i<uploads;++i)
        if(!grouped || i%capacity==0 || order[i]!=order[i-1]+1)++expected_calls;
    need(uploads==expected_uploads && calls==expected_calls,"copy/expert count differs");
    need(compat || (copies.size()==calls && requested_bytes==uploads*bytes),"copy payload includes gaps/overfetch");
    std::vector<Interval> copy_times,reader_times;uint64_t copy_ns=0,reader_ns=0;
    for(const auto& e:copies){auto x=interval(e);copy_ns+=x.end-x.begin;copy_times.push_back(x);}
    for(const auto& e:readers){auto x=interval(e);reader_ns+=x.end-x.begin;reader_times.push_back(x);}
    uint64_t first=UINT64_MAX,last=0;
    for(const auto& xs:{copy_times,reader_times})for(const auto& x:xs){first=std::min(first,x.begin);last=std::max(last,x.end);}
    std::cout<<"BATCH arm="<<(grouped?"batch":"serial")<<" bytes="<<bytes<<" loops="<<loops
        <<" pattern="<<pattern<<" capacity="<<capacity<<" repetition="<<repetition
        <<" experts="<<count<<" completed="<<completed<<" uploaded_experts="<<uploads
        <<" copy_calls="<<calls<<" requested_upload_bytes="<<uploads*bytes
        <<" copy_events="<<copies.size()<<" readers="<<readers.size()
        <<" wall_s="<<std::chrono::duration<double>(end-begin).count()<<" submission_s="<<submission_s
        <<" copy_s="<<double(copy_ns)/1e9<<" reader_s="<<double(reader_ns)/1e9
        <<" overlap_s="<<double(overlap(copy_times,reader_times))/1e9
        <<" device_span_s="<<double(last-first)/1e9<<" cancelled="<<int(cancelled)
        <<" admission_rejections="<<rejected<<" compat="<<int(compat)
        <<" source_usm=unknown source=PASS guards=PASS output=PASS drain=PASS\n";
}

int main(int argc,char** argv) {
    try {
        std::cout<<std::setprecision(12)<<std::unitbuf;
        const bool smoke=argc==2 && std::string(argv[1])=="--smoke";
        const bool compare=argc==3 && std::string(argv[1])=="--compare" &&
            (std::string(argv[2])=="0" || std::string(argv[2])=="1" || std::string(argv[2])=="2");
        need(smoke||compare,"choose --smoke or --compare0|1|2");
        gate("before-runtime");auto rt=sb::runtime_for();
        need(rt->device().is_gpu() && rt->device().get_info<sycl::info::device::vendor_id>()==0x8086,"Intel GPU required");
        std::cout<<"DEVICE name="<<rt->device().get_info<sycl::info::device::name>()<<'\n';
        gate("runtime-ready");timing_controls();
        if(smoke) {
            batch(false,4096,1,8,3,"contiguous",0);
            batch(true,4096,1,8,3,"contiguous",0,false,true);
            batch(true,4096,1,8,3,"runs4",0);
            batch(true,4096,1,8,3,"gapped",0);
            batch(true,4096,1,8,3,"contiguous",1,true);
            batch(true,4096,1,1,3,"contiguous",2);
            batch(true,4096,1,8,1,"contiguous",3);
            batch(true,4096,1,4,3,"edge",4);
            batch(true,4096,1,8,3,"contiguous",5,false,false,true);
        } else {
            const int process=argv[2][0]-'0';
            for(const size_t bytes:{size_t(1971200),size_t(2662400)})
                for(const std::string pattern:{"contiguous","runs4","gapped"})
                    for(const int loops:{1,64})
                        for(int repetition=-1;repetition<5;++repetition)
                            for(int arm=0;arm<2;++arm)
                                batch(bool((arm+repetition+1+process)%2),bytes,loops,160,128,pattern,repetition);
        }
        rt->wait();rt->check();std::cout<<"DONE mechanism_only=1 decode_changed=0\n";
        gate("before-exit");return 0;
    }catch(const std::exception& error){
        std::cerr<<"FAIL expert_batch="<<error.what()<<'\n';std::cerr.flush();std::_Exit(1);
    }
}
