#include "strata/sycl_upstream/memory.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <vector>
using namespace strata::sycl_upstream;
using namespace std::chrono_literals;
void check(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class F> void rejects(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid memory operation was accepted");
}
struct Mapping {
    Memory& memory;
    void* data;
    size_t bytes;
    std::vector<void*> registrations;
    Mapping(Memory& m, size_t bytes, int fd=-1, bool read_only=false): memory(m), bytes(bytes) {
        data = mmap(nullptr, bytes, read_only ? PROT_READ : PROT_READ|PROT_WRITE,
                    fd < 0 ? MAP_PRIVATE|MAP_ANONYMOUS : MAP_SHARED, fd, 0);
        check(data != MAP_FAILED, "mmap");
    }
    void add(size_t offset, size_t bytes, bool read_only=false) {
        void* p = static_cast<char*>(data) + offset;
        registrations.push_back(p); memory.register_host(p, bytes, read_only);
    }
    ~Mapping() {
        for (void* p : registrations) {
            auto info = memory.info(p);
            if (info && info->base == p && info->kind == Memory::Kind::registered_host) memory.unregister_host(p);
        }
        munmap(data, bytes);
    }
};
struct Gate {
    std::promise<void> release;
    std::shared_future<void> future = release.get_future().share();
    bool opened=false;
    void open() { if (!opened) { release.set_value(); opened=true; } }
    ~Gate() { open(); }
};
int main(int argc, char** argv) try {
    Runtime rt{sycl::device{sycl::gpu_selector_v}};
    Memory memory(rt);
    const auto stream = rt.create_stream(true);
    const auto page = memory.page_size();
    std::cout << "device=" << rt.device().get_info<sycl::info::device::name>() << " page=" << page << std::endl;
    if (argc == 2 && !std::strcmp(argv[1], "--expect-registration-rejected")) {
        Mapping map(memory,page);
        try { memory.register_host(map.data,page); }
        catch (const std::runtime_error& e) {
            check(std::strstr(e.what(), "NEO-19425"), "wrong rejection reason");
            check(!memory.info(map.data) && !memory.stats().imported_bytes, "rejected import changed ownership");
            *static_cast<int*>(map.data)=73;
            std::cout << "PASS unsupported_driver_rejected_before_import=1 reason=" << e.what() << '\n';
            return 0;
        }
        throw std::runtime_error("old driver rejection was expected");
    }
    int cases=0; size_t comparisons=0;
    auto* device = static_cast<uint32_t*>(memory.allocate_device(page * 4));
    auto* host = static_cast<uint32_t*>(memory.allocate_host(page * 4));
    check(uintptr_t(device)%256==0 && uintptr_t(host)%64==0, "allocation alignment");
    check(memory.info(device+1)->base==device && memory.info(host+1)->kind==Memory::Kind::host,
          "interior allocation lookup");
    check(memory.device_alias(host+3)==host+3, "host alias offset");
    check(!memory.info(host+page), "one-past-end lookup");
    check(memory.allocate_device(0)==nullptr && memory.allocate_host(0)==nullptr, "zero allocation");
    memory.free_host(nullptr); memory.free_device(nullptr);
    rejects([&]{ memory.free_device(host); }); rejects([&]{ memory.free_host(host+1); });
    rejects([&]{ memory.device_alias(device); });
    for (size_t i=0;i<page;++i) host[i]=uint32_t(i*7+3);
    rt.copy(stream,device,host,page*4);
    rt.enqueue(stream,[=](sycl::queue& q){return q.parallel_for(sycl::range<1>(page),[=](sycl::id<1> i){device[i]+=9;});});
    rt.copy(stream,host,device,page*4);rt.synchronize(stream);
    for(size_t i=0;i<page;++i){check(host[i]==i*7+12,"owned copy/kernel");++comparisons;}
    ++cases;
    {
        Mapping map(memory,page*5);
        const size_t aoff=64, abytes=page+64, boff=aoff+abytes, bbytes=page+128;
        map.add(aoff,abytes);
        auto before=memory.stats();
        rejects([&]{map.add(aoff,abytes);});
        rejects([&]{memory.register_host(static_cast<char*>(map.data)+aoff+4,16);});
        rejects([&]{memory.register_host(static_cast<char*>(map.data)+boff,bbytes,true);});
        check(memory.stats().imported_bytes==before.imported_bytes,"failed registration changed mappings");
        map.add(boff,bbytes);
        check(memory.stats().imported_bytes==page*3 && memory.stats().imported_ranges==2,"shared edge page imported twice");
        auto* a=static_cast<uint32_t*>(memory.device_alias(static_cast<char*>(map.data)+aoff));
        auto* b=static_cast<uint32_t*>(memory.device_alias(static_cast<char*>(map.data)+boff));
        auto* end=static_cast<char*>(map.data)+boff+bbytes;
        check(!memory.info(end),"one-past-registration lookup");
        for(size_t i=0;i<abytes/4;++i)a[i]=uint32_t(13+i);
        rt.copy(stream,device,a,abytes);
        rt.copy(stream,host,device,abytes);rt.synchronize(stream);
        for(size_t i=0;i<abytes/4;++i){check(host[i]==13+i,"imported H2D/D2H");++comparisons;}
        rt.begin_capture(stream);
        rt.enqueue(stream,[=](sycl::queue& q){return q.parallel_for(sycl::range<1>(bbytes/4),[=](sycl::id<1> i){b[i]+=17;});});
        auto graph=rt.end_capture(stream);
        for(int run=0;run<4;++run){
            for(size_t i=0;i<bbytes/4;++i)b[i]=uint32_t(run*100+i);
            if(run==2) {
                memory.unregister_host(a);
                rejects([&]{memory.device_alias(a);});
                check(memory.stats().registered_bytes==bbytes && memory.stats().imported_bytes==page*3,
                      "shared backing was released while neighbor needed its edge page");
            }
            rt.launch(graph,stream);rt.synchronize(stream);
            for(size_t i=0;i<bbytes/4;++i){check(b[i]==run*100+i+17,"registered graph replay");++comparisons;}
        }
        rejects([&]{memory.unregister_host(b+1);});
        memory.unregister_host(b);
        check(memory.stats().registered_bytes==0 && memory.stats().imported_bytes==0,"registration backing leaked");
        a[0]=41;b[0]=42;check(a[0]==41 && b[0]==42,"unregister freed caller storage");
        rejects([&]{memory.unregister_host(b);});
        for (int run=0;run<8;++run) {
            map.add(0,page);
            auto* p=static_cast<uint32_t*>(map.data);p[0]=uint32_t(run*11);
            rt.enqueue(stream,[=](sycl::queue& q){return q.single_task([=]{p[0]+=3;});});
            memory.unregister_host(p); // Waits for the kernel before native release.
            check(p[0]==run*11+3 && !memory.stats().imported_bytes, "same-address reimport");
            ++comparisons;
        }
        ++cases;
    }
    {
        FILE* file=tmpfile();check(file,"temporary file");
        check(ftruncate(fileno(file),page*2)==0,"resize file");
        {
            Mapping map(memory,page*2,fileno(file));map.add(0,page*2);
            auto* p=static_cast<uint32_t*>(memory.device_alias(map.data));
            std::fill(p,p+page/2,19);
            rt.enqueue(stream,[=](sycl::queue& q){return q.parallel_for(sycl::range<1>(page/2),[=](sycl::id<1> i){p[i]=uint32_t(i+31);});});
            rt.synchronize(stream);
            for(size_t i=0;i<page/2;++i){check(p[i]==i+31,"file-backed mapped kernel");++comparisons;}
        }
        {
            Mapping map(memory,page*2,fileno(file),true);map.add(0,page*2,true);
            const auto* p=static_cast<const uint32_t*>(memory.device_alias(map.data));
            check(memory.info(p)->read_only,"read-only registration metadata");
            rt.enqueue(stream,[=](sycl::queue& q){return q.parallel_for(sycl::range<1>(page/2),[=](sycl::id<1> i){device[i]=p[i]+7;});});
            rt.copy(stream,host,device,page*2);rt.synchronize(stream);
            for(size_t i=0;i<page/2;++i){check(host[i]==i+38,"read-only imported kernel");++comparisons;}
        }
        fclose(file);++cases;
    }
    {
        Mapping map(memory,page);map.add(0,page);
        auto* registered=static_cast<uint32_t*>(map.data);registered[0]=77;
        auto* temporary=static_cast<uint32_t*>(memory.allocate_host(64));temporary[0]=55;
        auto busy=rt.create_stream(true);
        Gate gate;const auto held=gate.future;
        rt.host_function(busy,[held]{held.wait();});
        rt.copy(busy,device,temporary,4);rt.copy(busy,device+1,registered,4);
        rt.destroy_stream(busy);
        std::promise<void> freeing,unregistering;
        auto started1=freeing.get_future(),started2=unregistering.get_future();
        auto freed=std::async(std::launch::async,[&]{freeing.set_value();memory.free_host(temporary);});
        auto unregistered=std::async(std::launch::async,[&]{unregistering.set_value();memory.unregister_host(registered);});
        started1.wait();started2.wait();
        bool waits=freed.wait_for(50ms)==std::future_status::timeout && unregistered.wait_for(50ms)==std::future_status::timeout;
        auto allocated=std::async(std::launch::async,[&]{return memory.allocate_host(32);});
        bool independent=allocated.wait_for(2s)==std::future_status::ready;
        gate.open();freed.get();unregistered.get();auto* other=allocated.get();
        memory.free_host(other);
        rt.copy(stream,host,device,8);rt.synchronize(stream);
        check(waits && independent && host[0]==55 && host[1]==77,"retired-queue release lifetime or memory mutex held across wait");
        registered[0]=91;check(registered[0]==91 && !memory.info(temporary),"release ownership");
        ++cases;
    }
    {
        rt.begin_capture(stream);
        rejects([&]{memory.allocate_host(64);});rejects([&]{rt.end_capture(stream);});
        rt.begin_capture(stream);
        rejects([&]{memory.free_device(device);});rejects([&]{rt.end_capture(stream);});
        check(memory.info(device).has_value(),"failed capture free lost allocation");
        rt.begin_capture(stream);
        rejects([&]{memory.register_host(host,64);});rejects([&]{rt.end_capture(stream);});
        rt.begin_capture(stream);
        rt.enqueue(stream,[=](sycl::queue& q){return q.single_task([=]{device[0]=111;});});
        auto outside=std::async(std::launch::async,[&]{return memory.allocate_device(64);});
        void* p=outside.get();auto graph=rt.end_capture(stream);
        rt.launch(graph,stream);rt.copy(stream,host,device,4);rt.synchronize(stream);
        check(host[0]==111,"other-thread allocation invalidated thread-local capture");
        memory.free_device(p);++cases;
    }
    rejects([&]{memory.register_host(nullptr,64);});
    rejects([&]{memory.register_host(host,0);});
    rejects([&]{memory.register_host(reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-3),8);});
    memory.free_host(host);memory.free_device(device);
    rejects([&]{memory.free_host(host);});
    const auto final=memory.stats();
    check(!final.device_bytes && !final.host_bytes && !final.registered_bytes && !final.imported_bytes,"memory domain leaked tracked storage");
    std::cout<<"PASS memory_cases="<<cases<<" exact_values="<<comparisons
             <<" registered_graph_replays=4 same_address_reimports=8 native_release_queries=1 shared_edge_page=1 readonly_file=1 release_waits_retired=1 capture_restrictions=1\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
