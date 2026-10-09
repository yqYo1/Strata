// Execute the actual extracted Stager with asynchronous CPU DMA/event stubs.
// This checks host lifetime/publication only; it cannot validate SYCL/Level Zero.
#include "strata/event_completion.hpp"
#include "strata/host_wait.hpp"
#include "strata/sycl_allocation.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <future>
#include <sstream>
#include <pthread.h>
#include <sched.h>
#include <cerrno>
using Clock=std::chrono::steady_clock;
namespace sycl {
namespace info { struct event { struct command_execution_status {}; }; enum class event_command_status {running,complete}; }
struct event {
    std::shared_ptr<std::atomic<bool>> done;
    template<class Tag> auto get_info() const {
        return !done || done->load(std::memory_order_acquire) ? info::event_command_status::complete : info::event_command_status::running;
    }
};
}
struct Queue {
    std::mutex mu;std::condition_variable cv;
    std::deque<std::function<void()>> jobs;
    bool quit=false,busy=false;
    std::thread worker;
    Queue():worker([this] {
        for (;;) {
            std::function<void()> job;
            {std::unique_lock lock(mu);cv.wait(lock,[&]{return quit||!jobs.empty();});
             if(quit&&jobs.empty())return;
             job=std::move(jobs.front());jobs.pop_front();busy=true;}
            std::this_thread::sleep_for(std::chrono::microseconds(40));job();
            {std::lock_guard lock(mu);busy=false;}cv.notify_all();
        }
    }){}
    sycl::event read(const uint8_t* p,size_t bytes,uint8_t expected) {
        sycl::event e{std::make_shared<std::atomic<bool>>(false)};
        {std::lock_guard lock(mu);jobs.emplace_back([=] {
            for(size_t i=0;i<bytes;++i)assert(p[i]==expected);
            e.done->store(true,std::memory_order_release);
        });}cv.notify_one();return e;
    }
    void drain() {std::unique_lock lock(mu);cv.wait(lock,[&]{return jobs.empty()&&!busy;});}
    ~Queue(){drain();{std::lock_guard lock(mu);quit=true;}cv.notify_all();worker.join();}
};
namespace dpct {
inline Queue& get_in_order_queue(){static Queue q;return q;}
inline int get_current_device_id(){return 0;}
inline void select_device(int){}
}
namespace sycl {
inline void* malloc_host(size_t n,Queue&){return std::malloc(n);}
inline void free(void* p,Queue&){std::free(p);}
}
namespace core {
struct ExpertSource {bool copy_blob(int32_t,int32_t,uint8_t*){return false;}};
struct GgufExpertSource {bool read_into(int32_t,int32_t,uint8_t*,size_t)const{return false;}};
}
namespace strata {
inline bool dma_completed(const EventCompletion<sycl::event>& s,uint64_t n) {
    return event_completed(s,n,[](const sycl::event& e){return e.get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete;});
}
}
inline bool force_pageable(){return false;}
#define DPCT_CHECK_ERROR(...) ([&]{__VA_ARGS__;return 0;}())
#include "actual-stager.hpp"
int main(int argc,char** argv) {
    assert(argc==4);
    const int ring=std::stoi(argv[1]),workers=std::stoi(argv[2]),generations=std::stoi(argv[3]);
    setenv("STRATA_STAGER_RING",argv[1],1);
    const int main_cpu=std::stoi(std::getenv("PREF_HOST_MAIN_CPU"));
    cpu_set_t parent_mask;CPU_ZERO(&parent_mask);CPU_SET(main_cpu,&parent_mask);
    assert(pthread_setaffinity_np(pthread_self(),sizeof(parent_mask),&parent_mask)==0);
    const bool expected_failure=std::getenv("PREF_HOST_EXPECT_REJECTION")!=nullptr;
    std::vector<int> expected;
    std::string cpu_text=std::getenv("PREF_HOST_EXPECT_CPUS");std::replace(cpu_text.begin(),cpu_text.end(),',',' ');
    std::istringstream cpu_stream(cpu_text);for(int cpu;cpu_stream>>cpu;)expected.push_back(cpu);
    constexpr size_t bytes=128;
    const int count=3*ring;
    std::vector<std::vector<uint8_t>> source(count,std::vector<uint8_t>(bytes));
    for(int i=0;i<count;++i)std::fill(source[i].begin(),source[i].end(),uint8_t(i+1));
    Queue q;
    std::unique_ptr<Stager> s;
    for(int gen=0;gen<generations;++gen) {
        if(!s){
            s=std::make_unique<Stager>();const bool ok=s->init(bytes,workers);
            if(expected_failure){assert(!ok);s.reset();std::puts("PASS rejected affinity safely");return 0;}
            assert(ok && !expected.empty());
            for(size_t i=0;i<s->threads.size();++i){
                cpu_set_t actual;CPU_ZERO(&actual);
                assert(pthread_getaffinity_np(s->threads[i].native_handle(),sizeof(actual),&actual)==0);
                assert(CPU_COUNT(&actual)==1 && CPU_ISSET(expected[i%expected.size()],&actual));
            }
        }
        std::vector<Stager::Job> js;
        for(int i=0;i<count;++i)js.push_back({source[i].data(),bytes});
        s->start(std::move(js));
        for(int i=0;i<count;++i){const auto* p=s->wait(i);s->issued_one(i,q.read(p,bytes,uint8_t(i+1)));}
        s->finish();
        // Last DMA reads stay pending until the next generation's worker waits.
        // At the phase boundary drain before freeing the actual Stager.
        if(gen%7==6){q.drain();s.reset();}
    }
    q.drain();s.reset();
    std::printf("PASS ring=%d workers=%d generations=%d DMA_reads=%d\n",ring,workers,generations,count*generations);
}
