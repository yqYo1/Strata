#pragma once
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
namespace strata_prefill_trace {
using Clock = std::chrono::steady_clock;
struct Item { const char* kind; sycl::event event; size_t bytes; int type; int64_t rows,cols,max_rows; double submit_us; };
struct State { std::atomic<bool> active{false}; std::mutex mutex; std::vector<Item> items; };
inline State& state() { static State* s = new State; return *s; }
inline void put(const char* kind, sycl::event e, Clock::time_point start, size_t bytes, int type, int64_t rows=0, int64_t cols=0, int64_t max_rows=0) {
    if (!state().active.load(std::memory_order_relaxed)) return;
    const double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
    std::lock_guard lock(state().mutex);
    state().items.push_back({kind,e,bytes,type,rows,cols,max_rows,us});
}
struct Scope {
    const char* path=std::getenv("STRATA_PREFILL_DEVICE_TRACE");
    Scope() { if(path && *path) state().active.store(true); }
    ~Scope() {
        if(!path || !*path) return;
        state().active.store(false);
        std::vector<Item> items;
        { std::lock_guard lock(state().mutex);items.swap(state().items); }
        std::FILE* f=std::fopen(path,"a");if(!f) return;
        if(std::ftell(f)==0) std::fprintf(f,"kind,type,bytes,rows,cols,max_rows,submit_us,start_ns,end_ns\n");
        try {
            for(const auto& i:items) {
                const auto a=i.event.get_profiling_info<sycl::info::event_profiling::command_start>();
                const auto b=i.event.get_profiling_info<sycl::info::event_profiling::command_end>();
                std::fprintf(f,"%s,%d,%zu,%lld,%lld,%lld,%.3f,%llu,%llu\n",i.kind,i.type,i.bytes,(long long)i.rows,(long long)i.cols,(long long)i.max_rows,i.submit_us,(unsigned long long)a,(unsigned long long)b);
            }
        } catch(const std::exception& e) { std::fprintf(stderr,"prefill trace failed: %s\n",e.what()); }
        std::fclose(f);
    }
};
}
