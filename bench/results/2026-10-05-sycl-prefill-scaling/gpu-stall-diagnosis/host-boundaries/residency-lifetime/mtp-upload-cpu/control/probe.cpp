
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
double ms_since(Clock::time_point t) {
    return std::chrono::duration<double,std::milli>(Clock::now()-t).count();
}
std::string mode;
struct Queue;
struct Event { Queue* queue; void wait_and_throw(); };
struct Queue {
    std::vector<std::function<void()>> jobs;
    int waits=0,launches=0;
    bool async_error=false;
    Event memcpy(void* destination,const void* source,size_t bytes) {
        if(mode=="upload-rejected" && jobs.empty() && launches==0)
            throw std::runtime_error("injected synchronous upload failure");
        jobs.push_back([=] { std::memcpy(destination,source,bytes); });
        return {this};
    }
    void wait() {
        ++waits;
        for(auto& job:jobs) job();
        jobs.clear();
    }
    void wait_and_throw() {
        wait();
        if(async_error) { async_error=false; throw std::runtime_error("injected async failure"); }
    }
    void ext_oneapi_graph(int) {
        if(++launches==2 && mode=="launch-failure")
            throw std::runtime_error("injected submission failure");
        if(mode=="async-final-error") async_error=true;
    }
};
void Event::wait_and_throw() { queue->wait_and_throw(); }
namespace dpct {
Queue queue;
Queue& get_in_order_queue() { return queue; }
const char* get_error_string_dummy(int) { return "injected error"; }
}
namespace sycl {
struct exception:std::runtime_error { using std::runtime_error::runtime_error; };
template<class T> T* malloc_device(size_t n,Queue&) { return new T[n]; }
template<class T> void free(T* p,Queue&) { delete[] p; }
}
namespace strata {
template<class T> T* checked_usm(T* p) { return p; }
}
#define DPCT_CHECK_ERROR(expr) [&](){ try { expr; return 0; } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; } }()
struct OnDevice { explicit OnDevice(int) {} };
class MtpDrafter {
public:
    struct Geometry { int64_t hc=1,n_embd=2,n_head=1; } geometry;
    Geometry* g_=&geometry;
    int device_=0,max_t_=4;
    int64_t window_=0,prompt_len_=0,pf_cap_=0;
    double ms_prefill=0;
    int32_t* pf_dev_=nullptr;
    Queue* cs_=&dpct::queue;
    int graphs[9]{};
    int* prefill_exec_[9]{}, *prefill_dev_exec_[9]{};
    int32_t tok[4]{},step[16]{},pos[4]{};
    float Rin[8]{};
    int32_t h_tok_[4]{},h_step_[16]{},h_pos_[4]{};
    int32_t* tok_=tok,*step_=step,*pos_=pos;
    float* Rin_=Rin;
    bool capture_prefill_dev(int t,std::string& e) {
        if(mode=="capture-failure") { e="injected capture failure"; return false; }
        prefill_dev_exec_[t]=&graphs[t]; return true;
    }
    bool capture_prefill(int t,std::string&) { prefill_exec_[t]=&graphs[t]; return true; }
    bool prefill(const float*,const int32_t*,int64_t,int64_t,std::string&);
    ~MtpDrafter() { assert(cs_->jobs.empty()); delete[] pf_dev_; }
};
#include "prefill.inc"
int main(int argc,char** argv) {
    assert(argc==2); mode=argv[1];
    MtpDrafter drafter; std::string error;
    float residual[16]{}; int32_t tokens[8]={1,2,3,4,5,6,7,8};
    bool ok=drafter.prefill(residual,tokens,8,0,error);
    if(mode=="healthy") assert(ok && error.empty());
    else if(mode=="async-final-error") {
#if CONTROL
        assert(ok && dpct::queue.async_error); // wait() missed the queued error.
#else
        assert(!ok && !error.empty() && !dpct::queue.async_error);
#endif
    } else assert(!ok && !error.empty());
    // The caller drains after return. The control's local rec no longer exists.
    dpct::queue.async_error=false; dpct::queue.wait_and_throw();
    std::printf("PASS %s: result=%d waits=%d launches=%d\n",mode.c_str(),ok,dpct::queue.waits,dpct::queue.launches);
}
