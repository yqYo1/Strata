#include "strata/sycl_upstream/cuda_backend.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/layer.hpp"
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <thread>
using namespace std::chrono_literals;
namespace backend=strata::sycl_upstream::cuda;
namespace ex=sycl::ext::oneapi::experimental;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorName(e))+": "+backend::backend_error_detail());}
struct Event {
    cudaEvent_t value{};
    explicit Event(unsigned flags=0){ck(cudaEventCreateWithFlags(&value,flags));}
    ~Event(){if(value)cudaEventDestroy(value);}
};
struct Stream {
    cudaStream_t value{};
    explicit Stream(unsigned flags=cudaStreamNonBlocking){ck(cudaStreamCreateWithFlags(&value,flags));}
    ~Stream(){cudaStreamCaptureStatus status{};if(cudaStreamIsCapturing(value,&status)==cudaSuccess && status!=cudaStreamCaptureStatusNone){cudaGraph_t g{};cudaStreamEndCapture(value,&g);if(g)cudaGraphDestroy(g);}cudaStreamDestroy(value);}
};
struct Gate {
    std::promise<void> release;std::shared_future<void> future=release.get_future().share();bool opened=false;
    static void hold(void* p){static_cast<Gate*>(p)->future.wait();}
    void open(){if(!opened){release.set_value();opened=true;}}
    ~Gate(){open();}
};
sycl::event tag(cudaStream_t stream){
    sycl::event result;
    ck(backend::submit(stream,[&](sycl::queue& q){
        check(!q.has_property<sycl::property::queue::enable_profiling>(),"profiling enabled on whole queue");
        result=ex::submit_profiling_tag(q);return result;
    }));
    return result;
}
uint64_t stamp(const sycl::event& e){return e.get_profiling_info<sycl::info::event_profiling::command_end>();}
int main() try {
    Stream stream,other,blocking(cudaStreamDefault);Event a,b,c(cudaEventBlockingSync),disabled(cudaEventDisableTiming);
    int cases=0;float measured=0;
    {
        cudaEvent_t created{};ck(cudaEventCreate(&created));ck(cudaEventDestroy(created));
        cudaEvent_t bad{};check(cudaEventCreateWithFlags(&bad,cudaEventInterprocess)==cudaErrorNotSupported && !bad,"interprocess admitted");cudaGetLastError();
        float ms=123;
        check(cudaEventElapsedTime(nullptr,a.value,b.value)==cudaErrorInvalidValue,"null time pointer");cudaGetLastError();
        check(cudaEventElapsedTime(&ms,nullptr,b.value)==cudaErrorInvalidResourceHandle,"null event");cudaGetLastError();
        check(cudaEventElapsedTime(&ms,a.value,b.value)==cudaErrorInvalidResourceHandle && ms==123,"unrecorded elapsed time");cudaGetLastError();
        ck(cudaEventQuery(a.value));ck(cudaEventSynchronize(a.value));
        ck(cudaEventRecord(disabled.value,stream.value));ck(cudaEventSynchronize(disabled.value));
        check(cudaEventElapsedTime(&ms,disabled.value,disabled.value)==cudaErrorInvalidResourceHandle,"disabled timing");cudaGetLastError();++cases;
    }
    {
        const auto outer_start=tag(stream.value);ck(cudaEventRecord(a.value,stream.value));ck(cudaEventSynchronize(a.value));
        Gate gate;ck(cudaLaunchHostFunc(stream.value,Gate::hold,&gate));
        auto record=std::async(std::launch::async,[&]{return cudaEventRecord(b.value,stream.value);});
        const bool record_ready=record.wait_for(2s)==std::future_status::ready;
        if(!record_ready)gate.open();ck(record.get());check(record_ready,"timed record blocked submitting host");
        auto query=std::async(std::launch::async,[&]{float ms=123;const auto e=cudaEventElapsedTime(&ms,a.value,b.value);
            check(ms==123 && cudaPeekAtLastError()==cudaSuccess,"NotReady changed output or sticky error");return e;});
        const bool query_ready=query.wait_for(2s)==std::future_status::ready;
        if(!query_ready)gate.open();check(query.get()==cudaErrorNotReady && query_ready,"elapsed time blocked pending event");
        check(cudaEventQuery(b.value)==cudaErrorNotReady,"timed event prematurely complete");
        std::this_thread::sleep_for(50ms);gate.open();ck(cudaEventSynchronize(b.value));
        ck(cudaEventElapsedTime(&measured,a.value,b.value));
        auto outer_end=tag(stream.value);outer_end.wait_and_throw();
        const double outer_ms=double(stamp(outer_end)-stamp(outer_start))/1e6;
        check(measured>=45 && measured<=outer_ms,"GPU interval disagrees with native timestamp bounds");
        float same=123,reverse=0;ck(cudaEventElapsedTime(&same,a.value,a.value));ck(cudaEventElapsedTime(&reverse,b.value,a.value));
        check(same==0 && reverse==-measured,"same/reverse event interval");
        std::cout<<"held_interval_ms="<<measured<<" native_outer_ms="<<outer_ms<<" record_nonblocking=1 elapsed_nonblocking=1\n";++cases;
    }
    {
        uint32_t* value=nullptr;ck(cudaMalloc(&value,sizeof(uint32_t)));
        const auto release=[](uint32_t* p){cudaFree(p);};
        std::unique_ptr<uint32_t,decltype(release)> storage(value,release);
        auto outer_start=tag(stream.value);ck(cudaEventRecord(a.value,stream.value));
        ck(backend::submit(stream.value,[=](sycl::queue& q){return q.single_task([=]{
            uint32_t x=0x13579bdu;
            for(int i=0;i<262144;++i){x^=x<<13;x^=x>>17;x^=x<<5;}
            *value=x;
        });}));
        ck(cudaEventRecord(b.value,stream.value));auto outer_end=tag(stream.value);outer_end.wait_and_throw();
        float ms=0;ck(cudaEventElapsedTime(&ms,a.value,b.value));
        const double outer_ms=double(stamp(outer_end)-stamp(outer_start))/1e6;
        check(ms>0 && ms<=outer_ms,"GPU kernel interval outside native bounds");
        uint32_t result=0,expected=0x13579bdu;ck(cudaMemcpy(&result,value,sizeof result,cudaMemcpyDeviceToHost));
        for(int i=0;i<262144;++i){expected^=expected<<13;expected^=expected>>17;expected^=expected<<5;}
        check(result==expected,"timed kernel result");
        std::cout<<"kernel_interval_ms="<<ms<<" native_outer_ms="<<outer_ms<<" result="<<result<<'\n';++cases;
    }
    {
        // Cross-stream timing follows the event dependency. Rerecording a after
        // the wait must not retarget that already-enqueued wait to new work.
        ck(cudaStreamWaitEvent(other.value,a.value));ck(cudaEventRecord(c.value,other.value));ck(cudaEventSynchronize(c.value));
        float ms=0;ck(cudaEventElapsedTime(&ms,a.value,c.value));check(ms>=0,"cross-stream clock epoch");
        Gate gate;ck(cudaLaunchHostFunc(stream.value,Gate::hold,&gate));ck(cudaEventRecord(a.value,stream.value));
        ck(cudaEventQuery(c.value));check(cudaEventElapsedTime(&ms,a.value,c.value)==cudaErrorNotReady,"rerecord did not replace timestamp");
        gate.open();ck(cudaEventSynchronize(a.value));ck(cudaEventElapsedTime(&ms,c.value,a.value));check(ms>=0,"rerecorded GPU timestamp");++cases;
    }
    {
        Event pending;Gate gate;ck(cudaLaunchHostFunc(stream.value,Gate::hold,&gate));ck(cudaEventRecord(pending.value,stream.value));
        const auto handle=pending.value;
        auto destroy=std::async(std::launch::async,[&]{return cudaEventDestroy(handle);});
        const bool ready=destroy.wait_for(2s)==std::future_status::ready;gate.open();ck(destroy.get());pending.value=nullptr;
        ck(cudaStreamSynchronize(stream.value));check(ready,"timed event destruction waited");
        float ms=0;check(cudaEventElapsedTime(&ms,handle,a.value)==cudaErrorInvalidResourceHandle,"stale timed handle");cudaGetLastError();++cases;
    }
    {
        // A timed default-stream record waits for blocking streams, as does
        // the previously implemented untimed record.
        Gate gate;ck(cudaLaunchHostFunc(blocking.value,Gate::hold,&gate));ck(cudaEventRecord(a.value));
        check(cudaEventQuery(a.value)==cudaErrorNotReady,"default record ignored blocking stream");
        gate.open();ck(cudaEventSynchronize(a.value));++cases;
    }
    {
        strata::core::CapturedGraph graph;std::string error;
        check(graph.begin(stream.value,error),error.c_str());ck(cudaEventRecord(a.value,stream.value));ck(cudaEventRecord(b.value,stream.value));
        check(graph.end(stream.value,error),error.c_str());check(graph.nodes()==2,"captured event dependencies");
        check(graph.launch(stream.value,error),error.c_str());check(graph.wait_ms(10000),"captured event replay");
        float ms=123;check(cudaEventElapsedTime(&ms,a.value,b.value)==cudaErrorInvalidResourceHandle && ms==123,"captured timestamp exported");cudaGetLastError();
        ck(cudaEventRecord(a.value,stream.value));ck(cudaEventRecord(b.value,stream.value));ck(cudaEventSynchronize(b.value));
        ck(cudaEventElapsedTime(&ms,a.value,b.value));check(ms>=0 && ms<measured,"rerecord after capture stale timestamp");++cases;
    }
    {
        // These functions are linked from the unchanged original layer.cpp.
        // The test linker discards its other, as-yet-unported engine functions.
        check(strata::core::stage_timing_enable(),"original stage timer init");
        strata::core::stage_timing_name(0,"SYCL timed event probe");
        strata::core::stage_mark_begin(0,0,stream.value);
        Gate gate;ck(cudaLaunchHostFunc(stream.value,Gate::hold,&gate));
        strata::core::stage_mark_end(0,0,stream.value);
        std::this_thread::sleep_for(20ms);gate.open();ck(cudaStreamSynchronize(stream.value));
        strata::core::stage_timing_report(1);++cases;
    }
    std::cout<<"PASS timing_cases="<<cases<<" original_stage_timer=1 queues_enable_profiling=0\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
