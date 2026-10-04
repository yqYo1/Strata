#include "strata/sycl_upstream/cuda_backend.hpp"
#include "strata/core/expert_cache.hpp"
#include <algorithm>
#include <future>
#include <iostream>
#include <thread>
#include <cstdlib>
using namespace std::chrono_literals;
namespace backend=strata::sycl_upstream::cuda;
using strata::sycl_upstream::Runtime;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorName(e))+": "+backend::backend_error_detail());}
struct Gate {
    std::promise<void> release;std::shared_future<void> future=release.get_future().share();bool opened=false;
    void open(){if(!opened){release.set_value();opened=true;}}
    ~Gate(){open();}
};
int main(int argc,char** argv) try {
    int count=-1;ck(cudaGetDeviceCount(&count));
    if(argc==2 && std::string(argv[1])=="--no-level-zero") {
        check(count==0,"frontend enumerated a non-Level-Zero GPU");int device=-1;
        check(cudaSetDevice(0)==cudaErrorInvalidDevice && cudaGetDevice(&device)==cudaErrorInvalidDevice && device==-1,"empty enumeration accepted ordinal");
        std::cout<<"PASS visible_level_zero_devices=0\n";return 0;
    }
    check(count>0,"no visible Level Zero GPU");
    if(argc==2 && std::string(argv[1])=="--legacy-sysman") {
        size_t free=123,total=456;
        // UR changes the legacy variable to 0 during device initialization.
        // Deliberately restore 1 afterwards to exercise our unsupported path.
        ck(cudaSetDevice(0));check(setenv("ZES_ENABLE_SYSMAN","1",1)==0,"set test Sysman environment");
        const auto status=cudaMemGetInfo(&free,&total);
        std::cout<<"legacy_status="<<cudaGetErrorName(status)<<" detail="<<backend::backend_error_detail()<<'\n';
        check(status==cudaErrorNotSupported && free==123 && total==456,"unsupported Sysman initialization fabricated capacity");
        std::cout<<"PASS legacy_sysman_rejected=1 outputs_untouched=1\n";return 0;
    }
    std::vector<sycl::device> visible;
    for(auto d:sycl::device::get_devices(sycl::info::device_type::gpu))
        if(d.get_backend()==sycl::backend::ext_oneapi_level_zero)visible.push_back(d);
    check(size_t(count)==visible.size(),"device count differs from native visible GPUs");
    int cases=0;
    {
        int device=-1;ck(cudaGetDevice(&device));check(device==0,"initial current device");
        check(cudaGetDeviceCount(nullptr)==cudaErrorInvalidValue && cudaGetDevice(nullptr)==cudaErrorInvalidValue,"null device query");cudaGetLastError();
        check(cudaSetDevice(-1)==cudaErrorInvalidDevice && cudaSetDevice(count)==cudaErrorInvalidDevice,"invalid ordinal admitted");cudaGetLastError();
        ck(cudaGetDevice(&device));check(device==0,"failed selection changed current device");
        ck(cudaSetDevice(count-1));
        std::vector<std::future<void>> jobs;
        for(int i=0;i<8;++i) jobs.push_back(std::async(std::launch::async,[&]{int value=-1;ck(cudaGetDevice(&value));check(value==0,"new host thread inherited another thread's device");ck(cudaSetDevice(0));cudaStream_t s{};ck(cudaStreamCreate(&s));ck(cudaStreamSynchronize(s));ck(cudaStreamDestroy(s));}));
        for(auto& job:jobs)job.get();ck(cudaGetDevice(&device));check(device==count-1,"another thread changed current device");ck(cudaSetDevice(0));++cases;
    }
    {
        size_t free=123,total=456;
        check(cudaMemGetInfo(nullptr,&total)==cudaErrorInvalidValue && total==456,"null free output");cudaGetLastError();
        check(cudaMemGetInfo(&free,nullptr)==cudaErrorInvalidValue && free==123,"null total output");cudaGetLastError();
        ck(cudaMemGetInfo(&free,&total));check(free>0 && free<=total && total==visible[0].get_info<sycl::info::device::global_mem_size>(),"invalid memory capacity");
        // An allocation outside the frontend ledger must affect its free-memory
        // result. Touch it on the GPU; metadata bookkeeping cannot pass this.
        sycl::queue q(visible[0]);auto* warm=sycl::malloc_device(4096,q);q.memset(warm,0,4096).wait_and_throw();sycl::free(warm,q);
        size_t before=0,after=0;ck(cudaMemGetInfo(&before,&total));
        constexpr size_t bytes=128*1024*1024;
        auto* p=sycl::malloc_device(bytes,q);check(p,"native allocation failed");
        q.memset(p,0x5a,bytes).wait_and_throw();ck(cudaMemGetInfo(&after,&total));sycl::free(p,q);
        check(before>after && before-after>=bytes/2,"external GPU allocation invisible to free-memory query");
        std::cout<<"external_allocation_bytes="<<bytes<<" free_before="<<before<<" free_after="<<after<<" total="<<total<<'\n';++cases;
    }
    {
        strata::core::ExpertCache cache;std::string err;size_t free=0,total=0;ck(cudaMemGetInfo(&free,&total));
        const auto ledger=backend::memory_stats();
        check(!cache.open(1,1,1,int64_t(total)+1,err) && err.find("Lower --expert-cache")!=std::string::npos && !cache.valid() && cache.slots()==0,"original cache did not reject capacity before allocation");
        check(backend::memory_stats().device_bytes==ledger.device_bytes,"refused cache allocated storage");
        check(cache.open(4,2,3,4096,err),err.c_str());
        uint8_t* host{};ck(cudaMallocHost(&host,4096));cudaStream_t stream{};ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for(int slot=0;slot<4;++slot) {
            for(int i=0;i<4096;++i)host[i]=uint8_t(slot*17+i*3);
            const auto admitted=cache.admit(slot/2,slot%2);check(admitted==slot && cache.slot_of(slot/2,slot%2)==slot,"original cache residency");
            if(slot==0)check(cache.fill_slot_blocking(slot,host,err),err.c_str());
            else if(slot==1){check(cache.fill_slot_queued(slot,host,err),err.c_str());check(cache.sync_queued(err),err.c_str());}
            else {check(cache.fill_slot(slot,host,stream,err),err.c_str());ck(cudaStreamSynchronize(stream));}
            check(cache.verify_slot(slot,host,err),err.c_str());
        }
        check(cache.fills()==4 && cache.resident()==4 && cache.admit(1,2)==-1 && !cache.device_slot(4),"original cache bounds");
        ck(cudaStreamDestroy(stream));ck(cudaFreeHost(host));cache.close();
        check(!cache.valid() && backend::memory_stats().device_bytes==ledger.device_bytes,"original cache release");++cases;
    }
    {
        // Two real runtimes on this GPU share a context. Exercise the owner's
        // event snapshot and the destination wait without inventing a GPU.
        sycl::context context(visible[0]);Runtime owner(visible[0],context),waiter(visible[0],context);
        auto source=owner.create_stream(true),target=waiter.create_stream(true);Runtime::Event event;
        auto* values=sycl::malloc_shared<int>(2,visible[0],context);values[0]=values[1]=0;
        Gate gate;owner.host_function(source,[&]{gate.future.wait();});
        owner.enqueue(source,[&](sycl::queue& q){return q.single_task([=]{values[0]=73;});});owner.record(event,source);
        auto snapshot=owner.snapshot_event(event);
        auto submit=std::async(std::launch::async,[&]{waiter.wait_event(target,snapshot);waiter.enqueue(target,[&](sycl::queue& q){return q.single_task([=]{values[1]=values[0];});});});
        bool ready=submit.wait_for(2s)==std::future_status::ready;if(!ready)gate.open();submit.get();check(ready && !waiter.query(target),"shared-context event wait blocked host or ignored dependency");
        owner.record(event,source);gate.open();waiter.synchronize(target);owner.synchronize(source);check(values[1]==73,"shared-context snapshot data");sycl::free(values,context);
        owner.begin_capture(source);owner.record(event,source);bool refused=false;
        try{owner.snapshot_event(event);}catch(const std::invalid_argument&){refused=true;}owner.abort_capture(source);check(refused,"exported a captured dependency");++cases;
    }
    // Runs only when real hardware provides another visible GPU. The B570
    // validation records this branch as skipped, not as multi-GPU validation.
    int switched=0;
    if(count>1) {
        cudaStream_t stream{};cudaEvent_t event{};ck(cudaStreamCreate(&stream));ck(cudaEventCreate(&event));uint8_t* p{};ck(cudaMalloc(&p,64));
        ck(cudaSetDevice(1));ck(cudaMemsetAsync(p,0x49,64,stream));ck(cudaEventRecord(event,stream));ck(cudaEventSynchronize(event));
        uint8_t output[64]{};ck(cudaMemcpyAsync(output,p,64,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));check(std::all_of(output,output+64,[](auto v){return v==0x49;}),"stream lost original GPU after current-device switch");
        check(cudaEventRecord(event)==cudaErrorInvalidResourceHandle,"record crossed GPU ownership");cudaGetLastError();
        ck(cudaFree(p));ck(cudaEventDestroy(event));ck(cudaStreamDestroy(stream));ck(cudaSetDevice(0));switched=1;
    }
    std::cout<<"PASS device_cases="<<cases<<" visible_devices="<<count<<" original_cache_bytes_verified=16384 shared_context_event_wait=1 multi_gpu_switch_executed="<<switched<<'\n';
} catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
