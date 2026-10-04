#include "strata/sycl_upstream/cuda_backend.hpp"
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
using strata::sycl_upstream::Runtime;
using strata::sycl_upstream::Memory;
namespace {
thread_local cudaError_t last_error=cudaSuccess;
thread_local std::string error_detail;
struct Error { cudaError_t code; };
void require(bool ok, cudaError_t e=cudaErrorInvalidValue) { if(!ok) throw Error{e}; }
cudaError_t remember(cudaError_t e) { if(e!=cudaSuccess && e!=cudaErrorNotReady) last_error=e; return e; }
template<class F> cudaError_t api(F&& f) noexcept {
    try { f(); return cudaSuccess; }
    catch(const Error& e) { return remember(e.code); }
    catch(const std::bad_alloc&) { return remember(cudaErrorMemoryAllocation); }
    catch(const sycl::exception& e) {
        try {error_detail=e.what();} catch(...) {}
        if(e.code()==sycl::make_error_code(sycl::errc::memory_allocation))return remember(cudaErrorMemoryAllocation);
        if(e.code()==sycl::make_error_code(sycl::errc::feature_not_supported))return remember(cudaErrorNotSupported);
        if(e.code()==sycl::make_error_code(sycl::errc::invalid))return remember(cudaErrorInvalidValue);
        return remember(cudaErrorUnknown);
    }
    catch(const std::invalid_argument& e) { try {error_detail=e.what();} catch(...) {} return remember(cudaErrorInvalidValue); }
    catch(const std::exception& e) { try {error_detail=e.what();} catch(...) {} return remember(cudaErrorUnknown); }
    catch(...) { return remember(cudaErrorUnknown); }
}
struct Stream { Runtime::Stream id; };
struct Event { Runtime::Event event; };
struct Definition { Runtime::GraphDefinition graph; };
struct Executable { Runtime::Graph graph; };
// Monotonic opaque tokens avoid reusing a destroyed handle's address. Tokens
// are keys only, never dereferenced. Runtime/Memory own native resource lifetime.
std::atomic<uintptr_t> next_handle{0x1000};
template<class H, class T> class Handles {
    std::mutex mutex;
    std::map<H,std::shared_ptr<T>> objects;
public:
    H insert(std::shared_ptr<T> object) {
        auto handle=reinterpret_cast<H>(next_handle.fetch_add(16));
        std::lock_guard lock(mutex);objects.emplace(handle,std::move(object));return handle;
    }
    std::shared_ptr<T> get(H handle) {
        std::lock_guard lock(mutex);auto it=objects.find(handle);
        require(it!=objects.end(),cudaErrorInvalidResourceHandle);return it->second;
    }
    void erase(H handle) {
        std::lock_guard lock(mutex);require(objects.erase(handle)==1,cudaErrorInvalidResourceHandle);
    }
};
struct Staging {
    std::shared_ptr<void> storage;
    std::optional<sycl::event> completion;
};
struct Domain {
    Runtime runtime{sycl::device{sycl::gpu_selector_v}};
    Memory memory{runtime};
    Handles<cudaStream_t,Stream> streams;
    Handles<cudaEvent_t,Event> events;
    Handles<cudaGraph_t,Definition> definitions;
    Handles<cudaGraphExec_t,Executable> executables;
    std::mutex staging_mutex;
    std::vector<Staging> staging;
    ~Domain() {
        // Staging must remain allocated until queued copies finish. Runtime
        // subsequently releases its retained graph/retired-stream resources.
        try { runtime.synchronize_device(); }
        catch(...) { std::terminate(); }
    }
    Runtime::Stream stream(cudaStream_t s) { return s ? streams.get(s)->id : 0; }
    void pageable_to_device(void* dst, const void* src, size_t bytes) {
        // CUDA's synchronous pageable H2D path synchronizes the default stream
        // before staging, then permits DMA to outlive the call.
        runtime.synchronize(0);
        auto context=runtime.context();
        std::shared_ptr<void> storage(sycl::malloc_host(bytes,context),[context](void* p){sycl::free(p,context);});
        if(!storage) throw std::bad_alloc();
        std::memcpy(storage.get(),src,bytes);
        std::lock_guard lock(staging_mutex);
        std::erase_if(staging,[](const Staging& s){return s.completion &&
            s.completion->get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete;});
        staging.push_back({storage,{}}); // Retain before a possibly partial submission.
        staging.back().completion=runtime.copy(0,dst,storage.get(),bytes);
    }
};
Domain& domain() { static Domain value; return value; }
std::optional<Memory::Info> range(const void* p, size_t bytes) {
    require(p || !bytes);
    auto info=domain().memory.info(p);
    if(info) {
        const auto offset=reinterpret_cast<uintptr_t>(p)-reinterpret_cast<uintptr_t>(info->base);
        require(bytes<=info->bytes-offset);
    }
    return info;
}
bool device_pointer(const void* p) {
    auto info=domain().memory.info(p);return info && info->kind==Memory::Kind::device;
}
cudaMemcpyKind direction(void* dst,const void* src,cudaMemcpyKind kind) {
    require(kind>=cudaMemcpyHostToHost && kind<=cudaMemcpyDefault);
    if(kind!=cudaMemcpyDefault) return kind;
    return device_pointer(dst) ? (device_pointer(src)?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice)
                               : (device_pointer(src)?cudaMemcpyDeviceToHost:cudaMemcpyHostToHost);
}
void copy(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream,bool async) {
    auto& d=domain();const auto s=d.stream(stream);
    range(dst,bytes);auto source=range(src,bytes);kind=direction(dst,src,kind);
    if(!bytes) return;
    if(kind==cudaMemcpyHostToHost) {
        // H2H is host-synchronous even for the Async spelling.
        d.runtime.synchronize(s);std::memcpy(dst,src,bytes);return;
    }
    if(!async && kind==cudaMemcpyHostToDevice && !source) {
        d.pageable_to_device(dst,src,bytes);return;
    }
    auto event=d.runtime.copy(s,dst,src,bytes);
    if(!async && kind!=cudaMemcpyDeviceToDevice) event.wait_and_throw();
}
}
const char* cudaGetErrorName(cudaError_t e) noexcept {
    switch(e) {
#define CASE(x) case x:return #x;
        CASE(cudaSuccess) CASE(cudaErrorInvalidValue) CASE(cudaErrorMemoryAllocation)
        CASE(cudaErrorInitializationError) CASE(cudaErrorInvalidDevice) CASE(cudaErrorInvalidResourceHandle)
        CASE(cudaErrorNotReady) CASE(cudaErrorNotSupported) CASE(cudaErrorStreamCaptureUnsupported)
        CASE(cudaErrorStreamCaptureInvalidated) CASE(cudaErrorUnknown)
#undef CASE
        default:return "unrecognized error code";
    }
}
const char* cudaGetErrorString(cudaError_t e) noexcept {
    switch(e) {
        case cudaSuccess:return "no error";
        case cudaErrorInvalidValue:return "invalid argument";
        case cudaErrorMemoryAllocation:return "out of memory";
        case cudaErrorInitializationError:return "initialization error";
        case cudaErrorInvalidDevice:return "invalid device ordinal";
        case cudaErrorInvalidResourceHandle:return "invalid resource handle";
        case cudaErrorNotReady:return "device not ready";
        case cudaErrorNotSupported:return "operation not supported by the SYCL frontend";
        case cudaErrorStreamCaptureUnsupported:return "operation not permitted during stream capture";
        case cudaErrorStreamCaptureInvalidated:return "stream capture invalidated";
        default:return "unknown error";
    }
}
cudaError_t cudaGetLastError() noexcept { const auto e=last_error;last_error=cudaSuccess;return e; }
cudaError_t cudaPeekAtLastError() noexcept { return last_error; }
cudaError_t cudaMalloc(void** p,size_t n) noexcept {return api([&]{require(p);*p=nullptr;*p=domain().memory.allocate_device(n);});}
cudaError_t cudaHostAlloc(void** p,size_t n,unsigned flags) noexcept {return api([&]{require(p);*p=nullptr;
    require((flags&~(cudaHostAllocPortable|cudaHostAllocMapped))==0,cudaErrorNotSupported);*p=domain().memory.allocate_host(n);});}
cudaError_t cudaMallocHost(void** p,size_t n) noexcept {return cudaHostAlloc(p,n,0);}
cudaError_t cudaFree(void* p) noexcept {return api([&]{domain().memory.free_device(p);});}
cudaError_t cudaFreeHost(void* p) noexcept {return api([&]{domain().memory.free_host(p);});}
cudaError_t cudaHostRegister(void* p,size_t n,unsigned flags) noexcept {return api([&]{
    require((flags&~(cudaHostRegisterPortable|cudaHostRegisterMapped|cudaHostRegisterReadOnly))==0,cudaErrorNotSupported);
    domain().memory.register_host(p,n,flags&cudaHostRegisterReadOnly);});}
cudaError_t cudaHostUnregister(void* p) noexcept {return api([&]{require(p);domain().memory.unregister_host(p);});}
cudaError_t cudaHostGetDevicePointer(void** out,void* p,unsigned flags) noexcept {return api([&]{require(out && flags==0);*out=nullptr;*out=domain().memory.device_alias(p);});}
cudaError_t cudaStreamCreate(cudaStream_t* s) noexcept {return cudaStreamCreateWithFlags(s,0);}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s,unsigned flags) noexcept {return api([&]{require(s);*s=nullptr;require((flags&~cudaStreamNonBlocking)==0);
    auto& d=domain();auto value=std::make_shared<Stream>(Stream{0});value->id=d.runtime.create_stream(flags&cudaStreamNonBlocking);
    try {*s=d.streams.insert(value);} catch(...) {d.runtime.destroy_stream(value->id);throw;}
});}
cudaError_t cudaStreamDestroy(cudaStream_t s) noexcept {return api([&]{auto& d=domain();auto value=d.streams.get(s);d.runtime.destroy_stream(value->id);d.streams.erase(s);});}
cudaError_t cudaStreamSynchronize(cudaStream_t s) noexcept {return api([&]{auto& d=domain();d.runtime.synchronize(d.stream(s));});}
cudaError_t cudaStreamQuery(cudaStream_t s) noexcept {return api([&]{auto& d=domain();require(d.runtime.query(d.stream(s)),cudaErrorNotReady);});}
cudaError_t cudaDeviceSynchronize() noexcept {return api([&]{domain().runtime.synchronize_device();});}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e,unsigned flags) noexcept {return api([&]{require(e);*e=nullptr;
    // Timing and interprocess event behavior are not silently approximated.
    require(flags==cudaEventDisableTiming || flags==(cudaEventDisableTiming|cudaEventBlockingSync),cudaErrorNotSupported);
    *e=domain().events.insert(std::make_shared<Event>());});}
cudaError_t cudaEventDestroy(cudaEvent_t e) noexcept {return api([&]{domain().events.erase(e);});}
cudaError_t cudaEventRecord(cudaEvent_t e,cudaStream_t s) noexcept {return api([&]{auto& d=domain();d.runtime.record(d.events.get(e)->event,d.stream(s));});}
cudaError_t cudaEventQuery(cudaEvent_t e) noexcept {return api([&]{auto& d=domain();require(d.runtime.query(d.events.get(e)->event),cudaErrorNotReady);});}
cudaError_t cudaEventSynchronize(cudaEvent_t e) noexcept {return api([&]{auto& d=domain();d.runtime.synchronize(d.events.get(e)->event);});}
cudaError_t cudaStreamWaitEvent(cudaStream_t s,cudaEvent_t e,unsigned flags) noexcept {return api([&]{require(flags==0,cudaErrorNotSupported);auto& d=domain();d.runtime.wait_event(d.stream(s),d.events.get(e)->event);});}
cudaError_t cudaLaunchHostFunc(cudaStream_t s,cudaHostFn_t fn,void* data) noexcept {return api([&]{require(fn);auto& d=domain();d.runtime.host_function(d.stream(s),[=]{fn(data);});});}
cudaError_t cudaMemcpy(void* dst,const void* src,size_t n,cudaMemcpyKind k) noexcept {return api([&]{copy(dst,src,n,k,nullptr,false);});}
cudaError_t cudaMemcpyAsync(void* dst,const void* src,size_t n,cudaMemcpyKind k,cudaStream_t s) noexcept {return api([&]{copy(dst,src,n,k,s,true);});}
cudaError_t cudaMemcpy2DAsync(void* dst,size_t dp,const void* src,size_t sp,size_t width,size_t height,cudaMemcpyKind k,cudaStream_t stream) noexcept {return api([&]{
    require(width<=dp && width<=sp);auto& d=domain();const auto s=d.stream(stream);k=direction(dst,src,k);
    if(!width || !height) return;
    require((height-1)<=(std::numeric_limits<size_t>::max()-width)/dp && (height-1)<=(std::numeric_limits<size_t>::max()-width)/sp);
    range(dst,(height-1)*dp+width);range(src,(height-1)*sp+width);
    if(k==cudaMemcpyHostToHost) {d.runtime.synchronize(s);for(size_t row=0;row<height;++row)
        std::memcpy(static_cast<char*>(dst)+row*dp,static_cast<const char*>(src)+row*sp,width);return;}
    d.runtime.enqueue(s,[=](sycl::queue& q){
        if(q.ext_oneapi_get_state()!=sycl::ext::oneapi::experimental::queue_state::recording)
            return q.ext_oneapi_memcpy2d(dst,dp,src,sp,width,height);
        // Graph recording currently rejects memcpy2d. The native-command
        // graph interop path deadlocks in this installed UR V2 adapter. Record
        // the same rectangle as ordered copy commands, preserving copy-engine
        // transfers, pitch and byte bounds. This adds one graph node per row.
        sycl::event last;
        for(size_t row=0;row<height;++row)
            last=q.memcpy(static_cast<char*>(dst)+row*dp,static_cast<const char*>(src)+row*sp,width);
        return last;
    });
});}
cudaError_t cudaMemsetAsync(void* p,int byte,size_t n,cudaStream_t s) noexcept {return api([&]{range(p,n);auto& d=domain();auto id=d.stream(s);if(n)d.runtime.memset(id,p,byte,n);});}
cudaError_t cudaMemset(void* p,int byte,size_t n) noexcept {return api([&]{auto info=range(p,n);if(!n)return;
    auto event=domain().runtime.memset(0,p,byte,n);if(info && info->kind!=Memory::Kind::device)event.wait_and_throw();});}
cudaError_t cudaStreamBeginCapture(cudaStream_t s,cudaStreamCaptureMode mode) noexcept {return api([&]{
    require(mode==cudaStreamCaptureModeThreadLocal,cudaErrorNotSupported);auto& d=domain();d.runtime.begin_capture(d.stream(s));});}
cudaError_t cudaStreamEndCapture(cudaStream_t s,cudaGraph_t* g) noexcept {return api([&]{require(g);*g=nullptr;auto& d=domain();
    auto value=std::make_shared<Definition>();value->graph=d.runtime.end_capture_definition(d.stream(s));*g=d.definitions.insert(std::move(value));});}
cudaError_t cudaStreamIsCapturing(cudaStream_t s,cudaStreamCaptureStatus* status) noexcept {return api([&]{require(status);auto& d=domain();
    const auto state=d.runtime.capture_status(d.stream(s));
    *status=state==Runtime::CaptureStatus::none?cudaStreamCaptureStatusNone:
        state==Runtime::CaptureStatus::invalidated?cudaStreamCaptureStatusInvalidated:cudaStreamCaptureStatusActive;});}
cudaError_t cudaGraphGetNodes(cudaGraph_t g,cudaGraphNode_t* nodes,size_t* n) noexcept {return api([&]{require(n);require(!nodes,cudaErrorNotSupported);*n=domain().definitions.get(g)->graph.node_count();});}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e,cudaGraph_t g,cudaGraphNode_t* bad,char* log,size_t bytes) noexcept {return api([&]{
    require(e);*e=nullptr;if(bad)*bad=nullptr;if(log && bytes)log[0]='\0';auto& d=domain();auto value=std::make_shared<Executable>();
    value->graph=d.runtime.instantiate(d.definitions.get(g)->graph);*e=d.executables.insert(std::move(value));});}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e,cudaGraph_t g,unsigned long long flags) noexcept {
    if(flags) {if(e)*e=nullptr;return remember(cudaErrorNotSupported);}return cudaGraphInstantiate(e,g,nullptr,nullptr,0);
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t e,cudaStream_t s) noexcept {return api([&]{auto& d=domain();d.runtime.launch(d.executables.get(e)->graph,d.stream(s));});}
cudaError_t cudaGraphDestroy(cudaGraph_t g) noexcept {return api([&]{domain().definitions.erase(g);});}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t e) noexcept {return api([&]{domain().executables.erase(e);});}
namespace strata::sycl_upstream::cuda {
cudaError_t submit(cudaStream_t s,const Runtime::Submit& fn) noexcept {return api([&]{auto& d=domain();d.runtime.enqueue(d.stream(s),fn);});}
Memory::Stats memory_stats(){return domain().memory.stats();}
std::optional<Memory::Info> allocation_info(const void* p){return domain().memory.info(p);}
const char* backend_error_detail() noexcept{return error_detail.c_str();}
}
