#include "strata/sycl_upstream/cuda_backend.hpp"
#include <atomic>
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
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
struct Domain;
struct Stream { Domain* owner; Runtime::Stream id; };
struct Event {
    Domain* owner;
    Runtime::Event event;
    Event(Domain& d,bool timing): owner(&d), event(timing) {}
};
struct Definition {
    Domain* owner;
    Runtime::GraphDefinition graph;
    std::vector<Runtime::NodeType> types;
    std::mutex mutex;
    std::vector<cudaGraphNode_t> nodes;
    bool alive = true;
};
struct Node { std::weak_ptr<Definition> definition; size_t index; };
struct Executable { Domain* owner; Runtime::Graph graph; };
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
    const int ordinal;
    Runtime runtime;
    Memory memory{runtime};
    std::mutex staging_mutex;
    std::vector<Staging> staging;
    Domain(int ordinal,const sycl::device& device,const sycl::context& context): ordinal(ordinal),runtime(device,context) {}
    ~Domain() {
        // Staging must remain allocated until queued copies finish. Runtime
        // subsequently releases its retained graph/retired-stream resources.
        // Fatal upstream launch checks use std::exit, so an unfinished capture
        // may still exist here. Discard it before draining submitted work;
        // otherwise the teardown exception hides the original launch error.
        try { runtime.prepare_teardown(); }
        catch(...) { std::terminate(); }
    }
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
// The frontend targets Level Zero. Enumerate visible GPUs once through SYCL,
// so OpenCL aliases never become additional physical CUDA ordinals. Named
// resources retain their creating domain; the default stream follows the
// current host thread's device.
thread_local int current_device=0;
struct Devices {
    std::vector<sycl::device> visible;
    std::vector<std::unique_ptr<Domain>> domains;
    std::vector<std::pair<sycl::platform,sycl::context>> contexts;
    std::mutex mutex;
    std::mutex peer_mutex;
    std::set<std::pair<int,int>> peers;
    Handles<cudaStream_t,Stream> streams;
    Handles<cudaEvent_t,Event> events;
    Handles<cudaGraph_t,Definition> definitions;
    Handles<cudaGraphExec_t,Executable> executables;
    Handles<cudaGraphNode_t,Node> nodes;
    Devices() {
        for(const auto& d:sycl::device::get_devices(sycl::info::device_type::gpu))
            if(d.get_backend()==sycl::backend::ext_oneapi_level_zero) visible.push_back(d);
        domains.resize(visible.size());
    }
    ~Devices() {
        // A platform's host ledger can be used by every runtime sharing its
        // context. Drain them all before any Domain destroys host mappings.
        try {for(auto& d:domains) if(d) d->runtime.prepare_teardown();}
        catch(...) {std::terminate();}
    }
    void validate(int ordinal) {
        require(ordinal>=0 && size_t(ordinal)<visible.size(),cudaErrorInvalidDevice);
    }
    Domain& get(int ordinal) {
        validate(ordinal);std::lock_guard lock(mutex);
        if(!domains[ordinal]) {
            const auto platform=visible[ordinal].get_platform();
            auto c=std::find_if(contexts.begin(),contexts.end(),[&](const auto& p){return p.first==platform;});
            if(c==contexts.end()) {
                std::vector<sycl::device> peers;
                for(const auto& d:visible) if(d.get_platform()==platform) peers.push_back(d);
                contexts.emplace_back(platform,sycl::context(peers));c=std::prev(contexts.end());
            }
            domains[ordinal]=std::make_unique<Domain>(ordinal,visible[ordinal],c->second);
        }
        return *domains[ordinal];
    }
    std::vector<Domain*> live() {
        std::lock_guard lock(mutex);std::vector<Domain*> result;
        for(const auto& d:domains) if(d) result.push_back(d.get());return result;
    }
    // All host mappings for a platform use a single ledger/context. This
    // prevents duplicate native page imports when callers switch devices.
    Domain& host(int ordinal) {
        validate(ordinal);
        for(size_t i=0;i<visible.size();++i)
            if(visible[i].get_platform()==visible[ordinal].get_platform()) return get(int(i));
        throw Error{cudaErrorInvalidDevice};
    }
};
Devices& devices() {static Devices value;return value;}
Domain& domain() {return devices().get(current_device);}
Domain& stream_domain(cudaStream_t s) {return s ? *devices().streams.get(s)->owner : domain();}
Runtime::Stream stream_id(cudaStream_t s) {return s ? devices().streams.get(s)->id : 0;}
struct Allocation {Domain* owner;Memory::Info info;};
std::optional<Allocation> allocation(const void* p) {
    for(auto* d:devices().live()) if(auto info=d->memory.info(p)) return Allocation{d,*info};
    return {};
}
void drain_context_users(Domain& owner) {
    for(auto* d:devices().live()) if(d->runtime.context()==owner.runtime.context()) d->runtime.synchronize_device();
}
void require_portable(unsigned flags,unsigned portable) {
    if(!(flags&portable)) return;
    auto& visible=devices().visible;devices().validate(current_device);
    for(const auto& d:visible) require(d.get_platform()==visible[current_device].get_platform(),cudaErrorNotSupported);
}
void accessible(Domain& d,const void* p) {
    if(auto a=allocation(p)) {
        require(a->owner->runtime.context()==d.runtime.context(),cudaErrorNotSupported);
        if(a->info.kind==Memory::Kind::device && a->owner!=&d) {
            std::lock_guard lock(devices().peer_mutex);
            require(devices().peers.contains({d.ordinal,a->owner->ordinal}),cudaErrorNotSupported);
        }
    }
}
std::optional<Memory::Info> range(const void* p, size_t bytes) {
    require(p || !bytes);
    auto a=allocation(p);auto info=a ? std::optional<Memory::Info>(a->info) : std::nullopt;
    if(info) {
        const auto offset=reinterpret_cast<uintptr_t>(p)-reinterpret_cast<uintptr_t>(info->base);
        require(bytes<=info->bytes-offset);
    }
    return info;
}
bool device_pointer(const void* p) {
    auto a=allocation(p);auto info=a ? std::optional<Memory::Info>(a->info) : std::nullopt;return info && info->kind==Memory::Kind::device;
}
cudaMemcpyKind direction(void* dst,const void* src,cudaMemcpyKind kind) {
    require(kind>=cudaMemcpyHostToHost && kind<=cudaMemcpyDefault);
    if(kind!=cudaMemcpyDefault) return kind;
    return device_pointer(dst) ? (device_pointer(src)?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice)
                               : (device_pointer(src)?cudaMemcpyDeviceToHost:cudaMemcpyHostToHost);
}
void copy(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream,bool async) {
    auto& d=stream_domain(stream);const auto s=stream_id(stream);
    accessible(d,dst);accessible(d,src);
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
    if(!async && kind!=cudaMemcpyDeviceToDevice) Runtime::synchronize_native(event);
}
}
const char* cudaGetErrorName(cudaError_t e) noexcept {
    switch(e) {
#define CASE(x) case x:return #x;
        CASE(cudaSuccess) CASE(cudaErrorInvalidValue) CASE(cudaErrorMemoryAllocation)
        CASE(cudaErrorInitializationError) CASE(cudaErrorInvalidDevice) CASE(cudaErrorInvalidResourceHandle)
        CASE(cudaErrorNotReady) CASE(cudaErrorNotSupported) CASE(cudaErrorStreamCaptureUnsupported)
        CASE(cudaErrorPeerAccessAlreadyEnabled) CASE(cudaErrorPeerAccessNotEnabled)
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
        case cudaErrorPeerAccessAlreadyEnabled:return "peer access already enabled";
        case cudaErrorPeerAccessNotEnabled:return "peer access not enabled";
        case cudaErrorNotSupported:return "operation not supported by the SYCL frontend";
        case cudaErrorStreamCaptureUnsupported:return "operation not permitted during stream capture";
        case cudaErrorStreamCaptureInvalidated:return "stream capture invalidated";
        default:return "unknown error";
    }
}
cudaError_t cudaGetLastError() noexcept { const auto e=last_error;last_error=cudaSuccess;return e; }
cudaError_t cudaPeekAtLastError() noexcept { return last_error; }
cudaError_t cudaGetDeviceCount(int* out) noexcept {return api([&]{require(out);*out=int(devices().visible.size());});}
cudaError_t cudaGetDevice(int* out) noexcept {return api([&]{require(out);devices().validate(current_device);*out=current_device;});}
cudaError_t cudaSetDevice(int ordinal) noexcept {return api([&]{devices().get(ordinal);current_device=ordinal;});}
cudaError_t cudaInitDevice(int ordinal,unsigned device_flags,unsigned flags) noexcept {return api([&]{
    devices().validate(ordinal);require(flags==0);
    // The installed SYCL runtime offers no per-device CUDA spin/yield policy.
    // Report that request explicitly; the original remote preflight already
    // clears this optional policy error and retains native default scheduling.
    require((device_flags&~cudaDeviceMapHost)==0,cudaErrorNotSupported);
    require(!(device_flags&cudaDeviceMapHost) || devices().visible[ordinal].has(sycl::aspect::usm_host_allocations),cudaErrorNotSupported);
    devices().get(ordinal); // Initialization does not change the calling thread's device.
});}
cudaError_t cudaDeviceCanAccessPeer(int* out,int ordinal,int peer) noexcept {return api([&]{
    require(out);devices().validate(ordinal);devices().validate(peer);
    *out=ordinal!=peer && devices().visible[ordinal].ext_oneapi_can_access_peer(devices().visible[peer]);
});}
cudaError_t cudaDeviceEnablePeerAccess(int peer,unsigned flags) noexcept {return api([&]{
    require(flags==0);devices().validate(peer);devices().validate(current_device);
    require(peer!=current_device,cudaErrorInvalidDevice);
    auto& from=devices().get(current_device);auto& to=devices().get(peer);
    require(from.runtime.context()==to.runtime.context(),cudaErrorNotSupported);
    auto& source=devices().visible[current_device];auto& target=devices().visible[peer];
    require(source.ext_oneapi_can_access_peer(target),cudaErrorInvalidDevice);
    from.runtime.check_memory_operation();to.runtime.check_memory_operation();
    std::lock_guard lock(devices().peer_mutex);
    auto [it,inserted]=devices().peers.emplace(current_device,peer);
    require(inserted,cudaErrorPeerAccessAlreadyEnabled);
    // Reserve ledger storage first; no allocation may fail after native access
    // succeeds. On native failure the attempted entry is rolled back.
    try {source.ext_oneapi_enable_peer_access(target);} catch(...) {devices().peers.erase(it);throw;}
});}
cudaError_t cudaDeviceDisablePeerAccess(int peer) noexcept {return api([&]{
    devices().validate(peer);devices().validate(current_device);require(peer!=current_device,cudaErrorInvalidDevice);
    auto& from=devices().get(current_device);from.runtime.synchronize_device();
    std::lock_guard lock(devices().peer_mutex);
    const auto it=devices().peers.find({current_device,peer});require(it!=devices().peers.end(),cudaErrorPeerAccessNotEnabled);
    devices().visible[current_device].ext_oneapi_disable_peer_access(devices().visible[peer]);devices().peers.erase(it);
});}
cudaError_t cudaMemGetInfo(size_t* free,size_t* total) noexcept {return api([&]{require(free && total);const auto value=domain().memory.available();*free=value.free;*total=value.total;});}
cudaError_t cudaMalloc(void** p,size_t n) noexcept {return api([&]{require(p);*p=nullptr;*p=domain().memory.allocate_device(n);});}
cudaError_t cudaHostAlloc(void** p,size_t n,unsigned flags) noexcept {return api([&]{require(p);*p=nullptr;
    require((flags&~(cudaHostAllocPortable|cudaHostAllocMapped))==0,cudaErrorNotSupported);require_portable(flags,cudaHostAllocPortable);
    domain().runtime.check_memory_operation();*p=devices().host(current_device).memory.allocate_host(n);});}
cudaError_t cudaMallocHost(void** p,size_t n) noexcept {return cudaHostAlloc(p,n,0);}
cudaError_t cudaFree(void* p) noexcept {return api([&]{if(p){auto a=allocation(p);require(bool(a));
    // Peer queues can use this device allocation too. Drain all runtimes in
    // the shared context before the owner's ledger releases the native USM.
    drain_context_users(*a->owner);a->owner->memory.free_device(p);}});}
cudaError_t cudaFreeHost(void* p) noexcept {return api([&]{if(p){auto a=allocation(p);require(bool(a));drain_context_users(*a->owner);a->owner->memory.free_host(p);}});}
cudaError_t cudaHostRegister(void* p,size_t n,unsigned flags) noexcept {return api([&]{
    require((flags&~(cudaHostRegisterPortable|cudaHostRegisterMapped|cudaHostRegisterReadOnly))==0,cudaErrorNotSupported);
    require_portable(flags,cudaHostRegisterPortable);domain().runtime.check_memory_operation();
    for(auto* d:devices().live())require(!d->memory.overlaps(p,n));
    devices().host(current_device).memory.register_host(p,n,flags&cudaHostRegisterReadOnly);});}
cudaError_t cudaHostUnregister(void* p) noexcept {return api([&]{require(p);auto a=allocation(p);require(bool(a));drain_context_users(*a->owner);a->owner->memory.unregister_host(p);});}
cudaError_t cudaHostGetDevicePointer(void** out,void* p,unsigned flags) noexcept {return api([&]{require(out && flags==0);*out=nullptr;auto a=allocation(p);require(bool(a));require(a->owner->runtime.context()==domain().runtime.context(),cudaErrorNotSupported);*out=a->owner->memory.device_alias(p);});}
cudaError_t cudaStreamCreate(cudaStream_t* s) noexcept {return cudaStreamCreateWithFlags(s,0);}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s,unsigned flags) noexcept {return api([&]{require(s);*s=nullptr;require((flags&~cudaStreamNonBlocking)==0);
    auto& d=domain();auto value=std::make_shared<Stream>(Stream{&d,0});value->id=d.runtime.create_stream(flags&cudaStreamNonBlocking);
    try {*s=devices().streams.insert(value);} catch(...) {d.runtime.destroy_stream(value->id);throw;}
});}
cudaError_t cudaStreamDestroy(cudaStream_t s) noexcept {return api([&]{auto value=devices().streams.get(s);auto& d=*value->owner;d.runtime.destroy_stream(value->id);devices().streams.erase(s);});}
cudaError_t cudaStreamSynchronize(cudaStream_t s) noexcept {return api([&]{auto& d=stream_domain(s);d.runtime.synchronize(stream_id(s));});}
cudaError_t cudaStreamQuery(cudaStream_t s) noexcept {return api([&]{auto& d=stream_domain(s);require(d.runtime.query(stream_id(s)),cudaErrorNotReady);});}
cudaError_t cudaDeviceSynchronize() noexcept {return api([&]{domain().runtime.synchronize_device();});}
cudaError_t cudaEventCreate(cudaEvent_t* e) noexcept {return cudaEventCreateWithFlags(e,cudaEventDefault);}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e,unsigned flags) noexcept {return api([&]{require(e);*e=nullptr;
    require((flags&~(cudaEventDisableTiming|cudaEventBlockingSync))==0,cudaErrorNotSupported);
    const bool timing=!(flags&cudaEventDisableTiming);
    auto& d=domain();
    require(!timing || d.runtime.device().has(sycl::aspect::ext_oneapi_queue_profiling_tag),cudaErrorNotSupported);
    *e=devices().events.insert(std::make_shared<Event>(d,timing));});}
cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t start,cudaEvent_t end) noexcept {return api([&]{
    require(ms);auto a=devices().events.get(start),b=devices().events.get(end);require(a->owner==b->owner,cudaErrorInvalidResourceHandle);
    const auto result=a->owner->runtime.elapsed_time(a->event,b->event);
    require(result.status!=Runtime::TimingStatus::invalid,cudaErrorInvalidResourceHandle);
    require(result.status!=Runtime::TimingStatus::not_ready,cudaErrorNotReady);
    *ms=result.milliseconds;
});}
cudaError_t cudaEventDestroy(cudaEvent_t e) noexcept {return api([&]{devices().events.erase(e);});}
cudaError_t cudaEventRecord(cudaEvent_t e,cudaStream_t s) noexcept {return api([&]{auto value=devices().events.get(e);auto& d=stream_domain(s);require(value->owner==&d,cudaErrorInvalidResourceHandle);d.runtime.record(value->event,stream_id(s));});}
cudaError_t cudaEventQuery(cudaEvent_t e) noexcept {return api([&]{auto value=devices().events.get(e);require(value->owner->runtime.query(value->event),cudaErrorNotReady);});}
cudaError_t cudaEventSynchronize(cudaEvent_t e) noexcept {return api([&]{auto value=devices().events.get(e);value->owner->runtime.synchronize(value->event);});}
cudaError_t cudaStreamWaitEvent(cudaStream_t s,cudaEvent_t e,unsigned flags) noexcept {return api([&]{require(flags==0,cudaErrorNotSupported);auto& d=stream_domain(s);auto value=devices().events.get(e);if(value->owner==&d)d.runtime.wait_event(stream_id(s),value->event);else {require(value->owner->runtime.context()==d.runtime.context(),cudaErrorNotSupported);d.runtime.wait_event(stream_id(s),value->owner->runtime.snapshot_event(value->event));}});}
cudaError_t cudaLaunchHostFunc(cudaStream_t s,cudaHostFn_t fn,void* data) noexcept {return api([&]{require(fn);auto& d=stream_domain(s);d.runtime.host_function(stream_id(s),[=]{fn(data);});});}
cudaError_t cudaMemcpy(void* dst,const void* src,size_t n,cudaMemcpyKind k) noexcept {return api([&]{copy(dst,src,n,k,nullptr,false);});}
cudaError_t cudaMemcpyAsync(void* dst,const void* src,size_t n,cudaMemcpyKind k,cudaStream_t s) noexcept {return api([&]{copy(dst,src,n,k,s,true);});}
cudaError_t cudaMemcpyPeerAsync(void* dst,int dst_device,const void* src,int src_device,size_t n,cudaStream_t s) noexcept {return api([&]{
    devices().validate(dst_device);devices().validate(src_device);stream_domain(s);stream_id(s);
    if(!n)return;
    const auto destination=allocation(dst),source=allocation(src);
    require(destination && source && destination->info.kind==Memory::Kind::device && source->info.kind==Memory::Kind::device);
    require(destination->owner->ordinal==dst_device && source->owner->ordinal==src_device);
    // Native copies keep this stream's ordering, capture and asynchronous
    // lifetime. Foreign USM requires enabled native access in its direction.
    // CUDA's no-P2P staging fallback remains unsupported, not a fake success.
    copy(dst,src,n,cudaMemcpyDeviceToDevice,s,true);
});}
cudaError_t cudaMemcpy2DAsync(void* dst,size_t dp,const void* src,size_t sp,size_t width,size_t height,cudaMemcpyKind k,cudaStream_t stream) noexcept {return api([&]{
    require(width<=dp && width<=sp);auto& d=stream_domain(stream);const auto s=stream_id(stream);
    accessible(d,dst);accessible(d,src);k=direction(dst,src,k);
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
cudaError_t cudaMemsetAsync(void* p,int byte,size_t n,cudaStream_t s) noexcept {return api([&]{range(p,n);auto& d=stream_domain(s);accessible(d,p);auto id=stream_id(s);if(n)d.runtime.memset(id,p,byte,n);});}
cudaError_t cudaMemset(void* p,int byte,size_t n) noexcept {return api([&]{auto info=range(p,n);if(!n)return;
    auto& d=domain();accessible(d,p);auto event=d.runtime.memset(0,p,byte,n);if(info && info->kind!=Memory::Kind::device)Runtime::synchronize_native(event);});}
cudaError_t cudaStreamBeginCapture(cudaStream_t s,cudaStreamCaptureMode mode) noexcept {return api([&]{
    require(mode==cudaStreamCaptureModeThreadLocal,cudaErrorNotSupported);auto& d=stream_domain(s);d.runtime.begin_capture(stream_id(s));});}
cudaError_t cudaStreamEndCapture(cudaStream_t s,cudaGraph_t* g) noexcept {return api([&]{require(g);*g=nullptr;auto& d=stream_domain(s);
    auto value=std::make_shared<Definition>();value->owner=&d;value->graph=d.runtime.end_capture_definition(stream_id(s));
    value->types=value->graph.node_types();*g=devices().definitions.insert(std::move(value));});}
cudaError_t cudaStreamIsCapturing(cudaStream_t s,cudaStreamCaptureStatus* status) noexcept {return api([&]{require(status);auto& d=stream_domain(s);
    const auto state=d.runtime.capture_status(stream_id(s));
    *status=state==Runtime::CaptureStatus::none?cudaStreamCaptureStatusNone:
        state==Runtime::CaptureStatus::invalidated?cudaStreamCaptureStatusInvalidated:cudaStreamCaptureStatusActive;});}
cudaError_t cudaGraphGetNodes(cudaGraph_t g,cudaGraphNode_t* nodes,size_t* n) noexcept {return api([&]{
    require(n);auto definition=devices().definitions.get(g);std::lock_guard lock(definition->mutex);
    require(definition->alive,cudaErrorInvalidResourceHandle);
    const size_t count=definition->types.size();
    if(!nodes){*n=count;return;}
    if(definition->nodes.empty() && count) {
        definition->nodes.reserve(count);
        try {for(size_t index=0;index<count;++index)definition->nodes.push_back(devices().nodes.insert(std::make_shared<Node>(Node{definition,index})));}
        catch(...) {for(auto handle:definition->nodes)devices().nodes.erase(handle);definition->nodes.clear();throw;}
    }
    const size_t capacity=*n,obtained=std::min(capacity,count);
    std::copy_n(definition->nodes.begin(),obtained,nodes);
    if(capacity>obtained)std::fill(nodes+obtained,nodes+capacity,nullptr);
    *n=obtained;
});}
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node,cudaGraphNodeType* type) noexcept {return api([&]{
    require(type);auto value=devices().nodes.get(node);auto definition=value->definition.lock();require(bool(definition),cudaErrorInvalidResourceHandle);
    std::lock_guard lock(definition->mutex);require(definition->alive,cudaErrorInvalidResourceHandle);
    switch(definition->types.at(value->index)) {
        case Runtime::NodeType::kernel:*type=cudaGraphNodeTypeKernel;break;
        case Runtime::NodeType::memcpy:*type=cudaGraphNodeTypeMemcpy;break;
        case Runtime::NodeType::memset:*type=cudaGraphNodeTypeMemset;break;
        case Runtime::NodeType::host:*type=cudaGraphNodeTypeHost;break;
        case Runtime::NodeType::subgraph:*type=cudaGraphNodeTypeGraph;break;
        case Runtime::NodeType::empty:*type=cudaGraphNodeTypeEmpty;break;
        default:throw Error{cudaErrorNotSupported};
    }
});}
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node,cudaKernelNodeParams* params) noexcept {
    if(!params)return remember(cudaErrorInvalidValue);
    cudaGraphNodeType type;const auto status=cudaGraphNodeGetType(node,&type);if(status!=cudaSuccess)return status;
    if(type!=cudaGraphNodeTypeKernel)return remember(cudaErrorInvalidValue);
    // SYCL's public node interface exposes no CUDA function pointer or
    // argument array. Keep this optional diagnostic unsupported; do not return
    // fabricated parameters. The original verify listing handles this error.
    return remember(cudaErrorNotSupported);
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e,cudaGraph_t g,cudaGraphNode_t* bad,char* log,size_t bytes) noexcept {return api([&]{
    require(e);*e=nullptr;if(bad)*bad=nullptr;if(log && bytes)log[0]='\0';auto definition=devices().definitions.get(g);auto& d=*definition->owner;auto value=std::make_shared<Executable>();value->owner=&d;
    value->graph=d.runtime.instantiate(definition->graph);*e=devices().executables.insert(std::move(value));});}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e,cudaGraph_t g,unsigned long long flags) noexcept {
    if(flags) {if(e)*e=nullptr;return remember(cudaErrorNotSupported);}return cudaGraphInstantiate(e,g,nullptr,nullptr,0);
}
cudaError_t cudaGraphLaunch(cudaGraphExec_t e,cudaStream_t s) noexcept {return api([&]{auto& d=stream_domain(s);auto value=devices().executables.get(e);require(value->owner==&d,cudaErrorInvalidResourceHandle);d.runtime.launch(value->graph,stream_id(s));});}
cudaError_t cudaGraphUpload(cudaGraphExec_t e,cudaStream_t s) noexcept {return api([&]{auto& d=stream_domain(s);auto value=devices().executables.get(e);require(value->owner==&d,cudaErrorInvalidResourceHandle);d.runtime.upload(value->graph,stream_id(s));});}
cudaError_t cudaGraphDestroy(cudaGraph_t g) noexcept {return api([&]{auto definition=devices().definitions.get(g);std::lock_guard lock(definition->mutex);
    require(definition->alive,cudaErrorInvalidResourceHandle);definition->alive=false;
    for(auto node:definition->nodes)devices().nodes.erase(node);
    devices().definitions.erase(g);});}
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t e) noexcept {return api([&]{devices().executables.erase(e);});}
namespace strata::sycl_upstream::cuda {
cudaError_t submit(cudaStream_t s,const Runtime::Submit& fn) noexcept {return api([&]{auto& d=stream_domain(s);d.runtime.enqueue(stream_id(s),fn);});}
cudaError_t stream_device(cudaStream_t s,sycl::device* output) noexcept {return api([&]{require(output);*output=stream_domain(s).runtime.device();});}
cudaError_t inspect_device(int ordinal,const std::function<void(const sycl::device&)>& fn) noexcept {
    return api([&]{devices().validate(ordinal);require(bool(fn));fn(devices().visible[ordinal]);});
}
cudaError_t device_context(int ordinal,sycl::context* output) noexcept {
    return api([&]{require(output);*output=devices().get(ordinal).runtime.context();});
}
cudaError_t validate_device_buffer(cudaStream_t s,const void* p,size_t bytes) noexcept {return api([&]{
    auto& d=stream_domain(s);stream_id(s);auto info=range(p,bytes);
    if(bytes){require(info && info->kind==Memory::Kind::device);accessible(d,p);}
});}
Memory::Stats memory_stats(){return domain().memory.stats();}
std::optional<Memory::Info> allocation_info(const void* p){auto a=allocation(p);return a ? std::optional<Memory::Info>(a->info) : std::nullopt;}
const char* backend_error_detail() noexcept{return error_detail.c_str();}
}
