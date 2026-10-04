#include "strata/sycl_upstream/cuda_backend.hpp"
#include "strata/core/graph.hpp"
#include "strata/sycl_upstream/handoff.hpp"
#include <immintrin.h>
#include "strata/core/pinned.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <unistd.h>
using namespace std::chrono_literals;
namespace backend=strata::sycl_upstream::cuda;
using strata::core::CapturedGraph;
using strata::core::GraphRegistry;
using strata::core::LayerType;
void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
void ok(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorName(e))+": "+backend::backend_error_detail());}
struct Buffer {
    void* p=nullptr;bool host;
    Buffer(size_t bytes,bool host):host(host){ok(host?cudaHostAlloc(&p,bytes,cudaHostAllocMapped):cudaMalloc(&p,bytes));}
    ~Buffer(){if(p){auto e=host?cudaFreeHost(p):cudaFree(p);if(e!=cudaSuccess)std::terminate();}}
    template<class T> T* as(){return static_cast<T*>(p);}
};
struct Stream {
    cudaStream_t value=nullptr;
    Stream(){ok(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){if(value){cudaStreamCaptureStatus status;
        if(cudaStreamIsCapturing(value,&status)==cudaSuccess && status!=cudaStreamCaptureStatusNone){
            cudaGraph_t graph=nullptr;cudaStreamEndCapture(value,&graph);if(graph)cudaGraphDestroy(graph);}
        cudaStreamDestroy(value);}}
};
struct Gate {
    std::promise<void> promise;std::shared_future<void> future=promise.get_future().share();bool opened=false;
    static void hold(void* p){static_cast<Gate*>(p)->future.wait();}
    void open(){if(!opened){promise.set_value();opened=true;}}
    ~Gate(){open();}
};
struct File {
    char path[48]="/tmp/strata-frontend-XXXXXX";int fd;
    File(){fd=mkstemp(path);check(fd>=0,"temporary file");}
    ~File(){close(fd);unlink(path);}
};
int main() try {
    setenv("STRATA_ARENA_PIN_GIB","0",1);setenv("STRATA_ARENA_LOCK","0",1);
    int cases=0;size_t values=0;
    Buffer input(16384,true),output(16384,true),device(16384,false),other(16384,false);
    auto* in=input.as<uint32_t>();auto* out=output.as<uint32_t>();auto* dev=device.as<uint32_t>();auto* tmp=other.as<uint32_t>();
    Stream stream;std::string err;
    {
        // Successful calls must not erase a previous error; error state is per thread.
        check(cudaStreamDestroy(nullptr)==cudaErrorInvalidResourceHandle,"invalid stream error");
        ok(cudaFree(nullptr));check(cudaPeekAtLastError()==cudaErrorInvalidResourceHandle,"error was cleared by success");
        auto worker=std::async(std::launch::async,[]{check(cudaGetLastError()==cudaSuccess,"error crossed threads");
            check(cudaMalloc(nullptr,4)==cudaErrorInvalidValue,"worker error");return cudaGetLastError();});
        check(worker.get()==cudaErrorInvalidValue && cudaGetLastError()==cudaErrorInvalidResourceHandle,"thread local errors");
        check(cudaGetLastError()==cudaSuccess,"error did not clear");
        cudaEvent_t event=nullptr;ok(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));ok(cudaEventQuery(event));
        Gate gate;ok(cudaLaunchHostFunc(stream.value,Gate::hold,&gate));ok(cudaEventRecord(event,stream.value));
        check(cudaEventQuery(event)==cudaErrorNotReady && cudaStreamQuery(stream.value)==cudaErrorNotReady,"pending query");
        check(cudaGetLastError()==cudaSuccess,"NotReady poisoned last error");
        gate.open();ok(cudaEventSynchronize(event));ok(cudaEventDestroy(event));
        check(cudaEventQuery(event)==cudaErrorInvalidResourceHandle,"destroyed event remained valid");cudaGetLastError();
        GraphRegistry registry(stream.value);
        check(!registry.record(LayerType::GDN,1,[&]{cudaMemsetAsync(nullptr,1,4,stream.value);},err),"capture body error was ignored");
        check(err.find("capture body left a CUDA error")!=std::string::npos && registry.size()==0,"original registry error path");
        CapturedGraph empty;check(empty.begin(stream.value,err),"empty begin");
        check(!empty.end(stream.value,err) && err.find("ZERO nodes")!=std::string::npos,"original empty graph rejection");
        ok(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
        void* rejected=nullptr;check(cudaMalloc(&rejected,4)!=cudaSuccess && !rejected,"allocation during capture");
        cudaStreamCaptureStatus status;ok(cudaStreamIsCapturing(stream.value,&status));
        check(status==cudaStreamCaptureStatusInvalidated,"invalidated status was hidden");
        cudaGraph_t invalid=nullptr;check(cudaStreamEndCapture(stream.value,&invalid)!=cudaSuccess && !invalid,"invalid capture survived");
        cudaGetLastError();++cases;
    }
    {
        GraphRegistry registry(stream.value);int bodies=0;
        for(auto type:{LayerType::GDN,LayerType::QSA})for(int tokens:{1,4}) {
            auto body=[&,type,tokens]{++bodies;ok(cudaMemcpyAsync(dev,in,512,cudaMemcpyHostToDevice,stream.value));
                const uint32_t factor=type==LayerType::GDN?3:5;
                ok(backend::submit(stream.value,[=](sycl::queue& q){return q.parallel_for(sycl::range<1>(128),[=](sycl::id<1> i){dev[i]=dev[i]*factor+tokens;});}));
                ok(cudaMemcpyAsync(out,dev,512,cudaMemcpyDeviceToHost,stream.value));};
            check(registry.record(type,tokens,body,err),err.c_str());
            check(registry.record(type,tokens,body,err),"cached record");
            check(registry.find(type,tokens)->nodes()==3,"original graph node count");
        }
        check(bodies==4 && registry.captures()==4 && registry.size()==4,"capture cache did not preserve original behavior");
        for(int run=0;run<32;++run){auto type=run%2?LayerType::QSA:LayerType::GDN;int tokens=run%4<2?1:4;
            for(int i=0;i<128;++i)in[i]=uint32_t(run*1000+i);
            check(registry.launch(type,tokens,2000,err),err.c_str());
            for(int i=0;i<128;++i){check(out[i]==in[i]*(type==LayerType::GDN?3:5)+tokens,"original registry replay result");++values;}
        }
        CapturedGraph source;check(source.begin(stream.value,err),"move graph begin");
        ok(cudaMemsetAsync(dev,0x12,4,stream.value));check(source.end(stream.value,err),err.c_str());
        CapturedGraph moved(std::move(source));check(!source.valid() && moved.valid(),"graph move ownership");
        check(moved.launch(stream.value,err) && moved.wait_ms(2000),"moved graph launch");++cases;
    }
    {
        // Distinct executable instances may run independently. Their shared
        // device counter uses an atomic so concurrent launches remain valid.
        ok(cudaMemset(dev,0,4));ok(cudaDeviceSynchronize());
        ok(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
        ok(backend::submit(stream.value,[=](sycl::queue& q){return q.single_task([=]{
            sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> count(*dev);count.fetch_add(1);});}));
        cudaGraph_t definition=nullptr;ok(cudaStreamEndCapture(stream.value,&definition));
        cudaGraphExec_t first=nullptr,second=nullptr;ok(cudaGraphInstantiate(&first,definition,0));ok(cudaGraphInstantiate(&second,definition,0));
        ok(cudaGraphDestroy(definition));Stream held,free;
        Gate gate;ok(cudaLaunchHostFunc(held.value,Gate::hold,&gate));ok(cudaGraphLaunch(first,held.value));
        ok(cudaGraphLaunch(second,free.value));ok(cudaMemcpyAsync(out,dev,4,cudaMemcpyDeviceToHost,free.value));
        auto completed=std::async(std::launch::async,[&]{return cudaStreamSynchronize(free.value);});
        const bool independent=completed.wait_for(2s)==std::future_status::ready;
        bool second_only=independent && out[0]==1;
        auto destroyed=std::async(std::launch::async,[&]{return cudaGraphExecDestroy(first);});
        const bool immediate=destroyed.wait_for(2s)==std::future_status::ready;
        gate.open();ok(completed.get());ok(destroyed.get());ok(cudaDeviceSynchronize());
        ok(cudaMemcpy(out,dev,4,cudaMemcpyDeviceToHost));
        check(independent && second_only && immediate && out[0]==2,"graph instantiation or asynchronous release lifetime");
        ok(cudaGraphExecDestroy(second));++values;++cases;
    }
    {
        // Device-to-device copy and device memset keep their asynchronous
        // synchronous-spelling behavior; pinned H2D must wait for completion.
        in[0]=71;ok(cudaMemcpy(dev,in,4,cudaMemcpyHostToDevice));
        Gate gate;ok(cudaLaunchHostFunc(nullptr,Gate::hold,&gate));
        auto submitted=std::async(std::launch::async,[&]{ok(cudaMemcpy(tmp,dev,4,cudaMemcpyDeviceToDevice));return cudaMemset(dev+1,0,4);});
        const bool async=submitted.wait_for(2s)==std::future_status::ready;
        gate.open();ok(submitted.get());ok(cudaDeviceSynchronize());check(async,"D2D or device memset blocked host");
        ok(cudaMemcpy(out,tmp,4,cudaMemcpyDeviceToHost));check(out[0]==71,"D2D result");++values;
        Gate pinned;ok(cudaLaunchHostFunc(nullptr,Gate::hold,&pinned));in[0]=83;
        auto copying=std::async(std::launch::async,[&]{return cudaMemcpy(dev,in,4,cudaMemcpyHostToDevice);});
        const bool waits=copying.wait_for(50ms)==std::future_status::timeout;
        pinned.open();ok(copying.get());check(waits,"pinned synchronous H2D returned too early");
        ok(cudaMemcpy(out,dev,4,cudaMemcpyDeviceToHost));check(out[0]==83,"pinned copy result");++values;
        std::vector<uint32_t> pageable(2048);
        for(int run=0;run<8;++run){for(size_t i=0;i<pageable.size();++i)pageable[i]=uint32_t(run*100+i);
            ok(cudaMemcpy(dev,pageable.data(),pageable.size()*4,cudaMemcpyHostToDevice));std::fill(pageable.begin(),pageable.end(),0);
            ok(cudaMemcpy(out,dev,pageable.size()*4,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<pageable.size();++i){check(out[i]==run*100+i,"pageable source lifetime after return");++values;}}
        Gate host;ok(cudaLaunchHostFunc(stream.value,Gate::hold,&host));in[0]=97;out[0]=0;
        auto h2h=std::async(std::launch::async,[&]{return cudaMemcpyAsync(out,in,4,cudaMemcpyHostToHost,stream.value);});
        const bool host_wait=h2h.wait_for(50ms)==std::future_status::timeout;
        host.open();ok(h2h.get());check(host_wait && out[0]==97,"Async H2H not synchronous");++values;++cases;
    }
    {
        constexpr size_t rows=7,width=13,sp=19,dp=31,xp=23,total=rows*dp;
        auto* src=input.as<unsigned char>();auto* dst=device.as<unsigned char>();auto* middle=other.as<unsigned char>();auto* got=output.as<unsigned char>();
        auto body=[&]{
            ok(cudaMemsetAsync(dst,0xcd,total,stream.value));
            ok(cudaMemsetAsync(middle,0xcd,rows*xp,stream.value));
            ok(cudaMemsetAsync(got,0xcd,total,stream.value));
            ok(cudaMemcpy2DAsync(dst,dp,src,sp,width,rows,cudaMemcpyDefault,stream.value));
            ok(cudaMemcpy2DAsync(middle,xp,dst,dp,width,rows,cudaMemcpyDefault,stream.value));
            ok(cudaMemcpy2DAsync(got,dp,middle,xp,width,rows,cudaMemcpyDefault,stream.value));
        };
        CapturedGraph graph;check(graph.begin(stream.value,err),"2D capture begin");body();
        check(graph.end(stream.value,err),err.c_str());
        check(graph.nodes()==rows*3+3,"pitched graph copy decomposition");
        for(int run=0;run<6;++run){for(size_t i=0;i<rows*sp;++i)src[i]=unsigned(i+run*17);
            if(run<2){body();ok(cudaStreamSynchronize(stream.value));}
            else check(graph.launch(stream.value,err) && graph.wait_ms(2000),"2D graph replay");
            for(size_t r=0;r<rows;++r)for(size_t col=0;col<dp;++col){
                check(got[r*dp+col]==(col<width?src[r*sp+col]:0xcd),"2D width/pitch or untouched padding");++values;}}
        ++cases;
    }
    {
        const size_t page=size_t(sysconf(_SC_PAGESIZE)),bytes=5*page,cap=3*page+128;
        const std::vector<uint64_t> bounds{0,page+64,cap,bytes};
        std::vector<unsigned char> fixture(bytes);for(size_t i=0;i<bytes;++i)fixture[i]=unsigned((i*37+11)%251);
        File file;check(write(file.fd,fixture.data(),bytes)==ssize_t(bytes),"write expert fixture");
        {
            strata::core::PinnedArena arena(bytes,bounds,cap);
            check(arena.valid() && arena.registered_bytes==cap && arena.registered_slices==2,"original capped arena registration");
            auto stats=backend::memory_stats();check(stats.registered_bytes==cap && stats.imported_bytes==page*4,"original slices did not share native pages");
            const std::vector<uint64_t> offsets{0,bounds[1],bounds[2]},sizes{bounds[1],bounds[2]-bounds[1],bytes-bounds[2]};
            auto loaded=strata::core::load_experts_ranges(file.path,arena.data(),offsets,sizes,3,257);
            check(loaded.ok && loaded.bytes==bytes && loaded.layers==3,"original expert loader");
            for(size_t i=0;i<bytes;++i){check(arena.data()[i]==fixture[i],"original loader bytes");++values;}
            for(size_t layer=0;layer<3;++layer){uint64_t hash=1469598103934665603ull;
                for(size_t i=offsets[layer];i<offsets[layer]+sizes[layer];++i){hash^=fixture[i];hash*=1099511628211ull;}
                check(loaded.layer_checksums[layer]==hash,"original loader checksums");}
            void* alias=nullptr;ok(cudaHostGetDevicePointer(&alias,arena.data()+bounds[1],0));
            check(alias==arena.data()+bounds[1],"original arena interior device alias");
            ok(cudaMemcpyAsync(dev,alias,128,cudaMemcpyHostToDevice,stream.value));
            ok(cudaMemcpyAsync(out,dev,128,cudaMemcpyDeviceToHost,stream.value));ok(cudaStreamSynchronize(stream.value));
            check(!std::memcmp(out,fixture.data()+bounds[1],128),"original arena transfer");values+=128;
            check(ftruncate(file.fd,bytes-3)==0,"truncate fixture");
            loaded=strata::core::load_experts_ranges(file.path,arena.data(),offsets,sizes,3,257);
            check(!loaded.ok && loaded.error.find("short read")!=std::string::npos,"original short-read refusal");
        }
        check(backend::memory_stats().registered_bytes==0 && backend::memory_stats().imported_bytes==0,"original arena teardown leaked imports");
        {strata::core::PinnedArena arena(page*2);check(arena.valid() && arena.registered_bytes==page*2 && arena.registered_slices==0,"original whole arena registration");}
        ++cases;
    }
    {
        using namespace strata::sycl_upstream;
        constexpr int n=128,k=2,layers=8;
        Buffer dx(n*4,false),di(k*4,false),dw(k*4,false),hx(n*4,true),hi(k*4,true),hw(k*4,true),reply(n*4,true),
            sequence(4,true),flag(4,true),timeouts(layers*4,true),polls(layers*4,true);
        auto* x=dx.as<float>();auto* ids=di.as<int32_t>();auto* weights=dw.as<float>();
        auto* cpu_x=hx.as<float>();auto* cpu_ids=hi.as<int32_t>();auto* cpu_weights=hw.as<float>();auto* cpu_reply=reply.as<float>();
        auto* seq=sequence.as<uint32_t>();auto* ack=flag.as<uint32_t>();auto* timed=timeouts.as<uint32_t>();auto* counts=polls.as<uint32_t>();
        *seq=0;*ack=0;
        int32_t initial_ids[k]={13,29};float initial_weights[k]={0.25f,0.75f};
        ok(cudaMemcpy(ids,initial_ids,sizeof initial_ids,cudaMemcpyHostToDevice));
        ok(cudaMemcpy(weights,initial_weights,sizeof initial_weights,cudaMemcpyHostToDevice));
        CapturedGraph bounded,unbounded;
        for(int mode=0;mode<2;++mode){auto& graph=mode?unbounded:bounded;check(graph.begin(stream.value,err),"handoff capture begin");
            for(int layer=0;layer<layers;++layer){
                ok(backend::submit(stream.value,[=](sycl::queue& q){return doorbell_publish(q,x,ids,weights,n,k,cpu_x,cpu_ids,cpu_weights,seq);}));
                const HandoffWaitLimit limit=mode?HandoffWaitLimit{}:HandoffWaitLimit{1000000,timed+layer,counts+layer};
                ok(backend::submit(stream.value,[=](sycl::queue& q){return doorbell_wait(q,ack,seq,limit);}));
                ok(backend::submit(stream.value,[=](sycl::queue& q){return copy_from_mapped(q,x,cpu_reply,n);}));
            }
            ok(cudaMemcpyAsync(out,x,n*4,cudaMemcpyDeviceToHost,stream.value));
            check(graph.end(stream.value,err) && graph.nodes()==layers*3+1,err.c_str());
        }
        size_t handoff_values=0;
        for(int run=0;run<4;++run){
            auto* initial=input.as<float>();for(int i=0;i<n;++i)initial[i]=float(run*100+i);
            std::fill(timed,timed+layers,0);std::fill(counts,counts+layers,0);
            ok(cudaMemcpy(x,initial,n*4,cudaMemcpyHostToDevice));
            std::promise<void> ready;auto started=ready.get_future();
            auto service=std::async(std::launch::async,[&]{ready.set_value();bool valid=true;size_t checked=0;
                for(int layer=0;layer<layers;++layer){const uint32_t wanted=uint32_t(run*layers+layer+1);
                    const auto deadline=std::chrono::steady_clock::now()+3s;
                    while(*reinterpret_cast<volatile uint32_t*>(seq)!=wanted){
                        if(std::chrono::steady_clock::now()>=deadline)return std::pair{false,checked};_mm_pause();}
                    std::atomic_thread_fence(std::memory_order_acquire);
                    for(int i=0;i<n;++i){valid&=cpu_x[i]==float(run*100+i+layer);cpu_reply[i]=cpu_x[i]+1; ++checked;}
                    for(int i=0;i<k;++i){valid&=cpu_ids[i]==initial_ids[i] && cpu_weights[i]==initial_weights[i];checked+=2;}
                    if(run<2 && layer==0)std::this_thread::sleep_for(1ms);
                    std::atomic_thread_fence(std::memory_order_seq_cst);_mm_sfence();
                    *reinterpret_cast<volatile uint32_t*>(ack)=wanted;
                }return std::pair{valid,checked};});
            started.wait();auto& graph=run<2?bounded:unbounded;
            check(graph.launch(stream.value,err),err.c_str());
            auto serviced=service.get(); // No query/sync during CPU service.
            check(graph.wait_ms(2000),"original CapturedGraph handoff completion");
            check(serviced.first,"frontend handoff payload or CPU timeout");handoff_values+=serviced.second;
            for(int i=0;i<layers;++i)check(timed[i]==0,"bounded GPU wait exhausted");
            if(run<2)check(counts[0]>0,"GPU did not actively wait for CPU reply");
            auto* result=output.as<float>();for(int i=0;i<n;++i){check(result[i]==float(run*100+i+layers),"connected frontend handoff output");++handoff_values;}
        }
        values+=handoff_values;++cases;
    }
    ok(cudaDeviceSynchronize());check(cudaGetLastError()==cudaSuccess,"unconsumed frontend error");
    std::cout<<"PASS frontend_cases="<<cases<<" exact_values="<<values
             <<" original_graph_registry_replays=32 independent_graph_execs=2 pitched_graph_replays=4 pitched_direct_runs=2 pitched_directions=3 pitched_copy_nodes=21 original_loader_layers=3 original_graph_handshakes=32 active_waits=2 unbounded_handoff_replays=2\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
