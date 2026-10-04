#include "strata/sycl_upstream/cuda_backend.hpp"
#include "strata/core/remote_experts.hpp"
#include <algorithm>
#include <future>
#include <iostream>
using namespace std::chrono_literals;
namespace backend=strata::sycl_upstream::cuda;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorName(e))+": "+backend::backend_error_detail());}
struct Gate {
    std::promise<void> release;std::shared_future<void> future=release.get_future().share();bool opened=false;
    static void hold(void* p){static_cast<Gate*>(p)->future.wait();}
    void open(){if(!opened){release.set_value();opened=true;}}
    ~Gate(){open();}
};
int main() try {
    int count=0,current=-1;ck(cudaGetDeviceCount(&count));ck(cudaGetDevice(&current));check(count>0,"no GPU");int cases=0;
    {
        ck(cudaInitDevice(0,0,0));ck(cudaInitDevice(0,cudaDeviceMapHost,0));int after=-1;ck(cudaGetDevice(&after));check(after==current,"initialization selected another device");
        check(cudaInitDevice(count,0,0)==cudaErrorInvalidDevice,"invalid initialization ordinal");cudaGetLastError();
        check(cudaInitDevice(0,0,1)==cudaErrorInvalidValue,"unsupported initialization flags");cudaGetLastError();
        check(cudaInitDevice(0,cudaDeviceScheduleSpin|cudaDeviceMapHost,0)==cudaErrorNotSupported,"spin scheduling fabricated success");cudaGetLastError();
        int capability=123;
        check(cudaDeviceCanAccessPeer(nullptr,0,0)==cudaErrorInvalidValue,"null peer output");cudaGetLastError();
        check(cudaDeviceCanAccessPeer(&capability,0,count)==cudaErrorInvalidDevice && capability==123,"invisible peer queried");cudaGetLastError();
        ck(cudaDeviceCanAccessPeer(&capability,0,0));check(capability==0,"same device advertised as peer");
        check(cudaDeviceEnablePeerAccess(0)==cudaErrorInvalidDevice && cudaDeviceDisablePeerAccess(0)==cudaErrorInvalidDevice,"self peer enabled");cudaGetLastError();
        check(cudaDeviceEnablePeerAccess(count)==cudaErrorInvalidDevice && cudaDeviceEnablePeerAccess(0,1)==cudaErrorInvalidValue,"invalid peer enable");cudaGetLastError();
        double gib=123;std::string error;
        check(!strata::core::RemoteExperts::preflight(count,gib,error) && gib==123 && error.find("not visible")!=std::string::npos,"original remote preflight admitted invisible device");++cases;
    }
    constexpr size_t bytes=4096;
    uint8_t *source{},*destination{},*host{};ck(cudaMalloc(&source,bytes));ck(cudaMalloc(&destination,bytes));ck(cudaMallocHost(&host,bytes));
    cudaStream_t stream{},other{};ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
    cudaEvent_t done{};ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
    for(size_t i=0;i<bytes;++i)host[i]=uint8_t(i*7+19);ck(cudaMemcpy(source,host,bytes,cudaMemcpyHostToDevice));ck(cudaStreamSynchronize(nullptr));
    {
        check(cudaMemcpyPeerAsync(destination,count,source,0,bytes,stream)==cudaErrorInvalidDevice,"invalid destination device");cudaGetLastError();
        check(cudaMemcpyPeerAsync(destination,0,source,-1,bytes,stream)==cudaErrorInvalidDevice,"invalid source device");cudaGetLastError();
        check(cudaMemcpyPeerAsync(destination,0,host,0,bytes,stream)==cudaErrorInvalidValue,"host pointer accepted as device source");cudaGetLastError();
        check(cudaMemcpyPeerAsync(destination+bytes-1,0,source,0,2,stream)==cudaErrorInvalidValue,"peer destination overflow");cudaGetLastError();
        ck(cudaMemcpyPeerAsync(nullptr,0,nullptr,0,0,stream));
        cudaStream_t stale{};ck(cudaStreamCreate(&stale));ck(cudaStreamDestroy(stale));check(cudaMemcpyPeerAsync(destination,0,source,0,bytes,stale)==cudaErrorInvalidResourceHandle,"stale peer stream admitted");cudaGetLastError();++cases;
    }
    {
        Gate gate;ck(backend::submit(stream,[future=gate.future](sycl::queue& q){return q.submit([&](sycl::handler& h){h.host_task([future]{future.wait();});});}));
        auto submit=std::async(std::launch::async,[&]{ck(cudaMemcpyPeerAsync(destination,0,source,0,bytes,stream));ck(cudaEventRecord(done,stream));});
        bool ready=submit.wait_for(2s)==std::future_status::ready;if(!ready)gate.open();submit.get();check(ready,"peer copy blocked the submitting host");
        check(cudaEventQuery(done)==cudaErrorNotReady,"peer copy did not follow held stream work");
        ck(cudaMemsetAsync(destination,0x11,bytes,other));ck(cudaStreamSynchronize(other));check(cudaEventQuery(done)==cudaErrorNotReady,"peer copy synchronized independent stream");
        gate.open();ck(cudaEventSynchronize(done));ck(cudaMemcpy(host,destination,bytes,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<bytes;++i)check(host[i]==uint8_t(i*7+19),"same-GPU peer copy data");++cases;
    }
    {
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ck(cudaMemcpyPeerAsync(destination,0,source,0,bytes,stream));cudaGraph_t graph{};ck(cudaStreamEndCapture(stream,&graph));size_t nodes=0;ck(cudaGraphGetNodes(graph,nullptr,&nodes));check(nodes==1,"peer copy was not recorded as native graph command");
        cudaGraphExec_t executable{};ck(cudaGraphInstantiate(&executable,graph,0ull));ck(cudaGraphDestroy(graph));
        for(int replay=0;replay<4;++replay){for(size_t i=0;i<bytes;++i)host[i]=uint8_t(i*3+replay*41);ck(cudaMemcpy(source,host,bytes,cudaMemcpyHostToDevice));ck(cudaStreamSynchronize(nullptr));ck(cudaGraphLaunch(executable,stream));ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(host,destination,bytes,cudaMemcpyDeviceToHost));for(size_t i=0;i<bytes;++i)check(host[i]==uint8_t(i*3+replay*41),"peer-copy graph reused stale input");}
        ck(cudaGraphExecDestroy(executable));++cases;
    }
    int physical_pairs=0,physical_copies=0;
    if(count>1) {
        int a=0,b=0;ck(cudaDeviceCanAccessPeer(&a,0,1));ck(cudaDeviceCanAccessPeer(&b,1,0));physical_pairs=1;
        std::cout<<"peer_0_to_1="<<a<<" peer_1_to_0="<<b<<'\n';
        if(a && b) {
            ck(cudaDeviceEnablePeerAccess(1));check(cudaDeviceEnablePeerAccess(1)==cudaErrorPeerAccessAlreadyEnabled,"duplicate native peer access");cudaGetLastError();
            ck(cudaSetDevice(1));ck(cudaDeviceEnablePeerAccess(0));uint8_t* remote{};ck(cudaMalloc(&remote,bytes));cudaStream_t remote_stream{};ck(cudaStreamCreateWithFlags(&remote_stream,cudaStreamNonBlocking));
            ck(cudaMemcpyPeerAsync(remote,1,source,0,bytes,remote_stream));ck(cudaStreamSynchronize(remote_stream));ck(cudaMemcpyPeerAsync(destination,0,remote,1,bytes,remote_stream));ck(cudaStreamSynchronize(remote_stream));ck(cudaSetDevice(0));ck(cudaMemcpy(host,destination,bytes,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<bytes;++i)check(host[i]==uint8_t(i*3+3*41),"two-GPU peer data");
            ck(cudaSetDevice(1));ck(cudaFree(remote));ck(cudaStreamDestroy(remote_stream));ck(cudaDeviceDisablePeerAccess(0));check(cudaDeviceDisablePeerAccess(0)==cudaErrorPeerAccessNotEnabled,"native peer disable state");cudaGetLastError();ck(cudaSetDevice(0));ck(cudaDeviceDisablePeerAccess(1));physical_copies=2;
        }
    }
    ck(cudaEventDestroy(done));ck(cudaStreamDestroy(other));ck(cudaStreamDestroy(stream));ck(cudaFreeHost(host));ck(cudaFree(destination));ck(cudaFree(source));
    std::cout<<"PASS peer_cases="<<cases<<" same_gpu_bytes_verified="<<bytes*5<<" original_remote_preflight=1 actual_device_pairs_queried="<<physical_pairs<<" physical_peer_copies="<<physical_copies<<'\n';
} catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
