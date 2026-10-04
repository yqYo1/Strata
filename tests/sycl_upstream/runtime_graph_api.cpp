// Access the unchanged original MTP capture-finishing helper. Other engine
// functions are discarded at link time; no device/GEMM test stubs are added.
#include "strata/sycl_upstream/cuda_backend.hpp"
#include "../../src/core/mtp.cpp"
#include <future>
#include <iostream>
using namespace std::chrono_literals;
namespace backend=strata::sycl_upstream::cuda;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string(cudaGetErrorName(e))+": "+backend::backend_error_detail());}
struct Gate {
    std::promise<void> release;std::shared_future<void> future=release.get_future().share();bool opened=false;
    void open(){if(!opened){release.set_value();opened=true;}}
    ~Gate(){open();}
};
void hold(cudaStream_t s,Gate& gate){ck(backend::submit(s,[future=gate.future](sycl::queue& q){return q.submit([&](sycl::handler& h){h.host_task([future]{future.wait();});});}));}
int main() try {
    ck(cudaSetDevice(0));cudaStream_t a{},b{},other{},blocking{};
    ck(cudaStreamCreateWithFlags(&a,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&b,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));ck(cudaStreamCreate(&blocking));
    int *device{},*host{};ck(cudaMalloc(&device,2*sizeof(int)));ck(cudaMallocHost(&host,2*sizeof(int)));ck(cudaMemset(device,0,2*sizeof(int)));ck(cudaDeviceSynchronize());
    int cases=0;std::atomic<int> callbacks{0};auto* calls=&callbacks;
    cudaGraphExec_t executable{};cudaGraph_t graph{};
    {
        ck(cudaStreamBeginCapture(a,cudaStreamCaptureModeThreadLocal));ck(cudaMemsetAsync(device,0,2*sizeof(int),a));
        ck(backend::submit(a,[=](sycl::queue& q){return q.single_task([=]{device[0]=43;});}));
        ck(cudaMemcpyAsync(host,device,sizeof(int),cudaMemcpyDeviceToHost,a));
        ck(backend::submit(a,[=](sycl::queue& q){return q.submit([&](sycl::handler& h){h.host_task([calls]{calls->fetch_add(1);});});}));
        ck(cudaStreamEndCapture(a,&graph));size_t n=99;ck(cudaGraphGetNodes(graph,nullptr,&n));check(n==4,"native mixed graph node count");
        std::vector<cudaGraphNode_t> nodes(n+2,reinterpret_cast<cudaGraphNode_t>(1));size_t capacity=nodes.size();ck(cudaGraphGetNodes(graph,nodes.data(),&capacity));check(capacity==n && !nodes[n] && !nodes[n+1],"graph oversized enumeration");
        std::vector<cudaGraphNode_t> partial(2);capacity=2;ck(cudaGraphGetNodes(graph,partial.data(),&capacity));check(capacity==2 && std::equal(partial.begin(),partial.end(),nodes.begin()),"graph stable partial enumeration");
        capacity=0;auto untouched=reinterpret_cast<cudaGraphNode_t>(3);ck(cudaGraphGetNodes(graph,&untouched,&capacity));check(!capacity && untouched==reinterpret_cast<cudaGraphNode_t>(3),"zero-capacity enumeration wrote output");
        std::map<cudaGraphNodeType,int> kinds;cudaGraphNode_t kernel{};
        for(size_t i=0;i<n;++i){cudaGraphNodeType type;ck(cudaGraphNodeGetType(nodes[i],&type));++kinds[type];if(type==cudaGraphNodeTypeKernel)kernel=nodes[i];}
        check(kinds[cudaGraphNodeTypeKernel]==1 && kinds[cudaGraphNodeTypeMemcpy]==1 && kinds[cudaGraphNodeTypeMemset]==1 && kinds[cudaGraphNodeTypeHost]==1,"native graph node kinds");
        cudaKernelNodeParams params{};params.func=reinterpret_cast<void*>(5);params.sharedMemBytes=123;
        check(cudaGraphKernelNodeGetParams(kernel,&params)==cudaErrorNotSupported && params.func==reinterpret_cast<void*>(5) && params.sharedMemBytes==123,"unsupported parameters fabricated or changed output");cudaGetLastError();
        check(cudaGraphNodeGetType(kernel,nullptr)==cudaErrorInvalidValue,"null node type output");cudaGetLastError();
        ck(cudaGraphInstantiate(&executable,graph,0ull));ck(cudaGraphDestroy(graph));
        cudaGraphNodeType type=cudaGraphNodeTypeEmpty;check(cudaGraphNodeGetType(kernel,&type)==cudaErrorInvalidResourceHandle && type==cudaGraphNodeTypeEmpty,"destroyed graph retained a valid node");cudaGetLastError();
        ck(cudaGraphUpload(executable,a));ck(cudaStreamSynchronize(a));ck(cudaMemcpy(host+1,device,sizeof(int),cudaMemcpyDeviceToHost));check(host[1]==0 && callbacks.load()==0,"upload executed captured commands");
        ck(cudaGraphLaunch(executable,a));ck(cudaStreamSynchronize(a));check(host[0]==43 && callbacks.load()==1,"mixed graph replay after definition release");++cases;
    }
    {
        Gate gate;hold(a,gate);cudaEvent_t done{};ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
        auto task=std::async(std::launch::async,[&]{ck(cudaGraphUpload(executable,a));ck(cudaGraphUpload(executable,b));ck(cudaEventRecord(done,b));});
        bool submitted=task.wait_for(2s)==std::future_status::ready;if(!submitted)gate.open();task.get();check(submitted,"graph upload blocked submitting host");
        check(cudaEventQuery(done)==cudaErrorNotReady,"uploads did not serialize across streams");
        ck(cudaMemsetAsync(device+1,0x22,sizeof(int),other));ck(cudaStreamSynchronize(other));check(cudaEventQuery(done)==cudaErrorNotReady,"upload synchronized unrelated nonblocking stream");
        ck(cudaGraphLaunch(executable,b));gate.open();ck(cudaStreamSynchronize(b));check(callbacks.load()==2 && host[0]==43,"launch following upload lost ordering");ck(cudaEventDestroy(done));++cases;
    }
    {
        Gate gate;cudaGraphExec_t held{};ck(cudaStreamBeginCapture(a,cudaStreamCaptureModeThreadLocal));hold(a,gate);
        ck(backend::submit(a,[=](sycl::queue& q){return q.single_task([=]{device[1]=91;});}));ck(cudaStreamEndCapture(a,&graph));ck(cudaGraphInstantiate(&held,graph,0ull));ck(cudaGraphDestroy(graph));
        ck(cudaGraphLaunch(held,a));cudaEvent_t done{};ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
        auto task=std::async(std::launch::async,[&]{ck(cudaGraphUpload(held,b));ck(cudaEventRecord(done,b));ck(cudaGraphExecDestroy(held));});
        bool submitted=task.wait_for(2s)==std::future_status::ready;if(!submitted)gate.open();task.get();check(submitted,"pending upload executable release blocked host");
        check(cudaEventQuery(done)==cudaErrorNotReady,"upload lost dependency on previous graph launch");gate.open();ck(cudaEventSynchronize(done));
        ck(cudaMemcpy(host+1,device+1,sizeof(int),cudaMemcpyDeviceToHost));check(host[1]==91,"executable died during pending upload/launch");ck(cudaEventDestroy(done));++cases;
    }
    {
        Gate gate;hold(blocking,gate);cudaEvent_t done{};ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
        auto task=std::async(std::launch::async,[&]{ck(cudaGraphUpload(executable,nullptr));ck(cudaEventRecord(done,nullptr));});
        bool submitted=task.wait_for(2s)==std::future_status::ready;if(!submitted)gate.open();task.get();check(submitted,"default upload blocked host");check(cudaEventQuery(done)==cudaErrorNotReady,"default upload ignored legacy blocking stream");
        gate.open();ck(cudaEventSynchronize(done));check(callbacks.load()==2,"default upload replayed graph");ck(cudaEventDestroy(done));++cases;
    }
    {
        ck(cudaStreamBeginCapture(a,cudaStreamCaptureModeThreadLocal));ck(cudaStreamEndCapture(a,&graph));size_t n=2;cudaGraphNode_t empty[2]{reinterpret_cast<cudaGraphNode_t>(1),reinterpret_cast<cudaGraphNode_t>(1)};ck(cudaGraphGetNodes(graph,empty,&n));check(n==0 && !empty[0] && !empty[1],"empty graph enumeration");ck(cudaGraphDestroy(graph));
        check(cudaGraphUpload(reinterpret_cast<cudaGraphExec_t>(1),a)==cudaErrorInvalidResourceHandle,"invalid upload executable");cudaGetLastError();
        ck(cudaStreamBeginCapture(a,cudaStreamCaptureModeThreadLocal));check(cudaGraphUpload(executable,a)!=cudaSuccess,"upload allowed during capture");cudaGetLastError();cudaStreamCaptureStatus status;ck(cudaStreamIsCapturing(a,&status));check(status==cudaStreamCaptureStatusInvalidated,"upload did not invalidate capture");check(cudaStreamEndCapture(a,&graph)!=cudaSuccess && !graph,"invalidated upload capture survived");cudaGetLastError();++cases;
    }
    {
        cudaGraphExec_t original{};std::string error;ck(cudaStreamBeginCapture(a,cudaStreamCaptureModeThreadLocal));
        ck(backend::submit(a,[=](sycl::queue& q){return q.single_task([=]{device[0]+=7;});}));
        check(strata::core::finish_capture(a,true,original,"SYCL probe",error),"original MTP finish capture failed");
        ck(cudaMemcpy(host,device,sizeof(int),cudaMemcpyDeviceToHost));check(host[0]==43,"original capture finisher executed graph while uploading");
        for(int replay=0;replay<4;++replay){ck(cudaGraphLaunch(original,a));ck(cudaStreamSynchronize(a));ck(cudaMemcpy(host,device,sizeof(int),cudaMemcpyDeviceToHost));check(host[0]==43+7*(replay+1),"original MTP finished graph replay output");}
        ck(cudaGraphExecDestroy(original));++cases;
    }
    int concurrent_wait_cases=0;
    for(int kind=0;kind<4;++kind) {
        Gate gate;const auto stream=kind==3 ? blocking : a;hold(stream,gate);
        cudaEvent_t done{};ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));ck(cudaEventRecord(done,stream));
        std::promise<void> entering;auto entered=entering.get_future();
        auto waiter=std::async(std::launch::async,[&]{entering.set_value();
            if(kind==0)ck(cudaEventSynchronize(done));else if(kind==1)ck(cudaStreamSynchronize(stream));
            else if(kind==2)ck(cudaDeviceSynchronize());else ck(cudaMemcpy(host+1,device,sizeof(int),cudaMemcpyDeviceToHost));
        });
        entered.wait();std::this_thread::sleep_for(20ms);
        std::atomic<int> serviced{0};
        auto submit=std::async(std::launch::async,[&]{ck(cudaLaunchHostFunc(other,[](void* p){static_cast<std::atomic<int>*>(p)->fetch_add(1);},&serviced));});
        bool ready=submit.wait_for(2s)==std::future_status::ready;
        bool still_waiting=waiter.wait_for(0s)!=std::future_status::ready;
        gate.open();submit.get();waiter.get();ck(cudaStreamSynchronize(other));ck(cudaEventDestroy(done));
        check(ready && still_waiting && serviced.load()==1,"native completion wait blocked an unrelated submission");++concurrent_wait_cases;
    }
    {
        ck(backend::submit(other,[](sycl::queue& q){return q.submit([](sycl::handler& h){h.host_task([]{throw std::runtime_error("SYCL async wait probe");});});}));
        cudaEvent_t done{};ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));ck(cudaEventRecord(done,other));
        check(cudaEventSynchronize(done)==cudaErrorUnknown && std::string(backend::backend_error_detail()).find("SYCL async wait probe")!=std::string::npos,"completion polling hid an asynchronous error");
        check(cudaGetLastError()==cudaErrorUnknown,"asynchronous error was not retained");ck(cudaEventDestroy(done));
        ck(cudaMemsetAsync(device+1,0x4c,sizeof(int),other));ck(cudaStreamSynchronize(other));
        ck(cudaMemcpy(host+1,device+1,sizeof(int),cudaMemcpyDeviceToHost));check(host[1]==0x4c4c4c4c,"queue unusable after async error delivery");
    }
    ck(cudaGraphExecDestroy(executable));ck(cudaStreamDestroy(blocking));ck(cudaStreamDestroy(other));ck(cudaStreamDestroy(b));ck(cudaStreamDestroy(a));ck(cudaFreeHost(host));ck(cudaFree(device));
    std::cout<<"PASS graph_api_cases="<<cases<<" native_node_kinds=4 stable_enumeration=1 upload_without_execution=1 asynchronous_upload=1 serialized_upload_launch=1 pending_destroy=1 original_mtp_finish_capture=1 original_mtp_replays=4 concurrent_native_wait_cases="<<concurrent_wait_cases<<" asynchronous_error_delivery=1\n";
} catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
