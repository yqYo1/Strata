// Compile the unchanged original translation unit to exercise its actual
// internal Stager. MMQ/fused/native declarations remain enabled. Unused engine
// functions are discarded at link time, not replaced by test implementations.
#include "strata/sycl_upstream/cuda_backend.hpp"
#include "../../src/prefill/prefill.cpp"
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
struct AssembledSource final: strata::core::ExpertSource {
    const std::vector<uint8_t>& data;const std::vector<size_t>& sizes;size_t stride,jobs;
    std::atomic<int> calls{0};
    AssembledSource(const std::vector<uint8_t>& data,const std::vector<size_t>& sizes,size_t blob,size_t jobs): data(data),sizes(sizes),stride(blob),jobs(jobs) {}
    const uint8_t* blob(int64_t,int64_t) override{return nullptr;}
    bool transient(int64_t,int64_t) const override{return true;}
    bool copy_blob(int64_t layer,int64_t expert,uint8_t* dst) override {
        if(layer<0 || expert<0 || size_t(expert)>=jobs)return false;
        const size_t id=size_t(layer)*jobs+size_t(expert);if(id>=sizes.size())return false;
        const size_t n=sizes[id],a=n/3,b=n/2;const auto* src=data.data()+id*stride;
        std::memcpy(dst,src,a);std::memcpy(dst+a,src+a,b);std::memcpy(dst+a+b,src+a+b,n-a-b);
        calls.fetch_add(1,std::memory_order_relaxed);return true;
    }
};
uint32_t checksum(const uint8_t* p,size_t bytes,uint32_t seed){uint32_t hash=2166136261u^seed;for(size_t i=0;i<bytes;++i)hash=(hash^p[i])*16777619u;return hash;}
int main(int argc,char** argv) try {
    check(setenv("STRATA_STAGER_RING","2",1)==0,"set stager ring");ck(cudaSetDevice(0));
    constexpr int generations=8,jobs=17;constexpr size_t total_jobs=generations*jobs;
    const bool model_sized=argc==2 && std::string(argv[1])=="--model-sized";
    const size_t blob=model_sized ? strata::kernels::cpu::BLOB : 65536;
    std::vector<uint8_t> source(total_jobs*blob),result(source.size());std::vector<size_t> sizes(total_jobs);
    std::vector<uint32_t> expected(total_jobs),observed(total_jobs);
    for(size_t id=0;id<total_jobs;++id){sizes[id]=id%7==0 ? blob : 257+(id*7919)%(blob-257);for(size_t i=0;i<blob;++i)source[id*blob+i]=uint8_t(id*23+i*31+11);expected[id]=checksum(source.data()+id*blob,sizes[id],uint32_t(id));}
    AssembledSource assembled(source,sizes,blob,jobs);
    uint8_t* device{};uint32_t* hashes{};ck(cudaMalloc(&device,source.size()));ck(cudaMalloc(&hashes,total_jobs*sizeof(uint32_t)));ck(cudaMemset(device,0xbd,source.size()));ck(cudaDeviceSynchronize());
    cudaStream_t copy{},compute{};ck(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking));
    {
        strata::prefill::Stager stager;check(stager.init(blob,3),"original stager initialization");check(stager.kRing==2 && std::all_of(stager.pinned.begin(),stager.pinned.end(),[](auto v){return bool(v);}),"stager did not use requested pinned ring");
        Gate gate;
        ck(backend::submit(copy,[future=gate.future](sycl::queue& q){return q.submit([&](sycl::handler& h){h.host_task([future]{future.wait();});});}));
        auto producer=std::async(std::launch::async,[&]{
            for(int generation=0;generation<generations;++generation){
                std::vector<strata::prefill::Stager::Job> plan;
                for(int j=0;j<jobs;++j){const size_t id=generation*jobs+j;if(id%2)plan.push_back({nullptr,sizes[id],&assembled,generation,j});else plan.push_back({source.data()+id*blob,sizes[id]});}
                stager.start(std::move(plan));
                for(int j=0;j<jobs;++j){const size_t id=generation*jobs+j,n=sizes[id];const auto* staged=stager.wait(j);
                    ck(cudaMemcpyAsync(device+id*blob,staged,n,cudaMemcpyHostToDevice,copy));stager.issued_one(j,copy);
                    ck(cudaStreamWaitEvent(compute,stager.dma_done[j%stager.kRing]));
                    ck(backend::submit(compute,[=](sycl::queue& q){return q.single_task([=]{uint32_t hash=2166136261u^uint32_t(id);for(size_t i=0;i<n;++i)hash=(hash^device[id*blob+i])*16777619u;hashes[id]=hash;});}));
                }
                stager.finish(); // No GPU synchronization between generations.
            }
        });
        // The first two jobs can be issued while the queue is held; job two
        // must not overwrite buffer zero before its first DMA has completed.
        const auto deadline=std::chrono::steady_clock::now()+2s;
        while(stager.issued.load(std::memory_order_acquire)<2 && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
        bool first_issued=stager.issued.load(std::memory_order_acquire)>=2;
        std::this_thread::sleep_for(50ms);
        bool protected_slot=first_issued && stager.ready[2].load(std::memory_order_acquire)==0;
        bool producer_pending=producer.wait_for(0s)!=std::future_status::ready;
        gate.open();producer.get();ck(cudaStreamSynchronize(copy));ck(cudaStreamSynchronize(compute));
        check(first_issued && protected_slot && producer_pending,"original stager reused a buffer before DMA completion");
        ck(cudaMemcpy(result.data(),device,result.size(),cudaMemcpyDeviceToHost));ck(cudaMemcpy(observed.data(),hashes,observed.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        check(observed==expected,"stager/compute event dependency changed checksum");
        for(size_t id=0;id<total_jobs;++id)for(size_t i=0;i<blob;++i)check(result[id*blob+i]==(i<sizes[id] ? source[id*blob+i] : uint8_t(0xbd)),"original stager payload or padding changed");
    }
    check(assembled.calls.load()==int(total_jobs/2),"original stager skipped source-copy jobs");
    ck(cudaStreamDestroy(compute));ck(cudaStreamDestroy(copy));ck(cudaFree(hashes));ck(cudaFree(device));
    std::cout<<"PASS original_stager_generations="<<generations<<" jobs="<<total_jobs<<" cpu_threads=3 pinned_ring=2 blob_stride="<<blob<<" exact_bytes="<<source.size()<<" gpu_checksums="<<observed.size()<<" held_dma_reuse_protected=1 generations_without_gpu_sync=1 transient_source_jobs=68\n";
} catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
