// Original Strata device fixture; repository LICENSE applies.
#include <cuda_runtime.h>
#include "strata/core/canonical_restore_copy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

static void need(bool value) {
    if(!value) {std::fputs("canonical restore copy fixture failed; owners retained\n",stderr);std::abort();}
}
static void gpu(cudaError_t value) {need(value==cudaSuccess);}
int main(int argc,char** argv) {
    need(argc==1 || (argc==2 && std::string_view(argv[1])=="--small"));
    constexpr size_t guard=65536;
    const size_t n=argc==2 ? (2ULL<<20) : (64ULL<<20)+17;
    // Construct both host buffers before creating any GPU owner.
    std::vector<unsigned char> reference(n),readback(n+2*guard);
    for(size_t i=0;i<n;++i)reference[i]=static_cast<unsigned char>((i*131u)^(i>>11));
    cudaStream_t queue=nullptr;gpu(cudaStreamCreate(&queue));
    unsigned char* destination=nullptr;gpu(cudaMalloc(&destination,n+2*guard));
    for(bool chunked:{false,true}) {
        gpu(cudaMemsetAsync(destination,0xa5,n+2*guard,queue));gpu(cudaStreamSynchronize(queue));
        size_t copies=99;
        need(strata::core::detail::submit_canonical_restore(n,chunked,
            [&](size_t at,size_t bytes) {
                need(at<=n && bytes && bytes<=n-at);
                gpu(cudaMemcpyAsync(destination+guard+at,reference.data()+at,bytes,cudaMemcpyHostToDevice,queue));
                return true;
            },copies));
        // This is the same single final drain as the production handoff.
        gpu(cudaStreamSynchronize(queue));
        gpu(cudaMemcpy(readback.data(),destination,n+2*guard,cudaMemcpyDeviceToHost));
        need(std::all_of(readback.begin(),readback.begin()+guard,[](auto x){return x==0xa5;}) &&
             std::all_of(readback.end()-guard,readback.end(),[](auto x){return x==0xa5;}) &&
             !std::memcmp(readback.data()+guard,reference.data(),n));
        need(copies==(chunked && argc==1?2u:1u));
        std::printf("canonical restore copy device: chunk64=%d bytes=%zu copies=%zu payload=1 guards=1\n",int(chunked),n,copies);
        std::fflush(stdout);
    }
    gpu(cudaStreamSynchronize(queue));gpu(cudaFree(destination));gpu(cudaStreamDestroy(queue));
    std::puts("canonical restore copy device: cases=2 passed=1 owners_retired=1");
}
