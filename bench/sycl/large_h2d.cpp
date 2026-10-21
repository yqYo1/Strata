// Authored for Strata; repository LICENSE (MIT) applies.
// Host-source/copy-size diagnostic, using production compatibility entrypoints.
#include <cuda_runtime.h>
#include "strata/sycl/cpu_wait_profile.hpp"
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>
#include <utility>
#include <cstdint>
#include <exception>
namespace hp = strata::sycl_backend::cpu_wait_profile;
[[noreturn]] void stop(const char* why) {
    std::fprintf(stderr,"large_h2d failed: %s; stopping with owners live\n",why);
    std::fflush(stderr);std::abort();
}
void gpu(cudaError_t status) {if(status!=cudaSuccess)stop(cudaGetErrorString(status));}
uint64_t number(const char* text) {
    uint64_t value=0;const char* end=text+std::strlen(text);
    auto p=std::from_chars(text,end,value);
    if(p.ec!=std::errc{} || p.ptr!=end)stop("invalid numeric argument");
    return value;
}
void stamp_ok(hp::Stamp s) {if(!s.valid)stop("thread CPU clock unavailable");}
int main(int argc,char** argv) try {
    if(argc!=5)stop("usage: large_h2d SOURCE CHUNK_BYTES SAMPLES BYTES; SOURCE=vector|host-usm|stage");
    const std::string_view source=argv[1];
    if(source!="vector" && source!="host-usm" && source!="stage")stop("unknown source kind");
    const uint64_t requested_chunk=number(argv[2]),samples=number(argv[3]),bytes=number(argv[4]);
    constexpr size_t guard=65536;
    if(bytes<(1<<20) || bytes>(2ULL<<30) || !samples || samples>6 ||
        (requested_chunk && (requested_chunk<(1<<20) || requested_chunk>bytes)))stop("bounded extent/sample limit");
    const size_t n=size_t(bytes),chunk=requested_chunk?size_t(requested_chunk):n;
    const auto setup_start=hp::stamp();stamp_ok(setup_start);
    // Allocate host reference/readback before creating any GPU owner. Source
    // remains immutable through each final queue drain, including staging.
    std::vector<unsigned char> reference(n),readback(n+2*guard);
    for(size_t i=0;i<n;++i)reference[i]=static_cast<unsigned char>((i*131u)^(i>>11)^(i>>23));
    cudaStream_t stream=nullptr;gpu(cudaStreamCreate(&stream));
    size_t free_before=0,total=0;gpu(cudaMemGetInfo(&free_before,&total));
    if(free_before<n+2*guard+(256ULL<<20))stop("insufficient unpressured device headroom");
    unsigned char* destination=nullptr;gpu(cudaMalloc(&destination,n+2*guard));
    unsigned char* host_usm=nullptr;
    if(source!="vector") {gpu(cudaMallocHost(&host_usm,n));
        if(source=="host-usm")std::memcpy(host_usm,reference.data(),n);}
    const auto setup_end=hp::stamp();stamp_ok(setup_end);
    double timer_max=0;
    for(int i=0;i<1000;++i){const auto a=hp::stamp(),b=hp::stamp();stamp_ok(a);stamp_ok(b);
        if(!hp::duration(b.wall-a.wall))stop("timer moved backward");timer_max=std::max(timer_max,b.wall-a.wall);}
    std::printf("{\"kind\":\"identity\",\"source\":\"%s\",\"bytes\":%llu,\"chunk\":%llu,\"samples\":%llu,"
        "\"free_before\":%zu,\"total\":%zu,\"setup_wall_s\":%.9f,\"setup_cpu_s\":%.9f,\"timer_max_s\":%.9f,"
        "\"unpressured\":true,\"source_mod_65536\":%zu,\"destination_mod_65536\":%zu,\"physical_bus_traffic_measured\":false,\"GPU_service_measured\":false}\n",
        argv[1],(unsigned long long)bytes,(unsigned long long)requested_chunk,(unsigned long long)samples,
        free_before,total,setup_end.wall-setup_start.wall,setup_end.cpu-setup_start.cpu,timer_max,
        size_t(reinterpret_cast<uintptr_t>(source=="vector"?reference.data():host_usm)%65536),
        size_t(reinterpret_cast<uintptr_t>(destination+guard)%65536));
    std::fflush(stdout);
    for(uint64_t sample=0;sample<samples;++sample) {
        // Reset/checks are outside timing. A full readback follows EVERY sample,
        // so a missing chunk cannot pass because an earlier copy filled it.
        gpu(cudaMemsetAsync(destination,0xa5,n+2*guard,stream));gpu(cudaStreamSynchronize(stream));
        const auto begin=hp::stamp();
        if(source=="stage")std::memcpy(host_usm,reference.data(),n);
        const auto staged=hp::stamp();
        const auto* src=source=="vector"?reference.data():host_usm;
        uint64_t copies=0;
        for(size_t at=0;at<n;) {
            const size_t count=std::min(chunk,n-at);
            gpu(cudaMemcpyAsync(destination+guard+at,src+at,count,cudaMemcpyHostToDevice,stream));
            at+=count;++copies;
        }
        const auto submitted=hp::stamp();gpu(cudaStreamSynchronize(stream));const auto drained=hp::stamp();
        for(auto s:{begin,staged,submitted,drained})stamp_ok(s);
        for(auto pair:{std::pair{begin,staged},std::pair{staged,submitted},std::pair{submitted,drained}})
            if(!hp::duration(pair.second.wall-pair.first.wall)||!hp::duration(pair.second.cpu-pair.first.cpu))stop("invalid phase clock");
        gpu(cudaMemcpy(readback.data(),destination,n+2*guard,cudaMemcpyDeviceToHost));
        if(!std::all_of(readback.begin(),readback.begin()+guard,[](auto x){return x==0xa5;}) ||
            !std::all_of(readback.end()-guard,readback.end(),[](auto x){return x==0xa5;}) ||
            std::memcmp(readback.data()+guard,reference.data(),n))stop("guard/payload mismatch");
        std::printf("{\"kind\":\"sample\",\"index\":%llu,\"first_copy\":%s,\"copies\":%llu,\"logical_bytes\":%llu,"
            "\"wall_s\":%.9f,\"cpu_s\":%.9f,\"stage_wall_s\":%.9f,\"stage_cpu_s\":%.9f,"
            "\"submit_wall_s\":%.9f,\"submit_cpu_s\":%.9f,\"drain_wall_s\":%.9f,\"drain_cpu_s\":%.9f,"
            "\"payload_match\":true,\"guards_match\":true}\n",(unsigned long long)sample,sample?"false":"true",
            (unsigned long long)copies,(unsigned long long)bytes,drained.wall-begin.wall,drained.cpu-begin.cpu,
            staged.wall-begin.wall,staged.cpu-begin.cpu,submitted.wall-staged.wall,submitted.cpu-staged.cpu,
            drained.wall-submitted.wall,drained.cpu-submitted.cpu);
        std::fflush(stdout);
    }
    gpu(cudaStreamSynchronize(stream));if(host_usm)gpu(cudaFreeHost(host_usm));gpu(cudaFree(destination));gpu(cudaStreamDestroy(stream));
    std::puts("{\"kind\":\"closed\",\"passed\":true,\"owners_retired\":true}");return 0;
} catch(const std::exception& e) {stop(e.what());}
