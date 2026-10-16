// Main-owned isolated product NativeHead lifecycle check, not a model benchmark.
// No environment mutation, graph capture or raw tensor/session files.
#include "strata/core/native_head.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
[[noreturn]] void stop(const char* why) noexcept {
    std::fprintf(stderr,"head component fail-stop: %s; owners remain live\n",why);
    std::fflush(stderr);std::_Exit(7);
}
void checked(cudaError_t status,const char* why) noexcept {
    if(status!=cudaSuccess)stop(why);
}
void need(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
uint64_t fingerprint(const void* data,size_t bytes) {
    const auto* p=static_cast<const uint8_t*>(data);
    uint64_t value=14695981039346656037ull;
    for(size_t i=0;i<bytes;++i){value^=p[i];value*=1099511628211ull;}
    return value; // diagnostic FNV1a64; exact memcmp is the acceptance check
}
struct Owners {
    strata::core::NativeHead head;
    cudaStream_t stream=nullptr;
    float* input=nullptr;
    float* logits=nullptr;
    void* filler=nullptr;
    std::vector<float> host_input,reference_logits,check_logits;
    std::vector<uint8_t> reference_weights,check_weights;
};
void logits(Owners& o) {
    std::string error;
    if(!o.head.run(o.input,o.logits,o.stream,error))stop(error.c_str());
    checked(cudaStreamSynchronize(o.stream),"head projection completion");
    checked(cudaMemcpy(o.check_logits.data(),o.logits,o.check_logits.size()*sizeof(float),
                       cudaMemcpyDeviceToHost),"logit readback");
    for(float v:o.check_logits)need(std::isfinite(v),"nonfinite projection output");
}
}

int main(int argc,char** argv) {
    // Heap ownership prevents stack unwinding of queued GPU/DMA owners on any
    // exceptional path. Only a fully drained success path deletes this object.
    Owners* o=nullptr;
    try {
        need(argc==3,"usage: head_fixture GGUF_SHARD_1 GGUF_SHARD_2");
        checked(cudaSetDevice(0),"select device");
        const auto runtime=strata::sycl_backend::runtime_for(0);
        const auto name=runtime->device().get_info<sycl::info::device::name>();
        need(name.find("B570")!=std::string::npos,"expected Arc B570");
        o=new Owners;
        checked(cudaStreamCreate(&o->stream),"create checked stream");
        std::string error;
        if(!o->head.load({argv[1],argv[2]},2560,248320,error))stop(error.c_str());
        need(o->head.type()==14 && o->head.weight_bytes()==521472000 && o->head.row_bytes()==2100,
             "actual unchanged Q6_K head geometry");
        const size_t bytes=size_t(o->head.weight_bytes());
        o->host_input.resize(2560);o->reference_logits.resize(248320);o->check_logits.resize(248320);
        o->reference_weights.resize(bytes);o->check_weights.resize(bytes);
        for(size_t i=0;i<2560;++i)o->host_input[i]=.001f*float(int(i*17%127)-63);
        checked(cudaMalloc(reinterpret_cast<void**>(&o->input),2560*sizeof(float)),"input allocation");
        checked(cudaMalloc(reinterpret_cast<void**>(&o->logits),248320*sizeof(float)),"logit allocation");
        checked(cudaMemcpy(o->input,o->host_input.data(),2560*sizeof(float),cudaMemcpyHostToDevice),"input upload");
        logits(*o);o->reference_logits=o->check_logits;
        checked(cudaMemcpy(o->reference_weights.data(),o->head.weights(),bytes,cudaMemcpyDeviceToHost),"reference packed bytes");
        need(o->head.prepare_prefill_weights(error),"prepare then cancel");
        need(!o->head.prepare_prefill_weights(error),"duplicate prepare rejected");
        o->head.cancel_prefill_weights();
        need(o->head.loaded() && !o->head.head_detached(),"cancel keeps original head");
        size_t total=0,free_before=0,free_detached=0,free_restored=0;
        uintptr_t old_address=0,new_address=0;
        for(int cycle=0;cycle<2;++cycle) {
            checked(cudaStreamSynchronize(o->stream),"retire sole projection reader");
            checked(cudaMemGetInfo(&free_before,&total),"memory before detach");
            old_address=reinterpret_cast<uintptr_t>(o->head.weights());
            need(o->head.prepare_prefill_weights(error),"prepare actual snapshot");
            o->head.detach_prefill_weights();
            need(!o->head.loaded() && o->head.head_detached() && o->head.weights()==nullptr,"detached owner guard");
            need(!o->head.run(o->input,o->logits,o->stream,error),"projection rejected while detached");
            checked(cudaMemGetInfo(&free_detached,&total),"memory after detach");
            // A bounded independent allocation exercises a different allocator
            // state. Address change is observed, never assumed or required.
            if(cycle==1)checked(cudaMalloc(&o->filler,64u<<20),"bounded detached filler");
            o->head.restore_prefill_weights();
            need(o->head.loaded() && o->head.head_detached(),"restore stays unreadable until resume");
            need(!o->head.run(o->input,o->logits,o->stream,error),"projection rejected before resume");
            o->head.resume_prefill_weights();
            new_address=reinterpret_cast<uintptr_t>(o->head.weights());
            if(o->filler){checked(cudaFree(o->filler),"retire filler");o->filler=nullptr;}
            checked(cudaMemcpy(o->check_weights.data(),o->head.weights(),bytes,cudaMemcpyDeviceToHost),"restored packed bytes");
            need(std::memcmp(o->reference_weights.data(),o->check_weights.data(),bytes)==0,"packed byte mismatch");
            logits(*o);
            need(std::memcmp(o->reference_logits.data(),o->check_logits.data(),248320*sizeof(float))==0,
                 "projection logit bits differ");
            checked(cudaMemGetInfo(&free_restored,&total),"memory after restored projection");
            std::printf("head_cycle %d packed_bytes=%zu weights_equal=1 logits_equal=1 free_before=%zu free_detached=%zu free_restored=%zu total=%zu address_changed=%d\n",
                cycle,bytes,free_before,free_detached,free_restored,total,int(old_address!=new_address));
            std::fflush(stdout);
        }
        const auto wh=fingerprint(o->reference_weights.data(),bytes);
        const auto lh=fingerprint(o->reference_logits.data(),248320*sizeof(float));
        checked(cudaStreamSynchronize(o->stream),"final reader drain");
        checked(cudaFree(o->input),"retire input");o->input=nullptr;
        checked(cudaFree(o->logits),"retire logits");o->logits=nullptr;
        checked(cudaStreamDestroy(o->stream),"retire sole stream");o->stream=nullptr;
        delete o;o=nullptr; // NativeHead checked teardown runs only after drain
        std::printf("head_component PASS cycles=2 weights_fnv1a64=%016llx logits_fnv1a64=%016llx\n",
            static_cast<unsigned long long>(wh),static_cast<unsigned long long>(lh));
        return 0;
    } catch(const std::exception& e) {
        // o and all referenced host/device storage remain allocated until exit.
        std::fprintf(stderr,"head component exception: %s; owners retained\n",e.what());
        std::fflush(stderr);std::_Exit(8);
    } catch(...) {stop("unknown exception");}
}
