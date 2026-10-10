#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <oneapi/mkl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <algorithm>

// Inclusive steady_clock wall with wait_and_throw: no GPU/host clock joining.
// All allocations are raw pointers; failures exit without unwinding live USM.
using Clock=std::chrono::steady_clock;
constexpr size_t bytes=256ull<<20, words=bytes/4;
constexpr int samples=7;
static uint32_t pattern(uint32_t x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
[[noreturn]]static void fail(const char* why){std::fprintf(stderr,"FAIL %s\n",why);std::fflush(nullptr);std::_Exit(70);}
static void need(bool ok,const char* why){if(!ok)fail(why);}
template<class F>static double span(sycl::queue* q,F f,int calls){
    q->wait_and_throw();const auto t0=Clock::now();
    for(int i=0;i<calls;++i)f(i);
    q->wait_and_throw();return std::chrono::duration<double>(Clock::now()-t0).count();
}
static void sample(const char* name,int id,double seconds,uint64_t payload,int calls){
    std::printf("{\"kind\":\"sample\",\"case\":\"%s\",\"sample\":%d,\"seconds\":%.9f,\"calls\":%d,\"logical_bytes\":%llu,\"logical_GBps\":%.6f}\n",name,id,seconds,calls,(unsigned long long)payload,payload/seconds/1e9);
}
int main(int argc,char**argv){
    setvbuf(stdout,nullptr,_IOLBF,0);
    need(argc==2 && (!std::strcmp(argv[1],"--qualify")||!std::strcmp(argv[1],"--measure")),"explicit_mode_required");
    const bool qualify=!std::strcmp(argv[1],"--qualify");
    try {
        auto* q=new sycl::queue(sycl::gpu_selector_v,[](sycl::exception_list e){for(auto x:e){try{std::rethrow_exception(x);}catch(const std::exception&v){std::fprintf(stderr,"async %s\n",v.what());}}fail("asynchronous_error");},sycl::property::queue::in_order{});
        need(q->get_backend()==sycl::backend::ext_oneapi_level_zero,"LevelZero_required");
        need(q->get_device().has(sycl::aspect::fp16)&&q->get_device().has(sycl::aspect::usm_device_allocations)&&q->get_device().has(sycl::aspect::usm_host_allocations),"required_aspects");
        auto dev=sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q->get_device());
        ze_device_properties_t prop{};prop.stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        need(zeDeviceGetProperties(dev,&prop)==ZE_RESULT_SUCCESS,"device_query");
        need(prop.vendorId==0x8086 && prop.deviceId==0xe20c && !(prop.flags&ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE),"B570_root_required");
        ze_pci_ext_properties_t pci{};pci.stype=ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        need(zeDevicePciGetPropertiesExt(dev,&pci)==ZE_RESULT_SUCCESS,"PCI_query");
        need(pci.address.domain==0&&pci.address.bus==5&&pci.address.device==0&&pci.address.function==0,"PCI_05_00_0_required");
        std::printf("{\"kind\":\"configuration\",\"device\":\"B570\",\"pci\":\"0000:05:00.0\",\"backend\":\"level_zero\",\"in_order\":true,\"profiling\":false,\"device_global_mem_bytes\":%llu,\"array_bytes\":%zu,\"mode\":\"%s\"}\n",(unsigned long long)q->get_device().get_info<sycl::info::device::global_mem_size>(),bytes,qualify?"qualify":"measure");
        uint32_t* pageable=(uint32_t*)std::malloc(bytes),*readback=(uint32_t*)std::malloc(bytes);
        auto* pinned=sycl::malloc_host<uint32_t>(words,*q);
        auto* a=sycl::malloc_device<uint32_t>(words,*q);auto* b=sycl::malloc_device<uint32_t>(words,*q);auto* c=sycl::malloc_device<uint32_t>(words,*q);
        need(pageable&&readback&&pinned&&a&&b&&c,"allocation");
        for(size_t i=0;i<words;++i)pageable[i]=pinned[i]=pattern((uint32_t)i+1);
        q->memcpy(a,pinned,bytes);
        q->parallel_for(sycl::range<1>(words),[=](sycl::id<1>i){b[i]=pattern((uint32_t)i+1+(uint32_t)words);});q->wait_and_throw();
        const int repetitions=qualify?1:samples;
        for(bool pin:{false,true})for(bool h2d:{true,false})for(size_t n:{1ull<<20,8ull<<20,64ull<<20,256ull<<20}){
            uint32_t* h=pin?pinned:pageable;
            const char* name=pin?(h2d?"H2D_host_USM":"D2H_host_USM"):(h2d?"H2D_pageable":"D2H_pageable");
            auto submit=[&](int){q->memcpy(h2d?(void*)a:(void*)h,h2d?(void*)h:(void*)a,n);};
            span(q,submit,1);const int calls=qualify?1:4;
            for(int r=0;r<repetitions;++r){double seconds=span(q,submit,calls);sample(name,r,seconds,n*calls,calls);}
            q->memcpy(readback,a,n).wait_and_throw();
            for(size_t i=0;i<n/4;++i)need(readback[i]==pattern((uint32_t)i+1),"H2D_full_validation");
            if(!h2d)for(size_t i=0;i<n/4;++i)need(h[i]==pattern((uint32_t)i+1),"D2H_full_validation");
            std::printf("{\"kind\":\"validation\",\"case\":\"%s\",\"payload_bytes_per_call\":%zu,\"passed\":true}\n",name,n);
        }
        constexpr size_t global=262144, local=256;
        auto* partial=sycl::malloc_device<uint32_t>(global/local,*q);
        auto* partial_h=(uint32_t*)std::malloc(global/local*4);need(partial&&partial_h,"partial_alloc");
        for(int op=0;op<4;++op){
            const char* name=op==0?"VRAM_queue_copy":op==1?"VRAM_kernel_copy":op==2?"VRAM_read_xor":"VRAM_triad_xor";
            auto submit=[&](int){
                if(op==0){q->memcpy(c,a,bytes);return;}
                if(op==2){q->parallel_for(sycl::nd_range<1>(global,local),[=](sycl::nd_item<1>it){
                    uint32_t v=0;for(size_t i=it.get_global_linear_id();i<words;i+=global)v^=a[i];
                    const auto out=sycl::reduce_over_group(it.get_group(),v,sycl::bit_xor<uint32_t>());
                    if(it.get_local_linear_id()==0)partial[it.get_group_linear_id()]=out;
                });return;}
                q->parallel_for(sycl::range<1>(global),[=](sycl::id<1>id){for(size_t i=id[0];i<words;i+=global)c[i]=op==1?a[i]:(a[i]^b[i]^0x7136a5c9u);});
            };
            span(q,submit,1);const int calls=qualify?1:16;
            const uint64_t traffic=bytes*(uint64_t)calls*(op==2?1:op==3?3:2);
            for(int r=0;r<repetitions;++r)sample(name,r,span(q,submit,calls),traffic,calls);
            if(op==2){q->memcpy(partial_h,partial,global/local*4).wait_and_throw();uint32_t expected=0,got=0;
                for(size_t i=0;i<words;++i)expected^=pattern((uint32_t)i+1);
                for(size_t i=0;i<global/local;++i)got^=partial_h[i];need(got==expected,"read_checksum");
            }else{q->memcpy(readback,c,bytes).wait_and_throw();for(size_t i=0;i<words;++i)need(readback[i]==(op==3?(pattern((uint32_t)i+1)^pattern((uint32_t)i+1+(uint32_t)words)^0x7136a5c9u):pattern((uint32_t)i+1)),"VRAM_full_validation");}
            std::printf("{\"kind\":\"validation\",\"case\":\"%s\",\"passed\":true}\n",name);
        }
        // Same column-major/transposed weight convention and data types as
        // production native GU/Down. Hot/rotating weights are distinct cases.
        constexpr int64_t maxT=8192, N=2560, maxW=N*1280, rotations=8;
        auto* W=sycl::malloc_device<sycl::half>(maxW*rotations,*q);
        auto* X=sycl::malloc_device<sycl::half>(maxT*N,*q);
        auto* Y=sycl::malloc_device<float>(maxT*N,*q);
        auto* Yh=(float*)std::malloc(maxT*N*sizeof(float));need(W&&X&&Y&&Yh,"gemm_alloc");
        q->parallel_for(sycl::range<1>(maxW*rotations),[=](sycl::id<1>i){W[i]=sycl::half((pattern((uint32_t)i+17)&1)?0.125f:-0.125f);});
        q->parallel_for(sycl::range<1>(maxT*N),[=](sycl::id<1>i){X[i]=sycl::half(1.0f);});q->wait_and_throw();
        for(bool down:{false,true})for(int64_t T:{1,4,8,32,80,160,320,640,1280,8192})for(bool rotate:{false,true}){
            const int64_t n=down?N:1280,k=down?640:N;
            auto submit=[&](int i){oneapi::mkl::blas::column_major::gemm(*q,oneapi::mkl::transpose::trans,oneapi::mkl::transpose::nontrans,n,T,k,1.0f,W+(rotate?(i%rotations)*maxW:0),k,X,k,0.0f,Y,n);};
            span(q,submit,rotate?rotations:2);const int calls=qualify?1:T>=640?16:64;
            for(int r=0;r<repetitions;++r){const double seconds=span(q,submit,calls);const double ops=2.0*n*T*k*calls;
                std::printf("{\"kind\":\"gemm_sample\",\"case\":\"%s\",\"weight_mode\":\"%s\",\"M\":%lld,\"N\":%lld,\"K\":%lld,\"sample\":%d,\"calls\":%d,\"seconds\":%.9f,\"TFLOPps\":%.6f}\n",down?"Down":"GU",rotate?"rotate8":"hot",(long long)T,(long long)n,(long long)k,r,calls,seconds,ops/seconds/1e12);}
            q->memcpy(Yh,Y,T*n*sizeof(float)).wait_and_throw();
            const int last_weight=rotate?(calls-1)%rotations:0;
            for(int64_t j=0;j<n;++j){int sum=0;for(int64_t kk=0;kk<k;++kk)sum+=(pattern((uint32_t)(last_weight*maxW+j*k+kk)+17)&1)?1:-1;
                for(int64_t t=0;t<T;++t)need(Yh[t*n+j]==sum*0.125f,"GEMM_full_exact_validation");}
            std::printf("{\"kind\":\"gemm_validation\",\"case\":\"%s\",\"M\":%lld,\"weight_mode\":\"%s\",\"passed\":true}\n",down?"Down":"GU",(long long)T,rotate?"rotate8":"hot");
        }
        q->wait_and_throw();for(void*p:{(void*)pinned,(void*)a,(void*)b,(void*)c,(void*)partial,(void*)W,(void*)X,(void*)Y})sycl::free(p,*q);
        std::free(pageable);std::free(readback);std::free(partial_h);std::free(Yh);delete q;return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"synchronous %s\n",e.what());fail("synchronous_error");}
    catch(...){fail("unknown_error");}
}
