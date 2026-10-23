// Main-authored bounded native tail/step readback; repository LICENSE applies.
#include <cuda_runtime.h>
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/prefill/locality_capture.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok,const char* text) { if (!ok) throw std::runtime_error(text); }
void check(cudaError_t status) { require(status==cudaSuccess,"locality fixture runtime operation"); }
struct Owners {
    cudaStream_t stream{};int32_t* device{};int32_t guards[4]{};
    ~Owners() noexcept {
        if (stream && cudaStreamSynchronize(stream)!=cudaSuccess)
            strata::sycl_backend::cleanup_fail_stop("locality fixture completion uncertain");
        if (device && cudaFree(device)!=cudaSuccess)
            strata::sycl_backend::cleanup_fail_stop("locality fixture free failed");
        if (stream && cudaStreamDestroy(stream)!=cudaSuccess)
            strata::sycl_backend::cleanup_fail_stop("locality fixture stream retirement failed");
    }
};
void verify(const std::string& path,int rows,int cap,int source_rows) {
    std::ifstream f(path,std::ios::binary);int32_t h[4]{};
    f.read(reinterpret_cast<char*>(h),sizeof(h));
    require(f.good() && h[0]==2 && h[1]==10000+source_rows-rows && h[2]==rows && h[3]==cap,"tail header");
    std::vector<int32_t> v(size_t(rows)*cap);
    f.read(reinterpret_cast<char*>(v.data()),std::streamsize(v.size()*sizeof(int32_t)));
    require(f.good() && f.peek()==std::char_traits<char>::eof(),"tail file extent");
    for (size_t i=0;i<v.size();++i)
        require(v[i]==int32_t(((size_t(source_rows-rows)*cap+i)*17+19)%262144),"tail payload/offset");
}
}
int main(int argc,char** argv) {
    try {
        require(argc==3,"use small|full PRIVATE_BASE");
        const std::string mode=argv[1],base=argv[2];
        require(mode=="small" || mode=="full","fixture mode");
        require(!std::ifstream(base+".ids").good() && !std::ifstream(base+".steps").good(),"fresh private files required");
        const int rows=mode=="small" ? 17 : 8192,cap=mode=="small" ? 33 : 2051,step_count=7;
        const size_t ids=size_t(rows)*cap,steps=size_t(rows)*step_count,count=ids+steps+4;
        constexpr int32_t marker=-1234567;Owners owned;
        check(cudaStreamCreate(&owned.stream));
        check(cudaMalloc(reinterpret_cast<void**>(&owned.device),count*sizeof(int32_t)));
        auto* device=owned.device;
        owned.stream->parallel_for(sycl::range<1>(count),[=](sycl::id<1> index) {
            const size_t i=index[0];
            if (i==0 || i==ids+1 || i==ids+2 || i==count-1) device[i]=marker;
            else {
                const size_t value=i<=ids ? i-1 : i-(ids+3);
                device[i]=int32_t((value*17+19)%262144);
            }
        });
        size_t copies=0,drains=0;std::string error;
        auto copy=[&](void* dst,const void* src,size_t bytes) {
            ++copies;return cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToHost,owned.stream)==cudaSuccess;
        };
        auto drain=[&] { ++drains;return cudaStreamSynchronize(owned.stream)==cudaSuccess; };
        auto fatal=[]() noexcept { strata::sycl_backend::cleanup_fail_stop("locality readback uncertain"); };
        const bool captured=strata::prefill::detail::locality_capture(base.c_str(),2,10000,rows,rows,cap,step_count,
            device+1,device+ids+3,copy,drain,fatal,error);
        require(captured,error.c_str());
        require(copies==2 && drains==2,"two checked copy/drain pairs");
        require(!strata::prefill::detail::locality_capture(base.c_str(),2,10000,rows,rows-1,cap,step_count,
            device+1,device+ids+3,copy,drain,fatal,error),"invalid geometry accepted");
        require(copies==2 && drains==3,"refusal must not copy and must drain inherited work");
        const size_t offsets[]={0,ids+1,ids+2,count-1};
        for (int i=0;i<4;++i)
            check(cudaMemcpyAsync(&owned.guards[i],device+offsets[i],sizeof(int32_t),cudaMemcpyDeviceToHost,owned.stream));
        check(cudaStreamSynchronize(owned.stream));
        for (auto v:owned.guards) require(v==marker,"source guards");
        const int take=std::min(512,rows);
        verify(base+".ids",take,cap,rows);verify(base+".steps",take,step_count,rows);
        std::cout<<"locality_capture mode="<<mode<<" source_rows="<<rows<<" captured_rows="<<take
            <<" cap="<<cap<<" exact=PASS guards=PASS invalid_geometry=PASS inherited_drain=PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<"locality fixture: "<<e.what()<<'\n';return 1; }
}
