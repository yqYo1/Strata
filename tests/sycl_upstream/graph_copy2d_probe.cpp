// Opt-in reproducer for the two unavailable rectangular graph-copy paths.
// Run the native variant under an external timeout: affected UR V2 builds hang.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <cstring>
#include <iostream>
namespace exp=sycl::ext::oneapi::experimental;
int main(int argc,char** argv) try {
    constexpr size_t rows=7,width=13,sp=19,dp=31;
    const bool native=argc==2 && !std::strcmp(argv[1],"native");
    sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
    auto* src=sycl::malloc_host<unsigned char>(rows*sp,q);
    auto* dst=sycl::malloc_device<unsigned char>(rows*dp,q);
    for(size_t i=0;i<rows*sp;++i)src[i]=unsigned(i);
    q.memset(dst,0xcd,rows*dp).wait_and_throw();
    exp::command_graph graph(q.get_context(),q.get_device());graph.begin_recording(q);
    try {
        if(native) q.submit([=](sycl::handler& h){
            h.ext_codeplay_enqueue_native_command([=](sycl::interop_handle interop){
                std::cout<<"native_handle_query=begin"<<std::endl;
                auto list=interop.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_level_zero>();
                std::cout<<"native_handle_query=returned"<<std::endl;
                const ze_copy_region_t region{0,0,0,width,rows,0};
                auto result=zeCommandListAppendMemoryCopyRegion(list,dst,&region,dp,0,src,&region,sp,0,nullptr,0,nullptr);
                if(result!=ZE_RESULT_SUCCESS)throw std::runtime_error("native append failed");
            });
        });
        else q.ext_oneapi_memcpy2d(dst,dp,src,sp,width,rows);
    } catch(const std::exception& e) {
        graph.end_recording();sycl::free(dst,q);sycl::free(src,q);
        std::cout<<"recording_failed="<<e.what()<<'\n';return 1;
    }
    graph.end_recording();std::cout<<"finalize=begin native="<<native<<std::endl;
    auto executable=graph.finalize();std::cout<<"finalize=returned"<<std::endl;
    q.ext_oneapi_graph(executable).wait_and_throw();
    unsigned char out[rows*dp];q.memcpy(out,dst,sizeof out).wait_and_throw();
    bool valid=true;for(size_t r=0;r<rows;++r)for(size_t c=0;c<dp;++c)valid&=out[r*dp+c]==(c<width?src[r*sp+c]:0xcd);
    sycl::free(dst,q);sycl::free(src,q);
    std::cout<<"rectangular_copy_values="<<valid<<'\n';return valid?0:1;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
