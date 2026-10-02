#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <optional>
namespace ex=sycl::ext::oneapi::experimental;
int main(){try{
 sycl::device d(sycl::gpu_selector_v);sycl::context c(d);
 for(int prof=0;prof<2;++prof){
 sycl::queue q(c,d,prof?sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}:sycl::property_list{sycl::property::queue::in_order{}});
 auto* p=sycl::malloc_device<float>(4096,q);auto* host=sycl::malloc_host<float>(4096,q);
 for(int native=0;native<2;++native){
 ex::command_graph g(c,d,native?sycl::property_list{ex::property::graph::enable_native_recording{}}:sycl::property_list{});
 auto start=std::chrono::steady_clock::now();g.begin_recording(q);q.memcpy(p,host,4096*4);for(int i=0;i<32;++i)q.parallel_for(sycl::range<1>(4096),[=](sycl::id<1> i){p[i]+=1.f;});q.ext_oneapi_submit_barrier();q.memcpy(host,p,4096*4);g.end_recording(q);auto exec=g.finalize();double capture=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 for(int run=0;run<12;++run){std::fill(host,host+4096,float(run*1024));auto e=q.ext_oneapi_graph(exec);e.wait_and_throw();for(int i=0;i<4096;++i)if(host[i]!=float(run*1024+32))throw std::runtime_error("changing host payload failure");}
 for(int dep=0;dep<2;++dep){std::optional<sycl::event> last;std::fill(host,host+4096,0.f);start=std::chrono::steady_clock::now();for(int i=0;i<300;++i){last=q.submit([&](sycl::handler& h){if(dep&&last)h.depends_on(*last);h.ext_oneapi_graph(exec);});}q.wait_and_throw();for(int i=0;i<4096;++i)if(host[i]!=9600.f)throw std::runtime_error("many replay failure");std::cout<<"profiling="<<prof<<" native="<<native<<" dependency="<<dep<<" capture_ms="<<capture<<" replay_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/300<<" changing_usm=PASS\n";}
 }
 sycl::free(p,q);sycl::free(host,q);
 }
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<'\n';return 1;}}
