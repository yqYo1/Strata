#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <atomic>
#include <iostream>
namespace g=sycl::ext::oneapi::experimental;
int main() try {
 sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
 auto* p=sycl::malloc_shared<int>(1,q);auto* host=sycl::malloc_host<int>(1,q);*p=-7;*host=-11;std::atomic<int> calls{0};auto* called=&calls;
 g::command_graph graph(q.get_context(),q.get_device());graph.begin_recording(q);
 q.memset(p,0,4);q.single_task([=]{*p=43;});q.memcpy(host,p,4);q.submit([&](sycl::handler& h){h.host_task([called]{called->fetch_add(1);});});graph.end_recording();
 std::cerr<<"PROBE_BEFORE_FINALIZE nodes="<<graph.get_nodes().size()<<" types=";for(auto n:graph.get_nodes())std::cerr<<int(n.get_type())<<',';std::cerr<<'\n';
 auto executable=graph.finalize();std::cerr<<"PROBE_AFTER_FINALIZE value="<<*p<<" copied="<<*host<<" callbacks="<<calls.load()<<'\n';
 if(*p!=-7 || *host!=-11 || calls.load()!=0)return 2;
 q.single_task<class PrepareMarker>([]{}).wait_and_throw();std::cerr<<"PROBE_BEFORE_FIRST_LAUNCH\n";
 q.submit([&](sycl::handler& h){h.ext_oneapi_graph(executable);}).wait_and_throw();std::cerr<<"PROBE_AFTER_FIRST_LAUNCH value="<<*p<<" copied="<<*host<<" callbacks="<<calls.load()<<'\n';
 bool ok=*p==43 && *host==43 && calls.load()==1;sycl::free(host,q);sycl::free(p,q);return ok ? 0 : 3;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
