#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/profiling_tag.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <future>
#include <thread>
#include <iostream>
using namespace std::chrono_literals;
namespace ex=sycl::ext::oneapi::experimental;
int main() try {
 sycl::queue q(sycl::gpu_selector_v,sycl::property::queue::in_order{});
 std::cout<<q.get_device().get_info<sycl::info::device::name>()<<" profiling_tag="<<q.get_device().has(sycl::aspect::ext_oneapi_queue_profiling_tag)<<" queue_profiling="<<q.has_property<sycl::property::queue::enable_profiling>()<<std::endl;
 auto a=ex::submit_profiling_tag(q);a.wait_and_throw();
 std::promise<void> release;auto gate=release.get_future().share();
 q.submit([&](sycl::handler& h){h.host_task([gate]{gate.wait();});});
 auto pending=std::async(std::launch::async,[&]{return ex::submit_profiling_tag(q);});
 bool ready=pending.wait_for(2s)==std::future_status::ready;
 std::this_thread::sleep_for(50ms);release.set_value();auto b=pending.get();b.wait_and_throw();
 auto as=a.get_profiling_info<sycl::info::event_profiling::command_start>();
 auto ae=a.get_profiling_info<sycl::info::event_profiling::command_end>();
 auto bs=b.get_profiling_info<sycl::info::event_profiling::command_start>();
 auto be=b.get_profiling_info<sycl::info::event_profiling::command_end>();
 std::cout<<"as="<<as<<" ae="<<ae<<" bs="<<bs<<" be="<<be<<" gap_ms="<<double(be-ae)/1e6<<" submit_ready="<<ready<<std::endl;
 ex::command_graph<ex::graph_state::modifiable> g(q.get_context(),q.get_device());
 g.begin_recording(q);
 try {auto c=ex::submit_profiling_tag(q);std::cout<<"capture_submission=accepted"<<std::endl;g.end_recording();auto e=g.finalize();std::cout<<"capture_finalize=accepted"<<std::endl;}
 catch(const std::exception& e){g.end_recording();std::cout<<"capture_error="<<e.what()<<std::endl;}
 return ready?0:2;
} catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
