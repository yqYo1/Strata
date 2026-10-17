// Authored for Strata; repository LICENSE (MIT) applies.
// Bounded profiling diagnostics: ordinary-host copies and a queued kernel burst.
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <sys/timex.h>
namespace {
using namespace strata::sycl_backend;
std::shared_ptr<Runtime> runtime;
void need(bool ok,const char* text) { if (!ok) throw std::runtime_error(text); }
void drain() noexcept {
  checked_cleanup([] { if (runtime) runtime->wait(); },[] {},"event stress drain");
}
void gate(const char* phase) {
  const char* value=std::getenv("STRATA_TEST_MAIN_GATES");
  if (!value || std::string(value)!="1") return;
  std::cout<<"EVENT_PROFILE_READY "<<phase<<'\n';need(std::cin.get()=='X',"Main gate token");
}
struct Saved {
  std::optional<sycl::event> event;
  const char* kind=nullptr;
  int job=-1;
};
void run(bool aged) {
  constexpr size_t Plane=163840,Guard=16,Count=2*Plane+2*Guard,Events=141;
  constexpr uint32_t Sentinel=0x1942ab57;
  auto& q=runtime->compute();
  need(q.has_property<sycl::property::queue::in_order>() &&
       q.has_property<sycl::property::queue::enable_profiling>(),"profiled in-order queue");
  need(q.get_backend()==sycl::backend::ext_oneapi_level_zero && q.get_device().is_gpu(),"Level Zero GPU");
  const auto resolution=q.get_device().get_info<sycl::info::device::profiling_timer_resolution>();
  std::vector<uint32_t> source(Count),result(Count);
  for (size_t i=0;i<Count;++i) source[i]=i<Guard || i>=Count-Guard ? Sentinel : uint32_t(i*1664525U+1013904223U);
  const auto original=source;
  Allocation device(runtime,Count*4,MemoryKind::Device);
  auto* base=device.as<uint32_t>();auto* data=base+Guard;
  std::array<Saved,Events> saved;
  std::array<uint32_t,64> additions{};
  size_t used=0,kernels=0;
  bool retired=false;
  struct Pending { bool& done;~Pending() noexcept { if (!done) drain(); } } pending{retired};
  auto keep=[&](const char* kind,int job,sycl::event event) {
    need(used<Events,"event budget");saved[used].kind=kind;saved[used].job=job;
    saved[used++].event.emplace(std::move(event));
  };
  runtime->wait(q.memcpy(base,source.data(),Count*4));
  // modes=0 reads the kernel clock adjustment; this never changes a clock.
  timex adjustment{};
  const int clock_status=adjtimex(&adjustment);
  std::cout<<"host_clock_query modes=0 status="<<clock_status
           <<" freq_scaled_ppm="<<adjustment.freq<<" freq_ppm="<<double(adjustment.freq)/65536.0<<'\n';
  if (aged) std::this_thread::sleep_for(std::chrono::seconds(20));
  keep("H2D_ordinary",-1,q.memcpy(data,source.data()+Guard,Plane*4));
  keep("H2D_ordinary",-1,q.memcpy(data+Plane,source.data()+Guard+Plane,Plane*4));
  constexpr const char* names[]={"dequant_gu","dequant_down","gather","gu","swiglu","down"};
  for (int job=0;job<512;++job) for (int kind=0;kind<6;++kind) {
    const size_t word=(size_t(job)+size_t(kind))%64;
    auto event=q.single_task([=] { data[word]+=1; });
    ++additions[word];++kernels;
    if ((kind<2 && job<4) || (kind>=2 && job%16==0)) keep(names[kind],job,std::move(event));
  }
  for (int i=0;i<3;++i) {
    const size_t word=size_t(i);
    auto event=q.single_task([=] { data[word]+=1; });
    ++additions[word];++kernels;
    keep(i==0 ? "combine" : "write",-1,std::move(event));
  }
  need(used==Events && kernels==3075,"planned command population");
  runtime->wait();retired=true;
  for (const auto& s:saved)
    need(s.event->get_info<sycl::info::event::command_execution_status>()==sycl::info::event_command_status::complete,"complete after drain");
  size_t invalid=0;
  for (size_t i=0;i<used;++i) {
    auto& s=saved[i];const auto before=std::chrono::steady_clock::now();
    const uint64_t submit=s.event->get_profiling_info<sycl::info::event_profiling::command_submit>();
    const uint64_t start=s.event->get_profiling_info<sycl::info::event_profiling::command_start>();
    const uint64_t end=s.event->get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto wall=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-before).count();
    const bool order=submit<=start,positive=end>start,granularity=positive && end-start>=resolution;
    const bool query=wall<1000000000LL;
    const bool valid=order && positive && granularity && query;invalid+=size_t(!valid);
    std::cout<<"sample="<<i<<" kind="<<s.kind<<" job="<<s.job<<" submit="<<submit
             <<" start="<<start<<" end="<<end<<" query_wall_ns="<<wall
             <<" submit_order="<<int(order)<<" positive_service="<<int(positive)
             <<" granularity="<<int(granularity)<<" query_budget="<<int(query)<<" valid="<<int(valid)<<'\n';
    s.event.reset();
  }
  runtime->wait(q.memcpy(result.data(),base,Count*4));
  for (size_t i=0;i<Count;++i) {
    uint32_t expected=original[i];
    if (i>=Guard && i<Guard+64) expected+=additions[i-Guard];
    need(result[i]==expected,"complete output/guard check");
  }
  need(source==original,"input immutable");
  std::cout<<"case status=PASS guards=PASS inputs=PASS capture=COMPLETE events="<<used
           <<" kernels="<<kernels<<" age_delay_ms="<<(aged ? 20000 : 0)<<" copy_bytes_each="<<Plane*4
           <<" invalid_timestamps="<<invalid<<" timestamp_qualified="<<int(invalid==0)
           <<" timer_resolution_ns="<<resolution<<" hardware_busy=UNMEASURED model_PP=UNRUN\n";
}
} // namespace
int main(int argc,char** argv) {
  try {
    std::cout.setf(std::ios::unitbuf);need(argc==2 && (std::string(argv[1])=="--baseline" || std::string(argv[1])=="--aged"),"age mode required");
    gate("before-runtime");runtime=runtime_for();
    std::cout<<"device="<<runtime->device().get_info<sycl::info::device::name>()
             <<" driver="<<runtime->device().get_info<sycl::info::device::driver_version>()<<'\n';
    gate("runtime-ready");run(std::string(argv[1])=="--aged");runtime->wait();runtime.reset();gate("after-cleanup");
    std::cout<<"fixture_status=PASS model_PP=UNRUN faults=EXTERNAL_REQUIRED\n";return 0;
  } catch (const std::exception& e) { drain();std::cerr<<"event stress failed: "<<e.what()<<'\n';return 1; }
}
