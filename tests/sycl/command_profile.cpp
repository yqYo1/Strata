// Authored for Strata; repository LICENSE (MIT) applies.
// Synthetic sampling/lifetime/counter test; tags are not operator throughput.
#include "strata/sycl/command_profile.hpp"
#include "strata/sycl/launch.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/compat/cuda_runtime.h"
#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
using namespace strata::sycl_backend;
namespace cp = strata::sycl_backend::command_profile;
namespace {
std::shared_ptr<Runtime> runtime;
void need(bool ok,const char* text) { if (!ok) throw std::runtime_error(text); }
void drain() noexcept { checked_cleanup([] { if (runtime) runtime->wait(); },[] {},"command-profile fixture drain"); }
void gate(const char* phase) {
  const char* value=std::getenv("STRATA_TEST_MAIN_GATES");
  if (!value || std::string(value)!="1") return;
  std::cout<<"EVENT_PROFILE_READY "<<phase<<'\n';need(std::cin.get()=='X',"Main gate token");
}
void run(bool inject) {
  constexpr size_t Guard=16,Words=64,Count=Words+Guard*2;
  constexpr uint32_t Sentinel=0x14ac28be;
  std::array<uint32_t,Count> source{},result{};
  std::array<uint32_t,Words> expected{};
  for (size_t i=0;i<Count;++i) source[i]=i<Guard || i>=Guard+Words ? Sentinel : 0;
  const auto original=source;
  Allocation memory(runtime,sizeof(source),MemoryKind::Device);
  auto& queue=runtime->compute();auto* base=memory.as<uint32_t>();auto* data=base+Guard;
  // Retain the event owner through the fallback checked drain on exceptions.
  std::unique_ptr<cp::Trace> trace;
  bool retired=false;
  struct Pending {
    bool& done;
    ~Pending() noexcept {
      if (!done) {
        std::cerr<<"fixture fallback drain begin\n";
        drain();
        std::cerr<<"fixture fallback drain end\n";
      }
    }
  } pending{retired};
  runtime->wait(queue.memcpy(base,source.data(),sizeof(source)));
  if (cp::requested()) trace.reset(new cp::Trace(&queue,1));
  {
    auto frame=trace ? trace->begin(0,0) : cp::Frame{};
    std::array<int64_t,64> rows{},offsets{};
    int64_t offset=0;
    for (unsigned bin=0;bin<8;++bin) for (unsigned j=0;j<8;++j) {
      const unsigned id=bin*8+j;
      rows[id]=(bin ? int64_t(16)<<(bin-1) : 1)+j;
      offsets[id]=offset;offset+=rows[id];
      frame.consider(int32_t(id),offsets[id],rows[id]);
    }
    {
      auto tag=frame.tag(cp::Kind::upload,-1,0,Words,1,4);
      need(cudaMemcpyAsync(data,source.data()+Guard,Words/2*4,cudaMemcpyHostToDevice,&queue)==cudaSuccess,"first actual compatibility copy");
      need(cudaMemcpyAsync(data+Words/2,source.data()+Guard+Words/2,Words/2*4,cudaMemcpyHostToDevice,&queue)==cudaSuccess,"second actual compatibility copy");
    }
    size_t launches=0;
    auto launch=[&](size_t word) {
      auto event=queue.single_task([=] { data[word]+=1; });
      finish(&queue,event);++expected[word];++launches;
    };
    for (unsigned id=0;id<64;++id) {
      for (auto kind:{cp::Kind::dequant_gu,cp::Kind::dequant_down,cp::Kind::gather,
                      cp::Kind::gu,cp::Kind::swiglu,cp::Kind::down}) {
        auto tag=frame.tag(kind,int32_t(id),offsets[id],rows[id],5120,640,21);
        launch((id+size_t(kind))%Words);
      }
    }
    {
      auto tag=frame.tag(cp::Kind::combine,-1,0,Words,5120,8);launch(1);
    }
    for (unsigned i=0;i<2;++i) {
      auto tag=frame.tag(cp::Kind::write,-1,i*32,32,10240);launch(i+2);
    }
    need(launches==387,"complete synthetic launch population");
    if (inject) throw std::runtime_error("injected-before-drain");
    runtime->wait();retired=true;
    frame.after_checked_drain();
  }
  runtime->wait(queue.memcpy(result.data(),base,sizeof(result)));
  for (size_t i=0;i<Count;++i)
    need(result[i]==(i<Guard || i>=Guard+Words ? Sentinel : expected[i-Guard]),"complete result/guard");
  need(source==original,"input unchanged");
  trace.reset(); // All events queried/cleared only after the checked sequence drain.
  std::cout<<"case status=PASS guards=PASS inputs=PASS expected_profile_records=141"
           <<" profile_requested="<<int(cp::requested())<<" synthetic_kernels=387 model_PP=UNRUN\n";
}
} // namespace
int main(int argc,char** argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    need(argc==2 && (std::string(argv[1])=="--fixture" || std::string(argv[1])=="--inject-before-drain"),"fixture mode required");
    gate("before-runtime");runtime=runtime_for();
    std::cout<<"device="<<runtime->device().get_info<sycl::info::device::name>()
             <<" driver="<<runtime->device().get_info<sycl::info::device::driver_version>()<<'\n';
    gate("runtime-ready");run(std::string(argv[1])=="--inject-before-drain");runtime->wait();runtime.reset();gate("after-cleanup");
    std::cout<<"fixture_status=PASS model_PP=UNRUN faults=EXTERNAL_REQUIRED\n";return 0;
  } catch (const std::exception& error) { drain();std::cerr<<error.what()<<'\n';return 1; }
}
