// Main-authored source-bound QSA diagnostic; repository LICENSE applies.
// Real selected IDs; synthetic finite Q/K/V. No engine route is changed.
#include "qsa-barrier.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace {
constexpr int Dim=256,Heads=24,KV=2,Page=4,Guard=16,Cap=2051;
std::shared_ptr<sycl_backend::Runtime> rt;
void require(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
struct Drain {
  ~Drain() noexcept { sycl_backend::checked_cleanup([] { if(rt) rt->wait(); },[] {},"QSA barrier fixture"); }
};
template<class T> struct Buffer {
  size_t count;
  sycl_backend::Allocation allocation;
  // Host transfer storage is a member: retained through checked retirement.
  std::vector<T> host;
  static size_t bytes(size_t n) {
    require(n <= std::numeric_limits<size_t>::max()/sizeof(T)-2*Guard,"allocation size overflow");
    const size_t value=(n+2*Guard)*sizeof(T);
    require(value<=rt->device().get_info<sycl::info::device::max_mem_alloc_size>(),"device single-allocation limit");
    return value;
  }
  explicit Buffer(size_t n):count(n),allocation(rt,bytes(n),sycl_backend::MemoryKind::Device) {}
  T* data() { return allocation.as<T>()+Guard; }
  void put(const std::vector<T>& input) {
    require(input.size()==count,"upload extent");
    host.assign(count+2*Guard,T(73));std::copy(input.begin(),input.end(),host.begin()+Guard);
    Drain drain;rt->wait(rt->compute().memcpy(allocation.as<T>(),host.data(),host.size()*sizeof(T)));
  }
  std::vector<T> get() {
    host.resize(count+2*Guard);Drain drain;
    rt->wait(rt->compute().memcpy(host.data(),allocation.as<T>(),host.size()*sizeof(T)));
    for(int i=0;i<Guard;++i) require(host[i]==T(73) && host[count+Guard+i]==T(73),"device guard");
    return {host.begin()+Guard,host.end()-Guard};
  }
};
uint16_t bf16(float x) {
  uint32_t bits=std::bit_cast<uint32_t>(x);
  return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
void gate(const char* stage) {
  std::cout<<"QSA_BARRIER_GATE "<<stage<<'\n';require(std::cin.get()=='X',"Main gate token");
}
uint64_t service(const sycl::event& event,const char* stage,int replica) {
  const auto begin=event.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
  if(end<=begin) std::cerr<<"event_clock_invalid stage="<<stage<<" replica="<<replica<<" start="<<begin<<" end="<<end<<'\n';
  require(end>begin,"attention event clock invalid");return end-begin;
}
struct Selected {
  std::array<int32_t,4> header;
  std::vector<int32_t> ids;
  explicit Selected(const char* file) {
    require(std::endian::native==std::endian::little,"capture integer ABI");
    std::ifstream stream(file,std::ios::binary|std::ios::ate);require(bool(stream),"open selected fixture");
    require(stream.tellg()==16+int64_t(6491)*Cap*4,"selected fixture extent");stream.seekg(0);
    stream.read(reinterpret_cast<char*>(header.data()),16);
    require(header==std::array<int32_t,4>{0,32768,6491,Cap},"source-bound capture header");
    ids.resize(size_t(6491)*Cap);stream.read(reinterpret_cast<char*>(ids.data()),ids.size()*4);
    require(bool(stream) && stream.peek()==std::char_traits<char>::eof(),"selected fixture payload");
    for(int q=0;q<6491;++q) {
      int previous=-1;
      for(int c=0;c<Cap;++c) {
        const int cell=ids[size_t(q)*Cap+c];require(previous<cell && cell<=32768+q,"causal sorted IDs");previous=cell;
      }
    }
  }
};
struct Packed {
  Buffer<uint16_t> k,v;
  Buffer<int32_t> bad;
  explicit Packed(size_t rows):k(rows*Dim),v(rows*Dim),bad(rows) {
    k.put(std::vector<uint16_t>(rows*Dim));v.put(std::vector<uint16_t>(rows*Dim));bad.put(std::vector<int32_t>(rows));
  }
  QsaBf16PackedScratch view() { return {k.data(),v.data(),bad.data()}; }
};

void run(const Selected& source,int queries,bool timing,int process) {
  const int cells=39259,pages=(cells+Page-1)/Page,rows=pages*KV*Page;
  require(queries>0 && queries<=6491,"fixture domain");
  std::vector<int8_t> codes(size_t(rows)*Dim),values(codes.size());
  for(size_t i=0;i<codes.size();++i) {codes[i]=int8_t(int((i*7+3)%15)-7);values[i]=int8_t(int((i*11+5)%31)-15);}
  std::vector<uint16_t> scales(size_t(rows)*4),vscales(scales.size()),query(size_t(queries)*Heads*Dim);
  for(size_t i=0;i<scales.size();++i) {
    scales[i]=half(std::ldexp(1.f,-5-int(i%4)));
    vscales[i]=half(std::ldexp(i%3?1.f:-1.f,-7-int(i%4)));
  }
  for(size_t i=0;i<query.size();++i) query[i]=bf16(float(int(i*13%17)-8)*.125f);
  std::vector<int32_t> table(pages),ids(size_t(queries)*Cap),steps(size_t(queries)*kStepCount);
  for(int p=0;p<pages;++p)table[p]=p;
  for(int q=0;q<queries;++q) {
    steps[q*kStepCount+kStepPos]=32768+q;steps[q*kStepCount+kStepNKv]=32769+q;
    steps[q*kStepCount+kStepNBid]=(32769+q)/4;steps[q*kStepCount+kStepWidth]=Cap;
    std::copy_n(source.ids.data()+size_t(q)*Cap,Cap,ids.data()+size_t(q)*Cap);
  }
  Buffer<int8_t> dk(codes.size()),dv(values.size());
  Buffer<uint16_t> dks(scales.size()),dvs(vscales.size()),dq(query.size()),out(query.size());
  Buffer<int32_t> dt(table.size()),di(ids.size()),ds(steps.size());
  dk.put(codes);dv.put(values);dks.put(scales);dvs.put(vscales);dq.put(query);
  dt.put(table);di.put(ids);ds.put(steps);out.put(std::vector<uint16_t>(query.size()));
  Packed packed(rows);Drain drain;auto shape=qsa_real_shapes();QsaAttnPools pools{};
  pools.page_table=dt.data();pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();
  QsaPromptBf16Bounds bounds{pages,pages};
  auto call=[&](int arm,sycl::event* submitted) {
    bool accepted=false;
    if(!arm) {
      require(!submitted,"unmodified production has no direct-event output");
      accepted=qsa_bf16_prefix_ordered_exp_batch(dq.data(),pools,di.data(),ds.data(),Cap,shape,out.data(),queries,bounds,cells,packed.view(),&rt->compute());
    } else {
      const auto fn=arm==1?barrier_control::qsa_bf16_prefix_ordered_exp_batch:barrier_merged::qsa_bf16_prefix_ordered_exp_batch;
      accepted=fn(dq.data(),pools,di.data(),ds.data(),Cap,shape,out.data(),queries,bounds,cells,packed.view(),&rt->compute(),submitted);
    }
    require(accepted,"selected barrier arm refused");
  };
  std::vector<uint16_t> expected,packed_k,packed_v;std::vector<int32_t> packed_bad;
  for(int arm=0;arm<3;++arm)for(int repeat=0;repeat<2;++repeat) {
    sycl::event event;call(arm,arm?&event:nullptr);rt->wait();
    if(arm)require(service(event,arm==1?"control":"merged",repeat)>0,"direct event clock");
    if(!arm && !repeat) {expected=out.get();packed_k=packed.k.get();packed_v=packed.v.get();packed_bad=packed.bad.get();}
    else require(out.get()==expected,"original/control/merged output bits differ");
    require(packed.k.get()==packed_k && packed.v.get()==packed_v && packed.bad.get()==packed_bad,"packed reference differs");
  }
  if(timing) {
    long long floor=std::numeric_limits<long long>::max();
    for(int i=0;i<128;++i) {
      const auto a=std::chrono::steady_clock::now(),b=std::chrono::steady_clock::now();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
      if(ns>0)floor=std::min(floor,static_cast<long long>(ns));
    }
    require(floor<std::numeric_limits<long long>::max(),"clock floor");
    const int repeats=queries==512?32:2;
    for(int round=0;round<6;++round)for(int offset=0;offset<3;++offset) {
      const int arm=(process+round+offset)%3;
      for(int i=0;i<2;++i)call(arm,nullptr);rt->wait();
      std::vector<sycl::event> events(arm?repeats:0);
      const auto begin=std::chrono::steady_clock::now();
      for(int i=0;i<repeats;++i)call(arm,arm?&events[size_t(i)]:nullptr);
      rt->wait();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
      require(ns>=100000000 && ns>=1000*floor,"timing batch below prospective floor");
      uint64_t device_ns=0,previous_end=0;int call_index=0;
      for(const auto& event:events) {
        const auto start=event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
        if(start<previous_end || end<=start)std::cerr<<"event_clock_invalid queries="<<queries<<" process="<<process<<" round="<<round<<" arm="<<arm<<" call="<<call_index<<" start="<<start<<" end="<<end<<" previous_end="<<previous_end<<'\n';
        require(start>=previous_end && end>start,"invalid/overlapping direct event");
        device_ns+=end-start;previous_end=end;++call_index;
      }
      if(arm)require(device_ns>0 && device_ns<=uint64_t(ns)+1000000,"event service exceeds host envelope");
      require(out.get()==expected,"timed output differs");
      std::cout<<"{\"kind\":\"sample\",\"process\":"<<process<<",\"queries\":"<<queries<<",\"round\":"<<round<<",\"arm\":"<<arm
        <<",\"repeats\":"<<repeats<<",\"batch_ns\":"<<ns<<",\"attention_device_ns\":";
      if(arm)std::cout<<device_ns;else std::cout<<"null";
      std::cout<<",\"event_calls\":"<<events.size()<<",\"timer_floor_ns\":"<<floor<<"}\n";
    }
  }
  require(dk.get()==codes && dv.get()==values && dks.get()==scales && dvs.get()==vscales && dq.get()==query && dt.get()==table && di.get()==ids && ds.get()==steps,"immutable input changed");
  require(packed.k.get()==packed_k && packed.v.get()==packed_v && packed.bad.get()==packed_bad,"packed operand bytes changed");
  require(out.get()==expected,"final output differs");
  std::cout<<"{\"kind\":\"checked\",\"queries\":"<<queries<<",\"cells\":"<<cells<<",\"guards\":true,\"inputs_immutable\":true,\"exact_original\":true}\n";
}
} // namespace
int main(int argc,char** argv) {
  try {
    require(argc==3 || argc==4,"--smoke|--bench FIXTURE [PROCESS]");
    const std::string mode=argv[1];require(mode=="--smoke" || mode=="--bench","unknown mode");
    require((mode=="--bench")==bool(argc==4),"process only for bench");
    const int process=argc==4?std::stoi(argv[3]):0;require(process>=0 && process<3,"process ordinal");
    const Selected source(argv[2]);std::cout.setf(std::ios::unitbuf);
    gate("before-runtime");rt=sycl_backend::runtime_for();
    require(rt->compute().has_property<sycl::property::queue::in_order>() && rt->compute().has_property<sycl::property::queue::enable_profiling>(),"queue properties");
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()<<'\n';gate("runtime-ready");
    if(mode=="--smoke")run(source,3,false,0);
    else {run(source,512,true,process);run(source,6491,true,process);}
    rt->wait();rt.reset();gate("after-cleanup");std::cout<<"{\"kind\":\"PASS\"}\n";
    return 0;
  }catch(const std::exception& e) {std::cerr<<"QSA barrier failure: "<<e.what()<<'\n';return 1;}
}
