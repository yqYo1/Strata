// Main-authored source-bound QSA diagnostic; repository LICENSE applies.
// Real selected IDs; synthetic finite Q/K/V. No engine route is changed.
#include "qsa-workingset.hpp"
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
#ifndef SYCL_EXT_ONEAPI_IN_ORDER_QUEUE_EVENTS
#error "Diagnostic requires the pinned SDK's in-order last-event extension"
#endif

using namespace strata;
using namespace strata::kernels;
namespace {
constexpr int Dim=256,Heads=24,KV=2,Page=4,Guard=16,Cap=2051;
std::shared_ptr<sycl_backend::Runtime> rt;
void require(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
struct Drain {
  ~Drain() noexcept { sycl_backend::checked_cleanup([] { if(rt) rt->wait(); },[] {},"QSA working-set fixture"); }
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
  std::cout<<"WS_GATE "<<stage<<'\n';require(std::cin.get()=='X',"Main gate token");
}
sycl::event last_attention() {
  auto event=rt->compute().ext_oneapi_get_last_event();
  require(event.has_value(),"last attention event missing");return *event;
}
uint64_t service(const sycl::event& event) {
  const auto begin=event.get_profiling_info<sycl::info::event_profiling::command_start>();
  const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
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

void run(const Selected& source,int queries,bool repeated,int copies,bool timing,int process,bool physical=false) {
  const int cells=physical?262144:39259,pages=(cells+Page-1)/Page,rows=pages*KV*Page;
  const int cap=physical?33:Cap;
  require(queries>0 && queries<=6491 && copies>=1 && copies<=8,"fixture domain");
  std::vector<int8_t> codes(size_t(rows)*Dim),values(codes.size());
  for(size_t i=0;i<codes.size();++i) { codes[i]=int8_t(int((i*7+3)%15)-7);values[i]=int8_t(int((i*11+5)%31)-15); }
  std::vector<uint16_t> scales(size_t(rows)*4),vscales(scales.size()),query(size_t(queries)*Heads*Dim);
  for(size_t i=0;i<scales.size();++i) {
    scales[i]=half(std::ldexp(1.f,-5-int(i%4)));
    vscales[i]=half(std::ldexp(i%3?1.f:-1.f,-7-int(i%4)));
  }
  for(int q=0;q<queries;++q) for(int d=0;d<Heads*Dim;++d)
    query[size_t(q)*Heads*Dim+d]=bf16(float(int(((repeated?0:q)*Heads*Dim+d)*13%17)-8)*.125f);
  std::vector<int32_t> table(pages),ids(size_t(queries)*cap),steps(size_t(queries)*kStepCount);
  for(int p=0;p<pages;++p) table[p]=p;
  for(int q=0;q<queries;++q) {
    const int original=repeated?0:q;
    const int position=physical?cells-1:32768+original;
    steps[q*kStepCount+kStepPos]=position;steps[q*kStepCount+kStepNKv]=position+1;
    steps[q*kStepCount+kStepNBid]=(position+1)/4;steps[q*kStepCount+kStepWidth]=cap;
    for(int c=0;c<cap;++c) ids[size_t(q)*cap+c]=physical?cells-cap+c:source.ids[size_t(original)*Cap+c];
  }
  Buffer<int8_t> dk(codes.size()),dv(values.size());Buffer<uint16_t> dks(scales.size()),dvs(vscales.size()),dq(query.size()),out(query.size());
  Buffer<int32_t> dt(table.size()),di(ids.size()),ds(steps.size());
  dk.put(codes);dv.put(values);dks.put(scales);dvs.put(vscales);dq.put(query);
  dt.put(table);di.put(ids);ds.put(steps);out.put(std::vector<uint16_t>(query.size()));
  std::vector<std::unique_ptr<Packed>> packed;
  for(int r=0;r<copies;++r) packed.push_back(std::make_unique<Packed>(rows));
  Drain drain;
  auto shape=qsa_real_shapes();QsaAttnPools pools{};
  pools.page_table=dt.data();pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();
  QsaPromptBf16Bounds bounds{pages,pages};
  auto call=[&](bool prepare,int replica) {
    const auto fn=prepare?qsa_bf16_prefix_ordered_exp_batch:working_set::qsa_bf16_prefix_ordered_exp_batch;
    require(fn(dq.data(),pools,di.data(),ds.data(),cap,shape,out.data(),queries,bounds,cells,packed[replica]->view(),&rt->compute()),"selected QSA entry refused");
  };
  // Original production pack creates every immutable replica outside timing.
  // It also supplies a bit-exact reference for this unchanged arithmetic.
  std::vector<uint16_t> expected;
  for(int r=0;r<copies;++r) {
    call(true,r);auto original_event=last_attention();rt->wait();require(service(original_event)>0,"original event");auto actual=out.get();
    if(!r) expected=actual;else require(actual==expected,"replica production operands differ");
    call(false,r);auto prepacked_event=last_attention();rt->wait();require(service(prepacked_event)>0,"prepacked event");
    require(out.get()==expected,"prepacked result differs from original");
  }
  // Source final-page padding is explicitly zeroed by the original pack.
  // Verify full prepared storage is identical, including bad-row metadata.
  const auto first_k=packed[0]->k.get(),first_v=packed[0]->v.get();const auto first_bad=packed[0]->bad.get();
  for(int r=1;r<copies;++r) {
    require(packed[r]->k.get()==first_k && packed[r]->v.get()==first_v && packed[r]->bad.get()==first_bad,"prepared replica bytes differ");
  }
  if(!timing) {
    bool rejected=false;auto alias=packed[0]->view();alias.v=alias.k;
    try { (void)working_set::qsa_bf16_prefix_ordered_exp_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,out.data(),queries,bounds,cells,alias,&rt->compute()); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"prepacked alias not rejected");
  } else {
    long long floor=std::numeric_limits<long long>::max();
    for(int i=0;i<128;++i) {
      const auto a=std::chrono::steady_clock::now(),b=std::chrono::steady_clock::now();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
      if(ns>0) floor=std::min(floor,static_cast<long long>(ns));
    }
    require(floor<std::numeric_limits<long long>::max(),"clock floor");
    const int repeats=repeated?64:2;
    for(int round=0;round<6;++round) for(int offset=0;offset<5;++offset) {
      const int arm=(process+round+offset)%5;
      const int rotation=arm==4?1:(1<<arm);const bool prepare=arm==4;
      require(rotation<=copies,"rotation exceeds prepared copies");
      for(int i=0;i<2*rotation;++i) call(prepare,i%rotation);
      rt->wait();
      std::vector<sycl::event> events;events.reserve(repeats);
      const auto begin=std::chrono::steady_clock::now();
      for(int i=0;i<repeats;++i) { call(prepare,i%rotation);events.push_back(last_attention()); }
      rt->wait();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
      require(ns>=100000000 && ns>=1000*floor,"timing batch below prospective floor");
      uint64_t device_ns=0,previous_end=0;
      for(const auto& event:events) {
        const auto start=event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
        require(start>=previous_end && end>start,"duplicate/overlapping attention events on in-order queue");
        device_ns+=end-start;previous_end=end;
      }
      require(device_ns>0 && device_ns<=uint64_t(ns)+1000000,"event service exceeds host envelope");
      require(out.get()==expected,"timed output differs");
      std::cout<<"{\"kind\":\"sample\",\"process\":"<<process<<",\"shape\":\""<<(repeated?"real1-repeated256":"real6491")
        <<"\",\"round\":"<<round<<",\"arm\":"<<arm<<",\"copies\":"<<rotation<<",\"prepare\":"<<int(prepare)
        <<",\"queries\":"<<queries<<",\"repeats\":"<<repeats<<",\"batch_ns\":"<<ns<<",\"attention_device_ns\":"<<device_ns
        <<",\"event_calls\":"<<events.size()<<",\"timer_floor_ns\":"<<floor<<"}\n";
    }
  }
  require(dk.get()==codes && dv.get()==values && dks.get()==scales && dvs.get()==vscales && dq.get()==query &&
    dt.get()==table && di.get()==ids && ds.get()==steps,"immutable input changed");
  for(auto& p:packed) require(p->k.get()==first_k && p->v.get()==first_v && p->bad.get()==first_bad,"immutable prepared storage changed");
  require(out.get()==expected,"final output guard/result");
  std::cout<<"{\"kind\":\"checked\",\"queries\":"<<queries<<",\"cells\":"<<cells<<",\"copies\":"<<copies
    <<",\"repeated\":"<<int(repeated)<<",\"guards\":true,\"inputs_immutable\":true,\"exact_original\":true}\n";
}
} // namespace
int main(int argc,char** argv) {
  try {
    require(argc==3 || argc==4,"--smoke|--check|--bench FIXTURE [PROCESS]");
    const std::string mode=argv[1];require(mode=="--smoke" || mode=="--check" || mode=="--bench","unknown mode");
    require((mode=="--bench")==(argc==4),"process argument only for bench");
    const int process=argc==4?std::stoi(argv[3]):0;require(process>=0 && process<3,"process ordinal");
    const Selected source(argv[2]);std::cout.setf(std::ios::unitbuf);
    gate("before-runtime");rt=sycl_backend::runtime_for();
    require(rt->compute().has_property<sycl::property::queue::in_order>() &&
      rt->compute().has_property<sycl::property::queue::enable_profiling>(),"qualified queue properties");
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()<<'\n';
    std::cout<<"{\"kind\":\"properties\",\"global_cache_bytes\":"<<rt->device().get_info<sycl::info::device::global_mem_cache_size>()
      <<",\"maximum_allocation_bytes\":"<<rt->device().get_info<sycl::info::device::max_mem_alloc_size>()
      <<",\"local_bytes\":"<<rt->device().get_info<sycl::info::device::local_mem_size>()<<"}\n";
    gate("runtime-ready");
    if(mode=="--smoke") run(source,3,false,2,false,0);
    else if(mode=="--check") { run(source,16,false,8,false,0);run(source,3,false,2,false,0,true); }
    else { run(source,256,true,8,true,process);run(source,6491,false,8,true,process); }
    rt->wait();rt.reset();gate("after-cleanup");std::cout<<"{\"kind\":\"PASS\"}\n";return 0;
  } catch(const std::exception& e) { std::cerr<<"QSA working-set failure: "<<e.what()<<'\n';return 1; }
}
