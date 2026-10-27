// Main-authored source-bound QSA diagnostic; repository LICENSE applies.
// Real selected IDs; synthetic finite Q/K/V. No engine route is changed.
// Own archived cfabc1ee2 buffers/capture reader; only logical ID topology varies.
#include "qsa-locality.hpp"
#include "qsa_locality_admission.hpp"
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
#include <iomanip>
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
  ~Drain() noexcept { sycl_backend::checked_cleanup([] { if(rt) rt->wait(); },[] {},"QSA locality fixture"); }
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
  std::cout<<"QSA_LOCALITY_GATE "<<stage<<'\n';require(std::cin.get()=='X',"Main gate token");
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


std::vector<int32_t> pattern(const Selected& source,int queries,int arm) {
  std::vector<int32_t> ids(size_t(queries)*Cap);
  for(int q=0;q<queries;++q) {
    const int pos=32768+q;
    if(arm<=1)std::copy_n(source.ids.data()+size_t(q)*Cap,Cap,ids.data()+size_t(q)*Cap);
    else {
      const int bank=arm==2?0:q%16;
      for(int c=0;c<2047;++c)ids[size_t(q)*Cap+c]=bank*2047+c;
      for(int c=0;c<4;++c)ids[size_t(q)*Cap+2047+c]=pos-3+c;
    }
    int previous=-1;
    for(int c=0;c<Cap;++c) {
      const int cell=ids[size_t(q)*Cap+c];
      require(previous<cell && cell<=pos,"locality IDs unordered/duplicate/noncausal");previous=cell;
    }
    require(previous==pos,"current-cell coverage");
  }
  return ids;
}
void run(const Selected& source,int queries,bool timing,int process) {
  require(queries==3 || queries==6491,"source-bound query extent");
  constexpr int cells=39259,pages=(cells+Page-1)/Page,rows=pages*KV*Page;
  std::vector<int8_t> codes(size_t(rows)*Dim),values(codes.size());
  std::vector<uint16_t> scales(size_t(rows)*4),vscales(scales.size()),query(size_t(queries)*Heads*Dim);
  // All logical cells have identical K/V within a KV head. Address topology
  // changes neither matrix operands, ordered softmax, masks nor output values.
  for(int row=0;row<rows;++row) {
    const int head=(row/Page)%KV;
    for(int d=0;d<Dim;++d) {
      codes[size_t(row)*Dim+d]=int8_t((head*3+d*7+3)%15-7);
      values[size_t(row)*Dim+d]=int8_t((head*5+d*11+5)%31-15);
    }
    for(int chunk=0;chunk<4;++chunk) {
      scales[size_t(row)*4+chunk]=half(std::ldexp(1.f,-5-chunk));
      vscales[size_t(row)*4+chunk]=half(std::ldexp(head? -1.f:1.f,-7-chunk));
    }
  }
  for(size_t i=0;i<query.size();++i)query[i]=bf16(float(int(i*13%17)-8)*.125f);
  std::vector<uint16_t> oracle(query.size());size_t negative_zero_sites=0;
  for(int q=0;q<queries;++q)for(int h=0;h<Heads;++h)for(int d=0;d<Dim;++d) {
    const int head=h/(Heads/KV),code=(head*5+d*11+5)%31-15;
    const float scale=float(std::bit_cast<_Float16>(half(std::ldexp(head?-1.f:1.f,-7-d/64))));
    const float value=float(code)*scale;
    // The FP32 PV accumulator starts at +0. An all-negative-zero V column
    // accumulates to +0 under round-to-nearest, rather than returning raw -0.
    // Correct the expected recurrence; never canonicalize observed GPU bits.
    const uint16_t raw=bf16(value),expected=bf16(0.f+value);
    require(raw==expected || (raw==0x8000 && expected==0),"nonzero uniform oracle change");
    negative_zero_sites+=raw!=expected;
    oracle[(size_t(q)*Heads+h)*Dim+d]=expected;
  }
  std::vector<int32_t> table(pages),steps(size_t(queries)*kStepCount);
  for(int p=0;p<pages;++p)table[p]=p;
  for(int q=0;q<queries;++q) {
    steps[q*kStepCount+kStepPos]=32768+q;steps[q*kStepCount+kStepNKv]=32769+q;
    steps[q*kStepCount+kStepNBid]=(32769+q)/4;
    steps[q*kStepCount+kStepWidth]=std::min(32769+q,Cap);
  }
  std::array<std::vector<int32_t>,4> ids;
  for(int arm=0;arm<4;++arm)ids[arm]=pattern(source,queries,arm);
  Buffer<int8_t> dk(codes.size()),dv(values.size());
  Buffer<uint16_t> dks(scales.size()),dvs(vscales.size()),dq(query.size()),out(query.size());
  Buffer<int32_t> dt(table.size()),di(ids[0].size()),ds(steps.size());
  dk.put(codes);dv.put(values);dks.put(scales);dvs.put(vscales);dq.put(query);
  dt.put(table);di.put(ids[0]);ds.put(steps);out.put(std::vector<uint16_t>(query.size()));
  Packed packed(rows);Drain drain;auto shape=qsa_real_shapes();QsaAttnPools pools{};
  pools.page_table=dt.data();pools.k_q=dk.data();pools.v_q=dv.data();
  pools.k_scale=dks.data();pools.v_scale=dvs.data();
  QsaPromptBf16Bounds bounds{pages,pages};
  auto call=[&](int arm,sycl::event* submitted) {
    bool accepted;
    if(!arm) {
      require(!submitted,"original production event is unavailable");
      accepted=qsa_bf16_prefix_ordered_exp_batch(dq.data(),pools,di.data(),ds.data(),Cap,shape,out.data(),queries,bounds,cells,packed.view(),&rt->compute());
    } else accepted=locality_control::qsa_bf16_prefix_ordered_exp_batch(dq.data(),pools,di.data(),ds.data(),Cap,shape,out.data(),queries,bounds,cells,packed.view(),&rt->compute(),submitted);
    require(accepted,"source-bound locality route refused");
  };
  auto check_output=[&](int arm) {
    const auto actual=out.get();
    if(actual!=oracle) {
      size_t changed=0,zero_sign=0;
      for(size_t i=0;i<actual.size();++i)if(actual[i]!=oracle[i]) {
        if(changed<8)std::cerr<<"uniform_oracle_delta arm="<<arm<<" index="<<i
          <<" expected_bits="<<oracle[i]<<" actual_bits="<<actual[i]<<'\n';
        ++changed;zero_sign+=(uint16_t(actual[i]^oracle[i])==0x8000 && (actual[i]&0x7fff)==0);
      }
      std::cerr<<"uniform_oracle_summary changed="<<changed<<" zero_sign_only="<<zero_sign<<'\n';
      throw std::runtime_error("uniform-cell full-output oracle differs");
    }
  };
  std::vector<uint16_t> packed_k,packed_v;std::vector<int32_t> packed_bad;
  for(int arm=0;arm<4;++arm) {
    di.put(ids[arm]);
    for(int repeat=0;repeat<2;++repeat) {
      sycl::event event;call(arm,arm?&event:nullptr);rt->wait();
      if(arm)require(service(event,"numerical",repeat)>0,"direct attention event");
      check_output(arm);
      if(!arm && !repeat){packed_k=packed.k.get();packed_v=packed.v.get();packed_bad=packed.bad.get();}
      else require(packed.k.get()==packed_k && packed.v.get()==packed_v && packed.bad.get()==packed_bad,"packed data changed across topology arms");
    }
    std::cout<<"{\"kind\":\"numerical\",\"arm\":"<<arm<<",\"queries\":"<<queries<<",\"compared\":"<<oracle.size()
      <<",\"negative_zero_sites_canonicalized\":"<<negative_zero_sites<<",\"all_bits_exact\":true}\n";
  }
  int final_arm=3;
  if(timing) {
    long long floor=std::numeric_limits<long long>::max();
    for(int i=0;i<128;++i) {
      const auto a=std::chrono::steady_clock::now(),b=std::chrono::steady_clock::now();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
      if(ns>0)floor=std::min(floor,static_cast<long long>(ns));
    }
    require(floor<std::numeric_limits<long long>::max(),"clock floor");
    constexpr int order[4][4]={{0,1,3,2},{1,2,0,3},{2,3,1,0},{3,0,2,1}};
    constexpr int repeats=8;
    for(int round=0;round<8;++round)for(int offset=0;offset<4;++offset) {
      const int arm=order[(round+process)%4][offset];
      di.put(ids[arm]);
      for(int i=0;i<2;++i)call(arm,nullptr);rt->wait();
      std::vector<sycl::event> events(arm?repeats:0);
      const auto begin=std::chrono::steady_clock::now();
      for(int i=0;i<repeats;++i)call(arm,arm?&events[size_t(i)]:nullptr);
      rt->wait();
      const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
      require(ns>=100000000 && ns>=1000*floor,"timing batch below prospective floor");
      uint64_t device_ns=0,previous_end=0;
      for(const auto& event:events) {
        const auto start=event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
        require(start>=previous_end && end>start,"invalid/overlapping direct attention event");
        device_ns+=end-start;previous_end=end;
      }
      if(arm)require(device_ns>0 && device_ns<=uint64_t(ns)+1000000,"service outside host envelope");
      require(out.get()==oracle,"timed full-output oracle differs");final_arm=arm;
      std::cout<<"{\"kind\":\"sample\",\"process\":"<<process<<",\"round\":"<<round<<",\"arm\":"<<arm
        <<",\"queries\":"<<queries<<",\"repeats\":"<<repeats<<",\"batch_ns\":"<<ns<<",\"attention_device_ns\":";
      if(arm)std::cout<<device_ns;else std::cout<<"null";
      std::cout<<",\"event_calls\":"<<events.size()<<",\"timer_floor_ns\":"<<floor<<"}\n";
    }
  }
  require(dk.get()==codes && dv.get()==values && dks.get()==scales && dvs.get()==vscales && dq.get()==query && dt.get()==table && di.get()==ids[final_arm] && ds.get()==steps,"immutable input changed");
  require(packed.k.get()==packed_k && packed.v.get()==packed_v && packed.bad.get()==packed_bad,"packed operands changed");
  require(out.get()==oracle,"final output differs");
  std::cout<<"{\"kind\":\"checked\",\"queries\":"<<queries<<",\"cells\":"<<cells<<",\"guards\":true,\"inputs_immutable\":true,\"all_outputs_exact\":true}\n";
}
} // namespace
int admission_main(int argc,char** argv) {
  try {
    require(argc==5,"--admit original private loader");std::cout.setf(std::ios::unitbuf);
    std::cout<<"admission_stage=before-runtime\n";
    auto runtime=sycl_backend::runtime_for();
    std::cout<<"admission_stage=runtime-ready device="<<runtime->device().get_info<sycl::info::device::name>()<<'\n';
    const auto properties=qsa_locality_admit(runtime->compute(),{argv[2],argv[3]},argv[4]);
    for(size_t arm=0;arm<properties.size();++arm) {
      const auto& p=properties[arm];
      std::cout<<"arm="<<arm<<" max_workgroup="<<p.maximum_workgroup<<" subgroup="<<p.subgroup
        <<" arguments="<<p.arguments<<" reported_local_bytes="<<p.reported_local_bytes
        <<" declared_dynamic_local_bytes="<<p.declared_local_bytes<<" spill_descriptor="<<p.spill_descriptor<<" status=PASS\n";
    }
    runtime->wait();runtime.reset();
    std::cout<<"admission_status=PASS kernel_submissions=0\n";return 0;
  }catch(const std::exception& e){std::cerr<<"admission_status=FAIL error="<<e.what()<<'\n';return 1;}
}
int main(int argc,char** argv) {
  try {
    if(argc==5 && std::string(argv[1])=="--admit")return admission_main(argc,argv);
    require(argc==3 || argc==4,"--smoke|--bench FIXTURE [PROCESS]");
    const bool timing=std::string(argv[1])=="--bench";
    require(timing || std::string(argv[1])=="--smoke","unknown mode");
    require(timing==bool(argc==4),"bench process required");
    const int process=argc==4?std::stoi(argv[3]):0;require(process>=0 && process<3,"process ordinal");
    const Selected source(argv[2]);std::cout.setf(std::ios::unitbuf);std::cout<<std::setprecision(10);
    gate("before-runtime");rt=sycl_backend::runtime_for();
    require(rt->compute().has_property<sycl::property::queue::in_order>() && rt->compute().has_property<sycl::property::queue::enable_profiling>(),"qualified queue required");
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()<<'\n';gate("runtime-ready");
    run(source,timing?6491:3,timing,process);
    rt->wait();rt.reset();gate("after-cleanup");std::cout<<"{\"kind\":\"PASS\"}\n";return 0;
  }catch(const std::exception& e){std::cerr<<"QSA locality failure: "<<e.what()<<'\n';return 1;}
}
