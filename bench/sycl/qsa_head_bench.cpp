// Authored for Strata; repository LICENSE applies.
// Component timing with varied nonzero operands. This is not model throughput.
#include "qsa-head.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> rt;
constexpr int Guard=16, Queries=8192, Cap=2051, Dim=256, Page=4, Batch=8;
void check(bool value,const char *message) {if(!value) throw std::runtime_error(message);}
void gate(const char *phase) {
  std::cout<<"HEAD_BENCH_GATE "<<phase<<std::endl;
  check(std::cin.get()=='X',"Main gate token");
}
struct Drain {
  ~Drain() noexcept {sycl_backend::checked_cleanup([]{if(rt)rt->wait();},[]{},"head benchmark drain");}
};
template<class T> struct Buffer {
  sycl_backend::Allocation memory;
  std::vector<T> upload,download;
  size_t size;
  T marker;
  Buffer(size_t n,T mark):memory(rt,(n+2*Guard)*sizeof(T),sycl_backend::MemoryKind::Device),size(n),marker(mark) {}
  T *base() {return memory.as<T>();}
  T *data() {return base()+Guard;}
  void put(const std::vector<T> &values) {
    check(values.size()==size,"payload size");rt->wait();upload.assign(size+2*Guard,marker);
    std::copy(values.begin(),values.end(),upload.begin()+Guard);
    rt->wait(rt->compute().memcpy(base(),upload.data(),memory.size()));
  }
  const std::vector<T> &get() {
    rt->wait();download.resize(size+2*Guard);
    rt->wait(rt->compute().memcpy(download.data(),base(),memory.size()));
    for(int i=0;i<Guard;++i)check(download[i]==marker && download[download.size()-Guard+i]==marker,"device guard changed");
    return download;
  }
  void immutable() {check(get()==upload,"input bytes changed");}
};
// Direct FP32 observation,not final BF16: a following DPAS could hide FTZ.
// 0x8200 is2^-134+2^-140,the finite online-state edge from the source review.
template<bool Multiply> class IdentityScaleEdge;
template<bool Multiply> void edge_submit(const float *input,const float *alpha,float *output) {
  rt->compute().submit([&](sycl::handler &handler) {
    handler.parallel_for<IdentityScaleEdge<Multiply>>(sycl::nd_range<1>(16,16),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      namespace mx=sycl::ext::oneapi::experimental::matrix;
      namespace mi=sycl::ext::intel::experimental::matrix;
      auto sg=item.get_sub_group();
      mx::joint_matrix<sycl::sub_group,float,mx::use::accumulator,8,16> value;
      mx::joint_matrix_load(sg,value,sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(input),16,mx::layout::row_major);
      if constexpr(Multiply)
        mi::joint_matrix_apply(sg,value,[=](float &v,size_t row,size_t) {v*=alpha[row];});
      mx::joint_matrix_store(sg,value,sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(output),16,mx::layout::row_major);
    });
  });
}
void identity_edge() {
  const auto ids=std::vector<sycl::kernel_id>{sycl::get_kernel_id<IdentityScaleEdge<false>>(),sycl::get_kernel_id<IdentityScaleEdge<true>>()};
  auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(rt->compute().get_context(),{rt->device()},ids);
  for(const auto &id:ids) {
    auto k=bundle.get_kernel(id);
    check(k.get_info<sycl::info::kernel_device_specific::compile_sub_group_size>(rt->device())==16 &&
          k.get_info<sycl::info::kernel_device_specific::work_group_size>(rt->device())>=16,"edge compiled geometry");
  }
  constexpr uint32_t bits[]={0,0x80000000,1,0x80000001,0x10000,0x80010000,0x400000,0x80400000,
    0x7fffff,0x807fffff,0x800000,0x80800000,0x3f800000,0xbf800000,0x8200,0x80008200};
  std::vector<float> values(128),scales(8,1.f);int subnormal=0;
  for(int i=0;i<128;++i) {values[i]=std::bit_cast<float>(bits[i%16]);subnormal+=std::fpclassify(values[i])==FP_SUBNORMAL;}
  check(subnormal==80,"host finite-subnormal construction");
  Buffer<float> input(128,42.5f),alpha(8,42.5f),scaled(128,42.5f),skipped(128,42.5f);Drain pending;
  input.put(values);alpha.put(scales);scaled.put(std::vector<float>(128,17.f));skipped.put(std::vector<float>(128,17.f));
  edge_submit<true>(input.data(),alpha.data(),scaled.data());rt->wait();
  edge_submit<false>(input.data(),alpha.data(),skipped.data());rt->wait();
  const auto original=scaled.get(),candidate=skipped.get(),unchanged=input.get();
  for(int i=0;i<128;++i) {
    const auto expected=bits[i%16],a=std::bit_cast<uint32_t>(original[Guard+i]),b=std::bit_cast<uint32_t>(candidate[Guard+i]);
    check(std::bit_cast<uint32_t>(unchanged[Guard+i])==expected,"edge input bit mutation");
    check(b==expected,"edge direct matrix load/store loses FP32bits");
    check(a==b,"FP32subnormal identity-rescale parity failed before timing");
  }
  alpha.immutable();
  std::cout<<"HEAD_EDGE finite=128 subnormal=80 FP32_bit_parity=1 direct_matrix_load_store_exact=1\n";
}

uint32_t next(uint32_t &s) {s^=s<<13;s^=s>>17;s^=s<<5;return s;}
uint16_t bf16(float x) {const auto b=std::bit_cast<uint32_t>(x);return uint16_t((b+0x7fff+((b>>16)&1))>>16);}
uint16_t fp16(float x) {return std::bit_cast<uint16_t>(_Float16(x));}
struct Sample {double wall_ms,pack_ms,attention_ms;};
void run_case(int context,int pattern,int process,int scenario) {
  const int pages=(context+Page-1)/Page,rows=pages*2*Page;
  std::vector<int8_t> hk(size_t(rows)*Dim),hv(hk.size());
  std::vector<uint16_t> hks(size_t(rows)*4),hvs(hks.size()),hq(size_t(Queries)*24*Dim);
  std::vector<int32_t> ht(pages),hi(size_t(Queries)*Cap),hs(size_t(Queries)*kStepCount);
  // Uniform byte codes and varied finite BF16/FP16 values defeat the previous
  // zero-Q/constant-V-only benchmark. These are synthetic,not model captures.
  uint32_t random=0x5d87a331u+uint32_t(context)*17+uint32_t(pattern)*1237;
  for(auto &v:hk)v=int8_t(next(random)>>24);
  for(auto &v:hv)v=int8_t(next(random)>>24);
  for(auto &v:hks) {
    const auto bits=next(random),exponent=next(random);
    v=fp16(std::ldexp(1.f+float(bits&15)/16,-7-int(exponent%4)));
  }
  for(auto &v:hvs) {
    const auto bits=next(random),exponent=next(random);
    v=fp16(std::ldexp(1.f+float(bits&31)/32,-7-int(exponent%4)));
  }
  for(auto &v:hq)v=bf16(std::ldexp(float(int(next(random)&65535)-32768),-12));
  for(int p=0;p<pages;++p)ht[p]=pages-1-p;
  for(int q=0;q<Queries;++q) {
    const int limit=context-Queries+q+1;
    int stride=239;while(std::gcd(stride,limit)!=1)stride+=2;
    const int start=int(next(random)%uint32_t(pattern?limit:limit-Cap+1));
    for(int j=0;j<Cap;++j)hi[size_t(q)*Cap+j]=pattern?int((int64_t(start)+int64_t(j)*stride)%limit):start+j;
    std::sort(hi.begin()+size_t(q)*Cap,hi.begin()+size_t(q+1)*Cap);
    check(std::adjacent_find(hi.begin()+size_t(q)*Cap,hi.begin()+size_t(q+1)*Cap)==hi.begin()+size_t(q+1)*Cap,"duplicate selected ID");
    hs[size_t(q)*kStepCount+kStepWidth]=Cap;
  }
  Buffer<int8_t> k(hk.size(),int8_t(0x55)),v(hv.size(),int8_t(0x55));
  Buffer<uint16_t> ks(hks.size(),0x59a3),vs(hvs.size(),0x59a3),query(hq.size(),0x59a3);
  Buffer<int32_t> table(ht.size(),0x12345678),ids(hi.size(),0x12345678),steps(hs.size(),0x12345678);
  Buffer<uint16_t> packed_k(size_t(rows)*Dim,0x59a3),packed_v(size_t(rows)*Dim,0x59a3),out(hq.size(),0x59a3);
  Buffer<int32_t> bad(rows,0x12345678);Drain pending;
  k.put(hk);v.put(hv);ks.put(hks);vs.put(hvs);query.put(hq);table.put(ht);ids.put(hi);steps.put(hs);
  packed_k.put(std::vector<uint16_t>(size_t(rows)*Dim,0x7fc1));packed_v.put(std::vector<uint16_t>(size_t(rows)*Dim,0x7fc1));
  bad.put(std::vector<int32_t>(rows,0x12345678));out.put(std::vector<uint16_t>(hq.size(),0x7fc1));
  QsaAttnPools pools{};pools.page_table=table.data();pools.k_q=k.data();pools.v_q=v.data();pools.k_scale=ks.data();pools.v_scale=vs.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;
  const QsaBf16PackedScratch scratch{packed_k.data(),packed_v.data(),bad.data()};
  auto invoke=[&](bool candidate,sycl::event *attention,sycl::event *pack) {
    return candidate?qsa_head_candidate::batch(query.data(),pools,ids.data(),steps.data(),Cap,shape,out.data(),Queries,{pages,pages},context,scratch,&rt->compute(),attention,pack):
                     qsa_head_original::batch(query.data(),pools,ids.data(),steps.data(),Cap,shape,out.data(),Queries,{pages,pages},context,scratch,&rt->compute(),attention,pack);
  };
  // Both entrypoints warm outside timing; compare every BF16 result bit.
  check(invoke(false,nullptr,nullptr),"original refused");rt->wait();const auto expected=out.get();
  check(invoke(true,nullptr,nullptr),"candidate refused");rt->wait();check(out.get()==expected,"full-output bit parity");
  auto measure=[&](bool candidate) {
    std::vector<sycl::event> attention(Batch),pack(Batch);rt->wait();
    const auto start=std::chrono::steady_clock::now();
    for(int i=0;i<Batch;++i)check(invoke(candidate,&attention[i],&pack[i]),"timed path refused");
    rt->wait();const auto end=std::chrono::steady_clock::now();
    Sample sample{std::chrono::duration<double,std::milli>(end-start).count(),0,0};
    for(int i=0;i<Batch;++i) {
      const auto duration=[](const sycl::event &event) {
        const uint64_t a=event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t b=event.get_profiling_info<sycl::info::event_profiling::command_end>();
        check(b>a,"event clock interval");return double(b-a)*1e-6;
      };
      sample.pack_ms+=duration(pack[i]);sample.attention_ms+=duration(attention[i]);
    }
    check(sample.wall_ms>=100,"ineligible batch below100ms");
    check(std::isfinite(sample.wall_ms) && std::isfinite(sample.pack_ms) && std::isfinite(sample.attention_ms),"nonfinite timing");
    check(sample.pack_ms+sample.attention_ms<=sample.wall_ms*1.01,"service exceeds enclosing in-order batch");
    check(out.get()==expected,"timed output bit parity");
    return sample;
  };
  for(int round=0;round<3;++round)for(int slot=0;slot<2;++slot) {
    const bool candidate=((process+scenario+round+slot)&1)!=0;
    const auto sample=measure(candidate);
    std::cout<<"HEAD_SAMPLE process="<<process<<" scenario="<<scenario<<" context="<<context<<" pattern="<<pattern
      <<" queries="<<Queries<<" cap="<<Cap<<" page="<<Page<<" batch="<<Batch<<" round="<<round<<" arm="<<(candidate?"candidate":"original")
      <<" wall_ms="<<sample.wall_ms<<" pack_ms="<<sample.pack_ms<<" attention_ms="<<sample.attention_ms<<" bit_parity=1\n";
  }
  k.immutable();v.immutable();ks.immutable();vs.immutable();query.immutable();table.immutable();ids.immutable();steps.immutable();
  (void)packed_k.get();(void)packed_v.get();(void)bad.get();
  std::cout<<"HEAD_CASE scenario="<<scenario<<" all_inputs_immutable=1 guards=1 bit_parity=1\n";
}
}
int main(int argc,char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);std::cout<<std::setprecision(12);
    check(argc==2,"process0..2 or --edge-only required");const bool edge_only=std::string(argv[1])=="--edge-only";
    check(edge_only || std::string(argv[1])=="0" || std::string(argv[1])=="1" || std::string(argv[1])=="2","explicit mode");
    const int process=edge_only?0:argv[1][0]-'0';gate("before-runtime");rt=sycl_backend::runtime_for();Drain pending;
    check(rt->device().get_info<sycl::info::device::name>()=="Intel(R) Arc(TM) B570 Graphics","B570 required");
    check(rt->compute().has_property<sycl::property::queue::in_order>() && rt->compute().has_property<sycl::property::queue::enable_profiling>(),"qualified profiling in-order queue");gate("runtime-ready");
    identity_edge(); // Must pass before any timing sample.
    if(edge_only) {rt->wait();gate("before-exit");std::cout<<"HEAD_EDGE_DONE passed=1\n";return 0;}
    static_assert(std::chrono::steady_clock::is_steady);
    timespec resolution{};check(clock_getres(CLOCK_MONOTONIC,&resolution)==0 && resolution.tv_sec==0 && resolution.tv_nsec<=1000,"clock resolution");
    std::vector<double> overhead;
    for(int i=0;i<10000;++i) {
      const auto a=std::chrono::steady_clock::now(),b=std::chrono::steady_clock::now();
      overhead.push_back(std::chrono::duration<double,std::nano>(b-a).count());
    }
    std::sort(overhead.begin(),overhead.end());
    check(overhead[5000]<=1000 && overhead[9900]<=10000,"clock control exceeds budget");
    std::cout<<"HEAD_CLOCK resolution_ns="<<resolution.tv_nsec<<" median_ns="<<overhead[5000]<<" p99_ns="<<overhead[9900]<<" max_ns="<<overhead.back()<<" floor_ms=100\n";
    int scenario=0;
    for(int context:{32768,117488})for(int pattern:{0,1})run_case(context,pattern,process,scenario++);
    rt->wait();gate("before-exit");std::cout<<"HEAD_BENCH_DONE passed=1 scenarios=4 samples=24\n";return 0;
  } catch(const std::exception &e) {sycl_backend::checked_cleanup([]{if(rt)rt->wait();},[]{},"headbenchmark failure drain");std::cerr<<"HEAD_BENCH_ERROR "<<e.what()<<'\n';return 1;}
}
