// Main-authored component qualification; repository LICENSE (MIT) applies.
// Independent host set/tensor oracles; no external implementation adapted.
#include "qsa_pair_probe.hpp"
#include "strata/kernels/qsa_pair_union.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace pair_probe { std::vector<sycl::event> events; }
namespace {
std::shared_ptr<sycl_backend::Runtime> rt;
constexpr int Guard=17,Dim=256,Heads=24,Page=16;
constexpr int32_t IntGuard=-123456789;
constexpr uint16_t WordGuard=0x4a9b;
void check(bool value,const char *what) {if (!value) throw std::runtime_error(what);}
struct Drain {
  ~Drain() noexcept {try {if (rt) rt->wait();} catch (...) {std::abort();}}
};
template<class T> struct Buffer {
  sycl_backend::Allocation owner;
  std::vector<T> host_put,host_get;
  explicit Buffer(size_t n):owner(rt,n*sizeof(T),sycl_backend::MemoryKind::Device) {}
  T *data() {return owner.as<T>();}
  void put(const std::vector<T> &v) {
    check(v.size()*sizeof(T)==owner.size(),"upload extent");
    rt->wait();host_put=v;
    rt->wait(rt->compute().memcpy(data(),host_put.data(),owner.size()));
  }
  std::vector<T> get() {
    rt->wait();host_get.resize(owner.size()/sizeof(T));
    rt->wait(rt->compute().memcpy(host_get.data(),data(),owner.size()));return host_get;
  }
};
template<class T> void guards(const std::vector<T> &v,T sentinel) {
  for (int i=0;i<Guard;++i)
    check(v[i]==sentinel && v[v.size()-Guard+i]==sentinel,"allocation guards");
}
uint16_t bf16(float x) {
  uint32_t b=std::bit_cast<uint32_t>(x);
  return std::isnan(x)?uint16_t((b>>16)|0x40):uint16_t((b+0x7fff+((b>>16)&1))>>16);
}
float from_bf16(uint16_t x) {return std::bit_cast<float>(uint32_t(x)<<16);}
uint16_t half(float x) {return std::bit_cast<uint16_t>(_Float16(x));}
float from_half(uint16_t x) {return float(std::bit_cast<_Float16>(x));}
float spacing(float x) {
  const auto b=bf16(std::fabs(x));return from_bf16(uint16_t(b+1))-from_bf16(b);
}
void gate(const char *stage) {
  std::cout<<"PAIR_GATE "<<stage<<std::endl;check(std::cin.get()=='X',"coordinator gate");
}
void properties(const char *loader_path) {
  // Queries only: no allocation or enqueue. Exactly the loaded namespace-0
  // loader; the controller checks live mappings/provider and the query trace.
  void *loader=dlopen(loader_path,RTLD_NOW|RTLD_LOCAL|RTLD_NOLOAD);
  check(loader!=nullptr,"qualified loader absent");
  const auto kp=reinterpret_cast<decltype(&zeKernelGetProperties)>(dlsym(loader,"zeKernelGetProperties"));
  const auto dp=reinterpret_cast<decltype(&zeDeviceGetComputeProperties)>(dlsym(loader,"zeDeviceGetComputeProperties"));
  check(kp && dp,"property API absent");
  ze_device_compute_properties_t device{};device.stype=ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  check(dp(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(rt->device()),&device)==ZE_RESULT_SUCCESS,
        "device property query");
  int found=0;
  for (const auto &id:sycl::get_kernel_ids()) {
    const std::string name=id.get_name();
    const bool u=name.find("qsa_pair_union_prefill")!=std::string::npos;
    const bool a=name.find("qsa_pair_attention_prefill")!=std::string::npos;
    const bool b=name.find("prefix_batch_impl")!=std::string::npos;
    if (!u && !a && !b) continue;
    const auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(rt->context(),{rt->device()},{id});
    const auto kernel=bundle.get_kernel(id);
    ze_kernel_properties_t p{};p.stype=ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    check(kp(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kernel),&p)==ZE_RESULT_SUCCESS,"kernel properties");
    // The selected prefix template also owns an ordinary64-lane pack kernel.
    if (b && p.requiredSubgroupSize!=16) continue;
    const size_t wg=a?192:128,sg=u?32:16,local=a?21856:u?0:20036;
    const auto maximum=kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(rt->device());
    check(maximum>=wg && p.requiredSubgroupSize==sg && p.maxNumSubgroups>=wg/sg,"compiled group geometry");
    check(!p.requiredGroupSizeX || p.requiredGroupSizeX==wg,"compiled WG requirement");
    check(!p.requiredNumSubGroups || p.requiredNumSubGroups==wg/sg,"compiled SG requirement");
    check(uint64_t(p.localMemSize)+local+4096<=device.maxSharedLocalMemory,"conservative SLM budget");
    std::cout<<"PAIR_PROPERTY kind="<<(u?"union":a?"attention":"selected")<<" wg="<<wg<<" sg="<<sg
        <<" maximum="<<maximum<<" args="<<p.numKernelArgs<<" static_slm="<<p.localMemSize
        <<" declared_slm="<<local<<" spill_descriptor="<<p.spillMemSize<<" name="<<name<<'\n';++found;
  }
  check(found==3,"three exact component kernels required");dlclose(loader);
}

void union_case(int cap,int nq,int pattern,int invalid=0) {
  const int n=262144;const int pairs=(nq+1)/2,stride=cap*2;
  std::vector<int32_t> ids(size_t(nq)*cap+2*Guard,IntGuard),steps(size_t(nq)*kStepCount,0);
  std::vector<std::map<int32_t,int>> oracle(pairs);
  for (int q=0;q<nq;++q) {
    const int width=pattern==0?0:pattern==1?(q%2?0:cap):cap;
    steps[q*kStepCount+kStepNKv]=n;steps[q*kStepCount+kStepNBid]=n/4;
    steps[q*kStepCount+kStepWidth]=width;
    for (int i=0;i<width;++i) {
      // Identical,disjoint,alternating and half-overlap sorted sets. Their
      // duplicates straddle both stable merge partitions and33-item chunks.
      int cell=pattern<=2?i:pattern==3?i*2+q%2:pattern==4?i+(q%2)*(cap/2):n-cap+i;
      ids[Guard+q*cap+i]=cell;oracle[q/2][cell]|=1<<(q%2);
    }
  }
  if (invalid==1) steps[kStepNBid]++;
  if (invalid==2) steps[kStepWidth]=cap+1;
  if (invalid==3) ids[Guard+cap-1]=-1;
  if (invalid==4) ids[Guard+cap-1]=n;
  if (invalid==5) ids[Guard+1]=ids[Guard];
  if (invalid==6) steps[kStepNKv]=-1;
  if (invalid==7) steps[kStepWidth]=-1;
  Buffer<int32_t> di(ids.size()),ds(steps.size()),du(size_t(pairs)*stride+2*Guard),dc(pairs+2*Guard);
  Buffer<uint8_t> dm(size_t(pairs)*stride+2*Guard);Drain pending;
  di.put(ids);ds.put(steps);du.put(std::vector<int32_t>(size_t(pairs)*stride+2*Guard,IntGuard));
  dc.put(std::vector<int32_t>(pairs+2*Guard,IntGuard));
  dm.put(std::vector<uint8_t>(size_t(pairs)*stride+2*Guard,0xa5));
  if (nq==129) {
    // Real transfer-queue input write,explicit compute dependency,no host wait
    // between queues. The Buffer owns its DMA source through checked drain.
    const auto copied=rt->transfer().memcpy(di.data(),di.host_put.data(),di.owner.size());
    rt->compute().submit([&](sycl::handler &h) {h.depends_on(copied);h.single_task([] {});});
  }
  qsa_pair_union_prefill(di.data()+Guard,ds.data(),nq,cap,du.data()+Guard,dm.data()+Guard,dc.data()+Guard,&rt->compute());
  rt->wait();const auto u=du.get(),counts=dc.get();const auto m=dm.get();
  guards(u,IntGuard);guards(counts,IntGuard);guards(m,uint8_t(0xa5));
  for (int p=0;p<pairs;++p) {
    const bool bad=invalid && p==0;const int count=bad?-1:int(oracle[p].size());
    check(counts[Guard+p]==count,"union count/oracle");int i=0;
    if (!bad) for (const auto &[cell,mask]:oracle[p]) {
      check(u[Guard+p*stride+i]==cell && m[Guard+p*stride+i]==mask,"union ID/membership oracle");++i;
    }
    for (;i<stride;++i) check(u[Guard+p*stride+i]==IntGuard && m[Guard+p*stride+i]==0xa5,"union uninitialized suffix");
  }
  check(di.get()==ids && ds.get()==steps,"union inputs immutable");
  std::cout<<"PAIR_CASE kind=union cap="<<cap<<" queries="<<nq<<" pattern="<<pattern<<" invalid="<<invalid<<" passed=1\n";
}

void attention_case(int cells,int cap,int nq,int variant) {
  const int pages=(cells+Page-1)/Page,rows=pages*2*Page,pairs=(nq+1)/2,stride=cap*2;
  const int live=variant==2?std::max(1,cells-3):cells;
  std::vector<int32_t> ids(size_t(nq)*cap,IntGuard),steps(size_t(nq)*kStepCount,0),table(pages);
  std::vector<std::map<int,int>> membership(pairs);
  for (int p=0;p<pages;++p) table[p]=variant==2 && p%7==2?-1:pages-1-p;
  for (int q=0;q<nq;++q) {
    const int width=variant==1 && q%2?0:std::min(cap,cells);
    steps[q*kStepCount+kStepPos]=cells-1;steps[q*kStepCount+kStepNKv]=cells;
    steps[q*kStepCount+kStepNBid]=cells/4;steps[q*kStepCount+kStepWidth]=width;
    // Sort a deterministic cyclic selection; no duplicate IDs within a row.
    std::vector<int> selected(cells);std::iota(selected.begin(),selected.end(),0);
    std::rotate(selected.begin(),selected.begin()+((q*7)%cells),selected.end());selected.resize(width);
    if (variant==10 && width) selected.back()=cells-1;
    std::sort(selected.begin(),selected.end());
    for (int i=0;i<width;++i) {ids[q*cap+i]=selected[i];membership[q/2][selected[i]]|=1<<(q%2);}
  }
  std::vector<int8_t> k(size_t(rows)*Dim),v(k.size());
  std::vector<uint16_t> ks(size_t(rows)*4),vs(ks.size()),query(size_t(nq)*Heads*Dim);
  for (size_t i=0;i<k.size();++i) {k[i]=int8_t(int((i*7+3)%15)-7);v[i]=int8_t(int((i*11+5)%31)-15);}
  for (size_t i=0;i<ks.size();++i) {ks[i]=half(std::ldexp(1.f,-5-int(i%4)));vs[i]=half(std::ldexp(i%3?1.f:-1.f,-7-int(i%4)));}
  for (size_t i=0;i<query.size();++i) query[i]=bf16(float(int(i*13%17)-8)*.125f);
  if (variant==3) { // Row0 is absent from query1's shifted selected set.
    const int source=(table[0]*2)*Page;vs[source*4]=0x7c00;
  }
  if (variant==4) query[0]=0x7fc0;
  if (variant>=11 && variant<=13) {
    const int cell=variant==12?7:cap+6;
    const int row=(table[cell/Page]*2)*Page+cell%Page;
    if (variant==13) ks[row*4]=0x7e00;
    else vs[row*4]=0x7c00;
  }
  const auto source_row=[&](int cell,int kv) {
    if (cell<0 || cell>=live || table[cell/Page]<0 || table[cell/Page]>=pages) return -1;
    return (table[cell/Page]*2+kv)*Page+cell%Page;
  };
  Buffer<int8_t> dk(k.size()),dv(v.size());Buffer<uint16_t> dks(ks.size()),dvs(vs.size()),dq(query.size());
  Buffer<int32_t> dt(table.size()),di(ids.size()),ds(steps.size()),du(size_t(pairs)*stride+2*Guard),dc(pairs+2*Guard);
  Buffer<uint8_t> dm(size_t(pairs)*stride+2*Guard);
  Buffer<uint16_t> pk(size_t(rows)*Dim+2*Guard),pv(size_t(rows)*Dim+2*Guard),out(query.size()+2*Guard);
  Buffer<int32_t> pb(rows+2*Guard);Drain pending;
  dk.put(k);dv.put(v);dks.put(ks);dvs.put(vs);dq.put(query);dt.put(table);di.put(ids);ds.put(steps);
  pk.put(std::vector<uint16_t>(size_t(rows)*Dim+2*Guard,WordGuard));
  pv.put(std::vector<uint16_t>(size_t(rows)*Dim+2*Guard,WordGuard));pb.put(std::vector<int32_t>(rows+2*Guard,IntGuard));
  du.put(std::vector<int32_t>(size_t(pairs)*stride+2*Guard,IntGuard));dc.put(std::vector<int32_t>(pairs+2*Guard,IntGuard));
  dm.put(std::vector<uint8_t>(size_t(pairs)*stride+2*Guard,0xa5));
  out.put(std::vector<uint16_t>(query.size()+2*Guard,WordGuard));
  QsaAttnPools pools;pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;const QsaPromptBf16Bounds bounds{pages,pages};
  const QsaBf16PackedScratch scratch{pk.data()+Guard,pv.data()+Guard,pb.data()+Guard};
  check(pair_probe::selected(dq.data(),pools,di.data(),ds.data(),cap,shape,out.data()+Guard,nq,bounds,live,scratch,false,&rt->compute()),"selected admission");
  rt->wait();const auto original=out.get(),packedk=pk.get(),packedv=pv.get();const auto packedbad=pb.get();
  out.put(std::vector<uint16_t>(query.size()+2*Guard,WordGuard));
  // No host readback between GPU metadata production and attention consumption.
  if (variant==5 || variant==6)
    rt->compute().fill(dc.data()+Guard,variant==5?-1:stride+1,pairs);
  else qsa_pair_union_prefill(di.data(),ds.data(),nq,cap,du.data()+Guard,dm.data()+Guard,dc.data()+Guard,&rt->compute());
  if (variant==7 || variant==14) rt->compute().fill(dm.data()+Guard,uint8_t(variant==7?0:4),1);
  if (variant==8) {
    auto ptr=du.data()+Guard;rt->compute().single_task([=] {ptr[1]=ptr[0];});
  }
  if (variant==9 || variant==15) rt->compute().fill(du.data()+Guard,variant==9?cells:-1,1);
  if (variant==16) rt->compute().fill(dm.data()+Guard+(pairs-1)*stride,uint8_t(2),1);
  check(qsa_pair_attention_prefill(dq.data(),pools,di.data(),ds.data(),cap,shape,out.data()+Guard,nq,bounds,live,scratch,
      du.data()+Guard,dm.data()+Guard,dc.data()+Guard,&rt->compute()),"pair attention admission");
  rt->wait();const auto result=out.get();guards(result,WordGuard);guards(original,WordGuard);
  double maximum=0;size_t compared=0,nans=0;
  for (int q=0;q<nq;++q) for (int h=0;h<Heads;++h) {
    const int kv=h/12;bool live_query=false,bad=false;
    for (int i=0;i<steps[q*kStepCount+kStepWidth];++i) {
      const int row=source_row(ids[q*cap+i],kv);if (row<0) continue;live_query=true;
      for (int g=0;g<4;++g) bad|=!std::isfinite(from_half(ks[row*4+g])) || !std::isfinite(from_half(vs[row*4+g]));
    }
    for (int d=0;d<Dim;++d) bad|=!std::isfinite(from_bf16(query[(q*Heads+h)*Dim+d]));
    const bool malformed=(variant==5 || variant==6) ||
        ((variant==7 || variant==8 || variant==9 || variant==14 || variant==15) && q<2) ||
        (variant==16 && q==nq-1);
    const bool nan=malformed || (live_query && bad);
    std::vector<int> cells_union;for (const auto &[cell,mask]:membership[q/2]) cells_union.push_back(cell);
    float m=-INFINITY,norm=0.f;std::vector<float> acc(Dim,0.f);
    if (!nan) for (size_t first=0;first<cells_union.size();first+=32) {
      float next=m;std::vector<float> scores(32,-INFINITY),prob(32,0.f);
      for (int c=0;c<32 && first+c<cells_union.size();++c) {
        const int cell=cells_union[first+c];if (!(membership[q/2].at(cell)&(1<<(q%2)))) continue;
        const int row=source_row(cell,kv);if (row<0) continue;double dot=0;
        for (int d=0;d<Dim;++d) {
          const float x=from_bf16(bf16((from_bf16(query[(q*Heads+h)*Dim+d])*.0625f)*1.4426950408f));
          const float key=from_bf16(bf16(float(k[row*Dim+d])*from_half(ks[row*4+d/64])));
          dot+=double(x)*key;
        }
        scores[c]=float(dot);next=std::max(next,scores[c]);
      }
      const float alpha=norm>0?std::exp2(m-next):0.f;float sum=0;
      for (int c=0;c<32;++c) {prob[c]=std::isfinite(scores[c])?std::exp2(scores[c]-next):0.f;sum+=prob[c];}
      for (int d=0;d<Dim;++d) {
        double value=0;
        for (int c=0;c<32 && first+c<cells_union.size();++c) if (prob[c]>0) {
          const int row=source_row(cells_union[first+c],kv);
          const float val=from_bf16(bf16(float(v[row*Dim+d])*from_half(vs[row*4+d/64])));
          value+=double(from_bf16(bf16(prob[c])))*val;
        }
        acc[d]=float(double(acc[d]*alpha)+value);
      }
      norm=std::fma(norm,alpha,sum);m=next;
    }
    for (int d=0;d<Dim;++d) {
      const float actual=from_bf16(result[Guard+(q*Heads+h)*Dim+d]);
      if (nan) {check(std::isnan(actual),"member diagnostic NaN");++nans;continue;}
      check(std::isfinite(actual),"finite nonmember output/NaN isolation");
      const float expected=norm>0?acc[d]/norm:0.f;const double error=std::fabs(double(actual)-expected);
      const double budget=spacing(expected)+2e-6;maximum=std::max(maximum,error/budget);
      check(error<=budget,"independent BF16 tensor/FP32 online oracle");++compared;
    }
  }
  check(dq.get()==query && di.get()==ids && ds.get()==steps && dt.get()==table && dk.get()==k && dv.get()==v && dks.get()==ks && dvs.get()==vs,"attention inputs immutable");
  check(pk.get()==packedk && pv.get()==packedv && pb.get()==packedbad,"attention packed operands immutable");
  guards(packedk,WordGuard);guards(packedv,WordGuard);guards(packedbad,IntGuard);
  guards(du.get(),IntGuard);guards(dm.get(),uint8_t(0xa5));guards(dc.get(),IntGuard);
  std::cout<<"PAIR_CASE kind=attention cells="<<cells<<" cap="<<cap<<" queries="<<nq<<" variant="<<variant
      <<" compared="<<compared<<" nans="<<nans<<" max_budget_ratio="<<maximum<<" passed=1\n";
}

void benchmark(const char *fixture,int first_arm) {
  check(first_arm==0 || first_arm==1,"benchmark first arm");
  std::ifstream input(fixture,std::ios::binary);check(bool(input),"captured fixture open");
  int32_t header[4];input.read(reinterpret_cast<char *>(header),sizeof(header));
  check(bool(input) && header[0]==0 && header[1]==32768 && header[2]==6491 && header[3]==2051,
        "exact selected layer0tail header");
  const int nq=header[2],cap=header[3],live=header[1]+nq;
  const int pages=(live+Page-1)/Page,rows=pages*2*Page,pairs=(nq+1)/2,stride=cap*2;
  std::vector<int32_t> ids(size_t(nq)*cap),steps(size_t(nq)*kStepCount,0),table(pages);
  input.read(reinterpret_cast<char *>(ids.data()),std::streamsize(ids.size()*4));
  check(bool(input) && input.peek()==EOF,"exact initialized captured payload");
  for (int q=0;q<nq;++q) {
    const int n=header[1]+q+1;
    steps[q*kStepCount+kStepNKv]=n;steps[q*kStepCount+kStepNBid]=n/4;
    steps[q*kStepCount+kStepWidth]=cap;steps[q*kStepCount+kStepPos]=n-1;
    for (int i=0;i<cap;++i) check(ids[q*cap+i]>=0 && ids[q*cap+i]<n &&
        (!i || ids[q*cap+i]>ids[q*cap+i-1]),"captured sorted unique causal IDs");
  }
  for (int p=0;p<pages;++p) table[p]=pages-1-p;
  // Constant finite V gives an exact all-output oracle for timing; nonuniform
  // Q/K/V and per-query membership/rounding are covered by the separate gate.
  std::vector<int8_t> k(size_t(rows)*Dim,0),v(k.size(),1);
  std::vector<uint16_t> ks(size_t(rows)*4,half(.0625f)),vs(ks),query(size_t(nq)*Heads*Dim,0);
  Buffer<int8_t> dk(k.size()),dv(v.size());Buffer<uint16_t> dks(ks.size()),dvs(vs.size()),dq(query.size());
  Buffer<int32_t> dt(table.size()),di(ids.size()),ds(steps.size()),du(size_t(pairs)*stride+2*Guard),dc(pairs+2*Guard);
  Buffer<uint8_t> dm(size_t(pairs)*stride+2*Guard);
  Buffer<uint16_t> pk(size_t(rows)*Dim+2*Guard),pv(size_t(rows)*Dim+2*Guard),out(query.size()+2*Guard);
  Buffer<int32_t> pb(rows+2*Guard);Drain pending;
  dk.put(k);dv.put(v);dks.put(ks);dvs.put(vs);dq.put(query);dt.put(table);di.put(ids);ds.put(steps);
  pk.put(std::vector<uint16_t>(size_t(rows)*Dim+2*Guard,WordGuard));
  pv.put(std::vector<uint16_t>(size_t(rows)*Dim+2*Guard,WordGuard));pb.put(std::vector<int32_t>(rows+2*Guard,IntGuard));
  du.put(std::vector<int32_t>(size_t(pairs)*stride+2*Guard,IntGuard));
  dm.put(std::vector<uint8_t>(size_t(pairs)*stride+2*Guard,0xa5));dc.put(std::vector<int32_t>(pairs+2*Guard,IntGuard));
  out.put(std::vector<uint16_t>(query.size()+2*Guard,WordGuard));
  QsaAttnPools pools;pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;const QsaPromptBf16Bounds bounds{pages,pages};
  const QsaBf16PackedScratch scratch{pk.data()+Guard,pv.data()+Guard,pb.data()+Guard};
  auto invoke=[&](int arm) {
    check(pair_probe::selected(dq.data(),pools,di.data(),ds.data(),cap,shape,out.data()+Guard,nq,bounds,live,scratch,arm!=0,&rt->compute()),"benchmark selected/pack admission");
    if (arm) {
      qsa_pair_union_prefill(di.data(),ds.data(),nq,cap,du.data()+Guard,dm.data()+Guard,dc.data()+Guard,&rt->compute());
      check(qsa_pair_attention_prefill(dq.data(),pools,di.data(),ds.data(),cap,shape,out.data()+Guard,nq,bounds,live,scratch,
          du.data()+Guard,dm.data()+Guard,dc.data()+Guard,&rt->compute()),"benchmark pair admission");
    }
  };
  auto output_oracle=[&] {
    const auto y=out.get();guards(y,WordGuard);
    for (size_t i=Guard;i<y.size()-Guard;++i) check(y[i]==bf16(.0625f),"every timed output exact constant-V oracle");
  };
  for (int arm:{first_arm,1-first_arm}) {pair_probe::events.clear();invoke(arm);rt->wait();output_oracle();}
  auto metadata_oracle=[&] {
    const auto u=du.get(),counts=dc.get();const auto m=dm.get();
    guards(u,IntGuard);guards(counts,IntGuard);guards(m,uint8_t(0xa5));
    for (int p=0;p<pairs;++p) {
      std::map<int32_t,int> expected;
      for (int q=0;q<2 && p*2+q<nq;++q) for (int i=0;i<cap;++i)
        expected[ids[(p*2+q)*cap+i]]|=1<<q;
      check(counts[Guard+p]==int(expected.size()),"actual captured union counts");int i=0;
      for (const auto &[cell,mask]:expected) {
        check(u[Guard+p*stride+i]==cell && m[Guard+p*stride+i]==mask,"actual captured union ID/mask");++i;
      }
      for (;i<stride;++i) check(u[Guard+p*stride+i]==IntGuard && m[Guard+p*stride+i]==0xa5,"captured union suffix untouched");
    }
  };
  metadata_oracle();
  constexpr int Repeat=4,Rounds=6;
  for (int round=0;round<Rounds;++round) for (int order=0;order<2;++order) {
    const int arm=(first_arm+round+order)%2;pair_probe::events.clear();
    const auto begin=std::chrono::steady_clock::now();
    for (int r=0;r<Repeat;++r) invoke(arm);
    rt->wait();const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    check(pair_probe::events.size()==size_t(Repeat*(arm?3:2)),"actual event count per dependency path");
    double service=0,pack=0,prep=0,attention=0;uint64_t first=0,last=0;
    for (size_t i=0;i<pair_probe::events.size();++i) {
      const auto &event=pair_probe::events[i];
      const auto start=event.get_profiling_info<sycl::info::event_profiling::command_start>();
      const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
      check(end>=start && (!i || start>=last),"sequential actual event clock");if (!i) first=start;last=end;
      const double seconds=double(end-start)*1e-9;service+=seconds;
      const size_t slot=i%size_t(arm?3:2);
      if (!slot) pack+=seconds;else if (arm && slot==1) prep+=seconds;else attention+=seconds;
    }
    check(wall>=.1 && service>=.1,"timing batch floor100ms;new plan required if short");
    output_oracle();
    std::cout<<"PAIR_SAMPLE round="<<round<<" order="<<order<<" arm="<<arm<<" repeat="<<Repeat
        <<" wall="<<wall<<" service="<<service<<" span="<<double(last-first)*1e-9
        <<" pack="<<pack<<" prep="<<prep<<" attention="<<attention<<" passed=1\n";
  }
  metadata_oracle();
  check(dq.get()==query && di.get()==ids && ds.get()==steps && dt.get()==table && dk.get()==k && dv.get()==v && dks.get()==ks && dvs.get()==vs,"benchmark inputs immutable");
  guards(pk.get(),WordGuard);guards(pv.get(),WordGuard);guards(pb.get(),IntGuard);
}
}
int main(int argc,char **argv) {
  try {
    check(argc>=2,"use properties <loader> | union | small | correctness");
    const std::string mode=argv[1];gate("before-runtime");rt=sycl_backend::runtime_for();Drain pending;
    check(rt->device().get_info<sycl::info::device::name>()=="Intel(R) Arc(TM) B570 Graphics","qualified device name");
    gate("runtime-ready");pair_probe::events.reserve(4096);std::cout<<std::setprecision(10);
    if (mode=="properties") {check(argc==3,"pinned loader required");properties(argv[2]);}
    else if (mode=="union") {
      for (int cap:{1,2,33,65,2051}) for (int p=0;p<6;++p) union_case(cap,3,p);
      for (int invalid=1;invalid<=7;++invalid) union_case(2051,3,2,invalid);
      union_case(2051,129,4);
    } else if (mode=="small") {
      attention_case(1,1,1,0);attention_case(33,17,3,0);attention_case(65,33,3,2);
      attention_case(65,33,3,1);attention_case(65,33,3,3);attention_case(65,33,3,4);
      attention_case(65,33,3,5);attention_case(65,33,3,6);
    } else if (mode=="correctness") {
      for (int cells:{15,32,33,257,4097}) attention_case(cells,std::min(2051,cells),3,0);
      for (int variant:{1,2,3,4,7,8,9,11,12,13,14,15,16}) attention_case(257,65,3,variant);
      attention_case(262144,2051,3,10);
    } else if (mode=="bench") {
      check(argc==4,"bench requires pinned captured IDs and first arm");benchmark(argv[2],std::stoi(argv[3]));
    } else throw std::runtime_error("unknown pair fixture mode");
    rt->wait();std::cout<<"PAIR_DONE mode="<<mode<<" passed=1\n";gate("before-exit");return 0;
  } catch (const std::exception &e) {std::cerr<<"PAIR_ERROR "<<e.what()<<'\n';return 1;}
}
