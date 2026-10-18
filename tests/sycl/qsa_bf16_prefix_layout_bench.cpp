// Authored for Strata; repository LICENSE (MIT) applies.
// Same numerical operations: cached Q versus four-adjacent-value K staging.
// Wall timing includes admission, prefix refresh, attention and checked drain;
// uploads/readbacks/checks are outside timing. No physical-bandwidth claim.
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> rt;
void require(bool x,const char *s) { if (!x) throw std::runtime_error(s); }
void drain() noexcept {
  sycl_backend::checked_cleanup([] { if (rt) rt->wait(); },[] {},"QSA pitch performance fixture drain");
}
struct Drain { ~Drain() noexcept { drain(); } };
template<class T> struct Buffer {
  sycl_backend::Allocation memory;
  std::vector<T> upload,download;
  explicit Buffer(size_t n):memory(rt,n*sizeof(T),sycl_backend::MemoryKind::Device) {}
  T *data() { return memory.as<T>(); }
  void put(const std::vector<T> &v) {
    require(memory.size()==v.size()*sizeof(T),"upload extent");rt->wait();upload=v;
    rt->wait(rt->compute().memcpy(data(),upload.data(),memory.size()));
  }
  const std::vector<T> &get() {
    rt->wait();download.resize(memory.size()/sizeof(T));
    rt->wait(rt->compute().memcpy(download.data(),data(),memory.size()));return download;
  }
};
uint16_t bf16(float x) {
  const uint32_t b=std::bit_cast<uint32_t>(x);
  return uint16_t((b+0x7fff+((b>>16)&1))>>16);
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
void gate(bool enabled,const char *phase) {
  if (!enabled) return;
  std::cout<<"QSA_BF16_READY "<<phase<<'\n';require(std::cin.get()=='X',"Main gate token");
}
void run(int cells,int queries,int cap,bool clustered) {
  constexpr int Dim=256,Heads=24,KV=2,Page=16;
  const int pages=(cells+Page-1)/Page,rows=pages*KV*Page;
  std::vector<int32_t> table(pages),ids(size_t(queries)*cap),steps(size_t(queries)*kStepCount);
  for (int p=0;p<pages;++p) table[p]=p%13==5 ? -1 : pages-1-p;
  std::vector<int8_t> k(size_t(rows)*Dim),v(k.size());
  for (size_t i=0;i<k.size();++i) {
    k[i]=int8_t(int((uint64_t(i)*1664525+1013904223)%255)-127);
    v[i]=int8_t(int((uint64_t(i)*1103515245+12345)%251)-125);
  }
  std::vector<uint16_t> ks(size_t(rows)*4),vs(ks.size()),q(size_t(queries)*Heads*Dim);
  for (size_t i=0;i<ks.size();++i) {
    ks[i]=half(std::ldexp(1.f,-10-int(i%4)));
    vs[i]=half(std::ldexp(i%3 ? 1.f : -1.f,-9-int(i%4)));
  }
  for (size_t i=0;i<q.size();++i) q[i]=bf16(float(int((uint64_t(i)*17+13)%127)-63)*.03125f);
  for (int b=0;b<queries;++b) {
    steps[size_t(b)*kStepCount+kStepPos]=cells-1;
    steps[size_t(b)*kStepCount+kStepNKv]=cells;
    steps[size_t(b)*kStepCount+kStepWidth]=cap;
    for (int c=0;c<cap;++c) {
      int32_t cell=clustered ? int32_t((uint64_t(b)*11+c)%cells)
          : int32_t((uint64_t(b)*8191+uint64_t(c)*7919+37)%cells);
      if (c%97==13) cell=-1;
      if (c%103==7) cell=cells+13;
      ids[size_t(b)*cap+c]=cell;
    }
  }
  const uint16_t marker=bf16(1234.f);const int32_t bad_marker=0x12345678;
  Buffer<uint16_t> dq(q.size()),dks(ks.size()),dvs(vs.size()),out(q.size()+32);
  Buffer<int8_t> dk(k.size()),dv(v.size());
  Buffer<int32_t> dt(table.size()),di(ids.size()),ds(steps.size());
  Buffer<uint16_t> pk(size_t(rows)*Dim+32),pv(size_t(rows)*Dim+32);
  Buffer<int32_t> pb(size_t(rows)+32);
  Drain pending;
  dq.put(q);dks.put(ks);dvs.put(vs);dk.put(k);dv.put(v);dt.put(table);di.put(ids);ds.put(steps);
  out.put(std::vector<uint16_t>(q.size()+32,marker));
  pk.put(std::vector<uint16_t>(size_t(rows)*Dim+32,marker));
  pv.put(std::vector<uint16_t>(size_t(rows)*Dim+32,marker));
  pb.put(std::vector<int32_t>(size_t(rows)+32,bad_marker));
  QsaAttnPools pools;pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;
  const QsaBf16PackedScratch scratch{pk.data()+16,pv.data()+16,pb.data()+16};
  auto invoke=[&](int variant) {
    rt->wait();const auto start=std::chrono::steady_clock::now();
    require((variant ? qsa_bf16_prefix_kvec4_batch : qsa_bf16_prefix_qcache_batch)(dq.data(),pools,di.data(),ds.data(),cap,shape,
        out.data()+16,queries,{pages,pages},cells,scratch,&rt->compute()),"matrix admission");
    rt->wait();return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  };
  auto guards=[&] {
    const auto &a=pk.get();const auto &b=pv.get();const auto &c=pb.get();const auto &y=out.get();
    for (size_t i=0;i<16;++i) require(a[i]==marker && a[a.size()-16+i]==marker &&
        b[i]==marker && b[b.size()-16+i]==marker && c[i]==bad_marker &&
        c[c.size()-16+i]==bad_marker && y[i]==marker && y[y.size()-16+i]==marker,"output/scratch guards");
  };
  invoke(0);guards();const auto reference=out.get();
  for (size_t i=16;i<reference.size()-16;++i)
    require(std::isfinite(std::bit_cast<float>(uint32_t(reference[i])<<16)),"baseline all finite");
  const std::array<int,2> variants{0,1};
  for (int variant:variants) {
    invoke(variant);guards();require(out.get()==reference,"warm full-output bit comparison");
  }
  for (int round=0;round<6;++round) for (int index=0;index<2;++index) {
    const int variant=variants[(round+index)%2];const double ms=invoke(variant);
    guards();require(out.get()==reference,"timed full-output bit comparison");
    std::cout<<"sample cells="<<cells<<" queries="<<queries<<" cap="<<cap
        <<" pattern="<<(clustered ? "clustered" : "gather")<<" round="<<round
        <<" order="<<index<<" variant="<<variant<<" milliseconds="<<std::setprecision(12)<<ms
        <<" compared="<<q.size()<<" exact_bits=PASS guards=PASS status=PASS\n";
  }
  require(dq.get()==q && dk.get()==k && dv.get()==v && dks.get()==ks && dvs.get()==vs &&
      dt.get()==table && di.get()==ids && ds.get()==steps,"all inputs immutable");
  std::cout<<"case cells="<<cells<<" queries="<<queries<<" cap="<<cap
      <<" pattern="<<(clustered ? "clustered" : "gather")<<" samples=12 full_bits=PASS guards=PASS inputs=PASS status=PASS\n";
}
}
int main(int argc,char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    require(argc==2 && (std::string(argv[1])=="--smoke" || std::string(argv[1])=="--component"),"--smoke|--component required");
    const char *g=std::getenv("STRATA_TEST_MAIN_GATES");const bool gated=g && std::string(g)=="1";
    gate(gated,"before-runtime");rt=sycl_backend::runtime_for();
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()
        <<" driver="<<rt->device().get_info<sycl::info::device::driver_version>()<<'\n';
    gate(gated,"runtime-ready");
    if (std::string(argv[1])=="--smoke") run(97,3,33,false);
    else { run(78453,8192,33,false);run(78453,8192,2051,false);run(78453,8192,2051,true); }
    rt->wait();rt.reset();gate(gated,"after-cleanup");
    std::cout<<"fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED\n";
    return 0;
  } catch (const std::exception &e) { drain();std::cerr<<e.what()<<'\n';return 1; }
}
