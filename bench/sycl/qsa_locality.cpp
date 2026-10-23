// Main-authored Strata diagnostic; repository LICENSE applies.
// Buffer ownership follows the independently authored fixture at234cd17d8.
// Identical production code/operands/order; only logical KV addresses differ.
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> rt;
void require(bool ok,const char *text) { if (!ok) throw std::runtime_error(text); }
void drain() noexcept {
  sycl_backend::checked_cleanup([] { if (rt) rt->wait(); },[] {},"QSA locality drain");
}
struct Drain { ~Drain() noexcept { drain(); } };
template<class T> struct Buffer {
  size_t count;
  sycl_backend::Allocation allocation;
  explicit Buffer(size_t n):count(n),allocation(rt,(n+32)*sizeof(T),sycl_backend::MemoryKind::Device) {}
  T *data() { return allocation.as<T>()+16; }
  void put(const std::vector<T> &v) {
    require(v.size()==count,"upload extent");
    std::vector<T> a(count+32,T(73));std::copy(v.begin(),v.end(),a.begin()+16);
    Drain pending;rt->wait(rt->compute().memcpy(allocation.data(),a.data(),a.size()*sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> a(count+32);Drain pending;
    rt->wait(rt->compute().memcpy(a.data(),allocation.data(),a.size()*sizeof(T)));
    for (size_t i=0;i<16;++i) require(a[i]==T(73) && a[count+16+i]==T(73),"buffer guards");
    return {a.begin()+16,a.end()-16};
  }
};
uint16_t half(float v) { return std::bit_cast<uint16_t>(_Float16(v)); }
uint16_t bf16(float v) {
  const auto b=std::bit_cast<uint32_t>(v);return uint16_t((b+0x7fff+((b>>16)&1))>>16);
}
uint64_t elapsed(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t).count();
}
void run_case(int cells,int queries,int cap,bool uniform,bool timing,int process) {
  constexpr int Dim=256,Page=4,KV=2,Heads=24;
  const int pages=(cells+Page-1)/Page,rows=pages*KV*Page,blocks=cells/32;
  require(blocks>=3 && cap<cells,"finite supported fixture geometry");
  std::vector<int8_t> k(size_t(rows)*Dim),v(k.size());
  std::vector<uint16_t> ks(size_t(rows)*4),vs(ks.size()),q(size_t(queries)*Heads*Dim);
  std::vector<int32_t> table(pages),steps(size_t(queries)*kStepCount);
  std::array<std::vector<int32_t>,3> masks;
  for (auto &m:masks) m.resize(size_t(queries)*cap);
  // Identity physical pages and a32-cell payload period keep every ordered
  // QK/PV operand bit identical when addresses move by multiples of32.
  for (int row=0;row<rows;++row) {
    const int cell=(row/(KV*Page))*Page+row%Page,head=row/Page%KV;
    for (int d=0;d<Dim;++d) {
      k[size_t(row)*Dim+d]=int8_t(((cell%32)*7+head*19+d*11)%251-125);
      v[size_t(row)*Dim+d]=int8_t(((cell%32)*13+head*23+d*17)%255-127);
    }
    for (int g=0;g<4;++g) {
      ks[size_t(row)*4+g]=half(std::ldexp(1.f,-8-g));
      vs[size_t(row)*4+g]=half(std::ldexp(1.f,-9-g));
    }
  }
  for (size_t i=0;i<q.size();++i) q[i]=bf16(float(int((i*17+23)%127)-63)*.03125f);
  for (int p=0;p<pages;++p) table[p]=p;
  for (int b=0;b<queries;++b) {
    steps[size_t(b)*kStepCount+kStepPos]=cells-1;
    steps[size_t(b)*kStepCount+kStepNKv]=cells;
    steps[size_t(b)*kStepCount+kStepWidth]=uniform ? cap : (b==0 ? 0 : b==1 ? cap : cap/2);
    for (int c=0;c<cap;++c) {
      masks[0][size_t(b)*cap+c]=c;
      masks[1][size_t(b)*cap+c]=int32_t(((uint64_t(b)*11+c/32)%blocks)*32+c%32);
      masks[2][size_t(b)*cap+c]=int32_t(((uint64_t(b)*8191+uint64_t(c)*7919)%blocks)*32+c%32);
    }
    for (const auto &m:masks) {
      std::set<int32_t> seen;
      for (int c=0;c<cap;++c) {
        const auto id=m[size_t(b)*cap+c];
        require(id>=0 && id<cells && id%32==c%32 && seen.insert(id).second,
                "valid unique ordered-equivalent payload IDs");
      }
    }
  }
  std::array<size_t,3> unions{};
  for (int arm=0;arm<3;++arm) unions[arm]=std::set<int32_t>(masks[arm].begin(),masks[arm].end()).size();
  Buffer<int8_t> dk(k.size()),dv(v.size());Buffer<uint16_t> dks(ks.size()),dvs(vs.size()),dq(q.size()),out(q.size());
  Buffer<int32_t> dt(table.size()),ds(steps.size());
  std::array<std::unique_ptr<Buffer<int32_t>>,3> di;
  for (auto &p:di) p=std::make_unique<Buffer<int32_t>>(size_t(queries)*cap);
  Buffer<uint16_t> pk(size_t(rows)*Dim),pv(size_t(rows)*Dim);Buffer<int32_t> bad(rows);
  Drain pending;
  dk.put(k);dv.put(v);dks.put(ks);dvs.put(vs);dq.put(q);dt.put(table);ds.put(steps);
  for (int arm=0;arm<3;++arm) di[arm]->put(masks[arm]);
  out.put(std::vector<uint16_t>(q.size(),73));pk.put(std::vector<uint16_t>(size_t(rows)*Dim,73));
  pv.put(std::vector<uint16_t>(size_t(rows)*Dim,73));bad.put(std::vector<int32_t>(rows,73));
  QsaAttnPools pools;pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;
  QsaBf16PackedScratch scratch{pk.data(),pv.data(),bad.data()};
  auto invoke=[&](int arm) {
    require(qsa_bf16_prefix_ordered_exp_batch(dq.data(),pools,di[arm]->data(),ds.data(),cap,shape,
        out.data(),queries,{pages,pages},cells,scratch,&rt->compute()),"production admission");
    rt->wait();
  };
  invoke(0);const auto reference=out.get();
  for (auto x:reference) require(std::isfinite(std::bit_cast<float>(uint32_t(x)<<16)),"finite output");
  auto check=[&] { require(out.get()==reference,"all production output bits across address patterns"); };
  for (int arm=0;arm<3;++arm) { invoke(arm);check(); }
  const auto saved=v;for (auto &x:v) x=int8_t(int(x)/2);dv.put(v);
  invoke(0);const auto perturbed=out.get();require(perturbed!=reference,"V perturbation must change observable output");
  for (int arm=1;arm<3;++arm) { invoke(arm);require(out.get()==perturbed,"all patterns observe identical V perturbation"); }
  v=saved;dv.put(v);
  for (int arm=0;arm<3;++arm) { invoke(arm);check(); }
  if (timing) {
    std::array<uint64_t,1000> clocks;
    for (auto &n:clocks) { auto t=std::chrono::steady_clock::now();n=elapsed(t); }
    std::sort(clocks.begin(),clocks.end());const auto clock_ns=clocks[500];
    constexpr int Calls=128;
    for (int round=0;round<5;++round) for (int order=0;order<3;++order) {
      const int arm=(process+round+order)%3;
      // Equal warmups after each switch;readback/validation excluded from timing.
      invoke(arm);invoke(arm);const auto start=std::chrono::steady_clock::now();
      for (int call=0;call<Calls;++call) invoke(arm);
      const auto ns=elapsed(start);check();const bool eligible=ns>=100000000 && ns>=clock_ns*1000;
      std::cout<<"sample process="<<process<<" arm="<<arm<<" round="<<round<<" order="<<order
          <<" calls="<<Calls<<" elapsed_ns="<<ns<<" timer_pair_ns="<<clock_ns<<" eligible="<<eligible<<" checks=PASS\n";
      require(eligible,"timing floor failed;no adaptive enlargement");
    }
  }
  require(dk.get()==k && dv.get()==v && dks.get()==ks && dvs.get()==vs && dq.get()==q &&
      dt.get()==table && ds.get()==steps,"immutable common inputs");
  for (int arm=0;arm<3;++arm) require(di[arm]->get()==masks[arm],"immutable IDs");
  (void)pk.get();(void)pv.get();(void)bad.get();
  std::cout<<"case cells="<<cells<<" queries="<<queries<<" cap="<<cap<<" uniform="<<uniform
      <<" output_bits=PASS perturbation=PASS guards=PASS immutable=PASS unique_ids=PASS unions="
      <<unions[0]<<','<<unions[1]<<','<<unions[2]<<'\n';
}
}
int main(int argc,char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);require(argc==2 || argc==3,"--smoke | --validate | --timing PROCESS");
    const std::string mode=argv[1];const bool timing=mode=="--timing";
    require(mode=="--smoke" || mode=="--validate" || timing,"invalid mode");
    const int process=argc==3 ? std::stoi(argv[2]) : 0;
    require(process>=0 && process<3 && (!timing || argc==3),"process ordinal");
    rt=sycl_backend::runtime_for();
    if (mode=="--smoke") run_case(97,3,33,false,false,process);
    else if (timing) run_case(117488,512,2051,true,true,process);
    else { run_case(117488,512,2051,false,false,process);run_case(117488,512,2051,true,false,process); }
    rt->wait();rt.reset();std::cout<<"fixture=PASS model=UNRUN physical_counters=UNMEASURED\n";return 0;
  } catch (const std::exception &e) { drain();std::cerr<<e.what()<<'\n';return 1; }
}
