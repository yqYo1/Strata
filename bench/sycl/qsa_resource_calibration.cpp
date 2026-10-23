// Main-authored diagnostic; repository LICENSE applies.
// Synthetic masks, not a model replay. Checks and transfers stay outside timing.
#include "qsa_resource_calibration.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace cal = strata::kernels::resource_calibration;
namespace {
std::shared_ptr<sycl_backend::Runtime> rt;
void require(bool ok,const char *message) { if (!ok) throw std::runtime_error(message); }
void drain() noexcept {
  sycl_backend::checked_cleanup([] { if (rt) rt->wait(); },[] {},"QSA calibration drain");
}
struct Drain { ~Drain() noexcept { drain(); } };
template<class T> struct Buffer {
  size_t count;
  sycl_backend::Allocation allocation;
  explicit Buffer(size_t n):count(n),allocation(rt,(n+32)*sizeof(T),sycl_backend::MemoryKind::Device) {}
  T *data() { return allocation.as<T>()+16; }
  static T marker() { return T(73); }
  void put(const std::vector<T> &v) {
    require(v.size()==count,"upload extent");
    std::vector<T> a(count+32,marker());std::copy(v.begin(),v.end(),a.begin()+16);
    Drain pending;rt->wait(rt->compute().memcpy(allocation.data(),a.data(),a.size()*sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> a(count+32);Drain pending;
    rt->wait(rt->compute().memcpy(a.data(),allocation.data(),a.size()*sizeof(T)));
    for (size_t i=0;i<16;++i) require(a[i]==marker() && a[count+16+i]==marker(),"buffer guards");
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
void run_case(int cells,int queries,int cap,bool clustered,bool timing,int process) {
  constexpr int Dim=256,Page=4,Heads=24,KV=2;
  const int pages=(cells+Page-1)/Page,rows=pages*KV*Page;
  std::vector<int8_t> k(size_t(rows)*Dim),v(k.size());
  std::vector<uint16_t> ks(size_t(rows)*4),vs(ks.size()),q(size_t(queries)*Heads*Dim);
  std::vector<int32_t> table(pages),ids(size_t(queries)*cap),steps(size_t(queries)*kStepCount);
  for (size_t i=0;i<k.size();++i) { k[i]=int8_t(int((i*7+19)%251)-125);v[i]=int8_t(int((i*11+13)%255)-127); }
  for (size_t i=0;i<ks.size();++i) { ks[i]=half(std::ldexp(1.f,-8-int(i%3)));vs[i]=half(std::ldexp(1.f,-9-int(i%4))); }
  for (size_t i=0;i<q.size();++i) q[i]=bf16(float(int((i*17+23)%127)-63)*.03125f);
  for (int p=0;p<pages;++p) table[p]=p%29==17 ? -1 : pages-1-p;
  for (int b=0;b<queries;++b) {
    steps[size_t(b)*kStepCount+kStepPos]=cells-1;
    steps[size_t(b)*kStepCount+kStepNKv]=cells;
    steps[size_t(b)*kStepCount+kStepWidth]=timing ? cap : (b==0 ? 0 : b==1 ? cap : cap/2);
    for (int c=0;c<cap;++c) {
      int32_t cell=clustered ? int32_t((uint64_t(b)*11+c)%cells)
          : int32_t((uint64_t(b)*8191+uint64_t(c)*7919+37)%cells);
      if (c%97==13) cell=-1;
      if (c%103==7) cell=cells+13;
      ids[size_t(b)*cap+c]=cell;
    }
  }
  Buffer<int8_t> dk(k.size()),dv(v.size());Buffer<uint16_t> dks(ks.size()),dvs(vs.size()),dq(q.size()),out(q.size());
  Buffer<int32_t> dt(table.size()),di(ids.size()),ds(steps.size());
  Buffer<uint16_t> pk(size_t(rows)*Dim),pv(size_t(rows)*Dim);Buffer<int32_t> bad(rows);
  Buffer<uint32_t> sink(size_t(queries)*KV);
  Drain pending;
  dk.put(k);dv.put(v);dks.put(ks);dvs.put(vs);dq.put(q);dt.put(table);di.put(ids);ds.put(steps);
  out.put(std::vector<uint16_t>(q.size(),73));pk.put(std::vector<uint16_t>(size_t(rows)*Dim,73));
  pv.put(std::vector<uint16_t>(size_t(rows)*Dim,73));bad.put(std::vector<int32_t>(rows,73));
  sink.put(std::vector<uint32_t>(size_t(queries)*KV,73));
  QsaAttnPools pools;pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;
  QsaBf16PackedScratch scratch{pk.data(),pv.data(),bad.data()};
  auto invoke=[&](int arm) {
    if (arm==0) require(qsa_bf16_prefix_ordered_exp_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
        out.data(),queries,{pages,pages},cells,scratch,&rt->compute()),"production admission");
    else require(cal::run(arm==3 ? cal::Arm::control : arm==1 ? cal::Arm::load : cal::Arm::stage,
        dq.data(),pools,di.data(),ds.data(),cap,shape,out.data(),queries,{pages,pages},cells,scratch,
        &rt->compute(),sink.data()),"calibration admission");
  };
  invoke(0);rt->wait();const auto reference=out.get();invoke(3);rt->wait();
  require(out.get()==reference,"namespace control must match all output bits");
  for (auto x:reference) require(std::isfinite(std::bit_cast<float>(uint32_t(x)<<16)),"production finite output");
  auto expected=[&] {
    std::vector<uint32_t> sums(size_t(queries)*KV);
    for (int b=0;b<queries;++b) for (int h=0;h<KV;++h) {
      uint32_t sum=0;const int width=steps[size_t(b)*kStepCount+kStepWidth];
      for (int c=0;c<width;++c) {
        const int cell=ids[size_t(b)*cap+c];if (cell<0 || cell>=cells || table[cell/Page]<0) continue;
        const size_t r=(size_t(table[cell/Page])*KV+h)*Page+cell%Page;
        for (int d=0;d<Dim;++d) {
          const float sk=float(std::bit_cast<_Float16>(ks[r*4+d/64]));
          const float sv=float(std::bit_cast<_Float16>(vs[r*4+d/64]));
          sum+=bf16(float(k[r*Dim+d])*sk);sum+=bf16(float(v[r*Dim+d])*sv);
        }
      }
      sums[size_t(b)*KV+h]=sum;
    }
    return sums;
  };
  const auto checksum=expected();
  auto check=[&](int arm) {
    require(out.get()==reference,"probe must not alter production output");
    (void)pk.get();(void)pv.get();(void)bad.get();
    if (arm==1 || arm==2) require(sink.get()==checksum,"all per-workgroup producer checksums");
  };
  for (int arm=0;arm<4;++arm) { invoke(arm);rt->wait();check(arm); }
  // Correctness-only runtime-operand perturbation: both probes must observe it.
  const auto base_sink=checksum;
  const auto saved=v;
  for (auto &x:v) x=int8_t(int(x)/2);
  dv.put(v);
  const auto changed=expected();require(changed!=base_sink,"fixture perturbation must change expected sink");
  for (int arm:{1,2}) { invoke(arm);rt->wait();require(sink.get()==changed,"probe must observe perturbed V bits"); }
  v=saved;dv.put(v);
  for (int arm=0;arm<3;++arm) { invoke(arm);rt->wait();check(arm); }
  if (timing) {
    std::array<uint64_t,1000> clocks;
    for (auto &n:clocks) { auto t=std::chrono::steady_clock::now();n=elapsed(t); }
    std::sort(clocks.begin(),clocks.end());const auto clock_ns=clocks[500];
    constexpr int Calls=128;
    for (int round=0;round<5;++round) for (int order=0;order<3;++order) {
      const int arm=(process+round+order)%3;rt->wait();const auto start=std::chrono::steady_clock::now();
      for (int call=0;call<Calls;++call) { invoke(arm);rt->wait(); }
      const auto ns=elapsed(start);check(arm);
      const bool eligible=ns>=100000000 && ns>=clock_ns*1000;
      std::cout<<"sample process="<<process<<" cells="<<cells<<" queries="<<queries<<" cap="<<cap
          <<" pattern="<<(clustered ? "clustered" : "gather")<<" round="<<round<<" order="<<order
          <<" arm="<<arm<<" calls="<<Calls<<" elapsed_ns="<<ns<<" timer_pair_ns="<<clock_ns
          <<" eligible="<<eligible<<" checks=PASS\n";
      require(eligible,"predeclared timing-batch duration floor failed;no adaptive enlargement");
    }
  }
  require(dk.get()==k && dv.get()==v && dks.get()==ks && dvs.get()==vs && dq.get()==q &&
      dt.get()==table && di.get()==ids && ds.get()==steps,"all inputs immutable");
  (void)sink.get();
  std::cout<<"case cells="<<cells<<" queries="<<queries<<" cap="<<cap<<" pattern="<<(clustered ? "clustered" : "gather")
      <<" control_bits=PASS producer_checksums=PASS perturbation=PASS guards=PASS immutable=PASS\n";
}
}
int main(int argc,char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    require(argc==2 || argc==3,"--smoke | --validate | --timing PROCESS required");
    const bool timing=std::string(argv[1])=="--timing";
    const bool validation=std::string(argv[1])=="--validate";
    require(timing || validation || std::string(argv[1])=="--smoke","invalid mode");
    const int process=argc==3 ? std::stoi(argv[2]) : 0;
    require(process>=0 && process<3 && (!timing || argc==3),"process ordinal");
    rt=sycl_backend::runtime_for();
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()<<'\n';
    if (timing || validation) { run_case(32768,512,2051,false,timing,process);run_case(117488,512,2051,false,timing,process);run_case(117488,512,2051,true,timing,process); }
    else run_case(97,3,33,false,false,process);
    rt->wait();rt.reset();std::cout<<"fixture=PASS model=UNRUN physical_counters=UNMEASURED\n";
    return 0;
  } catch (const std::exception &e) { drain();std::cerr<<e.what()<<'\n';return 1; }
}
