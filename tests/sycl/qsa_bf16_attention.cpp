// Authored for Strata; repository LICENSE (MIT) applies.
// Independent host tensor oracle for the configured BF16-operand/FP32-state
// online attention contract. It does not reproduce subgroup/SLM indexing.
#include "strata/kernels/qsa_bf16_attention.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> rt;
void require(bool x, const char *message) { if (!x) throw std::runtime_error(message); }
void drain() noexcept {
  sycl_backend::checked_cleanup([] { if (rt) rt->wait(); }, [] {}, "BF16 operand fixture drain");
}
struct Drain { ~Drain() noexcept { drain(); } };
template <class T> struct Buffer {
  sycl_backend::Allocation memory;
  // DMA spans must outlive a throwing wait and the outer checked drain.
  std::vector<T> upload, download;
  explicit Buffer(size_t n) : memory(rt,n*sizeof(T),sycl_backend::MemoryKind::Device) {}
  T *data() { return memory.as<T>(); }
  void put(const std::vector<T> &v) {
    require(memory.size()==v.size()*sizeof(T),"upload extent");
    rt->wait(); upload=v;
    rt->wait(rt->compute().memcpy(data(),upload.data(),memory.size()));
  }
  std::vector<T> get() {
    rt->wait(); download.resize(memory.size()/sizeof(T));
    rt->wait(rt->compute().memcpy(download.data(),data(),memory.size())); return download;
  }
};
uint16_t round_bf16(float x) {
  const uint32_t b=std::bit_cast<uint32_t>(x);
  if (std::isnan(x)) return uint16_t((b>>16)|0x40);
  return uint16_t((b+0x7fff+((b>>16)&1))>>16);
}
float load_bf16(uint16_t b) { return std::bit_cast<float>(uint32_t(b)<<16); }
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
float load_half(uint16_t b) { return float(std::bit_cast<_Float16>(b)); }
float spacing(float x) {
  const uint16_t b=round_bf16(std::fabs(x));
  return load_bf16(uint16_t(b+1))-load_bf16(b);
}
void gate(bool enabled,const char *phase) {
  if (!enabled) return;
  std::cout<<"QSA_BF16_READY "<<phase<<'\n';
  require(std::cin.get()=='X',"Main gate token");
}

void run(int cells,int cap,int queries,bool masked,bool extended) {
  constexpr int Heads=24,KV=2,Dim=256,Page=16;
  const int pages=(cells+Page-1)/Page, rows=pages*KV*Page;
  std::vector<int32_t> table(pages),ids(size_t(queries)*cap),steps(size_t(queries)*kStepCount);
  for (int p=0;p<pages;++p)
    table[p]=masked && p%7==2 ? -1 : masked && p%7==3 ? pages+9 : pages-1-p;
  std::vector<int8_t> k(size_t(rows)*Dim),v(k.size());
  for (size_t i=0;i<k.size();++i) { k[i]=int8_t(int((i*7+3)%15)-7); v[i]=int8_t(int((i*11+5)%31)-15); }
  std::vector<uint16_t> ks(size_t(rows)*4),vs(ks.size()),q(size_t(queries)*Heads*Dim);
  for (size_t i=0;i<ks.size();++i) {
    ks[i]=half(std::ldexp(1.f,-5-int(i%4)));
    vs[i]=half(std::ldexp(i%3 ? 1.f : -1.f,-7-int(i%4)));
  }
  for (size_t i=0;i<q.size();++i) q[i]=round_bf16(float(int(i*13%17)-8)*.125f);
  for (int b=0;b<queries;++b) {
    steps[b*kStepCount+kStepPos]=cells-1;
    steps[b*kStepCount+kStepNKv]=cells;
    steps[b*kStepCount+kStepWidth]=cap;
    for (int c=0;c<cap;++c) ids[b*cap+c]=(c*17+b*11)%cells;
    ids[b*cap]=cells-1;
    if (cap>1) ids[b*cap+1]=0;
    if (cap>5) ids[b*cap+5]=-1;
  }
  const uint16_t guard=round_bf16(1234.f);
  const std::vector<uint16_t> sentinel(q.size()+32,guard);
  Buffer<uint16_t> dq(q.size()),dks(ks.size()),dvs(vs.size()),out(sentinel.size());
  Buffer<int8_t> dk(k.size()),dv(v.size());
  Buffer<int32_t> dt(table.size()),di(ids.size()),ds(steps.size());
  Drain pending; // Destroy before every input/output owner on success and failure.
  dq.put(q);dks.put(ks);dvs.put(vs);dk.put(k);dv.put(v);dt.put(table);di.put(ids);ds.put(steps);out.put(sentinel);
  QsaAttnPools pools; pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;
  auto invoke=[&] {
    require(qsa_bf16_attention_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
        out.data()+16,queries,{pages,pages},&rt->compute()),"BF16 matrix unavailable");
    rt->wait();
  };
  auto guarded=[&] {
    auto y=out.get();
    for (size_t i=0;i<16;++i) require(y[i]==guard && y[y.size()-16+i]==guard,"output guard");
    return y;
  };
  auto immutable=[&] {
    require(dq.get()==q && dks.get()==ks && dvs.get()==vs && dk.get()==k && dv.get()==v &&
            dt.get()==table && di.get()==ids && ds.get()==steps,"all input bytes immutable");
  };
  auto row_of=[&](int cell,int head) -> int64_t {
    if (cell<0 || cell>=pages*Page) return -1;
    const int physical=table[cell/Page]; if (physical<0 || physical>=pages) return -1;
    return (int64_t(physical)*KV+head/(Heads/KV))*Page+cell%Page;
  };
  invoke();const auto y=guarded();immutable();
  double max_error=0,max_budget_ratio=0;size_t compared=0;
  for (int b=0;b<queries;++b) for (int h=0;h<Heads;++h) {
    // Scalar dense dot products and tensor vectors, independent of GPU tiling.
    std::vector<float> scores(cap,-INFINITY),values(size_t(cap)*Dim),acc(Dim,0.f);
    for (int c=0;c<cap;++c) {
      const int64_t row=row_of(ids[b*cap+c],h);if (row<0) continue;
      double dot=0;
      for (int d=0;d<Dim;++d) {
        float query=load_bf16(q[(b*Heads+h)*Dim+d])*.0625f;
        query=load_bf16(round_bf16(query*1.4426950408f));
        const float key=load_bf16(round_bf16(float(k[row*Dim+d])*load_half(ks[row*4+d/64])));
        dot+=double(query)*key;
        values[size_t(c)*Dim+d]=load_bf16(round_bf16(float(v[row*Dim+d])*load_half(vs[row*4+d/64])));
      }
      scores[c]=float(dot);
    }
    float m=-INFINITY,norm=0;
    // Probability rounding is a part of the reference's online numerical
    // contract. This host loop has no device addressing or matrix fragments.
    for (int first=0;first<cap;first+=32) {
      const int last=std::min(cap,first+32);float next=m;
      for (int c=first;c<last;++c) next=std::fmax(next,scores[c]);
      const float alpha=norm>0 ? std::exp2(m-next) : 0.f;float sum=0;
      std::vector<float> probability(size_t(last-first));
      for (int c=first;c<last;++c) {
        const float p=std::isfinite(scores[c]) ? std::exp2(scores[c]-next) : 0.f;
        sum+=p;probability[c-first]=load_bf16(round_bf16(p));
      }
      for (int d=0;d<Dim;++d) {
        // FP64 sum avoids reproducing DPAS reduction order. Only the sum
        // rounds to FP32; a final BF16 ULP covers its FP32-order difference.
        double term=0;for (int c=first;c<last;++c) term+=double(probability[c-first])*values[size_t(c)*Dim+d];
        acc[d]=float(double(acc[d]*alpha)+term);
      }
      norm=std::fma(norm,alpha,sum);m=next;
    }
    for (int d=0;d<Dim;++d) {
      const float ref=norm>0 ? load_bf16(round_bf16(acc[d]/norm)) : 0.f;
      const float actual=load_bf16(y[16+(b*Heads+h)*Dim+d]);
      require(std::isfinite(actual),"complete output finite");
      const double error=std::fabs(double(actual)-ref);
      const double budget=double(spacing(ref))+2e-6;
      max_error=std::max(max_error,error);max_budget_ratio=std::max(max_budget_ratio,error/budget);
      require(error<=budget,"BF16 operand contract oracle");++compared;
    }
  }
  invoke();require(guarded()==y,"unchanged input repeat deterministic");immutable();
  if (extended) {
    const auto saved_table=table;std::fill(table.begin(),table.end(),-1);dt.put(table);invoke();
    auto z=guarded();for (size_t i=16;i<z.size()-16;++i) require(z[i]==0,"all missing pages zero");immutable();
    table=saved_table;dt.put(table);
    const auto saved_steps=steps;
    for (int width : {0,-1,cap+1}) {
      for (int b=0;b<queries;++b) steps[b*kStepCount+kStepWidth]=width;
      ds.put(steps);invoke();z=guarded();for (size_t i=16;i<z.size()-16;++i) require(z[i]==0,"invalid width zero");immutable();
    }
    steps=saved_steps;ds.put(steps);
    const auto saved_q=q;q[0]=0x7fc1;dq.put(q);invoke();z=guarded();
    for (int d=0;d<Dim;++d) require(std::isnan(load_bf16(z[16+d])),"NaN query diagnostic");
    immutable();
    q=saved_q;dq.put(q);
    const auto saved_vs=vs;std::fill(vs.begin(),vs.end(),uint16_t(0x7c00));dvs.put(vs);invoke();z=guarded();
    for (size_t i=16;i<z.size()-16;++i) require(std::isnan(load_bf16(z[i])),"live invalid V scale diagnostic");
    immutable();vs=saved_vs;dvs.put(vs);
    const auto saved_ks=ks;std::fill(ks.begin(),ks.end(),uint16_t(0x7c00));dks.put(ks);invoke();z=guarded();
    for (size_t i=16;i<z.size()-16;++i) require(std::isnan(load_bf16(z[i])),"live invalid K scale diagnostic");
    immutable();ks=saved_ks;dks.put(ks);invoke();require(guarded()==y,"recover original output");immutable();
    bool rejected=false;
    try { (void)qsa_bf16_attention_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,dq.data(),queries,{pages,pages},&rt->compute()); }
    catch (const std::invalid_argument &) { rejected=true; }
    require(rejected,"output alias rejected before submission");
    require(!qsa_bf16_attention_batch(dq.data(),pools,di.data(),ds.data(),0,shape,out.data()+16,queries,{pages,pages},&rt->compute()),"unsupported cap refused");
    immutable();require(guarded()==y,"rejected submissions leave output intact");
  }
  std::cout<<"case cells="<<cells<<" cap="<<cap<<" queries="<<queries<<" masked="<<masked
           <<" compared="<<compared<<" max_abs="<<std::setprecision(10)<<max_error
           <<" max_budget_ratio="<<max_budget_ratio<<" oracle=BF16_operands_FP32_online"
           <<" guards=PASS inputs=PASS status=PASS\n";
}
} // namespace

int main(int argc,char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    require(argc==2 && (std::string(argv[1])=="--smoke" || std::string(argv[1])=="--component"),"--smoke|--component required");
    const char *g=std::getenv("STRATA_TEST_MAIN_GATES");const bool gates=g && std::string(g)=="1";
    gate(gates,"before-runtime");rt=sycl_backend::runtime_for();
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()
             <<" driver="<<rt->device().get_info<sycl::info::device::driver_version>()<<'\n';
    gate(gates,"runtime-ready");
    run(97,33,1,false,true);run(97,33,3,true,false);
    if (std::string(argv[1])=="--component") {
      for (int width : {1,17,31,32,63,65,2051}) run(2113,width,3,true,false);
      run(262144,2051,1,false,false);
    }
    rt->wait();rt.reset();gate(gates,"after-cleanup");
    std::cout<<"fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED\n";
    return 0;
  } catch (const std::exception &e) { drain();std::cerr<<e.what()<<'\n';return 1; }
}
