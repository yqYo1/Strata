// Main-authored Strata diagnostic; repository LICENSE applies.
// Replays captured selection/step rows with synthetic finite Q/K/V operands.
// This measures whole-query enumeration sensitivity, not model speed. CPU
// grouping/uploads/unpermutation are excluded and cannot justify adoption.
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace {
constexpr int Dim=256,Page=4,KV=2,Heads=24,Arms=3;
std::shared_ptr<sycl_backend::Runtime> rt;
void require(bool ok,const char *text) { if (!ok) throw std::runtime_error(text); }
void drain() noexcept {
  sycl_backend::checked_cleanup([] { if (rt) rt->wait(); },[] {},"QSA replay drain");
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
struct Record {
  int layer,first,queries,cap;
  std::vector<int32_t> ids,steps;
  int cells() const { return first+queries; }
};
// A bounded parser accepts the diagnostic's exact paired ABI and refuses
// trailing/short/duplicate/mismatched records before any GPU initialization.
bool header(std::ifstream &f,std::array<int32_t,4> &h) {
  f.read(reinterpret_cast<char*>(h.data()),sizeof(h));
  if (f.gcount()==0 && f.eof()) return false;
  require(f.gcount()==std::streamsize(sizeof(h)) && bool(f),"short capture header");return true;
}
void payload(std::ifstream &f,std::vector<int32_t> &v) {
  const auto n=std::streamsize(v.size()*sizeof(int32_t));
  f.read(reinterpret_cast<char*>(v.data()),n);
  require(f.gcount()==n && bool(f),"short capture payload");
}
void validate(const Record &r,bool actual) {
  require(r.layer>=0 && r.layer<12 && r.queries>0 && r.queries<=512 && r.cap>0 && r.cap<=2051 &&
      r.first>=0 && r.first<=262144-r.queries,"bounded capture geometry");
  require(r.ids.size()==size_t(r.queries)*r.cap && r.steps.size()==size_t(r.queries)*kStepCount,
      "capture vector extents");
  if (actual) require(r.queries==512 && r.cap==2051,"actual tail shape");
  for (int b=0;b<r.queries;++b) {
    const auto *s=r.steps.data()+size_t(b)*kStepCount;
    const int nkv=r.first+b+1,width=actual ? std::min(nkv,2051) : s[kStepWidth];
    require(s[kStepPos]==nkv-1 && s[kStepNKv]==nkv && s[kStepNBid]==nkv/4 &&
        s[kStepWidth]==width && width>=0 && width<=r.cap,"actual step contract");
    int previous=-1;
    for (int c=0;c<width;++c) {
      const auto id=r.ids[size_t(b)*r.cap+c];
      require(id>previous && id<nkv,"active IDs must be unique ascending and causal");previous=id;
    }
  }
}
std::vector<Record> load(const std::string &base) {
  std::ifstream ids(base+".ids",std::ios::binary),steps(base+".steps",std::ios::binary);
  require(bool(ids) && bool(steps),"paired capture open");
  std::vector<Record> records;std::array<int32_t,4> ih{},sh{};
  while (header(ids,ih)) {
    require(records.size()<12 && header(steps,sh),"paired bounded record count");
    require(ih[0]==int(records.size()) && ih[0]==sh[0] && ih[1]==sh[1] && ih[2]==sh[2] &&
        ih[1]>=0 && ih[2]==512 && ih[1]<=262144-ih[2] && ih[3]==2051 && sh[3]==kStepCount,
        "paired actual record headers");
    Record r{ih[0],ih[1],ih[2],ih[3],{}, {}};
    r.ids.resize(size_t(r.queries)*r.cap);r.steps.resize(size_t(r.queries)*kStepCount);
    payload(ids,r.ids);payload(steps,r.steps);validate(r,true);
    if (!records.empty()) require(r.first==records[0].first,"common final tail extent");
    records.push_back(std::move(r));
  }
  require(records.size()==12 && !header(steps,sh),"exactly twelve paired layers");return records;
}
Record smoke() {
  Record r{0,94,3,33,std::vector<int32_t>(3*33,-1),std::vector<int32_t>(3*kStepCount)};
  for (int b=0;b<3;++b) {
    auto *s=r.steps.data()+b*kStepCount;
    s[kStepPos]=r.first+b;s[kStepNKv]=s[kStepPos]+1;s[kStepNBid]=s[kStepNKv]/4;
    s[kStepWidth]=b==0 ? 0 : b==1 ? 33 : 17;
    for (int c=0;c<s[kStepWidth];++c) r.ids[b*33+c]=c*2+b;
  }
  validate(r,false);return r;
}
using Orders=std::array<std::vector<int>,Arms>;
Orders orders(const Record &r) {
  Orders o;for (auto &v:o) { v.resize(r.queries);std::iota(v.begin(),v.end(),0); }
  std::reverse(o[1].begin(),o[1].end());
  // Deterministic groups of16: choose maximal overlap with the current
  // group's union, ties by original row. CPU construction is diagnostic only.
  const size_t words=(r.cells()+63)/64;
  std::vector<uint64_t> bits(size_t(r.queries)*words),united(words);
  for (int b=0;b<r.queries;++b) for (int c=0;c<r.steps[size_t(b)*kStepCount+kStepWidth];++c) {
    const auto id=r.ids[size_t(b)*r.cap+c];bits[size_t(b)*words+id/64]|=uint64_t(1)<<(id%64);
  }
  std::vector<bool> used(r.queries);
  for (int at=0;at<r.queries;++at) {
    if (at%16==0) std::fill(united.begin(),united.end(),0);
    int best=-1,best_overlap=-1;
    for (int b=0;b<r.queries;++b) if (!used[b]) {
      int overlap=0;
      for (size_t w=0;w<words;++w) overlap+=std::popcount(united[w]&bits[size_t(b)*words+w]);
      if (overlap>best_overlap) {best=b;best_overlap=overlap;}
    }
    require(best>=0,"complete row permutation");used[best]=true;o[2][at]=best;
    for (size_t w=0;w<words;++w) united[w]|=bits[size_t(best)*words+w];
  }
  for (const auto &v:o) {auto sorted=v;std::sort(sorted.begin(),sorted.end());require(sorted==o[0],"row bijection");}
  return o;
}
template<class T> std::vector<T> permute(const std::vector<T> &v,const std::vector<int> &o,size_t width) {
  require(v.size()==o.size()*width,"permuted row extent");std::vector<T> p(v.size());
  for (size_t b=0;b<o.size();++b) std::copy_n(v.data()+size_t(o[b])*width,width,p.data()+b*width);
  return p;
}
template<class T> std::vector<T> restore(const std::vector<T> &v,const std::vector<int> &o,size_t width) {
  require(v.size()==o.size()*width,"restored row extent");std::vector<T> p(v.size());
  for (size_t b=0;b<o.size();++b) std::copy_n(v.data()+b*width,width,p.data()+size_t(o[b])*width);
  return p;
}
void run_case(const Record &r,bool timing,int process) {
  const int cells=r.cells(),pages=(cells+Page-1)/Page,rows=pages*KV*Page;
  const auto preparation=std::chrono::steady_clock::now();const auto o=orders(r);
  const auto prepare_ns=elapsed(preparation);
  std::vector<int8_t> k(size_t(rows)*Dim),v(k.size());
  std::vector<uint16_t> ks(size_t(rows)*4),vs(ks.size()),q(size_t(r.queries)*Heads*Dim);
  std::vector<int32_t> table(pages);std::iota(table.begin(),table.end(),0);
  // No periodic address remapping: preserve each logical cell's synthetic
  // payload and every original query/ID/step row in all three permutations.
  for (int row=0;row<rows;++row) {
    const int cell=(row/(KV*Page))*Page+row%Page,head=row/Page%KV;
    for (int d=0;d<Dim;++d) {
      k[size_t(row)*Dim+d]=int8_t(int((uint64_t(cell)*7+head*19+d*11)%251)-125);
      v[size_t(row)*Dim+d]=int8_t(int((uint64_t(cell)*13+head*23+d*17)%255)-127);
    }
    for (int g=0;g<4;++g) {ks[size_t(row)*4+g]=half(std::ldexp(1.f,-8-g));vs[size_t(row)*4+g]=half(std::ldexp(1.f,-9-g));}
  }
  for (size_t i=0;i<q.size();++i) q[i]=bf16(float(int((i*17+23)%127)-63)*.03125f);
  Buffer<int8_t> dk(k.size()),dv(v.size());Buffer<uint16_t> dks(ks.size()),dvs(vs.size()),dq(q.size()),out(q.size());
  Buffer<int32_t> dt(table.size()),di(r.ids.size()),ds(r.steps.size());
  Buffer<uint16_t> pk(size_t(rows)*Dim),pv(size_t(rows)*Dim);Buffer<int32_t> bad(rows);
  Drain pending;
  dk.put(k);dv.put(v);dks.put(ks);dvs.put(vs);dt.put(table);
  out.put(std::vector<uint16_t>(q.size(),73));pk.put(std::vector<uint16_t>(size_t(rows)*Dim,73));
  pv.put(std::vector<uint16_t>(size_t(rows)*Dim,73));bad.put(std::vector<int32_t>(rows,73));
  QsaAttnPools pools;pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  const auto shape=qsa_real_shapes();QsaBf16PackedScratch scratch{pk.data(),pv.data(),bad.data()};
  auto select=[&](int arm) {
    dq.put(permute(q,o[arm],Heads*Dim));di.put(permute(r.ids,o[arm],r.cap));ds.put(permute(r.steps,o[arm],kStepCount));
  };
  auto invoke=[&] {
    require(qsa_bf16_prefix_ordered_exp_batch(dq.data(),pools,di.data(),ds.data(),r.cap,shape,
        out.data(),r.queries,{pages,pages},cells,scratch,&rt->compute()),"production admission");rt->wait();
  };
  auto read=[&](int arm) {return restore(out.get(),o[arm],Heads*Dim);};
  select(0);invoke();const auto reference=read(0);
  for (auto x:reference) require(std::isfinite(std::bit_cast<float>(uint32_t(x)<<16)),"finite output");
  for (int arm=0;arm<Arms;++arm) {select(arm);invoke();require(read(arm)==reference,"unpermuted output bits");}
  const auto saved=v;for (auto &x:v) x=int8_t(int(x)/2);dv.put(v);
  select(0);invoke();const auto perturbed=read(0);require(perturbed!=reference,"V perturbation must change output");
  for (int arm=1;arm<Arms;++arm) {select(arm);invoke();require(read(arm)==perturbed,"same unpermuted V perturbation");}
  v=saved;dv.put(v);
  std::array<uint64_t,1000> clocks;
  for (auto &n:clocks) {auto t=std::chrono::steady_clock::now();n=elapsed(t);}
  std::sort(clocks.begin(),clocks.end());const auto clock_ns=clocks[500];
  for (int round=0;round<(timing ? 5 : 1);++round) for (int order=0;order<Arms;++order) {
    const int arm=(process+round+order)%Arms;select(arm);invoke();invoke();
    constexpr int Calls=32;const auto start=std::chrono::steady_clock::now();
    for (int call=0;call<(timing ? Calls : 1);++call) invoke();
    const auto ns=elapsed(start);
    require(read(arm)==reference,"timed output bits");
    require(dq.get()==permute(q,o[arm],Heads*Dim) && di.get()==permute(r.ids,o[arm],r.cap) &&
        ds.get()==permute(r.steps,o[arm],kStepCount),"immutable permuted inputs");
    if (timing) {
      const bool eligible=ns>=100000000 && ns>=clock_ns*1000;
      std::cout<<"sample layer="<<r.layer<<" process="<<process<<" arm="<<arm<<" round="<<round<<" order="<<order
          <<" calls="<<Calls<<" elapsed_ns="<<ns<<" timer_pair_ns="<<clock_ns<<" eligible="<<eligible<<" checks=PASS\n";
      require(eligible,"timing floor failed;no adaptive enlargement");
    }
  }
  require(dk.get()==k && dv.get()==v && dks.get()==ks && dvs.get()==vs && dt.get()==table,"immutable common inputs");
  (void)pk.get();(void)pv.get();(void)bad.get();
  std::cout<<"case layer="<<r.layer<<" first="<<r.first<<" cells="<<cells<<" queries="<<r.queries<<" cap="<<r.cap
      <<" cpu_order_ns="<<prepare_ns<<" output_bits=PASS perturbation=PASS guards=PASS immutable=PASS\n";
}
}
int main(int argc,char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);require(argc>=2,"--smoke | --host-check BASE | --validate BASE | --timing PROCESS BASE");
    const std::string mode=argv[1];const bool timing=mode=="--timing";
    require((mode=="--smoke" && argc==2) || ((mode=="--host-check" || mode=="--validate") && argc==3) ||
        (timing && argc==4),"exact replay mode/argument count");
    const int process=timing ? std::stoi(argv[2]) : 0;require(process>=0 && process<3,"process ordinal");
    auto records=mode=="--smoke" ? std::vector<Record>{smoke()} : load(argv[timing ? 3 : 2]);
    if (mode=="--host-check") {std::cout<<"capture_parser=PASS records="<<records.size()<<" gpu=UNRUN\n";return 0;}
    rt=sycl_backend::runtime_for();for (const auto &r:records) run_case(r,timing,process);
    rt->wait();rt.reset();std::cout<<"fixture=PASS model=UNRUN physical_counters=UNMEASURED cpu_order=EXCLUDED\n";return 0;
  } catch (const std::exception &e) {drain();std::cerr<<e.what()<<'\n';return 1;}
}
