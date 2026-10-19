// Authored for Strata; repository LICENSE (MIT) applies.
// Independent host tensor oracle for the configured BF16-operand/FP32-state
// online attention contract. It does not reproduce subgroup/SLM indexing.
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  // Keep host DMA spans inside the owner. If wait throws, a caller temporary
  // or local result must not disappear before the outer checked Drain retires
  // work (or aborts with these owners still alive).
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
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
void parallel_output(const std::vector<uint16_t> &candidate,
                     const std::vector<uint16_t> &reference) {
  require(candidate.size()==reference.size(),"parallel output extent");
  for (size_t i=0;i<candidate.size();++i) {
    if (i<16 || i>=candidate.size()-16) {
      require(candidate[i]==reference[i],"parallel output guard bits");continue;
    }
    const float a=load_bf16(candidate[i]),b=load_bf16(reference[i]);
    require(std::isnan(a)==std::isnan(b),"parallel diagnostic classification");
    if (std::isnan(b)) continue;
    require(std::isfinite(a) && std::isfinite(b),"parallel output finite");
    require(std::fabs(double(a)-b)<=double(spacing(b))+2e-6,
            "parallel BF16 output budget");
  }
}
#endif
void gate(bool enabled,const char *phase) {
  if (!enabled) return;
  std::cout<<"QSA_BF16_READY "<<phase<<'\n';
  require(std::cin.get()=='X',"Main gate token");
}

void run(int cells,int cap,int queries,bool masked,bool extended,int live_cells=0) {
  constexpr int Heads=24,KV=2,Dim=256,Page=16;
#ifdef STRATA_QSA_PACKED_GUARD
  constexpr int PackedGuard=STRATA_QSA_PACKED_GUARD;
#else
  constexpr int PackedGuard=16;
#endif
  if (!live_cells) live_cells=cells;
  require(live_cells>0 && live_cells<=cells,"fixture live prefix bounds");
  const int pages=(cells+Page-1)/Page, rows=pages*KV*Page;
  const int active_pages=(live_cells+Page-1)/Page, active_rows=active_pages*KV*Page;
  std::vector<int32_t> table(pages),ids(size_t(queries)*cap),steps(size_t(queries)*kStepCount);
  for (int p=0;p<pages;++p) table[p]=masked && p%7==2 ? -1 : masked && p%7==3 ? pages+9 : pages-1-p;
  std::vector<int8_t> k(size_t(rows)*Dim),v(k.size());
  for (size_t i=0;i<k.size();++i) { k[i]=int8_t(int((i*7+3)%15)-7); v[i]=int8_t(int((i*11+5)%31)-15); }
  std::vector<uint16_t> ks(size_t(rows)*4),vs(ks.size()),q(size_t(queries)*Heads*Dim);
  for (size_t i=0;i<ks.size();++i) {
    ks[i]=half(std::ldexp(1.f,-5-int(i%4)));
    vs[i]=half(std::ldexp(i%3 ? 1.f : -1.f,-7-int(i%4)));
  }
  // Poison every unmapped/suffix source row. A valid-prefix implementation
  // never consumes these scales, including the final physical page's tail.
  std::vector<bool> source_live(rows,false);
  for (int cell=0;cell<live_cells;++cell) {
    const int physical=table[cell/Page];if (physical<0 || physical>=pages) continue;
    for (int h=0;h<KV;++h) source_live[(physical*KV+h)*Page+cell%Page]=true;
  }
  for (int row=0;row<rows;++row) if (!source_live[row])
    for (int g=0;g<4;++g) ks[row*4+g]=vs[row*4+g]=0x7c00;
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
  Buffer<uint16_t> packed_k(size_t(active_rows)*Dim+2*PackedGuard),packed_v(size_t(active_rows)*Dim+2*PackedGuard);
  Buffer<int32_t> packed_bad(size_t(active_rows)+2*PackedGuard);
  const int32_t bad_guard=0x12345678;
#if defined(STRATA_QSA_COMPARE_PREFIX_STATE)
  const size_t state_values=size_t((cap+31)/32)*4;
  const float state_guard=-913.75f;
  Buffer<float> state(state_values+32);
  const std::vector<float> state_initial(state_values+32,state_guard);
#endif
  Drain pending; // Destroy before every input/output owner on success and failure.
  packed_k.put(std::vector<uint16_t>(size_t(active_rows)*Dim+2*PackedGuard,guard));
  packed_v.put(std::vector<uint16_t>(size_t(active_rows)*Dim+2*PackedGuard,guard));
  packed_bad.put(std::vector<int32_t>(size_t(active_rows)+2*PackedGuard,bad_guard));
  dq.put(q);dks.put(ks);dvs.put(vs);dk.put(k);dv.put(v);dt.put(table);di.put(ids);ds.put(steps);out.put(sentinel);
  QsaAttnPools pools; pools.k_q=dk.data();pools.v_q=dv.data();pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;
  const QsaBf16PackedScratch scratch{packed_k.data()+PackedGuard,packed_v.data()+PackedGuard,packed_bad.data()+PackedGuard};
  if constexpr (PackedGuard==17)
    require(reinterpret_cast<uintptr_t>(scratch.k)%8!=0,"packed K must exercise two-byte-only alignment");
  auto attention=[&](const uint16_t *qin,const QsaAttnPools &p,const int32_t *ii,const int32_t *ss,
                     int64_t c,const QsaShapes &sh,uint16_t *o,int64_t nq,
                     const QsaPromptBf16Bounds &b,void *stream) {
    return qsa_bf16_prefix_batch(qin,p,ii,ss,c,sh,o,nq,b,live_cells,scratch,stream);
  };
  auto packed_guards=[&] {
    auto a=packed_k.get(),b=packed_v.get();auto bad=packed_bad.get();
    for (size_t i=0;i<PackedGuard;++i)
      require(a[i]==guard && a[a.size()-PackedGuard+i]==guard && b[i]==guard && b[b.size()-PackedGuard+i]==guard &&
              bad[i]==bad_guard && bad[bad.size()-PackedGuard+i]==bad_guard,"packed output guards");
  };
  auto invoke=[&] {
    require(attention(dq.data(),pools,di.data(),ds.data(),cap,shape,
        out.data()+16,queries,{pages,pages},&rt->compute()),"BF16 matrix unavailable");
    rt->wait();
#if defined(STRATA_QSA_COMPARE_PREFIX33) || defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
    // Compare every output bit, including NaN diagnostics, masks and guards.
    // Both calls use the same inputs; the second refreshes identical scratch.
    const auto candidate=out.get();out.put(sentinel);
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
    require(qsa_bf16_prefix_ordered_exp_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
        out.data()+16,queries,{pages,pages},live_cells,scratch,&rt->compute()),"ordered reference unavailable");
    rt->wait();const auto reference=out.get();parallel_output(candidate,reference);
#else
    require(qsa_bf16_prefix_optimized_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
        out.data()+16,queries,{pages,pages},live_cells,scratch,&rt->compute()),"prefix33 reference unavailable");
    rt->wait();require(out.get()==candidate,"complete prefix33 bit identity");
#endif
#if defined(STRATA_QSA_COMPARE_PREFIX_STATE)
    // Capture both real and padded rows. The diagnostic specialization must
    // preserve the corresponding normal output, including NaNs and guards.
    // Compare raw FP32 bits before final BF16 output could hide a difference.
    for (int kv_head : {0,1}) for (int head : {0,15}) {
      std::vector<float> serial;
      for (bool ordered : {false,true}) {
        state.put(state_initial);out.put(sentinel);
        const QsaPrefixStateProbe probe{0,kv_head,head,state.data()+16};
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
        const bool admitted=ordered ? qsa_bf16_prefix_parallel_sum_probe_batch(
            dq.data(),pools,di.data(),ds.data(),cap,shape,out.data()+16,queries,
            {pages,pages},live_cells,scratch,probe,&rt->compute()) :
          qsa_bf16_prefix_state_probe_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
            out.data()+16,queries,{pages,pages},live_cells,scratch,true,probe,&rt->compute());
        require(admitted,"parallel state probe admission");
        rt->wait();require(out.get()==(ordered ? candidate : reference),
            "state probe changed its own normal output");
#else
        require(qsa_bf16_prefix_state_probe_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
            out.data()+16,queries,{pages,pages},live_cells,scratch,ordered,probe,&rt->compute()),
            "state probe admission");
        rt->wait();require(out.get()==candidate,"state probe changed final output");
#endif
        const auto observed=state.get();
        for (size_t i=0;i<16;++i)
          require(observed[i]==state_guard && observed[observed.size()-16+i]==state_guard,"state guards");
        const int width=steps[kStepWidth];
        const size_t used=(width>=0 && width<=cap) ? size_t((width+31)/32)*4 : 0;
        for (size_t i=used;i<state_values;++i)
          require(observed[16+i]==state_guard,"state capture wrote beyond selected width");
        if (!ordered) serial=observed;
        else {
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
          // Positive FP32 operands: any association of32 terms has at most31
          // rounding steps. Do not assume a vendor-specific reduction tree.
          constexpr double u=0x1p-24, gamma31=(31*u)/(1-31*u);
          double normalizer_budget=0;
          bool saw_positive=false;
          for (size_t i=0;i<used;i+=4) {
            const size_t at=16+i;
            require(std::bit_cast<uint32_t>(serial[at])==std::bit_cast<uint32_t>(observed[at]) &&
                std::bit_cast<uint32_t>(serial[at+1])==std::bit_cast<uint32_t>(observed[at+1]),
                "parallel maximum/alpha bits");
            const double a=observed[at+1], sn=serial[at+2], pn=observed[at+2],
                         ss=serial[at+3], ps=observed[at+3];
            require(std::isfinite(a) && a>=0 && a<=1 && std::isfinite(sn) && sn>=0 &&
                    std::isfinite(pn) && pn>=0 && std::isfinite(ss) && ss>=0 &&
                    std::isfinite(ps) && ps>=0,"parallel finite nonnegative state");
            require((ss>0)==(ps>0),"parallel positive tile classification");
            saw_positive |= ss>0;
            require(saw_positive ? (sn>0 && pn>0) : (sn==0 && pn==0),
                    "parallel empty/positive normalizer");
            const double sum_budget=2*gamma31/(1-gamma31)*std::max(ss,ps)+
                64*double(std::numeric_limits<float>::denorm_min());
            require(std::fabs(ss-ps)<=sum_budget,"parallel tile sum bound");
            normalizer_budget=a*normalizer_budget+sum_budget+
                2*u/(1-u)*std::max(sn,pn)+2*double(std::numeric_limits<float>::denorm_min());
            require(std::fabs(sn-pn)<=normalizer_budget,"parallel normalizer recurrence bound");
          }
#else
          require(std::memcmp(serial.data(),observed.data(),observed.size()*sizeof(float))==0,
                     "online maximum/alpha/normalizer/sum FP32 bits differ");
#endif
        }
      }
    }
#endif
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
    out.put(candidate);
#endif
#endif
  };
  auto guarded=[&] {
    packed_guards();
    auto y=out.get();
    for (size_t i=0;i<16;++i) require(y[i]==guard && y[y.size()-16+i]==guard,"output guard");
    return y;
  };
  auto immutable=[&] {
    require(dq.get()==q && dks.get()==ks && dvs.get()==vs && dk.get()==k && dv.get()==v &&
            dt.get()==table && di.get()==ids && ds.get()==steps,"all input bytes immutable");
  };
  auto row_of=[&](int cell,int head) -> int64_t {
    if (cell<0 || cell>=live_cells) return -1;
    const int physical=table[cell/Page]; if (physical<0 || physical>=pages) return -1;
    return (int64_t(physical)*KV+head/(Heads/KV))*Page+cell%Page;
  };
  invoke();const auto y=guarded();immutable();
  { // Check EVERY logical scratch element, including masked/padding rows.
    auto pk=packed_k.get(),pv=packed_v.get();auto pb=packed_bad.get();
    for (int64_t row=0;row<active_rows;++row) {
      const int page=int(row/(KV*Page)),head=int(row/Page%KV),offset=int(row%Page);
      const int64_t source=row_of(page*Page+offset,head*(Heads/KV));
      require(pb[PackedGuard+row]==0,"finite or masked prefix diagnostic row");
      for (int d=0;d<Dim;++d) {
        const uint16_t key=source>=0 ? round_bf16(float(k[source*Dim+d])*load_half(ks[source*4+d/64])) : 0;
        const uint16_t val=source>=0 ? round_bf16(float(v[source*Dim+d])*load_half(vs[source*4+d/64])) : 0;
        require(pk[PackedGuard+row*Dim+d]==key,"complete logical K prefix including zero padding/missing pages");
        require(pv[PackedGuard+row*Dim+d]==val,"complete logical V prefix including zero padding/missing pages");
      }
    }
  }

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
#if defined(STRATA_QSA_COMPARE_PREFIX33) || defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
    // Explicit physical-page alias: reads remain valid even when two logical
    // pages refer to one physical page. Compare all candidate/reference bits.
    require(pages>1 && live_cells>1 && cap>32,"row-publication diagnostic geometry");
    const auto original_table=table;table[1]=table[0];dt.put(table);
    invoke();(void)guarded();immutable();table=original_table;dt.put(table);
    // Duplicate IDs fill the first tile; a distinct ID appears ONLY in the
    // later partial tile. Poison each side separately to check both early
    // sticky diagnostics and the next-row publication path.
    const auto original_ids=ids;
    for (int b=0;b<queries;++b) {
      std::fill(ids.begin()+size_t(b)*cap,ids.begin()+size_t(b+1)*cap,0);
      ids[size_t(b+1)*cap-1]=1;
    }
    di.put(ids);invoke();(void)guarded();immutable();
    for (int invalid_cell : {0,1}) for (bool key : {false,true}) {
      auto &scales=key ? ks : vs;auto &device=key ? dks : dvs;
      const auto saved=scales;
      for (int head=0;head<KV;++head) {
        const size_t row=(size_t(table[0])*KV+head)*Page+invalid_cell;
        scales[row*4]=0x7c00;
      }
      device.put(scales);invoke();const auto diagnostic=guarded();
      for (size_t i=16;i<diagnostic.size()-16;++i)
        require(std::isnan(load_bf16(diagnostic[i])),"first/later-tile invalid scale diagnostic");
      immutable();scales=saved;device.put(scales);
      invoke();const auto finite=guarded();
      for (size_t i=16;i<finite.size()-16;++i)
        require(std::isfinite(load_bf16(finite[i])),"diagnostic clears on next call");
      immutable();
    }
    ids=original_ids;di.put(ids);invoke();require(guarded()==y,"restore original selection");immutable();
    std::cout<<"case explicit_duplicate_ids=PASS aliased_physical_pages=PASS first_later_KV_diagnostics=PASS status=PASS\n";
#endif
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

#if defined(STRATA_QSA_COMPARE_PREFIX_STATE)
    // Invalid state views are refused before either pack/attention submission.
    for (int invalid=0;invalid<8;++invalid) {
      QsaPrefixStateProbe probe{0,0,0,state.data()+16};
      if (invalid==0) probe.query=queries;
      if (invalid==1) probe.head=16;
      if (invalid==2) probe.kv_head=2;
      if (invalid==3) probe.states=nullptr;
      if (invalid==4) probe.states=reinterpret_cast<float *>(out.data()+16); // Actual output, after the guard.
      if (invalid==5) probe.states=state_initial.empty() ? nullptr : const_cast<float *>(state_initial.data());
      if (invalid==6) probe.states=reinterpret_cast<float *>(reinterpret_cast<char *>(state.data())+1);
      if (invalid==7) probe.query=-1;
      bool refused=false;
      try { (void)qsa_bf16_prefix_state_probe_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
          out.data()+16,queries,{pages,pages},live_cells,scratch,true,probe,&rt->compute()); }
      catch (const std::invalid_argument &) { refused=true; }
      require(refused,"invalid state capture submitted work");
      require(guarded()==y,"state rejection changed output");
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
      refused=false;
      try { (void)qsa_bf16_prefix_parallel_sum_probe_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
          out.data()+16,queries,{pages,pages},live_cells,scratch,probe,&rt->compute()); }
      catch (const std::invalid_argument &) { refused=true; }
      require(refused,"invalid parallel state capture submitted work");
      require(guarded()==y,"parallel state rejection changed output");
#endif
    }
    { Buffer<float> tiny(1); Drain pending_tiny;
      bool refused=false;
      try { (void)qsa_bf16_prefix_state_probe_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
          out.data()+16,queries,{pages,pages},live_cells,scratch,false,{0,0,0,tiny.data()},&rt->compute()); }
      catch (const std::invalid_argument &) { refused=true; }
      require(refused,"short state span submitted work");
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
      refused=false;
      try { (void)qsa_bf16_prefix_parallel_sum_probe_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
          out.data()+16,queries,{pages,pages},live_cells,scratch,{0,0,0,tiny.data()},&rt->compute()); }
      catch (const std::invalid_argument &) { refused=true; }
      require(refused,"short parallel state span submitted work");
#endif
    }
    std::cout<<"case state_admission_rejections=9 status=PASS\n";
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
    std::cout<<"case parallel_state_admission_rejections=9 status=PASS\n";
#endif
#endif
    bool rejected=false;
    try { (void)attention(dq.data(),pools,di.data(),ds.data(),cap,shape,dq.data(),queries,{pages,pages},&rt->compute()); }
    catch (const std::invalid_argument &) { rejected=true; }
    require(rejected,"output alias rejected before submission");
    require(!attention(dq.data(),pools,di.data(),ds.data(),0,shape,out.data()+16,queries,{pages,pages},&rt->compute()),"unsupported cap refused");
    immutable();require(guarded()==y,"rejected submissions leave output intact");
    for (int variant=0;variant<4;++variant) {
      auto bad=scratch;
      if (variant==0) bad.k=dq.data();
      if (variant==1) bad.v=bad.k;
      if (variant==2) bad.bad_rows=reinterpret_cast<int32_t *>(bad.k);
      if (variant==3) bad.bad_rows=table.data(); // Host memory is not a device allocation.
      bool refused=false;
      try { (void)qsa_bf16_prefix_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
          out.data()+16,queries,{pages,pages},live_cells,bad,&rt->compute()); }
      catch (const std::invalid_argument &) { refused=true; }
      require(refused,"scratch overlap/domain rejected before either kernel");
      immutable();require(guarded()==y,"scratch rejection preserves final output and guards");
    }
    { Buffer<uint16_t> tiny(1); Drain pending_tiny;auto bad=scratch;bad.k=tiny.data();
      bool refused=false;
      try { (void)qsa_bf16_prefix_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
          out.data()+16,queries,{pages,pages},live_cells,bad,&rt->compute()); }
      catch (const std::invalid_argument &) { refused=true; }
      require(refused,"complete scratch extent rejected before either kernel");
      require(guarded()==y,"short scratch rejection preserves output");
    }
    require(qsa_bf16_prefix_batch(nullptr,{},nullptr,nullptr,0,{},nullptr,0,{}, 0,{},&rt->compute()),"zero-query no-op");
    for (int64_t bad_live : {int64_t(0),int64_t(-1),int64_t(pages*Page+1)}) {
      bool refused=false;
      try { (void)qsa_bf16_prefix_batch(dq.data(),pools,di.data(),ds.data(),cap,shape,
          out.data()+16,queries,{pages,pages},bad_live,scratch,&rt->compute()); }
      catch (const std::invalid_argument &) { refused=true; }
      require(refused,"invalid live prefix rejected before either submission");
      require(guarded()==y,"rejected live extent preserves outputs");
    }

  }
  std::cout<<"case cells="<<cells<<" live_cells="<<live_cells<<" active_rows="<<active_rows<<" cap="<<cap<<" queries="<<queries<<" masked="<<masked
           <<" compared="<<compared<<" max_abs="<<std::setprecision(10)<<max_error
           <<" max_budget_ratio="<<max_budget_ratio<<" oracle=BF16_operands_FP32_online"
           <<" packed_guard_elements="<<PackedGuard<<" packed_k_mod8="<<reinterpret_cast<uintptr_t>(scratch.k)%8
           <<" packed_logical_prefix=PASS packed_guards=PASS guards=PASS inputs=PASS status=PASS\n";
#if defined(STRATA_QSA_COMPARE_PREFIX_STATE)
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
  std::cout<<"case parallel_FP32_state_bound=PASS exact_max_alpha=PASS real_padded_heads=PASS state_guards=PASS cells="<<cells<<" cap="<<cap<<'\n';
#else
  std::cout<<"case exact_FP32_online_state=PASS real_padded_heads=PASS state_guards=PASS cells="<<cells<<" cap="<<cap<<'\n';
#endif
#endif
#if defined(STRATA_QSA_COMPARE_PARALLEL_SUM)
  std::cout<<"case parallel_BF16_output_bound=PASS cells="<<cells<<" cap="<<cap<<" queries="<<queries<<'\n';
#endif
#if defined(STRATA_QSA_COMPARE_PREFIX33)
  std::cout<<"case exact_prefix33_bits=PASS cells="<<cells<<" cap="<<cap<<" queries="<<queries<<'\n';
#endif
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
    run(97,33,1,false,true,49);run(97,33,3,true,false,71);
    if (std::string(argv[1])=="--component") {
      for (int width : {1,17,31,32,63,65,2051}) run(2113,width,3,true,false);
      run(262144,2051,1,false,false);run(262144,33,1,false,false,13);
    }
    rt->wait();rt.reset();gate(gates,"after-cleanup");
    std::cout<<"fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED\n";
    return 0;
  } catch (const std::exception &e) { drain();std::cerr<<e.what()<<'\n';return 1; }
}
