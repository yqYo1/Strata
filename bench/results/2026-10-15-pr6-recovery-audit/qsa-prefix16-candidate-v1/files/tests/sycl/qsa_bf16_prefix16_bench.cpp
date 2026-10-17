// Authored for Strata; repository LICENSE (MIT) applies.
// Matched live-prefix baseline/candidate timing, not model PP or utilization.
// Every physical input is initialized; both arms allocate/initialize scratch.
// Identical prefix pack time and cache effects are included in every call.
#include "strata/kernels/qsa_bf16_prefix.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

using namespace strata;
using namespace strata::kernels;
#ifndef STRATA_QSA_PREFIX_CANDIDATE
#define STRATA_QSA_PREFIX_CANDIDATE qsa_bf16_prefix16_batch
#endif
#ifndef STRATA_QSA_PREFIX_CANDIDATE_NAME
#define STRATA_QSA_PREFIX_CANDIDATE_NAME "tile16"
#endif
namespace strata::kernels {
// A separately compiled candidate with the same public signature.
bool STRATA_QSA_PREFIX_CANDIDATE(const uint16_t *, const QsaAttnPools &,
    const int32_t *, const int32_t *, int64_t, const QsaShapes &, uint16_t *,
    int64_t, const QsaPromptBf16Bounds &, int64_t, const QsaBf16PackedScratch &, void *);
}
namespace {
std::shared_ptr<sycl_backend::Runtime> rt;
void require(bool x,const char *s) { if (!x) throw std::runtime_error(s); }
void drain() noexcept {
  sycl_backend::checked_cleanup([] { if (rt) rt->wait(); },[] {},"QSA matched benchmark drain");
}
struct Drain { ~Drain() noexcept { drain(); } };
template<class T> struct Buffer {
  sycl_backend::Allocation memory;
  explicit Buffer(size_t n):memory(rt,n*sizeof(T),sycl_backend::MemoryKind::Device) {}
  T *data() { return memory.as<T>(); }
  void put(const std::vector<T> &v) {
    require(v.size()*sizeof(T)==memory.size(),"upload extent");
    rt->wait(rt->compute().memcpy(data(),v.data(),memory.size()));
  }
  std::vector<T> get() {
    std::vector<T> v(memory.size()/sizeof(T));
    rt->wait(rt->compute().memcpy(v.data(),data(),memory.size()));return v;
  }
};
uint16_t bf(float x) {
  const auto b=std::bit_cast<uint32_t>(x);
  return uint16_t((b+0x7fff+((b>>16)&1))>>16);
}
float unbf(uint16_t b) { return std::bit_cast<float>(uint32_t(b)<<16); }
uint16_t hf(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
float unhf(uint16_t b) { return float(std::bit_cast<_Float16>(b)); }
template<class T> uint64_t fingerprint(const std::vector<T> &v) {
  const auto *p=reinterpret_cast<const unsigned char *>(v.data());uint64_t h=14695981039346656037ull;
  for (size_t i=0;i<v.size()*sizeof(T);++i) { h^=p[i];h*=1099511628211ull; }return h;
}
double cpu() {
  timespec t{};require(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0,"thread clock");
  return double(t.tv_sec)+double(t.tv_nsec)*1e-9;
}
void gate(bool enabled,const char *phase) {
  if (!enabled) return;
  std::cout<<"QSA_BF16_READY "<<phase<<'\n';require(std::cin.get()=='X',"Main gate token");
}
void run(const std::string &arm,int nq,int repetitions,int Cells) {
  constexpr int H=24,D=256,K=2,Page=16,Cap=2051;
  const int Pages=(Cells+Page-1)/Page;
  const size_t elements=size_t(nq)*H*D,rows=Pages*K*Page;
  std::vector<int8_t> k(rows*D),v(k.size());
  std::vector<uint16_t> ks(rows*4),vs(ks.size()),qb(elements);
  std::vector<float> qf(elements);
  std::vector<int32_t> table(Pages),ids(size_t(nq)*Cap),steps(size_t(nq)*kStepCount);
  for (int p=0;p<Pages;++p) table[p]=Pages-1-p;
  for (size_t i=0;i<k.size();++i) { k[i]=int8_t(int(i*7%255)-127);v[i]=int8_t(int(i*11%255)-127); }
  for (size_t i=0;i<ks.size();++i) { ks[i]=hf(.007f+float(i%7)*.0001f);vs[i]=hf(.008f+float(i%11)*.0002f); }
  // Both arms see identical BF16 Q. The extra FP32 backing is an unchanged
  // allocation/input control outside timing, not a precision reference.
  for (size_t i=0;i<elements;++i) { qb[i]=bf(float(std::sin(double(i)*.019)));qf[i]=unbf(qb[i]); }
  for (int b=0;b<nq;++b) {
    steps[b*kStepCount+kStepPos]=4096+b;steps[b*kStepCount+kStepNKv]=Cells;
    steps[b*kStepCount+kStepWidth]=Cap;
    for (int c=0;c<Cap;++c) ids[b*Cap+c]=int((int64_t(c)*Cells/Cap+int64_t(b)*17)%Cells);
    if (!b) ids[b*Cap+Cap-1]=Cells-1;
  }
  std::vector<int32_t> selected_stamp(size_t(Cells),0);
  for (int b=0;b<nq;++b) for (int c=0;c<Cap;++c) {
    const int cell=ids[b*Cap+c];
    require(cell>=0 && cell<Cells && selected_stamp[cell]!=b+1,"complete selected bounds and per-query uniqueness");
    selected_stamp[cell]=b+1;
  }
  const size_t selected_cells=std::count_if(selected_stamp.begin(),selected_stamp.end(),[](int32_t x){return x!=0;});
  require(nq!=8192 || selected_cells==size_t(Cells),"full physical-context logical coverage in timing cohort");
  // Allocate and initialize every arm's backing in every process. This makes
  // memory pressure identical while keeping unused outputs checked as well.
  Buffer<int8_t> dk(k.size()),dv(v.size());
  Buffer<uint16_t> dks(ks.size()),dvs(vs.size()),dqb(qb.size()),ob(elements+32);
  Buffer<float> dqf(qf.size()),of(elements+32);
  Buffer<int32_t> dt(table.size()),di(ids.size()),ds(steps.size());
  Buffer<uint16_t> pk(rows*D+32),pv(rows*D+32);Buffer<int32_t> bad(rows+32);
  Drain pending;
  const uint16_t guard=bf(1234.f);
  std::vector<uint16_t> initb(elements+32,guard);
  std::vector<float> initf(elements+32,1234.f);
  const int32_t bad_guard=0x12345678;
  pk.put(std::vector<uint16_t>(rows*D+32,guard));pv.put(std::vector<uint16_t>(rows*D+32,guard));
  bad.put(std::vector<int32_t>(rows+32,bad_guard));
  dk.put(k);dv.put(v);dks.put(ks);dvs.put(vs);dqb.put(qb);dqf.put(qf);
  ob.put(initb);of.put(initf);dt.put(table);di.put(ids);ds.put(steps);
  size_t bytes=0;
  for (size_t n : {dk.memory.size(),dv.memory.size(),dks.memory.size(),dvs.memory.size(),
      dqb.memory.size(),dqf.memory.size(),ob.memory.size(),of.memory.size(),
      dt.memory.size(),di.memory.size(),ds.memory.size(),pk.memory.size(),pv.memory.size(),bad.memory.size()}) bytes+=n;
  require(bytes<(size_t(2)<<30),"device footprint budget");
  QsaAttnPools pools{};pools.k_q=dk.data();pools.v_q=dv.data();
  pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;
  auto issue=[&] {
    bool ok=false;
    if (arm=="base") ok=qsa_bf16_prefix_optimized_batch(dqb.data(),pools,di.data(),ds.data(),Cap,shape,
        ob.data()+16,nq,{Pages,Pages},Cells,{pk.data()+16,pv.data()+16,bad.data()+16},&rt->compute());
    if (arm==STRATA_QSA_PREFIX_CANDIDATE_NAME) ok=STRATA_QSA_PREFIX_CANDIDATE(dqb.data(),pools,di.data(),ds.data(),Cap,shape,
        ob.data()+16,nq,{Pages,Pages},Cells,{pk.data()+16,pv.data()+16,bad.data()+16},&rt->compute());
    require(ok,"selected matrix entry unavailable; no fallback");
  };
  auto values=[&] {
    auto b=ob.get();auto f=of.get();
    { auto a=pk.get(),v=pv.get();auto flags=bad.get();
      for (size_t i=0;i<16;++i)
        require(a[i]==guard && a[a.size()-16+i]==guard && v[i]==guard && v[v.size()-16+i]==guard &&
            flags[i]==bad_guard && flags[flags.size()-16+i]==bad_guard,"packed allocation guards");
      for (size_t i=16;i<flags.size()-16;++i) require(flags[i]==0,"all logical finite-scale diagnostics");
    }
    for (size_t i=0;i<16;++i)
      require(b[i]==guard && b[b.size()-16+i]==guard && f[i]==1234.f && f[f.size()-16+i]==1234.f,"output guards");
    require(f==initf,"unused FP32 output immutable");
    std::vector<float> out(elements);
    for (size_t i=0;i<elements;++i) {
      out[i]=unbf(b[16+i]);
      require(std::isfinite(out[i]),"complete output finite");
    }
    return out;
  };
  std::cout<<"arm="<<arm<<" nq="<<nq<<" cap="<<Cap<<" workgroups="<<nq*2
           <<" cells="<<Cells<<" global_selected_cells="<<selected_cells<<" selected_set=per_query_rotated_uniform_spread_unique"
           <<" device_bytes="<<bytes<<" packed_scratch_payload_bytes="<<(rows*(D*4+4))
           <<" pack_cost=IN_TIMED_CALL"<<" warmups=2 repetitions="<<repetitions
           <<" fingerprints=FNV1a64_native_bytes q="<<fingerprint(qb)<<" k="<<fingerprint(k)
           <<" v="<<fingerprint(v)<<" ks="<<fingerprint(ks)<<" vs="<<fingerprint(vs)
           <<" table="<<fingerprint(table)<<" ids="<<fingerprint(ids)<<" steps="<<fingerprint(steps)<<'\n';
  issue();rt->wait();const auto initial=values();
  // Independent BF16-operand oracle at four complete queries, same criteria
  // for both layout arms. This bounded32K fixture does not reuse the known
  // strict-scalar256K case or waive its recorded numerical failure.
  double max_error=0,max_ratio=0;size_t compared=0;
  for (int b=0;b<nq;++b) {
    if (b!=0 && b!=1 && b!=2 && b!=nq-1) continue;
    for (int h=0;h<H;++h) {
      std::vector<float> score(Cap),value(size_t(Cap)*D),acc(D,0.f);
      for (int c=0;c<Cap;++c) {
        const int cell=ids[b*Cap+c];
        const int64_t row=(int64_t(table[cell/Page])*K+h/12)*Page+cell%Page;
        double dot=0;
        for (int d=0;d<D;++d) {
          const float q=unbf(bf((qf[(size_t(b)*H+h)*D+d]*.0625f)*1.4426950408f));
          dot+=double(q)*unbf(bf(float(k[row*D+d])*unhf(ks[row*4+d/64])));
          value[size_t(c)*D+d]=unbf(bf(float(v[row*D+d])*unhf(vs[row*4+d/64])));
        }
        score[c]=float(dot);
      }
      float maximum=-INFINITY,norm=0;
      const int tile_cells = arm=="base" ? 32 : 16;
      for (int first=0;first<Cap;first+=tile_cells) {
        const int last=std::min(Cap,first+tile_cells);float next=maximum,sum=0;
        for (int c=first;c<last;++c) next=std::max(next,score[c]);
        const float alpha=norm>0 ? std::exp2(maximum-next) : 0.f;
        std::vector<float> p(size_t(last-first));
        for (int c=first;c<last;++c) { const float x=std::exp2(score[c]-next);sum+=x;p[c-first]=unbf(bf(x)); }
        for (int d=0;d<D;++d) {
          double term=0;for (int c=first;c<last;++c) term+=double(p[c-first])*value[size_t(c)*D+d];
          acc[d]=float(double(acc[d]*alpha)+term);
        }
        norm=std::fma(norm,alpha,sum);maximum=next;
      }
      for (int d=0;d<D;++d) {
        const float ref=unbf(bf(acc[d]/norm));
        const double error=std::fabs(double(initial[(size_t(b)*H+h)*D+d])-ref);
        const uint16_t mag=bf(std::fabs(ref));
        const double budget=double(unbf(uint16_t(mag+1))-unbf(mag))+2e-6;
        max_error=std::max(max_error,error);max_ratio=std::max(max_ratio,error/budget);
        require(error<=budget,"BF16 operand contract at sampled queries");++compared;
      }
    }
  }
  for (int w=0;w<2;++w) { issue();rt->wait(); }
  using Clock=std::chrono::steady_clock;
  auto ms=[](auto a,auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
  for (int i=0;i<repetitions;++i) {
    const double c0=cpu();const auto t0=Clock::now();issue();const auto t1=Clock::now();
    const double c1=cpu();rt->wait();const auto t2=Clock::now();const double c2=cpu();
    std::cout<<"sample="<<i<<" whole_wall_ms="<<ms(t0,t2)<<" enqueue_wall_ms="<<ms(t0,t1)
             <<" wait_wall_ms="<<ms(t1,t2)<<" whole_thread_cpu_ms="<<(c2-c0)*1000
             <<" enqueue_thread_cpu_ms="<<(c1-c0)*1000<<" wait_thread_cpu_ms="<<(c2-c1)*1000
             <<" validation=PENDING_FINAL_CHECKS\n";
  }
  const auto final=values();
  for (size_t i=0;i<initial.size();++i)
    require(std::bit_cast<uint32_t>(initial[i])==std::bit_cast<uint32_t>(final[i]),"complete repeat bit identity including signed zero");
  require(dk.get()==k && dv.get()==v && dks.get()==ks && dvs.get()==vs && dqb.get()==qb && dqf.get()==qf &&
          dt.get()==table && di.get()==ids && ds.get()==steps,"all input bytes immutable");
  std::cout<<"case status=PASS guards=PASS inputs=PASS full_finite=PASS repeat_bits=PASS"
           <<" oracle="<<"BF16_operands_FP32_online_queries_0_1_2_last"
           <<" compared="<<compared<<" max_abs="<<max_error<<" max_budget_ratio="<<max_ratio
           <<" output_FNV1a64="<<fingerprint(final)
           <<" per_repeat_numerical=FINAL_FRAME_ONLY model_PP=UNRUN physical_traffic=UNMEASURED utilization=UNMEASURED\n";
}
} // namespace
int main(int argc,char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);std::cout<<std::setprecision(12);
    require(argc==9 && std::string(argv[1])=="--arm" && std::string(argv[3])=="--nq" &&
            std::string(argv[5])=="--repeats" && std::string(argv[7])=="--cells","--arm base|" STRATA_QSA_PREFIX_CANDIDATE_NAME " --nq 3|8192 --repeats 1|10 --cells 2113|32768|131072|262144");
    const std::string arm=argv[2];const int nq=std::stoi(argv[4]),reps=std::stoi(argv[6]),cells=std::stoi(argv[8]);
    require((arm=="base" || arm==STRATA_QSA_PREFIX_CANDIDATE_NAME) && (nq==3 || nq==8192) &&
            (reps==1 || reps==10) && (cells==2113 || cells==32768 || cells==131072 || cells==262144),"bounded benchmark arguments");
    for (auto [key,value] : {std::pair{"STRATA_SYCL_PROMPT_DIRECT","1"},
        {"STRATA_SYCL_PROMPT_K_CELL_MAJOR","0"},{"STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP","0"},
        {"STRATA_SYCL_PROMPT_NATIVE_KV_MIN","0"}}) {
      const char *found=std::getenv(key);require(found && std::string(found)==value,"fixed pre-exec legacy controls");
    }
    const char *g=std::getenv("STRATA_TEST_MAIN_GATES");const bool gates=g && std::string(g)=="1";
    gate(gates,"before-runtime");rt=sycl_backend::runtime_for();
    std::cout<<"device="<<rt->device().get_info<sycl::info::device::name>()
             <<" driver="<<rt->device().get_info<sycl::info::device::driver_version>()<<'\n';
    gate(gates,"runtime-ready");run(arm,nq,reps,cells);rt->wait();rt.reset();gate(gates,"after-cleanup");
    std::cout<<"fixture_status=PASS model_PP=UNRUN physical262144_model=UNRUN RESTORE=UNRUN faults=EXTERNAL_REQUIRED\n";return 0;
  } catch (const std::exception &e) { drain();std::cerr<<e.what()<<'\n';return 1; }
}
