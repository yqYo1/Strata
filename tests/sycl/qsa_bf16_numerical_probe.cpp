// Authored for Strata; repository LICENSE (MIT) applies.
// Capture an unresolved numerical failure; no warmups or timing samples.
// Diagnostic success is separate from numerical contract pass/failure.
// Every physical input is initialized; both arms allocate/initialize scratch.
// Pack time and its cache effects are included in every packed timed call.
#include "strata/kernels/qsa_bf16_attention_late_flags.hpp"
#include "strata/kernels/qsa_bf16_packed.hpp"
#include "strata/kernels/qsa_bf16_numerical_probe.hpp"
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
  // All three arms see exactly BF16-representable Q values. The FP32 arm
  // stores the same values in FP32; its internal precision contract differs.
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
  constexpr int Tiles=(Cap+31)/32;
  Buffer<float> score_probe(Cap+32),prob_probe(Cap+32),state_probe(Tiles*4+32);
  Buffer<uint16_t> round_probe(Cap+32);
  Drain pending;
  const uint16_t guard=bf(1234.f);
  std::vector<uint16_t> initb(elements+32,guard);
  std::vector<float> initf(elements+32,1234.f);
  const int32_t bad_guard=0x12345678;
  score_probe.put(std::vector<float>(Cap+32,1234.f));prob_probe.put(std::vector<float>(Cap+32,1234.f));
  state_probe.put(std::vector<float>(Tiles*4+32,1234.f));round_probe.put(std::vector<uint16_t>(Cap+32,guard));
  pk.put(std::vector<uint16_t>(rows*D+32,guard));pv.put(std::vector<uint16_t>(rows*D+32,guard));
  bad.put(std::vector<int32_t>(rows+32,bad_guard));
  dk.put(k);dv.put(v);dks.put(ks);dvs.put(vs);dqb.put(qb);dqf.put(qf);
  ob.put(initb);of.put(initf);dt.put(table);di.put(ids);ds.put(steps);
  size_t bytes=0;
  for (size_t n : {dk.memory.size(),dv.memory.size(),dks.memory.size(),dvs.memory.size(),
      dqb.memory.size(),dqf.memory.size(),ob.memory.size(),of.memory.size(),
      dt.memory.size(),di.memory.size(),ds.memory.size(),pk.memory.size(),pv.memory.size(),bad.memory.size(),score_probe.memory.size(),prob_probe.memory.size(),state_probe.memory.size(),round_probe.memory.size()}) bytes+=n;
  require(bytes<(size_t(2)<<30),"device footprint budget");
  QsaAttnPools pools{};pools.k_q=dk.data();pools.v_q=dv.data();
  pools.k_scale=dks.data();pools.v_scale=dvs.data();pools.page_table=dt.data();
  auto shape=qsa_real_shapes();shape.page_size=Page;
  auto issue=[&] {
    bool ok=false;
    if (arm=="base") ok=qsa_bf16_numerical_probe_batch(dqb.data(),pools,di.data(),ds.data(),Cap,shape,
        ob.data()+16,nq,{Pages,Pages},{2,13,score_probe.data()+16,prob_probe.data()+16,round_probe.data()+16,state_probe.data()+16},&rt->compute());
    if (arm=="packed") ok=qsa_bf16_packed_batch(dqb.data(),pools,di.data(),ds.data(),Cap,shape,
        ob.data()+16,nq,{Pages,Pages},{pk.data()+16,pv.data()+16,bad.data()+16},&rt->compute());
    require(ok,"selected matrix entry unavailable; no fallback");
  };
  auto values=[&] {
    auto b=ob.get();auto f=of.get();
    { auto a=pk.get(),v=pv.get();auto flags=bad.get();
      for (size_t i=0;i<16;++i)
        require(a[i]==guard && a[a.size()-16+i]==guard && v[i]==guard && v[v.size()-16+i]==guard &&
            flags[i]==bad_guard && flags[flags.size()-16+i]==bad_guard,"packed allocation guards");
      if (arm=="base") {
        require(std::all_of(a.begin(),a.end(),[=](uint16_t x){return x==guard;}) &&
                std::all_of(v.begin(),v.end(),[=](uint16_t x){return x==guard;}) &&
                std::all_of(flags.begin(),flags.end(),[=](int32_t x){return x==bad_guard;}),"base leaves every unused packed byte intact");
      } else for (size_t i=16;i<flags.size()-16;++i) require(flags[i]==0,"all physical finite-scale diagnostics");
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
           <<" pack_cost="<<(arm=="packed" ? "IN_TIMED_CALL" : "NONE")<<" warmups=2 repetitions="<<repetitions
           <<" fingerprints=FNV1a64_native_bytes q="<<fingerprint(qb)<<" k="<<fingerprint(k)
           <<" v="<<fingerprint(v)<<" ks="<<fingerprint(ks)<<" vs="<<fingerprint(vs)
           <<" table="<<fingerprint(table)<<" ids="<<fingerprint(ids)<<" steps="<<fingerprint(steps)<<'\n';
  issue();rt->wait();const auto initial=values();
  const auto gs=score_probe.get(),gp=prob_probe.get(),gst=state_probe.get();const auto gr=round_probe.get();
  for (size_t i=0;i<16;++i) require(gs[i]==1234.f && gs[gs.size()-16+i]==1234.f &&
      gp[i]==1234.f && gp[gp.size()-16+i]==1234.f && gst[i]==1234.f && gst[gst.size()-16+i]==1234.f &&
      gr[i]==guard && gr[gr.size()-16+i]==guard,"probe guards");
  for (int c=0;c<Cap;++c) require(std::isfinite(gs[16+c]) && std::isfinite(gp[16+c]) &&
      gp[16+c]>0.f && gp[16+c]<=1.f && gr[16+c]==bf(gp[16+c]),"captured finite probability and BF16 cast");
  std::cout<<"probe_identity scores_FNV1a64="<<fingerprint(gs)<<" probabilities_FNV1a64="<<fingerprint(gp)
           <<" rounded_FNV1a64="<<fingerprint(gr)<<" states_FNV1a64="<<fingerprint(gst)<<'\n';

  // Independent scalar tensor oracle at four complete queries. Only the new
  // BF16 path is judged against this numerical contract. Legacy paths retain
  // their earlier qualification; they are not reference outputs for BF16.
  double max_error=0,max_ratio=0;size_t compared=0,failed=0;
  for (int b=0;b<nq;++b) {
    if (b!=0 && b!=1 && b!=2 && b!=nq-1) continue;
    for (int h=0;h<H;++h) {
      std::vector<float> score(Cap),value(size_t(Cap)*D),acc(D,0.f),cpu_p(Cap),cpu_states(Tiles*4);
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
      for (int first=0;first<Cap;first+=32) {
        const int last=std::min(Cap,first+32);float next=maximum,sum=0;
        for (int c=first;c<last;++c) next=std::max(next,score[c]);
        const float alpha=norm>0 ? std::exp2(maximum-next) : 0.f;
        std::vector<float> p(size_t(last-first));
        for (int c=first;c<last;++c) { const float x=std::exp2(score[c]-next);sum+=x;p[c-first]=unbf(bf(x));cpu_p[c]=x; }
        for (int d=0;d<D;++d) {
          double term=0;for (int c=first;c<last;++c) term+=double(p[c-first])*value[size_t(c)*D+d];
          acc[d]=float(double(acc[d]*alpha)+term);
        }
        norm=std::fma(norm,alpha,sum);maximum=next;
        cpu_states[(first/32)*4]=next;cpu_states[(first/32)*4+1]=alpha;cpu_states[(first/32)*4+2]=norm;cpu_states[(first/32)*4+3]=sum;
      }
      for (int d=0;d<D;++d) {
        const float ref=unbf(bf(acc[d]/norm));
        const double error=std::fabs(double(initial[(size_t(b)*H+h)*D+d])-ref);
        const uint16_t mag=bf(std::fabs(ref));
        const double budget=double(unbf(uint16_t(mag+1))-unbf(mag))+2e-6;
        max_error=std::max(max_error,error);max_ratio=std::max(max_ratio,error/budget);
        if (error>budget) {
          if (failed<32) std::cout<<"mismatch b="<<b<<" head="<<h<<" d="<<d
              <<" ref="<<ref<<" actual="<<initial[(size_t(b)*H+h)*D+d]
              <<" abs="<<error<<" budget="<<budget<<" ratio="<<error/budget<<'\n';
          ++failed;
        }
        ++compared;
      }
      if (b==2 && h==13) {
        size_t score_differences=0,probability_cast_differences=0,score_path_cast_changes=0,exp2_cast_changes=0;
        double maximum_score_error=0,maximum_exp2_error=0;uint32_t maximum_exp2_ulps=0;
        for (int c=0;c<Cap;++c) {
          const int t=c/32;const float gpu_score=gs[16+c],gpu_p=gp[16+c];
          const float same_argument=std::exp2(gpu_score-gst[16+t*4]);
          score_differences+=score[c]!=gpu_score;
          maximum_score_error=std::max(maximum_score_error,std::fabs(double(score[c])-gpu_score));
          maximum_exp2_error=std::max(maximum_exp2_error,std::fabs(double(same_argument)-gpu_p));
          const auto u=std::bit_cast<uint32_t>(same_argument),v=std::bit_cast<uint32_t>(gpu_p);
          maximum_exp2_ulps=std::max(maximum_exp2_ulps,u>v ? u-v : v-u);
          score_path_cast_changes+=bf(cpu_p[c])!=bf(same_argument);
          exp2_cast_changes+=bf(same_argument)!=gr[16+c];
          if (bf(cpu_p[c])!=gr[16+c]) {
            if (probability_cast_differences<32)
              std::cout<<"probability_difference cell="<<c<<" cpu_score="<<score[c]<<" gpu_score="<<gpu_score
                       <<" cpu_max="<<cpu_states[t*4]<<" gpu_max="<<gst[16+t*4]
                       <<" cpu_p="<<cpu_p[c]<<" gpu_p="<<gpu_p<<" host_exp2_gpu_argument="<<same_argument
                       <<" cpu_bf16="<<bf(cpu_p[c])<<" gpu_bf16="<<gr[16+c]<<'\n';
            ++probability_cast_differences;
          }
        }
        std::vector<float> replay(D,0.f);size_t replay_failed=0,replay_exact_differences=0,original_failed=0,explained=0;
        double replay_max_error=0;
        for (int first=0;first<Cap;first+=32) {
          const int last=std::min(Cap,first+32),t=first/32;const float a=gst[16+t*4+1];
          for (int d=0;d<D;++d) {
            double term=0;for (int c=first;c<last;++c) term+=double(unbf(gr[16+c]))*value[size_t(c)*D+d];
            replay[d]=float(double(replay[d]*a)+term);
          }
        }
        const float final_norm=gst[16+(Tiles-1)*4+2];require(final_norm>0 && std::isfinite(final_norm),"captured final norm");
        for (int d=0;d<D;++d) {
          const float original=unbf(bf(acc[d]/norm)),ref=unbf(bf(replay[d]/final_norm));
          const float actual=initial[(size_t(b)*H+h)*D+d];
          const uint16_t mag=bf(std::fabs(ref)),omag=bf(std::fabs(original));
          const double budget=double(unbf(uint16_t(mag+1))-unbf(mag))+2e-6;
          const double original_budget=double(unbf(uint16_t(omag+1))-unbf(omag))+2e-6;
          const double error=std::fabs(double(actual)-ref);replay_max_error=std::max(replay_max_error,error);
          replay_failed+=error>budget;replay_exact_differences+=actual!=ref;
          if (std::fabs(double(actual)-original)>original_budget) { ++original_failed;explained+=error<=budget; }
        }
        std::cout<<"probe_metrics score_differences="<<score_differences<<" max_score_abs="<<maximum_score_error
                 <<" p_bf16_differences="<<probability_cast_differences<<" score_path_bf16_changes="<<score_path_cast_changes
                 <<" same_argument_exp2_bf16_changes="<<exp2_cast_changes<<" max_same_argument_exp2_abs="<<maximum_exp2_error
                 <<" max_same_argument_exp2_ulps="<<maximum_exp2_ulps<<'\n';
        std::cout<<"captured_coefficient_replay compared="<<D<<" failed="<<replay_failed<<" exact_differences="<<replay_exact_differences
                 <<" max_abs="<<replay_max_error<<" original_failed="<<original_failed<<" explained="<<explained
                 <<" original_FP64_oracle_status=UNCHANGED\n";
      }
    }
  }
  require(dk.get()==k && dv.get()==v && dks.get()==ks && dvs.get()==vs && dqb.get()==qb && dqf.get()==qf &&
          dt.get()==table && di.get()==ids && ds.get()==steps,"all input bytes immutable");
  require(values()==initial,"unchanged output after host oracle/input inspection");
  std::cout<<"case status=COMPLETE guards=PASS inputs=PASS full_finite=PASS"
           <<" numerical_contract="<<(failed ? "FAIL" : "PASS")<<" compared="<<compared
           <<" failed="<<failed<<" max_abs="<<max_error<<" max_budget_ratio="<<max_ratio
           <<" output_FNV1a64="<<fingerprint(initial)
           <<" speed_samples=NONE model_PP=UNRUN physical262144_model=UNRUN\n";

}
} // namespace
int main(int argc,char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);std::cout<<std::setprecision(12);
    require(argc==9 && std::string(argv[1])=="--arm" && std::string(argv[3])=="--nq" &&
            std::string(argv[5])=="--repeats" && std::string(argv[7])=="--cells","--arm base|packed --nq 3|8192 --repeats 1|10 --cells 2113|32768|262144");
    const std::string arm=argv[2];const int nq=std::stoi(argv[4]),reps=std::stoi(argv[6]),cells=std::stoi(argv[8]);
    require(arm=="base" && (nq==3 || nq==8192) &&
            (reps==1 || reps==10) && (cells==2113 || cells==32768 || cells==262144),"bounded benchmark arguments");
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
    std::cout<<"diagnostic_status=COMPLETE model_PP=UNRUN physical262144_model=UNRUN RESTORE=UNRUN faults=EXTERNAL_REQUIRED\n";return 0;
  } catch (const std::exception &e) { drain();std::cerr<<e.what()<<'\n';return 1; }
}
