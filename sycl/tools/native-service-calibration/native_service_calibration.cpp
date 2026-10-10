// Actual selected payloads; bounded host-only service characterization, not an engine benchmark.
#include "strata/artifact/native_role_plan.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/core/progress.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif
namespace c = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
static constexpr size_t kMaxNT=4,kSourceCases=384,kCohortCases=192,kHotCases=8;
static constexpr double kReferenceRmsFloor=1e-12;
static_assert(kCohortCases%kHotCases==0,"hot source ranks must divide source cohort");
static_assert(kMaxNT<=c::MAXT,"harness token bound exceeds production");
static void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
static void result(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }
static uint64_t fnv(const void* p,size_t n) { uint64_t h=14695981039346656037ULL; auto b=static_cast<const uint8_t*>(p); for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ULL;} return h; }
static std::string quote(const std::string& s) { std::string r="\""; for(char v:s){require(v!='\n'&&v!='\r',"newline in metadata");if(v=='\"')r+='\"';r+=v;}return r+'\"'; }
struct Affinity { c::ThreadAffinity old; explicit Affinity(int core):old(c::pin_current_thread(core)){require(old.valid,"host affinity pin failed");} ~Affinity(){c::restore_thread_affinity(old);} };
static int host_cpu() {
#ifdef __linux__
 return sched_getcpu();
#else
 return -1;
#endif
}
static void thread_placement(int tasks,const c::CpuTopology& topology){
#ifdef __linux__
 size_t threads=0;
 std::set<int> observed,expected{topology.host_core};
 require(topology.worker_cores.size()>=5,"five worker CPU plans required");
 for(int i=0;i<5;++i)require(expected.insert(topology.worker_cores[i]).second,"duplicate planned CPU");
 for(const auto& entry:std::filesystem::directory_iterator("/proc/self/task")){
  require(++threads<=128,"thread audit bound");
  std::ifstream file(entry.path()/"status",std::ios::binary);require(bool(file),"thread status open");
  std::array<char,16384> bytes{};file.read(bytes.data(),bytes.size());auto count=file.gcount();
  require(count>0&&count<static_cast<std::streamsize>(bytes.size()),"thread status byte bound");
  std::istringstream lines(std::string(bytes.data(),static_cast<size_t>(count)));
  std::string line,allowed;while(std::getline(lines,line))if(line.rfind("Cpus_allowed_list:",0)==0)allowed=line.substr(18);
  require(!allowed.empty(),"missing thread affinity");
  std::cout<<"PLACEMENT,observed_allowed_list,"<<tasks<<','<<quote(entry.path().filename().string())<<','<<quote(allowed)<<'\n';
  const auto begin=allowed.find_first_not_of(" \t");require(begin!=std::string::npos,"empty CPU list");
  const auto cpu_text=allowed.substr(begin);
  require(std::all_of(cpu_text.begin(),cpu_text.end(),[](char ch){return ch>='0'&&ch<='9';}),"thread must have singleton CPU affinity");
  require(observed.insert(std::stoi(cpu_text)).second,"threads share a planned CPU");
 }
 require(threads==6&&observed==expected,"actual thread affinity differs from six CPU plan");
#else
 (void)topology;
 std::cout<<"PLACEMENT,observed_allowed_list,unsupported,"<<tasks<<'\n';
#endif
}
template<size_t N> struct alignas(64) Floats { std::array<float,N> data{}; std::array<uint8_t,64> guard{}; void reset(){data.fill(std::numeric_limits<float>::quiet_NaN());guard.fill(0xa5);} void check()const{for(float v:data)require(std::isfinite(v),"nonfinite/incomplete output");for(auto v:guard)require(v==0xa5,"float row guard overwritten");} };
template<size_t N> struct alignas(64) Bytes { std::array<uint8_t,N+64> data{}; void reset(){data.fill(0xa5);} void check(size_t used)const{require(used<=N,"quant buffer too small");for(size_t j=used;j<data.size();++j)require(data[j]==0xa5,"quant tail overwritten");} };
struct Case {int layer,expert; c::NativeFmt fmt; std::vector<uint8_t> blob;};
struct alignas(64) Q2Buffer {
 c::ActQ value{};
 std::array<uint8_t,64> guard{};
 void reset(){value={};guard.fill(0xa5);}
 void check()const{for(auto v:guard)require(v==0xa5,"ActQ guard overwritten");}
};
static_assert(sizeof(Q2Buffer)<=(64ULL<<10),"local quantizer scratch bound");
static_assert(sizeof(c::ExpertJobMulti)*6<=(64ULL<<10),"local pool-job array bound");
static_assert(sizeof(c::ExpertPool)<=(256ULL<<10),"local pool owner bound");
static void check_q2(const float* ff,const Q2Buffer& actual){
 Q2Buffer ref;ref.reset();
 if(c::cpu_avx512_ok())c::act_quant_q8_1(ff,c::FF,ref.value);
 else c::act_quant_q8_1_avx2(ff,c::FF,ref.value);
 actual.check();ref.check();const auto& a=actual.value;const auto& r=ref.value;
 require(a.nchunks==c::FF/c::QKA&&a.nchunks==r.nchunks,"Q2 chunk geometry");
 require(!std::memcmp(a.q,r.q,c::FF)&&!std::memcmp(a.scale,r.scale,a.nchunks*sizeof(float))&&
 !std::memcmp(a.sum,r.sum,a.nchunks*sizeof(int32_t))&&!std::memcmp(a.hx,r.hx,a.nchunks*sizeof(float)),"Q2 meaningful fields differ from selected production quantizer");
 require(a.bp_pairs==r.bp_pairs&&a.bp_pairs>=0&&a.bp_pairs<=c::FF/128,"Q2 bitplane geometry");
 if(a.bp_pairs)require(!std::memcmp(a.qp,r.qp,a.bp_pairs*128)&&!std::memcmp(a.psum,r.psum,a.bp_pairs*8*sizeof(int32_t))&&
 !std::memcmp(a.pscale,r.pscale,a.bp_pairs*8*sizeof(float)),"Q2 bitplane fields differ");
}
static void environment_metadata(){
 require(strata::core::release_gpu_fn().load()==nullptr,"GPU release callback registered");
 for(const char* key:{"STRATA_FORCE_ISA","STRATA_FORCE_AVX2","STRATA_NO_AVXVNNI","STRATA_NO_Q8K_AVX2","STRATA_IQ_MT_MIN","STRATA_IQ3S_MT1","STRATA_IQ_PREFETCH","STRATA_NO_IQ512","STRATA_NO_IQ256","STRATA_NO_IQ4NL","STRATA_KQ256","STRATA_IQ256_GATHER","STRATA_Q2_BITPLANE","STRATA_POOL_SPIN_US","STRATA_HOST_CORE","STRATA_NATIVE_DISPATCH_HISTOGRAM"}){
  const char* value=std::getenv(key);std::cout<<"ENV,"<<key<<','<<quote(value?value:"<unset>")<<'\n';
 }
 require(std::getenv("STRATA_FORCE_ISA")==nullptr&&std::getenv("STRATA_FORCE_AVX2")==nullptr,"feature-forcing environments forbidden");
 require(std::getenv("STRATA_NATIVE_DISPATCH_HISTOGRAM")==nullptr,"diagnostic histogram must be unset for this service fixture");
}
// No shared engine state: fixed inputs are keyed by layer/token, never expert or cohort.
struct LayerInput {
 std::array<std::array<float,c::H>,kMaxNT> x{};
 std::array<Bytes<c::kNativeActBytes>,kMaxNT> a;
};
struct Buffers {
 std::array<Bytes<c::kNativeHBytes>,kMaxNT> hq,hq_ref,hq_chain_ref,hq_expected;
 std::array<Q2Buffer,kMaxNT> q2,q2_expected;
 std::array<Floats<c::FF>,kMaxNT> ff,ff_ref,ff_expected;
 std::array<Floats<c::H>,kMaxNT> out,pout,down_ref,chain_ref,out_expected;
 const void* ap[kMaxNT]{};const void* hp[kMaxNT]{};const c::ActQ* qp[kMaxNT]{};
 float* fp[kMaxNT]{};float* op[kMaxNT]{};
 void bind(const LayerInput& input,int nt){require(nt>=1&&nt<=int(kMaxNT),"bind NT bound");for(int t=0;t<nt;++t){ap[t]=input.a[t].data.data();hp[t]=hq[t].data.data();qp[t]=&q2[t].value;fp[t]=ff[t].data.data();op[t]=out[t].data.data();}}
};
static void quant_ff(const c::NativeFmt& f,Buffers& b,int nt){
 for(int t=0;t<nt;++t) {
  if(f.d_type==42)c::act_quant_any(b.ff[t].data.data(),c::FF,b.q2[t].value);
  else c::native_quant_h(f,b.ff[t].data.data(),b.hq[t].data.data());
 }
}
static void down(const Case& item,Buffers& b,int nt){
 if(item.fmt.d_type==42)c::q2_rows_any(item.blob.data()+item.fmt.down_off,item.fmt.d_row,c::FF/c::QK,b.qp,nt,b.op,0,c::H);
 else c::native_down_rows(item.fmt,item.blob.data(),b.hp,nt,b.op,0,c::H);
}
static void gu(const Case& item,Buffers& b,int nt){c::native_gu_rows(item.fmt,item.blob.data(),b.ap,nt,b.fp,0,c::FF);}
static void full(const Case& item,Buffers& b,int nt){gu(item,b,nt);quant_ff(item.fmt,b,nt);down(item,b,nt);}
static void check_quant(const c::NativeFmt& f,Buffers& b,int nt){
 for(int t=0;t<nt;++t){
  b.ff[t].check();
  if(f.d_type==42)check_q2(b.ff[t].data.data(),b.q2[t]);
  else {
   auto tr=ggml_get_type_traits_cpu(static_cast<ggml_type>(f.d_act));require(tr&&tr->from_float,"Down quantizer absent");
   b.hq_ref[t].reset();tr->from_float(b.ff[t].data.data(),b.hq_ref[t].data.data(),c::FF);
   b.hq[t].check(f.h_bytes);b.hq_ref[t].check(f.h_bytes);
   require(!std::memcmp(b.hq[t].data.data(),b.hq_ref[t].data.data(),f.h_bytes),"Down quantization differs from ggml");
  }
 }
}
static void q2_repeat(const Q2Buffer& actual,const Q2Buffer& expected){
 const auto& a=actual.value;const auto& e=expected.value;actual.check();expected.check();
 require(a.nchunks==e.nchunks&&a.nchunks==c::FF/c::QKA&&a.bp_pairs==e.bp_pairs,"Q2 repeat geometry");
 require(!std::memcmp(a.q,e.q,c::FF)&&!std::memcmp(a.scale,e.scale,a.nchunks*sizeof(float))&&
 !std::memcmp(a.sum,e.sum,a.nchunks*sizeof(int32_t))&&!std::memcmp(a.hx,e.hx,a.nchunks*sizeof(float)),"Q2 repeat fields mismatch");
 if(a.bp_pairs)require(!std::memcmp(a.qp,e.qp,a.bp_pairs*128)&&!std::memcmp(a.psum,e.psum,a.bp_pairs*8*sizeof(int32_t))&&
 !std::memcmp(a.pscale,e.pscale,a.bp_pairs*8*sizeof(float)),"Q2 repeat bitplane mismatch");
}
static_assert(sizeof(Buffers)*kSourceCases<=(512ULL<<20),"work buffer heap budget");
static_assert(sizeof(LayerInput)*48<=(8ULL<<20),"layer input heap budget");
struct ErrorMetrics {double maxabs=0,nrms=0,relative_refmax=0,ref_rms=0,error_rms=0,ref_max=0;size_t rows=0;};
struct Tolerance {double maxabs=0,nrms=0;bool have_abs=false,have_nrms=false;};
using Gates=std::array<Tolerance,3>; // GU/SwiGLU, staged Down, independent full chain.
static ErrorMetrics error_metrics(const float* actual,const float* reference,size_t n){
 require(n>0&&n<=size_t(c::H),"reference output count bound");
 ErrorMetrics result;double error2=0,reference2=0;
 for(size_t row=0;row<n;++row){
  require(std::isfinite(actual[row])&&std::isfinite(reference[row]),"nonfinite reference/actual row");
  const double a=actual[row],r=reference[row],d=a-r;
  result.maxabs=std::max(result.maxabs,std::abs(d));result.ref_max=std::max(result.ref_max,std::abs(r));
  error2+=d*d;reference2+=r*r;++result.rows;
 }
 require(std::isfinite(error2)&&std::isfinite(reference2),"reference error accumulation nonfinite");
 result.error_rms=std::sqrt(error2/n);result.ref_rms=std::sqrt(reference2/n);
 result.nrms=result.error_rms/std::max(result.ref_rms,kReferenceRmsFloor);
 result.relative_refmax=result.maxabs/std::max(result.ref_max,kReferenceRmsFloor);
 require(std::isfinite(result.nrms)&&std::isfinite(result.relative_refmax),"reference normalization nonfinite");
 return result;
}
static const char* reference_stage(int stage){return stage==0?"GU_SwiGLU":stage==1?"Down_actual_FF":"full_chain_independent_FF";}
static std::array<ErrorMetrics,3> ggml_reference(const Case& item,Buffers& b,int nt,size_t case_index,const Gates& gates){
 auto gu_traits=ggml_get_type_traits_cpu(static_cast<ggml_type>(item.fmt.gu_type));
 auto down_traits=ggml_get_type_traits_cpu(static_cast<ggml_type>(item.fmt.d_type));
 auto htr=ggml_get_type_traits_cpu(static_cast<ggml_type>(item.fmt.d_act));
 require(gu_traits&&gu_traits->vec_dot&&down_traits&&down_traits->vec_dot&&htr&&htr->from_float,"ggml reference traits absent");
 std::array<ErrorMetrics,3> worst{};
 for(int t=0;t<nt;++t){
  b.ff_ref[t].reset();b.down_ref[t].reset();b.chain_ref[t].reset();
  for(int r=0;r<c::FF;++r){float g=0,u=0;
   gu_traits->vec_dot(c::H,&g,0,item.blob.data()+r*item.fmt.gu_row,0,b.ap[t],0,1);
   gu_traits->vec_dot(c::H,&u,0,item.blob.data()+item.fmt.up_off+r*item.fmt.gu_row,0,b.ap[t],0,1);
   require(std::isfinite(g)&&std::isfinite(u),"nonfinite GGML gate/up");
   b.ff_ref[t].data[r]=(g/(1.f+std::exp(-g)))*u;
  }
  b.ff_ref[t].check();b.ff[t].check();b.out[t].check();
  b.hq_ref[t].reset();htr->from_float(b.ff[t].data.data(),b.hq_ref[t].data.data(),c::FF);b.hq_ref[t].check(item.fmt.h_bytes);
  b.hq_chain_ref[t].reset();htr->from_float(b.ff_ref[t].data.data(),b.hq_chain_ref[t].data.data(),c::FF);b.hq_chain_ref[t].check(item.fmt.h_bytes);
  for(int r=0;r<c::H;++r){
   down_traits->vec_dot(c::FF,&b.down_ref[t].data[r],0,item.blob.data()+item.fmt.down_off+r*item.fmt.d_row,0,b.hq_ref[t].data.data(),0,1);
   down_traits->vec_dot(c::FF,&b.chain_ref[t].data[r],0,item.blob.data()+item.fmt.down_off+r*item.fmt.d_row,0,b.hq_chain_ref[t].data.data(),0,1);
  }
  b.down_ref[t].check();b.chain_ref[t].check();
  const std::array<ErrorMetrics,3> metrics={error_metrics(b.ff[t].data.data(),b.ff_ref[t].data.data(),c::FF),
   error_metrics(b.out[t].data.data(),b.down_ref[t].data.data(),c::H),error_metrics(b.out[t].data.data(),b.chain_ref[t].data.data(),c::H)};
  for(int stage=0;stage<3;++stage){const auto& m=metrics[stage];
   worst[stage].maxabs=std::max(worst[stage].maxabs,m.maxabs);worst[stage].nrms=std::max(worst[stage].nrms,m.nrms);
   worst[stage].relative_refmax=std::max(worst[stage].relative_refmax,m.relative_refmax);
   std::cout<<std::setprecision(17)<<"REFERENCE_TOKEN,"<<case_index<<','<<item.layer<<','<<item.expert<<','<<nt<<','<<t<<','<<reference_stage(stage)<<','<<m.maxabs<<','<<m.nrms<<','<<m.relative_refmax<<','<<m.ref_rms<<','<<m.error_rms<<','<<m.ref_max<<','<<m.rows<<'\n';
   if(nt>1){std::cout.flush();require(gates[stage].have_abs&&gates[stage].have_nrms,"NT2-4 missing predeclared tolerance");
    require(m.maxabs<=gates[stage].maxabs&&m.nrms<=gates[stage].nrms,"predeclared GGML maxabs/NRMS gate failed");}
  }
  if(nt==1){
   require(!std::memcmp(b.ff_ref[t].data.data(),b.ff[t].data.data(),sizeof(float)*c::FF),"NT1 GU differs from independent ggml");
   if(item.fmt.d_type==20){
    require(!std::memcmp(b.down_ref[t].data.data(),b.out[t].data.data(),sizeof(float)*c::H),"NT1 Down20 differs from independent ggml");
    require(!std::memcmp(b.chain_ref[t].data.data(),b.out[t].data.data(),sizeof(float)*c::H),"NT1 full-chain Down20 differs from independent ggml");
   }
  }
 }
 return worst;
}
static uint64_t decimal(const std::string& s){
 require(!s.empty()&&s.size()<=20,"decimal width");uint64_t value=0;
 for(char ch:s){require(ch>='0'&&ch<='9',"nondecimal argument");unsigned d=ch-'0';require(value<=(UINT64_MAX-d)/10,"decimal overflow");value=value*10+d;}
 return value;
}
struct Selection {int cohort,layer,expert;};
static std::vector<Selection> read_cohort(const std::string& path){
 require(std::filesystem::is_regular_file(path),"cohort not regular");auto size=std::filesystem::file_size(path);
 require(size>0&&size<=16384,"cohort size bound");std::ifstream in(path,std::ios::binary);require(bool(in),"cohort open");
 std::string text(size,'\0');in.read(text.data(),static_cast<std::streamsize>(size));require(in.gcount()==static_cast<std::streamsize>(size)&&in.peek()==EOF,"cohort short/changed");
 require(text.back()=='\n',"cohort final LF required");std::istringstream lines(text);std::string line;
 require(bool(std::getline(lines,line))&&line=="cohort\tlayer\texpert","cohort header");
 std::set<std::pair<int,int>> seen;std::array<std::vector<Selection>,2> split;
 while(std::getline(lines,line)){
  auto a=line.find('\t'),b=line.find('\t',a==std::string::npos?0:a+1);
  require(a!=std::string::npos&&b!=std::string::npos&&line.find('\t',b+1)==std::string::npos,"cohort columns");
  auto label=line.substr(0,a);require(label=="train"||label=="holdout","cohort label");
  uint64_t l=decimal(line.substr(a+1,b-a-1)),e=decimal(line.substr(b+1));require(l<48&&e<512,"cohort ID range");
  require(seen.emplace(int(l),int(e)).second,"duplicate/global train-holdout overlap");int cidx=label=="train"?0:1;
  split[cidx].push_back({cidx,int(l),int(e)});require(split[cidx].size()<=192,"cohort count cap");
 }
 require(split[0].size()==192&&split[1].size()==192,"exact192 train and holdout required");
 std::vector<Selection> all;all.reserve(384);for(const auto& rows:split)all.insert(all.end(),rows.begin(),rows.end());
 std::cout<<"META,cohort_fnv,"<<fnv(text.data(),text.size())<<'\n';return all;
}
// SplitMix64 modulo Fisher-Yates: explicit deterministic bounded arithmetic, not unbiased statistical sampling.
static uint64_t rng(uint64_t& state){state+=0x9e3779b97f4a7c15ULL;uint64_t z=state;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
using Order=std::vector<size_t>;
using Schedule=std::array<std::array<Order,28>,2>;
static void order_check(const Order& order,int cohort,bool hot){
 require(order.size()==(hot?kHotCases:kCohortCases),"active order count");std::array<bool,kCohortCases> seen{};
 for(size_t index:order){require(index<kSourceCases&&index/kCohortCases==size_t(cohort),"active order cohort/index");
  size_t rank=index%kCohortCases;require(!seen[rank]&&(!hot||rank%(kCohortCases/kHotCases)==0),"active order duplicate/not selected hot ID");seen[rank]=true;}
}
static uint64_t order_hash(const Order& order){uint64_t h=14695981039346656037ULL;for(size_t v:order){for(unsigned shift=0;shift<32;shift+=8){h^=(v>>shift)&255;h*=1099511628211ULL;}}return h;}
static Schedule schedule(uint64_t seed,bool hot){
 Schedule result{};uint64_t state=seed;const size_t count=hot?kHotCases:kCohortCases;
 for(int cohort=0;cohort<2;++cohort)for(int round=0;round<28;++round){auto& order=result[cohort][round];order.resize(count);
  for(size_t i=0;i<count;++i)order[i]=size_t(cohort)*kCohortCases+(hot?i*(kCohortCases/kHotCases):i);
  for(size_t i=count-1;i>0;--i){size_t j=rng(state)%(i+1);std::swap(order[i],order[j]);}
 }
 return result;
}

enum class Arm {GU,Q,Down,Full,Pool};
static const char* name(Arm a){switch(a){case Arm::GU:return "GU";case Arm::Q:return "FFquant";case Arm::Down:return "Down";case Arm::Full:return "direct_complete";case Arm::Pool:return "pool";}return "invalid";}
static void prepare(Arm arm,const Case& item,Buffers& b,int nt){
 for(int t=0;t<nt;++t){
  if(arm==Arm::GU||arm==Arm::Full)b.ff[t].reset();else b.ff[t]=b.ff_expected[t];
  if(arm==Arm::Q||arm==Arm::Full){b.hq[t].reset();b.q2[t].reset();}
  else {b.hq[t]=b.hq_expected[t];b.q2[t]=b.q2_expected[t];}
  b.out[t].reset();b.pout[t].reset();
 }
 (void)item;
}
static uint64_t check_round(Arm arm,const Case& item,Buffers& b,int nt){
 uint64_t checksum=14695981039346656037ULL;
 if(arm==Arm::Q||arm==Arm::Full)check_quant(item.fmt,b,nt);
 for(int t=0;t<nt;++t){
  b.ff[t].check();b.ff_expected[t].check();b.out_expected[t].check();
  b.ff_ref[t].check();b.down_ref[t].check();b.chain_ref[t].check();
  b.hq_ref[t].check(item.fmt.h_bytes);b.hq_chain_ref[t].check(item.fmt.h_bytes);
  if(item.fmt.d_type==20)b.hq[t].check(item.fmt.h_bytes);else b.q2[t].check();
  if(arm!=Arm::Pool){b.ff[t].check();require(!std::memcmp(b.ff[t].data.data(),b.ff_expected[t].data.data(),c::FF*sizeof(float)),"GU repeat mismatch");}
  if(arm==Arm::Q||arm==Arm::Full){
   if(item.fmt.d_type==20)require(!std::memcmp(b.hq[t].data.data(),b.hq_expected[t].data.data(),item.fmt.h_bytes),"FFquant repeat mismatch");
   else q2_repeat(b.q2[t],b.q2_expected[t]);
  }
  if(arm==Arm::Down||arm==Arm::Full||arm==Arm::Pool){
   const auto& output=arm==Arm::Pool?b.pout[t]:b.out[t];output.check();
   require(!std::memcmp(output.data.data(),b.out_expected[t].data.data(),c::H*sizeof(float)),"Down/pool repeat/partition mismatch");
   checksum^=fnv(output.data.data(),c::H*sizeof(float));
  }else checksum^=fnv(b.ff[t].data.data(),c::FF*sizeof(float));
  checksum*=1099511628211ULL;
 }
 return checksum;
}
static void input_check(const c::NativeFmt& f,const LayerInput& input,int nt){
 auto tr=ggml_get_type_traits_cpu(static_cast<ggml_type>(f.gu_act));require(tr&&tr->from_float,"GU quantizer absent");
 for(int t=0;t<nt;++t){for(float value:input.x[t])require(std::isfinite(value),"nonfinite layer input");Bytes<c::kNativeActBytes> ref;ref.reset();tr->from_float(input.x[t].data(),ref.data.data(),c::H);ref.check(f.act_bytes);input.a[t].check(f.act_bytes);require(!std::memcmp(input.a[t].data.data(),ref.data.data(),f.act_bytes),"input quantization differs from ggml");}
}
int main(int argc,char** argv){try{
 require(argc>=10&&argc<=24,"usage: native_service_calibration PACK PRIMARY TSV GU DOWN NT TASKS BATCH SEED [--correctness-only] [--hot-eight] [--gu-maxabs V --gu-nrms V --down-maxabs V --down-nrms V --chain-maxabs V --chain-nrms V]; all six gates required NT2-4");
 for(int path_arg=1;path_arg<=3;++path_arg)require(std::strlen(argv[path_arg])>0&&std::strlen(argv[path_arg])<=4096,"source path argument width");
 bool correctness_only=false,hot=false;Gates gates{};
 for(int arg=10;arg<argc;++arg){const std::string option=argv[arg];
  if(option=="--correctness-only"){require(!correctness_only,"duplicate correctness-only");correctness_only=true;continue;}
  if(option=="--hot-eight"){require(!hot,"duplicate hot-eight");hot=true;continue;}
  int stage=-1;bool absolute=false;
  for(int candidate=0;candidate<3;++candidate){const std::string prefix=candidate==0?"--gu-":candidate==1?"--down-":"--chain-";
   if(option==prefix+"maxabs"){stage=candidate;absolute=true;}if(option==prefix+"nrms"){stage=candidate;absolute=false;}
  }
  require(stage>=0&&arg+1<argc,"unknown option or missing tolerance value");const std::string text=argv[++arg];
  require(!text.empty()&&text.size()<=64&&std::all_of(text.begin(),text.end(),[](char v){return(v>='0'&&v<='9')||v=='.'||v=='e'||v=='E'||v=='+'||v=='-';}),"tolerance syntax/width");
  char* end=nullptr;errno=0;const double value=std::strtod(text.c_str(),&end);
  require(end==text.c_str()+text.size()&&errno!=ERANGE&&std::isfinite(value)&&value>=0&&value<=1e6,"finite nonnegative tolerance <=1e6 required");
  auto& gate=gates[stage];if(absolute){require(!gate.have_abs,"duplicate maxabs gate");gate.maxabs=value;gate.have_abs=true;}
  else {require(!gate.have_nrms,"duplicate NRMS gate");gate.nrms=value;gate.have_nrms=true;}
 }
 uint64_t gu_type=decimal(argv[4]),down_type=decimal(argv[5]),nt_value=decimal(argv[6]),tasks_value=decimal(argv[7]),batch_value=decimal(argv[8]),seed=decimal(argv[9]);
 require(gu_type==18||gu_type==21||gu_type==22||gu_type==23,"GU cell type");require(down_type==20||down_type==42,"Down cell type");
 require(nt_value>=1&&nt_value<=kMaxNT,"NT cell1-4");require(tasks_value==0||tasks_value==6,"tasks cell");require(batch_value==1||batch_value==6,"batch cell");
 require(!hot||batch_value==1,"hot-eight pool batch must be1; no padding/dropped jobs");
 require(kCohortCases%batch_value==0&&(!hot||kHotCases%batch_value==0),"pool batch must exactly divide active cohort");
 if(nt_value>1)for(const auto& gate:gates)require(gate.have_abs&&gate.have_nrms,"all six predeclared tolerance values required before NT2-4 preparation");
 for(int stage=0;stage<3;++stage){const auto& gate=gates[stage];require(gate.have_abs==gate.have_nrms,"partial tolerance pair");
  std::cout<<std::setprecision(17)<<"GATE,"<<reference_stage(stage)<<','<<gate.have_abs<<','<<gate.maxabs<<','<<gate.nrms<<','<<kReferenceRmsFloor<<'\n';}
 std::cout<<"META,harness_version,native_service_capacity_v2\nMETA,harness_token_bound,4\nMETA,tolerance_policy,NT1_bitwise_controls_NT2-4_predeclared_per_expert_token_maxabs_AND_NRMS\n";
 int nt=int(nt_value),tasks=int(tasks_value),batch=int(batch_value);require(sizeof(void*)==8&&c::cpu_avx2_ok(),"64-bit AVX2 host required");
 require(std::getenv("LD_PRELOAD")==nullptr&&std::getenv("LD_AUDIT")==nullptr,"preload/audit forbidden");
 if(correctness_only){const char* debug=std::getenv("LD_DEBUG");require(debug==nullptr||std::string(debug)=="libs","correctness loader observation accepts LD_DEBUG=libs only");}
 else require(std::getenv("LD_DEBUG")==nullptr,"loader debugging forbidden in timing process");
 require(HOST_Q2_AVXVNNI==1,"frozen Q2 compile probe must be1");environment_metadata();ggml_cpu_init();
 auto selected=read_cohort(argv[3]);std::string error;bool loaded=c::expert_layout_load(argv[1],48,512,error);result(loaded,error);
 const auto& layout=c::expert_layout();require(layout.native&&layout.n_layers==48&&layout.n_expert==512&&layout.fmt.size()==48,"native geometry");
 strata::NativeRolePlan plan;bool opened=plan.open(argv[2],layout,error);result(opened,error);
 std::vector<Case> cases;cases.reserve(384);size_t aggregate=0;std::array<uint64_t,2> gu_bytes{},down_bytes{};
 for(const auto& id:selected){auto f=layout.fmt[id.layer];
  require(f.gu_type==int(gu_type)&&f.d_type==int(down_type)&&f.n_embd==c::H&&f.n_ff==c::FF,"nonhomogeneous format/dimensions");
  require(f.act_bytes>0&&f.act_bytes<=c::kNativeActBytes&&f.h_bytes>0&&f.h_bytes<=c::kNativeHBytes,"activation bound");
  require(f.up_off==f.gu_row*c::FF&&f.down_off==2*f.up_off&&f.bytes==f.down_off+f.d_row*c::H&&f.bytes==plan.blob_bytes(id.layer),"row/blob geometry");
  if(!cases.empty()){const auto& p=cases.front().fmt;require(f.gu_row==p.gu_row&&f.d_row==p.d_row&&f.act_bytes==p.act_bytes&&f.h_bytes==p.h_bytes&&f.gu_act==p.gu_act&&f.d_act==p.d_act,"nonhomogeneous row/activation geometry");}
  require(f.bytes<=(1ULL<<30)-aggregate,"all owned blobs exceed1GiB");aggregate+=f.bytes;gu_bytes[id.cohort]+=f.down_off;down_bytes[id.cohort]+=f.bytes-f.down_off;
  Case item{id.layer,id.expert,f,std::vector<uint8_t>(f.bytes)};bool copied=plan.copy_blob(id.layer,id.expert,item.blob.data(),item.blob.size(),error);result(copied,error);
  for(int role=0;role<3;++role){const auto* e=plan.extent(id.layer,role);require(e,"extent absent");std::cout<<"EXTENT,"<<id.cohort<<','<<id.layer<<','<<id.expert<<','<<role<<','<<quote(e->path)<<','<<quote(e->tensor)<<','<<e->offset+uint64_t(id.expert)*e->bytes_per_expert<<','<<e->bytes_per_expert<<'\n';}
  std::cout<<"ID,"<<id.cohort<<','<<cases.size()<<','<<id.layer<<','<<id.expert<<','<<fnv(item.blob.data(),item.blob.size())<<'\n';cases.push_back(std::move(item));
 }
 for(int cohort=0;cohort<2;++cohort)require(gu_bytes[cohort]>(64ULL<<20)&&down_bytes[cohort]>(64ULL<<20),"each cohort GU and Down must exceed64MiB");
 plan.close();auto topology=c::detect_cpu_topology(true,c::PoolAffinity::All);require(topology.host_core>=0&&topology.worker_cores.size()>=5,"six physical-core plan");
 require(c::planned_host_core(c::PoolAffinity::All)==topology.host_core,"host plan mismatch");
 std::vector<LayerInput> inputs(48);std::array<bool,48> initialized{};
 for(const auto& item:cases)if(!initialized[item.layer]){
  auto& input=inputs[item.layer];for(int t=0;t<nt;++t){for(int i=0;i<c::H;++i)input.x[t][i]=float(((i*37+item.layer*11+t*101)%509)-254)/257.0f;input.a[t].reset();c::native_quant_act(item.fmt,input.x[t].data(),input.a[t].data.data());}
  input_check(item.fmt,input,nt);initialized[item.layer]=true;
 }
 std::vector<Buffers> work(kSourceCases);double gu_delta=0,down_delta=0,chain_delta=0;
 std::array<double,3> worst_nrms{},worst_relative{};
 size_t reference_tokens=0,reference_rows=0;
 for(size_t i=0;i<cases.size();++i){auto& b=work[i];const auto& item=cases[i];b.bind(inputs[item.layer],nt);
  for(int t=0;t<nt;++t){b.ff[t].reset();b.hq[t].reset();b.q2[t].reset();b.out[t].reset();}
  full(item,b,nt);check_quant(item.fmt,b,nt);for(int t=0;t<nt;++t)b.out[t].check();auto delta=ggml_reference(item,b,nt,i,gates);gu_delta=std::max(gu_delta,delta[0].maxabs);down_delta=std::max(down_delta,delta[1].maxabs);chain_delta=std::max(chain_delta,delta[2].maxabs);
  for(int stage=0;stage<3;++stage){worst_nrms[stage]=std::max(worst_nrms[stage],delta[stage].nrms);worst_relative[stage]=std::max(worst_relative[stage],delta[stage].relative_refmax);}
  reference_tokens+=size_t(nt);reference_rows+=size_t(nt)*(c::FF+2*c::H);
  std::cout<<std::setprecision(17)<<"REFERENCE,"<<i<<','<<item.layer<<','<<item.expert<<','<<nt<<','<<delta[0].maxabs<<','<<delta[1].maxabs<<','<<delta[2].maxabs<<'\n';
  for(int t=0;t<nt;++t){b.ff_expected[t]=b.ff[t];b.hq_expected[t]=b.hq[t];b.q2_expected[t]=b.q2[t];b.out_expected[t]=b.out[t];}
 }
 require(cases.size()==kSourceCases&&work.size()==cases.size(),"source/work count reconciliation");
 require(reference_tokens==kSourceCases*size_t(nt)&&reference_rows==kSourceCases*size_t(nt)*(c::FF+2*c::H),"reference token/row reconciliation");
 std::cout<<"REFERENCE_COUNTS,"<<reference_tokens<<','<<reference_rows<<','<<reference_tokens*3<<'\n';
 for(int stage=0;stage<3;++stage)std::cout<<std::setprecision(17)<<"REFERENCE_WORST,"<<reference_stage(stage)<<','<<worst_nrms[stage]<<','<<worst_relative[stage]<<'\n';
 auto orders=schedule(seed,hot);
 for(int cohort=0;cohort<2;++cohort)for(const auto& order:orders[cohort])order_check(order,cohort,hot);
 std::cout<<std::setprecision(17)<<"META,scope,"<<(hot?"intended_hot_eight_CPU_cell_not_cache_residency_or_adoption":"host_resident_streaming_CPU_cell_not_cold_RAM_or_adoption")<<"\nMETA,GU,"<<gu_type<<"\nMETA,Down,"<<down_type<<"\nMETA,NT,"<<nt<<"\nMETA,tasks,"<<tasks<<"\nMETA,batch_jobs,"<<batch<<"\nMETA,seed,"<<seed<<"\nMETA,rng,splitmix64_modulo_fisher_yates_v1\nMETA,input_formula,((i*37+layer*11+token*101)%509-254)/257.0f\nMETA,input_quant,outside_service_intervals_once_per_layer_token\nMETA,cpu,"<<quote(c::cpu_name())<<"\nMETA,owned_blob_bytes,"<<aggregate<<"\nMETA,ggml_GU_maxabs,"<<gu_delta<<"\nMETA,ggml_Down_maxabs,"<<down_delta<<"\nMETA,ggml_full_chain_maxabs,"<<chain_delta<<"\nMETA,q2_avxvnni,1\nMETA,iq_avxvnni,0\nMETA,iq2s_gcc,off\nMETA,precision,IntelLLVM_precise\n";
 std::cout<<"META,source_case_count,"<<kSourceCases<<"\nMETA,active_jobs_per_cohort,"<<(hot?kHotCases:kCohortCases)<<"\nMETA,hot_selection,source_cohort_ranks_0_24_48_72_96_120_144_168\nMETA,normalization,error_rms/max(reference_rms_1e-12);maxabs/max(reference_maxabs_1e-12)\nMETA,input_heap_bytes,"<<sizeof(LayerInput)*inputs.size()<<"\nMETA,work_heap_bytes,"<<sizeof(Buffers)*work.size()<<'\n';
 for(int cohort=0;cohort<2;++cohort){
  std::cout<<"WORKING_SET,"<<cohort<<','<<gu_bytes[cohort]<<','<<down_bytes[cohort]<<'\n';
  uint64_t active_gu=0,active_down=0,active_blob=0;const auto& active=orders[cohort][0];
  for(size_t index:active){require(index<cases.size()&&index/kCohortCases==size_t(cohort),"active cohort index");const auto& item=cases[index];active_gu+=item.fmt.down_off;active_down+=item.fmt.bytes-item.fmt.down_off;active_blob+=item.fmt.bytes;
   if(hot)std::cout<<"HOT_ID,"<<cohort<<','<<index<<','<<item.layer<<','<<item.expert<<','<<fnv(item.blob.data(),item.blob.size())<<'\n';}
  std::cout<<"ACTIVE_LOGICAL_WORKING_SET,"<<cohort<<','<<active.size()<<','<<active_gu<<','<<active_down<<','<<active_blob<<'\n';
 }

 std::cout<<"ROUTE,"<<int(c::native_gu_dispatch(int(gu_type),nt))<<','<<(down_type==42?int(c::cpu_avx512_ok()?c::NativeDispatch::Q2Avx512:c::NativeDispatch::Q2Avx2):int(c::native_down_dispatch(int(down_type),nt)))<<'\n';
 std::cout<<"ROUND,cohort,arm,round,GU,Down,NT,tasks,batch_jobs,batches,logical_jobs,order_fnv,outer_ms,pool_GU_ms,pool_Q_ms,pool_Down_ms,output_fnv,host_cpu\n";
 // One pool lifetime; tasks0 and6 belong in different owner-launched processes.
 c::ExpertPool pool(5,true,true,c::PoolAffinity::All,tasks);require(pool.workers()==5,"pool workers");
 // Construct workers while the caller still has its original allowed CPU set.
 // The pool rescans caller affinity; pinning first made all workers inherit CPU0.
 Affinity affinity(topology.host_core);
 std::cout<<"PLACEMENT,planned_host,"<<topology.host_core<<','<<host_cpu()<<'\n';for(int i=0;i<5;++i)std::cout<<"PLACEMENT,planned_worker,"<<i<<','<<topology.worker_cores[i]<<'\n';
 std::cout<<"TASK_PLAN,"<<(tasks?tasks:18)<<','<<(tasks?tasks:18)<<'\n';
 // Untimed pool/reference parity gate over all owned experts before timing any arm.
 for(size_t i=0;i<cases.size();i+=size_t(batch)){
  std::array<c::ExpertJobMulti,6> jobs{};for(int j=0;j<batch;++j){auto k=i+j;auto& b=work[k];jobs[j].blob=cases[k].blob.data();jobs[j].nt=nt;for(int t=0;t<nt;++t){b.pout[t].reset();jobs[j].nact[t]=b.ap[t];jobs[j].out[t]=b.pout[t].data.data();}}
  pool.run_split_multi_native(cases.front().fmt,jobs.data(),batch);for(int j=0;j<batch;++j)check_round(Arm::Pool,cases[i+j],work[i+j],nt);
 }
 thread_placement(tasks,topology);pool.diag(stderr);
 if(correctness_only){
  for(size_t i=0;i<cases.size();++i)input_check(cases[i].fmt,inputs[cases[i].layer],nt);
  require(strata::core::release_gpu_fn().load()==nullptr,"GPU callback changed");
  std::cout<<"RESULT,correctness_only_pass,no_calibration_rounds\n";
  std::cout.flush();std::cerr.flush();require(bool(std::cout)&&bool(std::cerr),"output failure");return 0;
 }
 std::cout<<"META,arm_order,round_rotated_GU_Q_Down_Full_Pool_seed_mod5\n";
 size_t warmup_records=0,measurement_records=0;std::array<std::array<size_t,5>,2> warmup_count{},measurement_count{};
 for(int cohort=0;cohort<2;++cohort)for(int r=0;r<28;++r)for(int arm_index=0;arm_index<5;++arm_index){
  Arm arm=static_cast<Arm>((arm_index+r+int(seed%5))%5);
  const auto& order=orders[cohort][r];order_check(order,cohort,hot);require(order.size()%size_t(batch)==0,"timing pool job count reconciliation");for(size_t k:order)prepare(arm,cases[k],work[k],nt);
  std::vector<std::array<c::ExpertJobMulti,6>> batches;
  if(arm==Arm::Pool){batches.resize(order.size()/size_t(batch));for(size_t i=0;i<order.size();++i){size_t k=order[i];auto& job=batches[i/batch][i%batch];job.blob=cases[k].blob.data();job.nt=nt;for(int t=0;t<nt;++t){job.nact[t]=work[k].ap[t];job.out[t]=work[k].pout[t].data.data();}}}
  // Only required production calls and bounded traversal occur between these clocks.
  double pg=pool.ms_multi_gu,pq=pool.ms_multi_q,pd=pool.ms_multi_down;
  auto start=Clock::now();
  switch(arm){
   case Arm::GU:for(size_t k:order)gu(cases[k],work[k],nt);break;
   case Arm::Q:for(size_t k:order)quant_ff(cases[k].fmt,work[k],nt);break;
   case Arm::Down:for(size_t k:order)down(cases[k],work[k],nt);break;
   case Arm::Full:for(size_t k:order)full(cases[k],work[k],nt);break;
   case Arm::Pool:for(auto& jobs:batches)pool.run_split_multi_native(cases.front().fmt,jobs.data(),batch);break;
  }
  auto end=Clock::now();double dg=pool.ms_multi_gu-pg,dq=pool.ms_multi_q-pq,dd=pool.ms_multi_down-pd;
  uint64_t checksum=14695981039346656037ULL;for(size_t k:order){checksum^=check_round(arm,cases[k],work[k],nt);checksum*=1099511628211ULL;input_check(cases[k].fmt,inputs[cases[k].layer],nt);}
  require(strata::core::release_gpu_fn().load()==nullptr,"GPU callback changed");
  if(arm==Arm::Pool){std::cerr<<"DIAG,"<<cohort<<','<<r-3<<'\n';pool.diag(stderr);}
  // warmups remain explicitly labelled;25 measurement rows per cohort/arm.
  if(r<3){++warmup_records;++warmup_count[cohort][static_cast<size_t>(arm)];}
  else {++measurement_records;++measurement_count[cohort][static_cast<size_t>(arm)];}
  std::cout<<(r<3?"WARMUP,":"ROUND,")<<cohort<<','<<name(arm)<<','<<r-3<<','<<gu_type<<','<<down_type<<','<<nt<<','<<tasks<<','<<(arm==Arm::Pool?batch:1)<<','<<(arm==Arm::Pool?order.size()/size_t(batch):order.size())<<','<<order.size()<<','<<order_hash(order)<<','<<ms(start,end)<<','<<dg<<','<<dq<<','<<dd<<','<<checksum<<','<<host_cpu()<<'\n';
 }
 require(warmup_records==30&&measurement_records==250,"warmup/measurement output count reconciliation");
 for(int cohort=0;cohort<2;++cohort)for(size_t arm=0;arm<5;++arm)require(warmup_count[cohort][arm]==3&&measurement_count[cohort][arm]==25,"per-arm25-round reconciliation");
 std::cout<<"ROUND_COUNTS,"<<warmup_records<<','<<measurement_records<<'\n';
 std::cout<<"RESULT,correctness_pass,statistical_holdout_adoption_and_full_lifecycle_not_qualified\n";std::cout.flush();std::cerr.flush();require(bool(std::cout)&&bool(std::cerr),"output failure");return 0;
 }catch(const std::exception& e){std::cerr<<"RESULT,fail,"<<quote(e.what())<<'\n';return 1;}}
