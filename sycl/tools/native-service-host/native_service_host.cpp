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
static void thread_placement(int tasks){
#ifdef __linux__
 size_t threads=0;
 for(const auto& entry:std::filesystem::directory_iterator("/proc/self/task")){
  require(++threads<=128,"thread audit bound");
  std::ifstream file(entry.path()/"status",std::ios::binary);require(bool(file),"thread status open");
  std::array<char,16384> bytes{};file.read(bytes.data(),bytes.size());auto count=file.gcount();
  require(count>0&&count<static_cast<std::streamsize>(bytes.size()),"thread status byte bound");
  std::istringstream lines(std::string(bytes.data(),static_cast<size_t>(count)));
  std::string line,allowed;while(std::getline(lines,line))if(line.rfind("Cpus_allowed_list:",0)==0)allowed=line.substr(18);
  require(!allowed.empty(),"missing thread affinity");
  std::cout<<"PLACEMENT,observed_allowed_list,"<<tasks<<','<<quote(entry.path().filename().string())<<','<<quote(allowed)<<'\n';
 }
#else
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
 for(const char* key:{"STRATA_FORCE_ISA","STRATA_FORCE_AVX2","STRATA_NO_AVXVNNI","STRATA_NO_Q8K_AVX2","STRATA_IQ_MT_MIN","STRATA_IQ3S_MT1","STRATA_NO_IQ512","STRATA_NO_IQ256","STRATA_NO_IQ4NL","STRATA_KQ256","STRATA_IQ256_GATHER","STRATA_Q2_BITPLANE","STRATA_POOL_SPIN_US","STRATA_HOST_CORE","STRATA_NATIVE_DISPATCH_HISTOGRAM"}){
  const char* value=std::getenv(key);std::cout<<"ENV,"<<key<<','<<quote(value?value:"<unset>")<<'\n';
 }
 require(std::getenv("STRATA_FORCE_ISA")==nullptr&&std::getenv("STRATA_FORCE_AVX2")==nullptr,"feature-forcing environments forbidden");
 require(std::getenv("STRATA_NATIVE_DISPATCH_HISTOGRAM")==nullptr,"diagnostic histogram must be unset for this service fixture");
}
struct Buffers {
 std::array<std::array<float,c::H>,2> x{};
 std::array<Bytes<c::kNativeActBytes>,2> a,aq_ref;
 std::array<Bytes<c::kNativeHBytes>,2> hq,hq_ref;
 std::array<Q2Buffer,2> q2{};
 std::array<Floats<c::FF>,2> ff,ff_ref;
 std::array<Floats<c::H>,2> out,pout,down_ref,expected;
 const void* ap[2]{};const void* hp[2]{};const c::ActQ* qp[2]{};
 float* fp[2]{};float* op[2]{};
 Buffers(const Case& item,int nt){for(int t=0;t<nt;++t){for(int i=0;i<c::H;++i)x[t][i]=float(((i*37+item.layer*11+item.expert*3+t*101)%509)-254)/257.0f; ap[t]=a[t].data.data();hp[t]=hq[t].data.data();qp[t]=&q2[t].value;fp[t]=ff[t].data.data();op[t]=out[t].data.data();}}
};
static void quant_input(const c::NativeFmt& f,Buffers& b,int nt){for(int t=0;t<nt;++t)c::native_quant_act(f,b.x[t].data(),b.a[t].data.data());}
static void check_input(const c::NativeFmt& f,Buffers& b,int nt){auto tr=ggml_get_type_traits_cpu(static_cast<ggml_type>(f.gu_act));require(tr&&tr->from_float,"missing GU activation quantizer");for(int t=0;t<nt;++t){b.a[t].check(f.act_bytes);b.aq_ref[t].reset();tr->from_float(b.x[t].data(),b.aq_ref[t].data.data(),c::H);b.aq_ref[t].check(f.act_bytes);require(!std::memcmp(b.a[t].data.data(),b.aq_ref[t].data.data(),f.act_bytes),"GU quantization differs from ggml");}}
static void quant_ff(const c::NativeFmt& f,Buffers& b,int nt){for(int t=0;t<nt;++t){if(f.d_type==42)c::act_quant_any(b.ff[t].data.data(),c::FF,b.q2[t].value);else c::native_quant_h(f,b.ff[t].data.data(),b.hq[t].data.data());}}
static void check_ff(const c::NativeFmt& f,Buffers& b,int nt){for(int t=0;t<nt;++t){b.ff[t].check();if(f.d_type==20){auto tr=ggml_get_type_traits_cpu(static_cast<ggml_type>(f.d_act));require(tr&&tr->from_float,"missing Down quantizer");b.hq_ref[t].reset();tr->from_float(b.ff[t].data.data(),b.hq_ref[t].data.data(),c::FF);b.hq[t].check(f.h_bytes);b.hq_ref[t].check(f.h_bytes);require(!std::memcmp(b.hq[t].data.data(),b.hq_ref[t].data.data(),f.h_bytes),"Down quantization differs from ggml");}else check_q2(b.ff[t].data.data(),b.q2[t]);}}
static void direct_down(const Case& item,Buffers& b,int nt){if(item.fmt.d_type==42)c::q2_rows_any(item.blob.data()+item.fmt.down_off,item.fmt.d_row,c::FF/c::QK,b.qp,nt,b.op,0,c::H);else c::native_down_rows(item.fmt,item.blob.data(),b.hp,nt,b.op,0,c::H);}
struct Timings {double input,gu,q,down,total;};
static Timings direct(const Case& item,Buffers& b,int nt){for(int t=0;t<nt;++t){b.a[t].reset();b.hq[t].reset();b.q2[t].reset();b.ff[t].reset();b.out[t].reset();}auto i0=Clock::now();quant_input(item.fmt,b,nt);auto i1=Clock::now();check_input(item.fmt,b,nt);auto s=Clock::now();c::native_gu_rows(item.fmt,item.blob.data(),b.ap,nt,b.fp,0,c::FF);auto g=Clock::now();quant_ff(item.fmt,b,nt);auto q=Clock::now();direct_down(item,b,nt);auto d=Clock::now();check_ff(item.fmt,b,nt);for(int t=0;t<nt;++t)b.out[t].check();return {ms(i0,i1),ms(s,g),ms(g,q),ms(q,d),ms(s,d)};}
// Independent per-row pinned GGML reference, not the resolver or pool as its own oracle.
static std::array<double,2> ggml_reference(const Case& item,Buffers& b,int nt){auto gu=ggml_get_type_traits_cpu(static_cast<ggml_type>(item.fmt.gu_type));auto down=ggml_get_type_traits_cpu(static_cast<ggml_type>(item.fmt.d_type));auto htr=ggml_get_type_traits_cpu(static_cast<ggml_type>(item.fmt.d_act));require(gu&&gu->vec_dot&&down&&down->vec_dot&&htr&&htr->from_float,"ggml reference traits absent");double fd=0,od=0;for(int t=0;t<nt;++t){b.ff_ref[t].reset();b.down_ref[t].reset();for(int r=0;r<c::FF;++r){float g=0,u=0;gu->vec_dot(c::H,&g,0,item.blob.data()+r*item.fmt.gu_row,0,b.ap[t],0,1);gu->vec_dot(c::H,&u,0,item.blob.data()+item.fmt.up_off+r*item.fmt.gu_row,0,b.ap[t],0,1);b.ff_ref[t].data[r]=(g/(1.f+std::exp(-g)))*u;fd=std::max(fd,double(std::abs(b.ff_ref[t].data[r]-b.ff[t].data[r])));}b.ff_ref[t].check();b.hq_ref[t].reset();htr->from_float(b.ff[t].data.data(),b.hq_ref[t].data.data(),c::FF);b.hq_ref[t].check(item.fmt.h_bytes);for(int r=0;r<c::H;++r){down->vec_dot(c::FF,&b.down_ref[t].data[r],0,item.blob.data()+item.fmt.down_off+r*item.fmt.d_row,0,b.hq_ref[t].data.data(),0,1);od=std::max(od,double(std::abs(b.down_ref[t].data[r]-b.out[t].data[r])));}b.down_ref[t].check();if(nt==1){require(!std::memcmp(b.ff_ref[t].data.data(),b.ff[t].data.data(),sizeof(float)*c::FF),"NT1 GU differs from independent ggml");if(item.fmt.d_type==20)require(!std::memcmp(b.down_ref[t].data.data(),b.out[t].data.data(),sizeof(float)*c::H),"NT1 Down20 differs from independent ggml");}}return {fd,od};}
static std::vector<std::pair<int,int>> cohort(const std::string& path){require(std::filesystem::is_regular_file(path),"cohort is not regular");auto size=std::filesystem::file_size(path);require(size>0&&size<=8192,"cohort byte bound");std::ifstream in(path,std::ios::binary);require(bool(in),"cohort open");std::string text(size,'\0');in.read(text.data(),static_cast<std::streamsize>(size));require(in.gcount()==static_cast<std::streamsize>(size)&&in.peek()==EOF,"cohort changed/short read");std::istringstream lines(text);std::string line;require(bool(std::getline(lines,line))&&line=="layer\texpert","cohort header must be layer TAB expert");std::set<std::pair<int,int>> seen;std::vector<std::pair<int,int>> result;while(std::getline(lines,line)){require(!line.empty(),"empty cohort row");size_t tab=line.find('\t');require(tab!=std::string::npos&&line.find('\t',tab+1)==std::string::npos,"cohort columns");auto number=[](const std::string& s){require(!s.empty()&&s.size()<=3,"cohort integer width");int v=0;for(char ch:s){require(ch>='0'&&ch<='9',"cohort nondecimal integer");v=v*10+ch-'0';}return v;};int l=number(line.substr(0,tab)),e=number(line.substr(tab+1));require(l<48&&e<512,"cohort range");require(seen.emplace(l,e).second,"duplicate cohort pair");result.emplace_back(l,e);require(result.size()<=96,"cohort entry cap");}require(!result.empty(),"empty cohort");std::cout<<"META,cohort_fnv,"<<fnv(text.data(),text.size())<<'\n';return result;}
int main(int argc,char** argv){try{require(argc==4,"usage: native_service_host PACK_DIR PRIMARY_GGUF COHORT_TSV");require(sizeof(void*)==8,"64-bit host required");require(c::cpu_avx2_ok(),"AVX2 production Q2 path required");environment_metadata();ggml_cpu_init();std::string error;auto pairs=cohort(argv[3]);bool loaded=c::expert_layout_load(argv[1],48,512,error);result(loaded,error);const auto& layout=c::expert_layout();require(layout.native&&layout.n_layers==48&&layout.n_expert==512&&layout.fmt.size()==48,"native layout geometry");strata::NativeRolePlan plan;bool opened=plan.open(argv[2],layout,error);result(opened,error);std::vector<Case> cases;size_t aggregate=0;for(auto [l,e]:pairs){auto f=layout.fmt[l];require(f.n_embd==c::H&&f.n_ff==c::FF,"expected H2560 FF640");require(f.gu_type==18||f.gu_type==21||f.gu_type==22||f.gu_type==23,"unsupported GU format");require(f.d_type==20||f.d_type==42,"unsupported Down format");require(f.act_bytes>0&&f.act_bytes<=c::kNativeActBytes&&f.h_bytes>0&&f.h_bytes<=c::kNativeHBytes,"activation geometry bound");require(f.up_off==f.gu_row*c::FF&&f.down_off==2*f.up_off&&f.bytes==f.down_off+f.d_row*c::H&&f.bytes==plan.blob_bytes(l),"blob/row geometry");require(f.bytes<=(1ULL<<30)-aggregate,"selected blob cap 1GiB");aggregate+=f.bytes;Case item{l,e,f,std::vector<uint8_t>(f.bytes)};bool copied=plan.copy_blob(l,e,item.blob.data(),item.blob.size(),error);result(copied,error);for(int role=0;role<3;++role){auto extent=plan.extent(l,role);require(extent!=nullptr,"missing extent");std::cout<<"EXTENT,"<<l<<','<<e<<','<<role<<','<<quote(extent->path)<<','<<quote(extent->tensor)<<','<<extent->offset+uint64_t(e)*extent->bytes_per_expert<<','<<extent->bytes_per_expert<<','<<extent->tensor_bytes<<'\n';}std::cout<<"BLOB,"<<l<<','<<e<<','<<f.gu_type<<','<<f.d_type<<','<<f.bytes<<','<<fnv(item.blob.data(),item.blob.size())<<'\n';cases.push_back(std::move(item));}plan.close();auto topology=c::detect_cpu_topology(true,c::PoolAffinity::All);require(topology.host_core>=0&&topology.worker_cores.size()>=5,"six physical-core plan unavailable");require(c::planned_host_core(c::PoolAffinity::All)==topology.host_core,"host plan disagreement");Affinity affinity(topology.host_core);std::cout<<std::setprecision(17)<<"META,input_formula,((i*37+layer*11+expert*3+token*101)%509-254)/257.0f\nMETA,scope,controlled_hot_cohort_not_calibration_or_engine_performance\nMETA,cpu,"<<quote(c::cpu_name())<<"\nMETA,planned_host,"<<topology.host_core<<"\nMETA,observed_host,"<<host_cpu()<<"\nMETA,q2_avxvnni,"<<HOST_Q2_AVXVNNI<<"\nMETA,iq_avxvnni,0\nMETA,iq2s_gcc,off\nMETA,precision,IntelLLVM_precise\nMETA,owned_blob_bytes,"<<aggregate<<'\n';for(int i=0;i<5;++i)std::cout<<"PLACEMENT,planned_worker,"<<i<<','<<topology.worker_cores[i]<<'\n';std::cout<<"SAMPLE,layer,expert,GU,Down,NT,tasks,workers,repeat,input_quant_ms,direct_GU_ms,direct_FFquant_ms,direct_Down_ms,direct_outer_ms,pool_GU_ms,pool_FFquant_ms,pool_Down_ms,pool_outer_ms,out_fnv,ggml_GU_maxabs,ggml_Down_maxabs,host_cpu\n";
// Separate pool lifetimes: no competing workers or shared diagnostic owner.
for(int tasks:{0,6}){c::ExpertPool pool(5,true,true,c::PoolAffinity::All,tasks);require(pool.workers()==5,"pool worker count");for(const auto& item:cases)for(int nt:{1,2}){Buffers b(item,nt);direct(item,b,nt);auto deltas=ggml_reference(item,b,nt);for(int t=0;t<nt;++t)b.expected[t]=b.out[t];std::cout<<"TASK_PLAN,"<<tasks<<","<<(tasks?tasks:18)<<","<<(tasks?tasks:18)<<"\n";std::cout<<"ROUTE,"<<item.layer<<','<<item.expert<<','<<nt<<','<<tasks<<','<<int(c::native_gu_dispatch(item.fmt.gu_type,nt))<<','<<(item.fmt.d_type==42?int(c::cpu_avx512_ok()?c::NativeDispatch::Q2Avx512:c::NativeDispatch::Q2Avx2):int(c::native_down_dispatch(item.fmt.d_type,nt)))<<'\n';for(int repeat=-3;repeat<5;++repeat){auto times=direct(item,b,nt);for(int t=0;t<nt;++t){require(!std::memcmp(b.out[t].data.data(),b.expected[t].data.data(),sizeof(float)*c::H),"direct repeat differs");b.pout[t].reset();}c::ExpertJobMulti job{};job.blob=item.blob.data();job.nt=nt;for(int t=0;t<nt;++t){job.nact[t]=b.ap[t];job.out[t]=b.pout[t].data.data();}double pg=pool.ms_multi_gu,pq=pool.ms_multi_q,pd=pool.ms_multi_down;auto s=Clock::now();pool.run_split_multi_native(item.fmt,&job,1);auto end=Clock::now();double dg=pool.ms_multi_gu-pg,dq=pool.ms_multi_q-pq,dd=pool.ms_multi_down-pd;std::cerr<<"DIAG,"<<item.layer<<','<<item.expert<<','<<nt<<','<<tasks<<','<<repeat<<'\n';pool.diag(stderr);if(repeat==-3&&nt==1&&item.layer==cases.front().layer&&item.expert==cases.front().expert)thread_placement(tasks);require(strata::core::release_gpu_fn().load()==nullptr,"GPU callback changed");check_input(item.fmt,b,nt);uint64_t checksum=14695981039346656037ULL;for(int t=0;t<nt;++t){b.pout[t].check();require(!std::memcmp(b.pout[t].data.data(),b.out[t].data.data(),sizeof(float)*c::H),"pool/direct output mismatch");checksum^=fnv(b.pout[t].data.data(),sizeof(float)*c::H);checksum*=1099511628211ULL;}if(repeat>=0)std::cout<<"SAMPLE,"<<item.layer<<','<<item.expert<<','<<item.fmt.gu_type<<','<<item.fmt.d_type<<','<<nt<<','<<tasks<<','<<pool.workers()<<','<<repeat<<','<<times.input<<','<<times.gu<<','<<times.q<<','<<times.down<<','<<times.total<<','<<dg<<','<<dq<<','<<dd<<','<<ms(s,end)<<','<<checksum<<','<<deltas[0]<<','<<deltas[1]<<','<<host_cpu()<<'\n';}}}std::cout<<"RESULT,pass,partition_and_repeat_bitexact_only\n";std::cout.flush();std::cerr.flush();require(bool(std::cout)&&bool(std::cerr),"output write failure");return 0;}catch(const std::exception& e){std::cerr<<"RESULT,fail,"<<quote(e.what())<<'\n';return 1;}}
