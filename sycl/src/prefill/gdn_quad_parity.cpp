// Candidate-specific exact GPU differential test; explicit synthetic service timing only. No model execution.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/gdn_variant.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
constexpr size_t G=32, NS=128*48*128;
constexpr float sentinel=-98765.f;
constexpr uint16_t half_sentinel=0x5a5a;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
// A disposable diagnostic process cannot unwind host memcpy endpoints when a
// submit/wait error leaves completion unknown. All referenced host vectors live
// outside these lambdas; terminate here before their scopes can unwind.
[[noreturn]] void fail_stop(const char* stage,const char* detail){
 std::fprintf(stderr,"FAIL-STOP stage=%s completion_unknown=true host_unwind=false explicit_USM_release=false detail=%s\n",stage,detail);
 std::fflush(stdout);std::fflush(stderr);std::_Exit(2);
}
template<class Fn> decltype(auto) gpu_stage(const char* stage,Fn&& fn){
 try{return fn();}catch(const std::exception& e){fail_stop(stage,e.what());}catch(...){fail_stop(stage,"non-standard exception");}
}
uint64_t fnv(const void* data,size_t n){uint64_t h=14695981039346656037ULL;auto p=static_cast<const uint8_t*>(data);for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;}
size_t si(int r,int h,int c){return (size_t(r)*48+h)*128+c;}
void early_contract(){
 std::array<float,8> state{1,2,-3,4,5,-6,7,8},out=state;
 std::array<uint16_t,8> half{1,2,3,4,5,6,7,8};auto old=state,oldout=out;auto oldhalf=half;
 strata::prefill::GdnQuadReport report;
 bool empty=strata::prefill::gdn_recurrence_quad_variant(state.data(),nullptr,nullptr,nullptr,nullptr,nullptr,1e-6f,out.data(),half.data(),0,nullptr,0,&report);
 require(empty&&report.status==strata::prefill::GdnQuadReport::Status::Empty,"empty contract");
 strata::prefill::gdn_recurrence_pipeline_reference(state.data(),nullptr,nullptr,nullptr,nullptr,nullptr,1e-6f,out.data(),half.data(),0,nullptr);
 bool denied=strata::prefill::gdn_recurrence_quad_variant(state.data(),nullptr,nullptr,nullptr,nullptr,nullptr,1e-6f,out.data(),half.data(),1,nullptr,0,&report,true);
 require(!denied&&report.status==strata::prefill::GdnQuadReport::Status::Denied,"forced pre-submit denial");
 require(state==old&&out==oldout&&half==oldhalf,"early contract touched host marker buffers");
 bool negative=false;try{strata::prefill::gdn_recurrence_quad_variant(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1e-6f,nullptr,nullptr,-1,nullptr);}catch(const std::invalid_argument&){negative=true;}
 require(negative,"negative T not rejected");
 volatile float big=16777216.f,one=1.f;float p01=big+one;float p012=p01-big;float ordered=p012+one;float changed=(big-big)+(one+one);
 require(ordered==1.f&&changed==2.f,"cancellation pattern does not distinguish association");
 std::printf("PASS early_empty_and_denial host_markers_unchanged state_fnv=%llu ordered_partial=%g changed_partial=%g\n",(unsigned long long)fnv(state.data(),sizeof(state)),ordered,changed);
}
struct Buffers {
 sycl::queue& q;std::vector<void*> owned;
 float *as,*bs,*h,*g,*b,*z,*gamma,*ay,*by;
 uint16_t *ah,*bh;
 size_t capacity;
 template<class T> T* alloc(size_t count){auto p=sycl::malloc_device<T>(count,q);require(p!=nullptr,"USM allocation failure");owned.push_back(p);return p;}
 explicit Buffers(sycl::queue& queue,size_t cap):q(queue),capacity(cap){
  // All allocations happen before any recurrence submission; fixed cap<=2048.
  require(cap>0&&cap<=2048,"allocation chunk cap");owned.reserve(11);
  try{as=alloc<float>(NS+2*G);bs=alloc<float>(NS+2*G);h=alloc<float>(cap*10240);g=alloc<float>(cap*48);b=alloc<float>(cap*48);z=alloc<float>(cap*6144);gamma=alloc<float>(128);ay=alloc<float>(cap*6144+2*G);by=alloc<float>(cap*6144+2*G);ah=alloc<uint16_t>(cap*6144+2*G);bh=alloc<uint16_t>(cap*6144+2*G);}
  catch(...){auto error=std::current_exception();for(auto p:owned)try{sycl::free(p,q);}catch(...){std::fprintf(stderr,"partial-allocation cleanup failed; original allocation failure retained\n");}std::rethrow_exception(error);}
 }
 ~Buffers() noexcept(false){// Normal calls drain before destruction; never retry a failed recurrence.
  gpu_stage("final queue drain",[&]{q.wait_and_throw();});
  std::exception_ptr error;
  for(auto p:owned)try{sycl::free(p,q);}catch(...){std::fprintf(stderr,"USM release failed after successful drain\n");if(!error)error=std::current_exception();}
  if(error){if(std::uncaught_exceptions()==0)std::rethrow_exception(error);else std::fprintf(stderr,"teardown failure during original error; result remains failed\n");}
 }
 Buffers(const Buffers&)=delete;
};
void initialize(Buffers& d,bool zero,bool cancel){
 std::vector<float> state(NS,0.f);
 if(!zero)for(int r=0;r<128;++r)for(int head=0;head<48;++head)for(int col=0;col<128;++col){
  float value=float((r*7+head*11+col*13)%31-15)*.001f;
  if(cancel){value=0.f;if(r==0)value=16777216.f;else if(r==32||r==96)value=1.f;else if(r==64)value=-16777216.f;}
  state[si(r,head,col)]=value;
 }
 gpu_stage("initialize host transfers",[&]{
  d.q.fill(d.as,sentinel,NS+2*G);d.q.fill(d.bs,sentinel,NS+2*G);
  d.q.memcpy(d.as+G,state.data(),NS*sizeof(float));d.q.memcpy(d.bs+G,state.data(),NS*sizeof(float));d.q.wait_and_throw();
 });
}
void prepare(Buffers& d,size_t T,size_t offset,bool cancel){
 require(T>0&&T<=d.capacity,"live input bound");
 std::vector<float> h(T*10240),gate(T*48),beta(T*48),z(T*6144),gamma(128);
 for(size_t t=0;t<T;++t){size_t token=offset+t;
  for(int kh=0;kh<16;++kh)for(int r=0;r<128;++r){
   float query=float(int((token*3+kh*11+r*7)%29)-14)*.003f;
   float key=float(int((token*5+kh*17+r*13)%23)-11)*.003f;
   if(cancel){query=key=(r%32==0)?1.f:0.f;}
   h[t*10240+kh*128+r]=query;h[t*10240+16*128+kh*128+r]=key;
  }
  for(int head=0;head<48;++head){gate[t*48+head]=cancel?0.f:-.08f-float((token*7+head*3)%80)*.0005f;beta[t*48+head]=cancel?.25f:.15f+float((token*11+head*5)%400)*.001f;
   for(int col=0;col<128;++col){h[t*10240+2*16*128+head*128+col]=cancel?0.f:float(int((token*13+head*7+col*17)%37)-18)*.01f;z[t*6144+head*128+col]=cancel?0.f:float(int((token*3+head*5+col*7)%31)-15)*.01f;}
  }
 }
 for(int col=0;col<128;++col)gamma[col]=cancel?1.f:1.f+float(col%17)*.002f;
 // Allocate guards at the live end, and keep unused capacity as a checked sentinel tail.
 gpu_stage("prepare host transfers",[&]{
  d.q.fill(d.ay,sentinel,d.capacity*6144+2*G);d.q.fill(d.by,sentinel,d.capacity*6144+2*G);d.q.fill(d.ah,half_sentinel,d.capacity*6144+2*G);d.q.fill(d.bh,half_sentinel,d.capacity*6144+2*G);
  d.q.memcpy(d.h,h.data(),h.size()*4);d.q.memcpy(d.g,gate.data(),gate.size()*4);d.q.memcpy(d.b,beta.data(),beta.size()*4);d.q.memcpy(d.z,z.data(),z.size()*4);d.q.memcpy(d.gamma,gamma.data(),128*4);d.q.wait_and_throw();
 });
}
void floats(Buffers& d,float* a,float* b,size_t live,size_t allocated,const char* label){
 std::vector<float> x(allocated+2*G),y(x.size());gpu_stage("float receive",[&]{d.q.memcpy(x.data(),a,x.size()*4);d.q.memcpy(y.data(),b,y.size()*4);d.q.wait_and_throw();});
 size_t different=0,bounds=0;bool finite=true;
 for(size_t i=0;i<x.size();++i){if(i<G||i>=G+live)bounds+=x[i]!=sentinel||y[i]!=sentinel;else{different+=std::bit_cast<uint32_t>(x[i])!=std::bit_cast<uint32_t>(y[i]);finite=finite&&std::isfinite(x[i])&&std::isfinite(y[i])&&x[i]!=sentinel&&y[i]!=sentinel;}}
 if(different||bounds||!finite)std::fprintf(stderr,"FAIL %s different=%zu guards=%zu finite=%d\n",label,different,bounds,finite);
 require(different==0&&bounds==0&&finite,"full float bitwise/finite/guard gate");
}
void compare(Buffers& d,size_t T){
 floats(d,d.as,d.bs,NS,NS,"state");floats(d,d.ay,d.by,T*6144,d.capacity*6144,"FP32y");
 std::vector<uint16_t> x(d.capacity*6144+2*G),y(x.size());gpu_stage("half receive",[&]{d.q.memcpy(x.data(),d.ah,x.size()*2);d.q.memcpy(y.data(),d.bh,y.size()*2);d.q.wait_and_throw();});
 size_t different=0,bounds=0;bool finite=true;for(size_t i=0;i<x.size();++i){if(i<G||i>=G+T*6144)bounds+=x[i]!=half_sentinel||y[i]!=half_sentinel;else{different+=x[i]!=y[i];finite=finite&&(x[i]&0x7c00)!=0x7c00&&(y[i]&0x7c00)!=0x7c00&&x[i]!=half_sentinel&&y[i]!=half_sentinel;}}
 require(different==0&&bounds==0&&finite,"full half bitwise/finite/guard gate");
}
void empty_device(Buffers& d){
 prepare(d,1,0,false);
 std::vector<float> before(NS+2*G),after(before.size());gpu_stage("empty before snapshot",[&]{d.q.memcpy(before.data(),d.bs,before.size()*4);d.q.wait_and_throw();});
 strata::prefill::GdnQuadReport report;
 bool empty=gpu_stage("device empty candidate",[&]{return strata::prefill::gdn_recurrence_quad_variant(d.bs+G,nullptr,nullptr,nullptr,nullptr,nullptr,1e-6f,d.by+G,d.bh+G,0,&d.q,0,&report);});
 require(empty&&report.status==strata::prefill::GdnQuadReport::Status::Empty,"device empty contract");
 gpu_stage("state after snapshot",[&]{d.q.memcpy(after.data(),d.bs,after.size()*4);d.q.wait_and_throw();});require(!std::memcmp(before.data(),after.data(),before.size()*4),"empty candidate changed full state bits");
 floats(d,d.by,d.by,0,d.capacity*6144,"empty FP32 output");
 std::vector<uint16_t> half(d.capacity*6144+2*G);gpu_stage("sentinel half receive",[&]{d.q.memcpy(half.data(),d.bh,half.size()*2);d.q.wait_and_throw();});require(std::all_of(half.begin(),half.end(),[](auto v){return v==half_sentinel;}),"empty candidate changed half output");
 std::printf("PASS empty_device full_state_fnv=%llu candidate_calls=0 norm_calls=0\n",(unsigned long long)fnv(before.data(),before.size()*4));
}
void call(Buffers& d,size_t T,size_t prefix,bool cancel,bool force_deny=false){
 prepare(d,T,prefix,cancel);strata::prefill::GdnQuadReport report;
 size_t candidate_calls=0,legacy_fallback_calls=0;
 // Forced denial has its own device snapshot gate before any explicit legacy fallback.
 std::vector<float> before;if(force_deny){before.resize(NS+2*G);gpu_stage("denied before snapshot",[&]{d.q.memcpy(before.data(),d.bs,before.size()*4);d.q.wait_and_throw();});}
 bool selected=gpu_stage("candidate recurrence submission",[&]{return strata::prefill::gdn_recurrence_quad_variant(d.bs+G,d.h,d.g,d.b,d.z,d.gamma,1e-6f,d.by+G,d.bh+G,int64_t(T),&d.q,0,&report,force_deny);});
 if(!selected&&!force_deny)std::fprintf(stderr,"DENIED exact_quad reason=%s compiledSG=%u deviceWG=%zu kernelWG=%zu\n",report.reason,report.compiled_subgroup,report.device_max_workgroup,report.kernel_max_workgroup);
 if(force_deny){
  require(!selected&&report.status==strata::prefill::GdnQuadReport::Status::Denied,"denial submitted candidate");
  std::vector<float> after(before.size());gpu_stage("state after snapshot",[&]{d.q.memcpy(after.data(),d.bs,after.size()*4);d.q.wait_and_throw();});require(!std::memcmp(before.data(),after.data(),before.size()*4),"denied candidate changed full state bits");
  // Output allocations must remain entirely sentinel until caller chooses fallback.
  floats(d,d.by,d.by,0,d.capacity*6144,"denied FP32 output");
  std::vector<uint16_t> half(d.capacity*6144+2*G);gpu_stage("sentinel half receive",[&]{d.q.memcpy(half.data(),d.bh,half.size()*2);d.q.wait_and_throw();});require(std::all_of(half.begin(),half.end(),[](auto v){return v==half_sentinel;}),"denied candidate changed half output");
  gpu_stage("denied legacy fallback",[&]{strata::prefill::gdn_recurrence_pipeline_reference(d.bs+G,d.h,d.g,d.b,d.z,d.gamma,1e-6f,d.by+G,d.bh+G,int64_t(T),&d.q);d.q.wait_and_throw();});++legacy_fallback_calls;
 }else{require(selected&&report.status==strata::prefill::GdnQuadReport::Status::Submitted,"candidate unavailable: this is not candidate parity PASS");++candidate_calls;}
 gpu_stage("reference recurrence and final drain",[&]{strata::prefill::gdn_recurrence_pipeline_reference(d.as+G,d.h,d.g,d.b,d.z,d.gamma,1e-6f,d.ay+G,d.ah+G,int64_t(T),&d.q);d.q.wait_and_throw();});compare(d,T);
 std::printf("PASS call T=%zu prefix=%zu pattern=%s candidate=%zu fallback=%zu compiledSG=%u deviceWG=%zu kernelWG=%zu private_known=%d private=%zu spill_known=%d spill=%zu reason=%s\n",T,prefix,cancel?"ordered_cancellation":"48head_directed",candidate_calls,legacy_fallback_calls,report.compiled_subgroup,report.device_max_workgroup,report.kernel_max_workgroup,report.private_known,report.private_bytes,report.spill_known,report.spill_bytes,report.reason);std::fflush(stdout);
}
size_t decimal(const char* text){require(text&&*text,"empty argument");size_t n=0;for(const char* p=text;*p;++p){require(*p>='0'&&*p<='9',"nondecimal argument");require(n<=262144/10,"argument overflow/bound");n=n*10+size_t(*p-'0');require(n<=262144,"argument bound");}return n;}
// Opt-in quiet component timing. Existing parity modes never call these helpers.
struct TimingOptions {size_t total,chunk,samples;bool quad_first;};
void quiet_environment(){
 // Presence, including a literal zero, is rejected for these diagnostic hooks:
 // callers must remove them, not guess how each runtime interprets a value.
 for(const char* key:{"STRATA_TRACE","STRATA_TRACE_SYNC","STRATA_PROFILE_API",
  "UR_ENABLE_LAYERS","UR_LOG_LOADER","UR_LOG_LEVEL_ZERO","UR_LOG_TRACING",
  "UR_LOG_OPENCL","UR_LOG_CUDA","UR_LOG_HIP","UR_LOG_UMF","UR_LOG_ADAPTER",
  "SYCL_PI_TRACE","SYCL_UR_TRACE","SYCL_TRACE","SYCL_CACHE_TRACE","SYCL_PROGRAM_COMPILE_OPTIONS",
  "SYCL_PROGRAM_LINK_OPTIONS","ZE_ENABLE_TRACING_LAYER","ZE_ENABLE_VALIDATION_LAYER",
  "ZE_ENABLE_PARAMETER_VALIDATION","ZEL_ENABLE_LOADER_LOGGING","ZEL_LOADER_LOG_CONSOLE",
  "ZEL_LOADER_LOGGING_LEVEL","ZEL_LOADER_LOGGING_ENABLE_SUCCESS_PRINT",
  "LD_PRELOAD","LD_AUDIT","LD_DEBUG","LD_DEBUG_OUTPUT","IGC_ExtraOCLOptions",
  "IGC_EnableDump","IGC_DumpToCustomDir","IGC_DumpVISAASM","IGC_DumpASM",
  "ForceLargeGrfCompilationMode","OverrideDefaultFP64Settings"}){
  if(std::getenv(key)){std::fprintf(stderr,"quiet timing refused: unset %s\n",key);require(false,"diagnostic/compiler/profiler environment in quiet mode");}
 }
 // This is a finite known-hook guard, not proof that an external profiler or
 // system configuration is absent. The owner must freeze the complete env.
}
TimingOptions timing_options(int argc,char** argv){
 require(argc==9&&std::string(argv[1])=="--timing-prefix"&&std::string(argv[3])=="--chunk"&&std::string(argv[5])=="--order"&&std::string(argv[7])=="--samples",
 "timing usage: --timing-prefix 32768..262144 --chunk 1024|2048 --order legacy-first|quad-first --samples 1..9");
 for(int index:{2,4,8})require(std::strlen(argv[index])<=6,"timing decimal byte bound");
 TimingOptions o{decimal(argv[2]),decimal(argv[4]),decimal(argv[8]),false};
 require(o.total>=32768&&o.total<=262144&&(o.chunk==1024||o.chunk==2048)&&o.samples>=1&&o.samples<=9,"timing finite bounds");
 const std::string order(argv[6]);require(order=="legacy-first"||order=="quad-first","timing order");o.quad_first=order=="quad-first";
 quiet_environment();return o;
}
using Clock=std::chrono::steady_clock;
static_assert(Clock::is_steady);
struct TimingResult {
 std::array<double,2> seconds{}; // index0 explicit legacy, index1 explicit quad
 std::array<uint64_t,2> state{},fp32{},fp16{};
 uint64_t initial_state=0;
 size_t calls=0,tokens=0;
 strata::prefill::GdnQuadReport quad;
};
template<class T> std::array<uint64_t,2> timing_hash_pair(Buffers& d,T* a,T* b,size_t count){
 std::vector<T> x(count),y(count); // host endpoints remain alive through gpu_stage
 gpu_stage("timing untimed digest transfers",[&]{d.q.memcpy(x.data(),a,count*sizeof(T));d.q.memcpy(y.data(),b,count*sizeof(T));d.q.wait_and_throw();});
 std::array<uint64_t,2> hashes{fnv(x.data(),count*sizeof(T)),fnv(y.data(),count*sizeof(T))};
 require(hashes[0]==hashes[1],"timing post-comparison digest discrepancy");return hashes;
}
uint64_t digest_join(uint64_t current,uint64_t chunk_hash,size_t offset,size_t live){
 // Canonical little-endian uint64 tuple, no pointer/host padding representation.
 for(uint64_t word:{uint64_t(offset),uint64_t(live),chunk_hash})for(unsigned k=0;k<8;++k){current^=(word>>(8*k))&255;current*=1099511628211ULL;}return current;
}
TimingResult timing_pair(Buffers& d,const TimingOptions& o,bool quad_first){
 initialize(d,false,false); // same nonzero finite state in separate device buffers
 floats(d,d.as,d.bs,NS,NS,"timing identical initial states");
 TimingResult result;result.initial_state=timing_hash_pair(d,d.as,d.bs,NS+2*G)[0];
 result.fp32.fill(14695981039346656037ULL);result.fp16.fill(14695981039346656037ULL);
 for(size_t offset=0;offset<o.total;){
  const size_t live=std::min(o.chunk,o.total-offset);
  prepare(d,live,offset,false); // input/guard fill/transfer + drain, all untimed
  for(size_t position=0;position<2;++position){
   const bool quad=quad_first?(position==0):(position==1);
   bool selected=false;strata::prefill::GdnQuadReport report;
   // Queue is idle after prepare or the preceding arm's successful wait.
   // Includes wrapper overhead, candidate admission queries/bundle acquisition,
   // host submit, recurrence+norm, check(), and completion wait. No event claim.
   const auto start=Clock::now();
   gpu_stage(quad?"timed quad call and drain":"timed legacy call and drain",[&]{
    if(quad)selected=strata::prefill::gdn_recurrence_quad_variant(d.bs+G,d.h,d.g,d.b,d.z,d.gamma,1e-6f,d.by+G,d.bh+G,int64_t(live),&d.q,0,&report);
    else strata::prefill::gdn_recurrence_pipeline_reference(d.as+G,d.h,d.g,d.b,d.z,d.gamma,1e-6f,d.ay+G,d.ah+G,int64_t(live),&d.q);
    d.q.wait_and_throw();
   });
   const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
   require(std::isfinite(elapsed)&&elapsed>0,"invalid service clock interval");
   if(quad){require(selected&&report.status==strata::prefill::GdnQuadReport::Status::Submitted&&report.compiled_subgroup==32,"timing requires actually admitted quad, no fallback");result.quad=report;}
   result.seconds[quad?1:0]+=elapsed;
  }
  // Mandatory complete comparison after every corresponding carried chunk.
  // Readback/hash work is intentionally outside both service clock intervals.
  compare(d,live);
  result.state=timing_hash_pair(d,d.as,d.bs,NS+2*G);
  const auto y=timing_hash_pair(d,d.ay,d.by,d.capacity*6144+2*G);
  const auto half=timing_hash_pair(d,d.ah,d.bh,d.capacity*6144+2*G);
  for(size_t arm=0;arm<2;++arm){result.fp32[arm]=digest_join(result.fp32[arm],y[arm],offset,live);result.fp16[arm]=digest_join(result.fp16[arm],half[arm],offset,live);}
  offset+=live;++result.calls;result.tokens+=live;
 }
 require(result.tokens==o.total&&result.calls==(o.total+o.chunk-1)/o.chunk,"timing carried prefix counters");
 for(double v:result.seconds)require(std::isfinite(v)&&v>0,"invalid summed prefix service");
 return result;
}
void timing_emit(const TimingResult& r,const TimingOptions& o,const char* phase,size_t pair,bool quad_first){
 for(size_t position=0;position<2;++position){const bool quad=quad_first?(position==0):(position==1);const size_t arm=quad?1:0;
  std::printf("TIMING,%s,%zu,%s,%zu,%s,%zu,%zu,%zu,%.9f,%zu,%zu,%zu,%zu,%016llx,%016llx,%016llx,%016llx,%u,%d,%zu,%d,%zu,pass\n",
   phase,pair,quad_first?"quad-first":"legacy-first",position+1,quad?"quad":"legacy",o.total,o.chunk,r.calls,r.seconds[arm],r.tokens,
   r.calls,quad?r.calls:size_t(0),r.calls,
   (unsigned long long)r.initial_state,(unsigned long long)r.state[arm],(unsigned long long)r.fp32[arm],(unsigned long long)r.fp16[arm],
   quad?r.quad.compiled_subgroup:0,quad&&r.quad.private_known,quad?r.quad.private_bytes:0,quad&&r.quad.spill_known,quad?r.quad.spill_bytes:0);
 }
 require(std::fflush(stdout)==0&&std::ferror(stdout)==0,"timing sample output failure");
}
void timing_run(sycl::queue& q,const TimingOptions& o){
 require(q.has_property<sycl::property::queue::in_order>(),"timing in-order queue");
 std::printf("TIMING_META,total=%zu,chunk=%zu,samples=%zu,warmup_pairs=2,clock=host_steady_service_sum,admission_included=true,interleaved_matching_chunks=true,queue_profiling_property=%d,model=false,kernel_only=false,adopted=false\n",o.total,o.chunk,o.samples,q.has_property<sycl::property::queue::enable_profiling>());
 std::puts("TIMING_HEADER,phase,pair,order,position,arm,total,chunk,chunks,service_seconds,tokens,recurrence_calls,admission_attempts,norm_calls,initial_state_fnv,final_state_fnv,guarded_fp32_chunk_chain_fnv,guarded_fp16_chunk_chain_fnv,compiled_sg,private_known,private,spill_known,spill,exact");
 Buffers d(q,o.chunk);TimingResult reference;
 // Full-prefix warmup for BOTH kernels and norm, in both execution orders.
 // Warmup samples are labelled and are not recorded trial samples.
 for(size_t warm=0;warm<2;++warm){bool first=o.quad_first!=(warm%2!=0);const auto r=timing_pair(d,o,first);if(warm==0)reference=r;
  else require(r.initial_state==reference.initial_state&&r.state==reference.state&&r.fp32==reference.fp32&&r.fp16==reference.fp16,"warmup prefix repeat digest");
  timing_emit(r,o,"warmup",warm+1,first);
 }
 for(size_t sample=0;sample<o.samples;++sample){bool first=o.quad_first!=(sample%2!=0);const auto r=timing_pair(d,o,first);
  require(r.initial_state==reference.initial_state&&r.state==reference.state&&r.fp32==reference.fp32&&r.fp16==reference.fp16,"recorded prefix repeat digest");timing_emit(r,o,"sample",sample+1,first);
 }
 // Buffers destructor drains/releases before caller prints aggregate success.
}
} // namespace
int main(int argc,char** argv){try{
 if(argc==2&&std::string(argv[1])=="--host-fail-stop-probe"){
  struct HostLifetimeProbe {~HostLifetimeProbe(){std::fputs("FAIL host fail-stop probe unwound\n",stderr);}} probe;
  std::vector<float> endpoint(8,1.f);
  gpu_stage("host injected exception",[&]{require(endpoint[0]==1.f,"probe endpoint");throw std::runtime_error("injected host-only exception; no GPU API");});
  require(false,"host fail-stop probe returned");
 }
 if(argc==2&&std::string(argv[1])=="--host-only"){
  early_contract();
  std::puts("PASS host-only empty/denial/negative/ordered-partial contracts; queue_lookup=false GPU_submission=false");
  require(std::fflush(stdout)==0&&std::ferror(stdout)==0&&std::fflush(stderr)==0&&std::ferror(stderr)==0,"final output failure");
  return 0;
 }
 if(argc>1&&std::string(argv[1])=="--timing-prefix"){
  const auto options=timing_options(argc,argv); // reject args/env before queue lookup
  auto& timing_queue=*strata::q_of(nullptr);timing_run(timing_queue,options);
  std::printf("TIMING_SUMMARY,samples=%zu,warmup_pairs=2,recorded_arms=%zu,exact=true,synthetic_component_only=true,model=false,full_lifecycle=false,adopted=false,pass\n",options.samples,2*options.samples);
  require(std::fflush(stdout)==0&&std::ferror(stdout)==0&&std::fflush(stderr)==0&&std::ferror(stderr)==0,"final timing output failure");return 0;
 }
 size_t total=0,chunk=2048;
 if(argc!=1){require(argc==5&&std::string(argv[1])=="--prefix"&&std::string(argv[3])=="--chunk","usage: gdn_quad_parity [--host-only | --host-fail-stop-probe | --prefix 32768|262144 --chunk 1..2048]");total=decimal(argv[2]);chunk=decimal(argv[4]);require((total==32768||total==262144)&&chunk>0&&chunk<=2048&&(total+chunk-1)/chunk<=2048,"bounded prefix/chunk admission");}
 early_contract();auto& q=*strata::q_of(nullptr);require(q.has_property<sycl::property::queue::in_order>(),"parity requires same in-order queue");
 std::printf("device=%s prefix=%zu chunk_cap=%zu numerical_gate=bitwise model=false performance=false\n",q.get_device().get_info<sycl::info::device::name>().c_str(),total,chunk);
 if(total){Buffers d(q,chunk);initialize(d,false,false);size_t prefix=0;while(prefix<total){size_t live=std::min(chunk,total-prefix);call(d,live,prefix,false);prefix+=live;}require(prefix==total,"prefix carry length");}
 else {
  for(size_t T:{size_t(1),size_t(3),size_t(4),size_t(5),size_t(17),size_t(257)})for(bool zero:{true,false})for(bool cancel:{false,true}){
   Buffers d(q,std::max(T,size_t(5)));initialize(d,zero,cancel);std::printf("CASE T=%zu initial=%s pattern=%s\n",T,zero?"zero":"nonzero",cancel?"cancellation":"directed");size_t prefix=0;
   for(size_t live:{T,size_t(3),size_t(5)}){call(d,live,prefix,cancel);prefix+=live;}
  }
  Buffers denied(q,5);initialize(denied,false,false);empty_device(denied);call(denied,5,0,false,true);
 }
 std::puts("PASS complete candidate-specific synthetic GPU differential; no model/full-lifecycle/performance qualification");
 require(std::fflush(stdout)==0&&std::ferror(stdout)==0&&std::fflush(stderr)==0&&std::ferror(stderr)==0,"final output failure");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
