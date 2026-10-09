// Candidate-specific exact GPU differential test. No timing or model execution.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/gdn_variant.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
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
  catch(...){auto error=std::current_exception();for(auto p:owned)try{sycl::free(p,q);}catch(...){}std::rethrow_exception(error);}
 }
 ~Buffers() noexcept(false){// Normal calls drain before destruction; never retry a failed recurrence.
  std::exception_ptr error;
  try{q.wait_and_throw();}catch(...){error=std::current_exception();}
  for(auto p:owned)try{sycl::free(p,q);}catch(...){if(!error)error=std::current_exception();}
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
 d.q.fill(d.as,sentinel,NS+2*G);d.q.fill(d.bs,sentinel,NS+2*G);
 d.q.memcpy(d.as+G,state.data(),NS*sizeof(float));d.q.memcpy(d.bs+G,state.data(),NS*sizeof(float));
 d.q.wait_and_throw();
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
 d.q.fill(d.ay,sentinel,d.capacity*6144+2*G);d.q.fill(d.by,sentinel,d.capacity*6144+2*G);d.q.fill(d.ah,half_sentinel,d.capacity*6144+2*G);d.q.fill(d.bh,half_sentinel,d.capacity*6144+2*G);
 d.q.memcpy(d.h,h.data(),h.size()*4);d.q.memcpy(d.g,gate.data(),gate.size()*4);d.q.memcpy(d.b,beta.data(),beta.size()*4);d.q.memcpy(d.z,z.data(),z.size()*4);d.q.memcpy(d.gamma,gamma.data(),128*4);d.q.wait_and_throw();
}
void floats(Buffers& d,float* a,float* b,size_t live,size_t allocated,const char* label){
 std::vector<float> x(allocated+2*G),y(x.size());d.q.memcpy(x.data(),a,x.size()*4);d.q.memcpy(y.data(),b,y.size()*4);d.q.wait_and_throw();
 size_t different=0,bounds=0;bool finite=true;
 for(size_t i=0;i<x.size();++i){if(i<G||i>=G+live)bounds+=x[i]!=sentinel||y[i]!=sentinel;else{different+=std::bit_cast<uint32_t>(x[i])!=std::bit_cast<uint32_t>(y[i]);finite=finite&&std::isfinite(x[i])&&std::isfinite(y[i])&&x[i]!=sentinel&&y[i]!=sentinel;}}
 if(different||bounds||!finite)std::fprintf(stderr,"FAIL %s different=%zu guards=%zu finite=%d\n",label,different,bounds,finite);
 require(different==0&&bounds==0&&finite,"full float bitwise/finite/guard gate");
}
void compare(Buffers& d,size_t T){
 floats(d,d.as,d.bs,NS,NS,"state");floats(d,d.ay,d.by,T*6144,d.capacity*6144,"FP32y");
 std::vector<uint16_t> x(d.capacity*6144+2*G),y(x.size());d.q.memcpy(x.data(),d.ah,x.size()*2);d.q.memcpy(y.data(),d.bh,y.size()*2);d.q.wait_and_throw();
 size_t different=0,bounds=0;bool finite=true;for(size_t i=0;i<x.size();++i){if(i<G||i>=G+T*6144)bounds+=x[i]!=half_sentinel||y[i]!=half_sentinel;else{different+=x[i]!=y[i];finite=finite&&(x[i]&0x7c00)!=0x7c00&&(y[i]&0x7c00)!=0x7c00&&x[i]!=half_sentinel&&y[i]!=half_sentinel;}}
 require(different==0&&bounds==0&&finite,"full half bitwise/finite/guard gate");
}
void empty_device(Buffers& d){
 prepare(d,1,0,false);
 std::vector<float> before(NS+2*G),after(before.size());d.q.memcpy(before.data(),d.bs,before.size()*4);d.q.wait_and_throw();
 strata::prefill::GdnQuadReport report;
 bool empty=strata::prefill::gdn_recurrence_quad_variant(d.bs+G,nullptr,nullptr,nullptr,nullptr,nullptr,1e-6f,d.by+G,d.bh+G,0,&d.q,0,&report);
 require(empty&&report.status==strata::prefill::GdnQuadReport::Status::Empty,"device empty contract");
 d.q.memcpy(after.data(),d.bs,after.size()*4);d.q.wait_and_throw();require(!std::memcmp(before.data(),after.data(),before.size()*4),"empty candidate changed full state bits");
 floats(d,d.by,d.by,0,d.capacity*6144,"empty FP32 output");
 std::vector<uint16_t> half(d.capacity*6144+2*G);d.q.memcpy(half.data(),d.bh,half.size()*2);d.q.wait_and_throw();require(std::all_of(half.begin(),half.end(),[](auto v){return v==half_sentinel;}),"empty candidate changed half output");
 std::printf("PASS empty_device full_state_fnv=%llu candidate_calls=0 norm_calls=0\n",(unsigned long long)fnv(before.data(),before.size()*4));
}
void call(Buffers& d,size_t T,size_t prefix,bool cancel,bool force_deny=false){
 prepare(d,T,prefix,cancel);strata::prefill::GdnQuadReport report;
 size_t candidate_calls=0,legacy_fallback_calls=0;
 // Forced denial has its own device snapshot gate before any explicit legacy fallback.
 std::vector<float> before;if(force_deny){before.resize(NS+2*G);d.q.memcpy(before.data(),d.bs,before.size()*4);d.q.wait_and_throw();}
 bool selected=strata::prefill::gdn_recurrence_quad_variant(d.bs+G,d.h,d.g,d.b,d.z,d.gamma,1e-6f,d.by+G,d.bh+G,int64_t(T),&d.q,0,&report,force_deny);
 if(!selected&&!force_deny)std::fprintf(stderr,"DENIED exact_quad reason=%s compiledSG=%u deviceWG=%zu kernelWG=%zu\n",report.reason,report.compiled_subgroup,report.device_max_workgroup,report.kernel_max_workgroup);
 if(force_deny){
  require(!selected&&report.status==strata::prefill::GdnQuadReport::Status::Denied,"denial submitted candidate");
  std::vector<float> after(before.size());d.q.memcpy(after.data(),d.bs,after.size()*4);d.q.wait_and_throw();require(!std::memcmp(before.data(),after.data(),before.size()*4),"denied candidate changed full state bits");
  // Output allocations must remain entirely sentinel until caller chooses fallback.
  floats(d,d.by,d.by,0,d.capacity*6144,"denied FP32 output");
  std::vector<uint16_t> half(d.capacity*6144+2*G);d.q.memcpy(half.data(),d.bh,half.size()*2);d.q.wait_and_throw();require(std::all_of(half.begin(),half.end(),[](auto v){return v==half_sentinel;}),"denied candidate changed half output");
  strata::prefill::gdn_recurrence_pipeline_reference(d.bs+G,d.h,d.g,d.b,d.z,d.gamma,1e-6f,d.by+G,d.bh+G,int64_t(T),&d.q);++legacy_fallback_calls;
 }else{require(selected&&report.status==strata::prefill::GdnQuadReport::Status::Submitted,"candidate unavailable: this is not candidate parity PASS");++candidate_calls;}
 strata::prefill::gdn_recurrence_pipeline_reference(d.as+G,d.h,d.g,d.b,d.z,d.gamma,1e-6f,d.ay+G,d.ah+G,int64_t(T),&d.q);
 d.q.wait_and_throw();compare(d,T);
 std::printf("PASS call T=%zu prefix=%zu pattern=%s candidate=%zu fallback=%zu compiledSG=%u deviceWG=%zu kernelWG=%zu private_known=%d private=%zu spill_known=%d spill=%zu reason=%s\n",T,prefix,cancel?"ordered_cancellation":"48head_directed",candidate_calls,legacy_fallback_calls,report.compiled_subgroup,report.device_max_workgroup,report.kernel_max_workgroup,report.private_known,report.private_bytes,report.spill_known,report.spill_bytes,report.reason);std::fflush(stdout);
}
size_t decimal(const char* text){require(text&&*text,"empty argument");size_t n=0;for(const char* p=text;*p;++p){require(*p>='0'&&*p<='9',"nondecimal argument");require(n<=262144/10,"argument overflow/bound");n=n*10+size_t(*p-'0');require(n<=262144,"argument bound");}return n;}
} // namespace
int main(int argc,char** argv){try{
 size_t total=0,chunk=2048;
 if(argc!=1){require(argc==5&&std::string(argv[1])=="--prefix"&&std::string(argv[3])=="--chunk","usage: gdn_quad_parity [--prefix 32768|262144 --chunk 1..2048]");total=decimal(argv[2]);chunk=decimal(argv[4]);require((total==32768||total==262144)&&chunk>0&&chunk<=2048&&(total+chunk-1)/chunk<=2048,"bounded prefix/chunk admission");}
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
