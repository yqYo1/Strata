// Standalone proposal; all builds/tests/measurements NOT_RUN by author.
#include "../parallel-iq-dequant-v1/fixture_common.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
using Clock=std::chrono::steady_clock;
void need(bool b,const char *s){if(!b) throw std::runtime_error(s);}
// Heap-owned runtime and USM: no destructor releases unknown-completion storage.
struct Runtime {
  sycl::queue *q=nullptr; std::vector<void*> allocations;
  std::atomic<bool> fault{false}; std::mutex mutex; std::exception_ptr first;
  Runtime(){allocations.reserve(8);} ~Runtime()=delete;
  void initialize(){q=new sycl::queue(sycl::gpu_selector_v,[this](sycl::exception_list es) noexcept {
    if(es.size()==0)return; fault.store(true,std::memory_order_release);
    try{std::lock_guard<std::mutex> l(mutex);if(!first)first=*es.begin();}catch(...){}
  },sycl::property::queue::in_order{});}
  template<class T>T *allocate(size_t n){need(allocations.size()<allocations.capacity(),"owner registry full");
    T *p=sycl::malloc_shared<T>(n,*q);if(!p)throw std::bad_alloc();allocations.push_back(p);return p;}
  void drain(){q->wait_and_throw();if(fault.load(std::memory_order_acquire)){
    std::lock_guard<std::mutex> l(mutex);if(first)std::rethrow_exception(first);throw std::runtime_error("sticky asynchronous fault");}}
  void retire(){drain();for(void *p:allocations)sycl::free(p,*q);allocations.clear();delete q;q=nullptr;
    need(!fault.load(std::memory_order_acquire),"fault during retirement");}
};
template<class T>struct Guard {
  size_t n;T *base,*p;
  Guard(Runtime &r,size_t count):n(count),base(r.allocate<T>(n+128)),p(base+64){std::memset(base,0xa5,(n+128)*sizeof(T));}
  bool valid()const{auto b=reinterpret_cast<const uint8_t*>(base);for(size_t i=0;i<64*sizeof(T);++i)
    if(b[i]!=0xa5||b[(n+64)*sizeof(T)+i]!=0xa5)return false;return true;}
};
uint64_t fingerprint(const void *v,size_t n){auto p=static_cast<const uint8_t*>(v);uint64_t h=14695981039346656037ull;
  for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;}
std::string quoted(const std::string &s){std::ostringstream o;o<<'"';for(unsigned char c:s){
  if(c=='"'||c=='\\')o<<'\\'<<char(c);else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else o<<char(c);}o<<'"';return o.str();}
struct Trial{int64_t submit,wall;};
int main(int argc,char **argv){Runtime *r=nullptr;try{
  need(argc==9||argc==10,"usage: timing 0|1 TYPE gu|down NE dequant|gemm REPEATS TRIALS WARMUP [NEW_CAPTURE]");
  std::string arm=argv[1],role=argv[3],mode=argv[5];int type=std::stoi(argv[2]),ne=std::stoi(argv[4]);
  int repeats=std::stoi(argv[6]),trials=std::stoi(argv[7]),warmup=std::stoi(argv[8]);
  const char *gate=std::getenv("STRATA_SYCL_PARALLEL_IQ_DEQUANT");
  need((arm=="0"||arm=="1")&&gate&&arm==gate,"exact immutable 0/1 environment gate required before provider init");
  bool gu=role=="gu";need(gu||role=="down","role");need(mode=="dequant"||mode=="gemm","mode");
  need((gu&&(type==18||type==22||type==21||type==23))||(!gu&&(type==20||type==42)),"type/role");
  need(ne==4||ne==16||ne==64||ne==128||ne==512,"ne");
  need(repeats>=1&&repeats<=100&&trials>=3&&trials<=30&&warmup>=1&&warmup<=10,"bounded repetitions/trials/warmup");
  need(argc==9||!std::filesystem::exists(argv[9]),"capture must be new");
  size_t rows=gu?640:2560,k=gu?2560:640,n=rows*(gu?2:1),wn=n*k;
  auto f=strata::sycl_backend::block_format(type);size_t bytes=rows*k/f.width*f.bytes;
  r=new Runtime;r->initialize();auto &q=*r->q;
  need(q.get_device().has(sycl::aspect::fp16)&&q.get_device().has(sycl::aspect::usm_shared_allocations),"device admission");
  Guard<uint8_t> w(*r,bytes),u(*r,bytes);Guard<uint16_t> dense(*r,wn),reference(*r,wn),x(*r,size_t(ne)*k);
  Guard<float> y(*r,size_t(ne)*n),expected(*r,size_t(ne)*n);
  for(size_t b=0;b<bytes/f.bytes;++b){fixture::make_block(type,w.p+b*f.bytes,unsigned(b),0x1800);
    fixture::make_block(type,u.p+b*f.bytes,unsigned(b+337),0x9800);}
  for(size_t i=0;i<x.n;++i)x.p[i]=strata::kernels::f16_from_f32(float(int((i*73+19)%257)-128)/128.f);
  std::vector<uint8_t> wi(w.p,w.p+w.n),ui(u.p,u.p+u.n);std::vector<uint16_t> xi(x.p,x.p+x.n);
  // Independent existing scalar CPU decoder and explicit repository half RNE.
  auto rw=fixture::oracle(type,w.p,rows*k);auto ru=gu?fixture::oracle(type,u.p,rows*k):std::vector<uint16_t>{};
  for(size_t row=0;row<rows;++row){std::copy_n(rw.data()+row*k,k,reference.p+row*k*(gu?2:1));
    if(gu)std::copy_n(ru.data()+row*k,k,reference.p+(row*2+1)*k);}
  auto gemm=[&](const uint16_t *a,float *b){oneapi::mkl::blas::row_major::gemm(q,
    oneapi::mkl::transpose::nontrans,oneapi::mkl::transpose::trans,int64_t(ne),int64_t(n),int64_t(k),1.f,
    reinterpret_cast<const sycl::half*>(x.p),int64_t(k),reinterpret_cast<const sycl::half*>(a),int64_t(k),0.f,b,int64_t(n));};
  if(mode=="gemm"){gemm(reference.p,expected.p);r->drain();}
  auto work=[&]{if(gu)strata::kernels::iq_dequant_gu_f16(type,w.p,u.p,int64_t(rows),int64_t(k),dense.p,&q);
    else strata::kernels::iq_dequant_f16(type,w.p,int64_t(wn),dense.p,&q);
    if(mode=="gemm")gemm(dense.p,y.p);};
  double maxerror=0;size_t checked=0;
  auto verify=[&]{r->drain();need(w.valid()&&u.valid()&&dense.valid()&&reference.valid()&&x.valid()&&y.valid()&&expected.valid(),"guards");
    need(std::memcmp(w.p,wi.data(),w.n)==0&&std::memcmp(u.p,ui.data(),u.n)==0&&std::memcmp(x.p,xi.data(),x.n*2)==0,"inputs changed");
    need(std::memcmp(dense.p,reference.p,wn*2)==0,"full half oracle mismatch");
    for(size_t i=0;i<wn;++i)need(std::isfinite(strata::kernels::f32_from_f16(dense.p[i])),"nonfinite half");
    if(mode=="gemm")for(size_t i=0;i<y.n;++i){need(std::isfinite(y.p[i])&&std::isfinite(expected.p[i]),"nonfinite GEMM");
      maxerror=std::max(maxerror,std::abs(double(y.p[i])-expected.p[i]));need(y.p[i]==expected.p[i],"full GEMM reference mismatch");}++checked;};
  work();verify(); // Full pre-timing correctness, then untimed warmup.
  for(int i=0;i<warmup;++i)work();r->drain();
  std::vector<Trial> samples;samples.reserve(trials);
  auto ns=[](auto a,auto b){return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();};
  for(int t=0;t<trials;++t){r->drain();auto a=Clock::now();for(int i=0;i<repeats;++i)work();auto b=Clock::now();
    r->drain();auto c=Clock::now();samples.push_back({ns(a,b),ns(a,c)});verify();}
  verify();uint64_t wh=fingerprint(w.p,w.n),uh=fingerprint(u.p,u.n),xh=fingerprint(x.p,x.n*2),dh=fingerprint(dense.p,wn*2);
  uint64_t yh=mode=="gemm"?fingerprint(y.p,y.n*4):0;
  if(argc==10){size_t count=mode=="gemm"?y.n*4:wn*2;need(count<=32*1024*1024,"capture budget");
    std::ofstream out(argv[9],std::ios::binary);out.write(mode=="gemm"?reinterpret_cast<char*>(y.p):reinterpret_cast<char*>(dense.p),count);out.close();need(bool(out),"capture write");}
  auto dev=q.get_device();std::string name=dev.get_info<sycl::info::device::name>(),driver=dev.get_info<sycl::info::device::driver_version>(),vendor=dev.get_info<sycl::info::device::vendor>();
  r->retire();
  std::cout<<std::setprecision(17)<<"{\"schema\":1,\"complete\":true,\"arm\":"<<quoted(arm)<<",\"type\":"<<type<<",\"role\":"<<quoted(role)<<",\"mode\":"<<quoted(mode)
    <<",\"ne\":"<<ne<<",\"rows\":"<<rows<<",\"N\":"<<n<<",\"K\":"<<k<<",\"device\":"<<quoted(name)<<",\"vendor\":"<<quoted(vendor)<<",\"driver\":"<<quoted(driver)
    <<",\"warmup\":"<<warmup<<",\"repeats_per_batch\":"<<repeats<<",\"product_dequant_calls_per_repeat\":1,\"gemm_calls_per_repeat\":"<<(mode=="gemm"?1:0)
    <<",\"materialized_half_elements_per_repeat\":"<<wn<<",\"packed_input_bytes_per_repeat\":"<<bytes*(gu?2:1)<<",\"gemm_flops_per_repeat\":"<<(mode=="gemm"?uint64_t(2)*ne*n*k:0)
    <<",\"w_fnv1a64\":"<<wh<<",\"u_fnv1a64\":"<<uh<<",\"x_fnv1a64\":"<<xh<<",\"dense_fnv1a64\":"<<dh<<",\"output_fnv1a64\":"<<yh
    <<",\"full_checks\":"<<checked<<",\"max_output_error\":"<<maxerror<<",\"full_half_oracle\":true,\"full_gemm_reference\":"<<(mode=="gemm"?"true":"false")
    <<",\"finite\":true,\"guards\":true,\"inputs_unchanged\":true,\"queue_retired\":true,\"trials\":[";
  for(size_t i=0;i<samples.size();++i){if(i)std::cout<<',';std::cout<<"{\"index\":"<<i<<",\"host_submit_ns\":"<<samples[i].submit<<",\"completed_batch_wall_ns\":"<<samples[i].wall<<'}';}
  std::cout<<"]}\n";std::cout.flush();need(bool(std::cout),"result write");return 0;
}catch(const std::exception &e){std::fprintf(stderr,"ORIGINAL_ERROR: %s\nFAIL_STOP: no unknown-completion owner unwind/free/retry; Main must reap and check faults\n",e.what());std::fflush(stderr);std::_Exit(86);
}catch(...){std::fprintf(stderr,"FAIL_STOP: non-std exception\n");std::fflush(stderr);std::_Exit(86);}}
