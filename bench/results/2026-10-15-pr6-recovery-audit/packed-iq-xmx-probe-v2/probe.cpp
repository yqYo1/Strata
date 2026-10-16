// Standalone proposal. No product dispatch. Validation status: NOT_RUN.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <oneapi/mkl/blas.hpp>
#include "../parallel-iq-dequant-v1/fixture_common.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <limits>
#include <atomic>
#include <mutex>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
namespace mx = sycl::ext::oneapi::experimental::matrix;
using namespace strata::sycl_backend;
using strata::kernels::f16_from_f32;
// Reuses the 8x16x16 FP16/FP32 tile in tools/sycl/matrix_probe.cpp and
// configured src/kernels/sycl/prompt_attention.cpp, not a new tile assumption.
constexpr int TM=8,TN=16,TK=16,SG=16;
struct Shape { int m,n,k,type; bool gu; };
bool admit(const sycl::device &d, Shape s) {
  if(s.m<1 || s.m>1280 || s.n<16 || s.n>2560 || s.k<16 || s.k>2560 ||
     s.n%TN || s.k%TK) return false;
  if(s.type!=18 && s.type!=20 && s.type!=21 && s.type!=22 && s.type!=23 && s.type!=42) return false;
  auto f=block_format(s.type);
  if(s.k%f.width || (s.gu && s.n%2)) return false;
  auto sizes=d.get_info<sycl::info::device::sub_group_sizes>();
  if(!d.has(sycl::aspect::usm_shared_allocations) || !d.has(sycl::aspect::fp16) || !d.has(sycl::aspect::ext_intel_matrix) ||
     std::find(sizes.begin(),sizes.end(),SG)==sizes.end() ||
     d.get_info<sycl::info::device::max_work_group_size>()<SG ||
     d.get_info<sycl::info::device::local_mem_size>()<1280) return false;
  for(auto c:d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
    if(c.atype==mx::matrix_type::fp16 && c.btype==mx::matrix_type::fp16 &&
       c.ctype==mx::matrix_type::fp32 && c.dtype==mx::matrix_type::fp32 &&
       c.msize==TM && c.nsize==TN && c.ksize==TK) return true;
  return false; // v2 deliberately declines max-only advertisements.
}
// Packed GU storage: [gate rows][up rows]; output columns alternate gate/up.
inline size_t source_row(size_t col, Shape s) {
  return s.gu ? col/2+(col%2)*size_t(s.n/2) : col;
}
template<int Type> sycl::event packed(sycl::queue &q,Shape s,const sycl::half *x,const uint8_t *w,float *y) {
  constexpr int BW=parallel_iq_width<Type>,BS=parallel_iq_bytes<Type>;
  const size_t rb=size_t(s.k/BW)*BS;
  size_t groups=size_t((s.m+TM-1)/TM)*(s.n/TN);
  return q.submit([&](sycl::handler &h){
    sycl::local_accessor<sycl::half,1> a(TM*TK,h),b(TK*TN,h);
    sycl::local_accessor<float,1> c(TM*TN,h);
    h.parallel_for(sycl::nd_range<1>{groups*SG,SG},[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
      auto sg=it.get_sub_group(); int lane=it.get_local_linear_id();
      size_t group=it.get_group_linear_id();
      int m0=int(group/(s.n/TN))*TM,n0=int(group%(s.n/TN))*TN;
      mx::joint_matrix<sycl::sub_group,sycl::half,mx::use::a,TM,TK,mx::layout::row_major> ja;
      mx::joint_matrix<sycl::sub_group,sycl::half,mx::use::b,TK,TN,mx::layout::row_major> jb;
      mx::joint_matrix<sycl::sub_group,float,mx::use::accumulator,TM,TN> jc;
      mx::joint_matrix_fill(sg,jc,0.f);
      for(int k0=0;k0<s.k;k0+=TK) {
        for(int i=lane;i<TM*TK;i+=SG) {
          int row=m0+i/TK;
          a[i]=row<s.m ? x[size_t(row)*s.k+k0+i%TK] : sycl::half(0.f);
        }
        for(int i=lane;i<TK*TN;i+=SG) {
          int kk=k0+i/TN,col=n0+i%TN;
          const uint8_t *block=w+source_row(col,s)*rb+size_t(kk/BW)*BS;
          // Explicit repository RNE conversion; do not use implicit float->half.
          b[i]=sycl::bit_cast<sycl::half>(f16_from_f32(parallel_iq_element<Type>(block,kk%BW)));
        }
        it.barrier(sycl::access::fence_space::local_space);
        mx::joint_matrix_load(sg,ja,a.template get_multi_ptr<sycl::access::decorated::no>(),TK);
        mx::joint_matrix_load(sg,jb,b.template get_multi_ptr<sycl::access::decorated::no>(),TN);
        mx::joint_matrix_mad(sg,jc,ja,jb,jc);
        // Uniform barrier before any lane overwrites A/B for the next K tile.
        it.barrier(sycl::access::fence_space::local_space);
      }
      mx::joint_matrix_store(sg,jc,c.template get_multi_ptr<sycl::access::decorated::no>(),TN,mx::layout::row_major);
      it.barrier(sycl::access::fence_space::local_space);
      for(int i=lane;i<TM*TN;i+=SG) if(m0+i/TN<s.m)
        y[size_t(m0+i/TN)*s.n+n0+i%TN]=c[i];
    });
  });
}
sycl::event dispatch(sycl::queue &q,Shape s,const sycl::half *x,const uint8_t *w,float *y) {
  if(!admit(q.get_device(),s)) throw std::invalid_argument("DECLINE: shape/type/matrix admission; caller must use original fallback");
  switch(s.type) {
#define ARM(T) case T:return packed<T>(q,s,x,w,y);
    ARM(18) ARM(20) ARM(21) ARM(22) ARM(23) ARM(42)
#undef ARM
  } throw std::invalid_argument("DECLINE: unsupported format");
}
// Runtime and queue live on the heap; no stack unwinding owns USM or queue.
// On ANY exception after context creation, fail-stop with all owners intact.
struct Runtime {
  sycl::queue *q=nullptr;
  std::vector<void*> allocations;
  std::atomic<bool> async_fault{false};
  std::mutex fault_mutex;
  std::exception_ptr first_async;
  Runtime() { allocations.reserve(8); }
  ~Runtime()=delete;
  void initialize() {
    q=new sycl::queue(sycl::gpu_selector_v,[this](sycl::exception_list list) noexcept {
      if(list.size()==0) return;
      async_fault.store(true,std::memory_order_release);
      try { std::lock_guard<std::mutex> lock(fault_mutex);
        if(!first_async) first_async=*list.begin();
      } catch(...) { /* sticky flag still forbids retirement/free */ }
    },sycl::property::queue::in_order{});
  }
  template<class T> T *allocate(size_t count) {
    // Registry capacity is fixed before any provider initialization. No vector
    // growth exception can strand a successfully allocated, unregistered pointer.
    if(allocations.size()==allocations.capacity()) throw std::runtime_error("owner registry full");
    T *p=sycl::malloc_shared<T>(count,*q);
    if(!p) throw std::bad_alloc();
    allocations.push_back(p); return p;
  }
  void checked_wait() {
    q->wait_and_throw();
    if(async_fault.load(std::memory_order_acquire)) {
      std::lock_guard<std::mutex> lock(fault_mutex);
      if(first_async) std::rethrow_exception(first_async);
      throw std::runtime_error("asynchronous SYCL fault (detail unavailable)");
    }
  }
  // The only free/delete path. Called once, after tensor captures, before success metadata.
  // If wait, async delivery, free or destruction throws, outer fail-stop applies.
  void retire() {
    checked_wait();
    for(void *p:allocations) sycl::free(p,*q);
    allocations.clear();
    delete q; q=nullptr;
    if(async_fault.load(std::memory_order_acquire))
      throw std::runtime_error("asynchronous fault during queue retirement");
  }
};
[[noreturn]] void fail_stop(const char *original) noexcept {
  std::fprintf(stderr,"ORIGINAL_ERROR: %s\nFAIL_STOP: probe failed; no implicit unwind/free/retry; unknown-completion owners retained; Main must reap and check faults\n",original);
  std::fflush(stderr);
  std::_Exit(86);
}
template<class T> struct Guard {
  Runtime &runtime; size_t n; T *base,*p;
  Guard(Runtime &rt,size_t count):runtime(rt),n(count) {
    base=runtime.allocate<T>(n+128);
    p=base+64; std::memset(base,0xa5,(n+128)*sizeof(T));
  }
  bool valid() const { auto b=reinterpret_cast<const uint8_t*>(base);
    for(size_t i=0;i<64*sizeof(T);++i) if(b[i]!=0xa5 || b[(n+64)*sizeof(T)+i]!=0xa5) return false;
    return true;
  }
  ~Guard()=default; // Runtime retains allocation ownership beyond this scope.
  Guard(const Guard&)=delete;
};
void dump(const std::string &p,const void *v,size_t n) {
  std::ofstream f(p,std::ios::binary); f.write(static_cast<const char*>(v),n);
  f.close();
  if(!f) throw std::runtime_error("capture write failed");
}
// ARM TYPE ROLE M N K PREFIX. Exactly one case/arm per immutable process.
int main(int argc,char **argv) { Runtime *runtime=nullptr; try {
  if(argc!=8) throw std::invalid_argument("probe ref|packed TYPE gu|down M N K PREFIX");
  std::string arm=argv[1],role=argv[3],prefix=argv[7];
  if((arm!="ref" && arm!="packed") || (role!="gu" && role!="down")) throw std::invalid_argument("bad arm/role");
  Shape s{std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[2]),role=="gu"};
  runtime=new Runtime; runtime->initialize(); auto &q=*runtime->q;
  // Reject before sizes/allocation, including ref: comparisons use same domain.
  if(!admit(q.get_device(),s)) { std::cerr<<"DECLINE\n"; runtime->retire(); return 77; }
  auto f=block_format(s.type); size_t rb=size_t(s.k/f.width)*f.bytes;
  Guard<uint8_t> w(*runtime,size_t(s.n)*rb); Guard<sycl::half> x(*runtime,size_t(s.m)*s.k);
  Guard<float> y(*runtime,size_t(s.m)*s.n);
  std::fill(y.p,y.p+y.n,std::numeric_limits<float>::quiet_NaN());
  for(size_t i=0;i<size_t(s.n)*s.k/f.width;++i)
    fixture::make_block(s.type,w.p+i*f.bytes,unsigned(i),uint16_t(0x0800+(i%4)*0x0400));
  for(size_t i=0;i<x.n;++i) x.p[i]=sycl::half(float(int((i*73+19)%257)-128)/128.f);
  std::vector<uint8_t> wi(w.p,w.p+w.n);
  std::vector<sycl::half> xi(x.p,x.p+x.n);
  dump(prefix+".w.bin",w.p,w.n); dump(prefix+".x.bin",x.p,x.n*2);
  double maxw=0,maxx=0; for(auto v:xi) maxx=std::max(maxx,std::abs(double(v)));
  // Scalar host decode only for operand bound; candidate never allocates a full FP16 W.
  for(size_t off=0;off<w.n;off+=f.bytes) { float v[256]; decode_block(s.type,wi.data()+off,v);
    for(int j=0;j<f.width;++j) maxw=std::max(maxw,std::abs(double(strata::kernels::f32_from_f16(f16_from_f32(v[j]))))); }
  auto t0=std::chrono::steady_clock::now();
  bool scratch_guard=true;
  if(arm=="packed") { dispatch(q,s,x.p,w.p,y.p); runtime->checked_wait(); }
  else {
    Guard<sycl::half> dense(*runtime,size_t(s.n)*s.k);
    // Independent original work-item-per-block scalar decoder, then same GEMM
    // transpose/alpha/beta/leading dimensions as Gemm::f16 in current gemm.cpp.
    auto *dst=dense.p; auto *src=w.p;
    q.parallel_for(sycl::range<1>(size_t(s.n)*s.k/f.width),[=](sycl::id<1> id){
      size_t block=id[0],row=block/(s.k/f.width),bi=block%(s.k/f.width);
      float values[256]; decode_block(s.type,src+source_row(row,s)*rb+bi*f.bytes,values);
      for(int j=0;j<f.width;++j) dst[block*f.width+j]=sycl::bit_cast<sycl::half>(f16_from_f32(values[j]));
    }); runtime->checked_wait();
    oneapi::mkl::blas::row_major::gemm(q,oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans,s.m,s.n,s.k,1.f,x.p,s.k,dense.p,s.k,0.f,y.p,s.n); runtime->checked_wait();
    scratch_guard=dense.valid();
  }
  auto t1=std::chrono::steady_clock::now(); runtime->checked_wait();
  bool guards=w.valid() && x.valid() && y.valid() && scratch_guard;
  bool unchanged=std::memcmp(w.p,wi.data(),w.n)==0 && std::memcmp(x.p,xi.data(),x.n*2)==0;
  bool finite=true; for(size_t i=0;i<y.n;++i) finite &= std::isfinite(y.p[i]);
  dump(prefix+".y.bin",y.p,y.n*4);
  auto device_name=q.get_device().get_info<sycl::info::device::name>();
  runtime->retire(); // Certify completion before publishing success metadata.
  std::ofstream meta(prefix+".json");
  meta<<std::setprecision(std::numeric_limits<double>::max_digits10)<<"{\"arm\":\""<<arm<<"\",\"type\":"<<s.type<<",\"gu\":"<<s.gu<<",\"m\":"<<s.m<<",\"n\":"<<s.n<<",\"k\":"<<s.k
      <<",\"queue_retired\":true,\"guards\":"<<guards<<",\"unchanged\":"<<unchanged<<",\"finite\":"<<finite<<",\"max_w\":"<<maxw<<",\"max_x\":"<<maxx
      <<",\"cold_completed_wall_ms\":"<<std::chrono::duration<double,std::milli>(t1-t0).count()<<"}\n";
  meta.close();
  if(!meta) throw std::runtime_error("metadata write failed");
  std::cout<<device_name<<" guards="<<guards<<" unchanged="<<unchanged<<" finite="<<finite<<"\n";
  return guards && unchanged && finite ? 0:1;
 } catch(const std::exception &e){
   if(runtime) fail_stop(e.what());
   std::fprintf(stderr,"ORIGINAL_ERROR: %s\n",e.what());return 2;
 } catch(...) { fail_stop("non-std exception"); } }
