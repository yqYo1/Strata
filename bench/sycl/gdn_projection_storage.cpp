// Main-authored; repository LICENSE (MIT). Resident component diagnostic only.
// Extends the independently authored oracle at 44ca73c5; no model dispatch edits.
#include "strata/sycl/runtime.hpp"
#include <oneapi/mkl/blas.hpp>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace s = strata::sycl_backend;
namespace {
constexpr int K=2560, batch=128, samples=5;
void need(bool ok,const char *why) { if(!ok)throw std::runtime_error(why); }
struct CheckedDrain {
  s::Runtime &rt; bool armed=true;
  void finish() { rt.wait(); armed=false; }
  ~CheckedDrain() noexcept {
    if(armed)try { finish(); } catch(...) {
      std::cerr<<"checked retirement failed; owners remain unreleased\n"<<std::flush;
      std::_Exit(86);
    }
  }
};
struct Buffer {
  s::Allocation allocation;
  std::vector<uint32_t> initial, observed;
  size_t begin,bytes;
  Buffer(const std::shared_ptr<s::Runtime> &rt,size_t length)
      :allocation(rt,((length+255)/64)*64,s::MemoryKind::Device),
       initial(allocation.size()/4,0x7fc055a5u),observed(initial.size()),
       begin(64+(64-uintptr_t(allocation.data())%64)%64),bytes(length) {
    need(begin+bytes+64<=allocation.size(),"guarded extent");
  }
  void *data() { return static_cast<unsigned char *>(allocation.data())+begin; }
  void input(size_t i,uint16_t bits) {
    need(2*i+2<=bytes,"half input extent");
    std::memcpy(reinterpret_cast<unsigned char *>(initial.data())+begin+2*i,&bits,2);
  }
  double input(size_t i) const {
    uint16_t bits;
    std::memcpy(&bits,reinterpret_cast<const unsigned char *>(initial.data())+begin+2*i,2);
    return float(sycl::bit_cast<sycl::half>(bits));
  }
  double value(size_t i,int width) const {
    const auto *p=reinterpret_cast<const unsigned char *>(observed.data())+begin+i*width;
    if(width==4) { float v;std::memcpy(&v,p,4);return v; }
    uint16_t v;std::memcpy(&v,p,2);return float(sycl::bit_cast<sycl::half>(v));
  }
  void upload(sycl::queue &q) { q.memcpy(allocation.data(),initial.data(),allocation.size()); }
  void download(sycl::queue &q) { q.memcpy(observed.data(),allocation.data(),allocation.size()); }
  void immutable() const { need(initial==observed,"input/guard changed"); }
  void guards(int t,int n,int ld,int width) const {
    const auto *a=reinterpret_cast<const unsigned char *>(initial.data());
    const auto *b=reinterpret_cast<const unsigned char *>(observed.data());
    for(size_t i=0;i<allocation.size();++i) {
      const size_t pos=i>=begin?(i-begin)/size_t(width):size_t(-1);
      if(i<begin||i>=begin+bytes||pos/size_t(ld)>=size_t(t)||pos%size_t(ld)>=size_t(n))
        need(a[i]==b[i],"output guard/padding changed");
    }
  }
};
uint32_t mix(uint32_t v) {
  v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16);
}
uint16_t general(int r,int k,uint32_t salt) {
  const auto h=mix(uint32_t(r)*0x9e3779b9u^uint32_t(k)*0x85ebca6bu^salt);
  // Normal half operands with magnitude <1: every exact absolute dot sum <K.
  // This is a bounded storage diagnostic, not proof of production tensor range.
  return uint16_t(((h>>16)&0x8000u)|((7+(h>>24)%8)<<10)|(h&1023u));
}
struct Stats { double sum=0.,maximum=0.; };
double up(double v) { return std::nextafter(v,std::numeric_limits<double>::infinity()); }
constexpr double gamma(double u) { return (K-1)*u/(1.-(K-1)*u); }
constexpr double g32=gamma(0x1p-24),g64=gamma(0x1p-53),u16=0x1p-11;
// Observable FP32-accumulation-plus-final-half-rounding diagnostic. Passing it
// does not prove the vendor's accumulation type or the model's quality contract.
double half_round_bound(double v) { return up(u16*std::abs(v)+0x1p-25); }

struct Case {
  std::shared_ptr<s::Runtime> rt;
  int t,n,ld;std::string pattern;
  Buffer a,w,direct,ref,converted;
  std::vector<Stats> astats,wstats;
  Case(const std::shared_ptr<s::Runtime> &runtime,int rows,int columns,const std::string &kind)
      :rt(runtime),t(rows),n(columns),ld(n+16),pattern(kind),a(rt,size_t(t)*K*2),
       w(rt,size_t(n)*K*2),direct(rt,size_t(t)*ld*2),ref(rt,size_t(t)*ld*4),
       converted(rt,size_t(t)*ld*2),astats(t),wstats(n) {
    for(int r=0;r<t;++r)for(int k=0;k<K;++k) {
      auto bits=general(r,kind=="cancel"?k/2:k,0x31415926u);
      if(kind=="zero")bits=0;
      if(kind=="onehot")bits=k==int(mix(uint32_t(r))%K)?uint16_t(0x3d00):0;
      a.input(size_t(r)*K+k,bits);
      const double v=std::abs(a.input(size_t(r)*K+k));
      astats[r].sum+=v;astats[r].maximum=std::max(astats[r].maximum,v);
    }
    for(int r=0;r<n;++r)for(int k=0;k<K;++k) {
      auto bits=general(r,kind=="cancel"?k/2:k,0x27182818u);
      if(kind=="cancel"&&k%2)bits^=0x8000u;
      w.input(size_t(r)*K+k,bits);
      const double v=std::abs(w.input(size_t(r)*K+k));
      wstats[r].sum+=v;wstats[r].maximum=std::max(wstats[r].maximum,v);
    }
    for(auto *stats:{&astats,&wstats})for(auto &v:*stats)v.sum=up(v.sum/(1.-g64));
  }
  sycl::event launch(int arm) {
    auto &q=rt->compute();
    const auto *x=static_cast<const sycl::half *>(a.data());
    const auto *weight=static_cast<const sycl::half *>(w.data());
    using oneapi::mkl::transpose;
    if(arm==1)return oneapi::mkl::blas::row_major::gemm(q,transpose::nontrans,
        transpose::trans,t,n,K,sycl::half(1),x,K,weight,K,sycl::half(0),
        static_cast<sycl::half *>(direct.data()),ld);
    auto event=oneapi::mkl::blas::row_major::gemm(q,transpose::nontrans,transpose::trans,
        t,n,K,1.f,x,K,weight,K,0.f,static_cast<float *>(ref.data()),ld);
    if(arm==0)return event;
    auto *src=static_cast<const float *>(ref.data());
    auto *dst=static_cast<sycl::half *>(converted.data());
    const size_t columns=size_t(n),pitch=size_t(ld);
    return q.parallel_for(sycl::range<1>(size_t(t)*n),[=](sycl::id<1> id) {
      const size_t i=id[0],pos=(i/columns)*pitch+i%columns;
      dst[pos]=sycl::half(src[pos]);
    }); // Qualified in-order compute queue orders conversion after GEMM.
  }
  void upload() { for(Buffer *b:{&a,&w,&direct,&ref,&converted})b->upload(rt->compute());rt->wait(); }
  void validate() {
    for(Buffer *b:{&a,&w,&direct,&ref,&converted})b->download(rt->compute());rt->wait();
    a.immutable();w.immutable();direct.guards(t,n,ld,2);ref.guards(t,n,ld,4);converted.guards(t,n,ld,2);
    double maxdiff=0.,maxfraction=0.,maxspot=0.;
    for(int r=0;r<t;++r)for(int c=0;c<n;++c) {
      const size_t pos=size_t(r)*ld+c;
      const double actual=direct.value(pos,2),other=ref.value(pos,4),cast=converted.value(pos,2);
      need(std::isfinite(actual)&&std::isfinite(other)&&std::isfinite(cast),"nonfinite projection output");
      const double sb=std::min(up(astats[r].maximum*wstats[c].sum),up(astats[r].sum*wstats[c].maximum));
      need(sb<65504.,"diagnostic range certificate failed");
      const double bound=up(2*g32*sb+half_round_bound(other)),error=std::abs(actual-other);
      need(error<=bound,"direct16 comparator diagnostic exceeded");
      need(std::abs(cast-other)<=half_round_bound(other),"explicit conversion rounding exceeded");
      maxdiff=std::max(maxdiff,error);maxfraction=std::max(maxfraction,error/bound);
      if(pattern=="zero")need(actual==0.&&other==0.&&cast==0.,"zero mismatch");
      if(pattern=="onehot") {
        const int k=int(mix(uint32_t(r))%K);
        const double exact=a.input(size_t(r)*K+k)*w.input(size_t(c)*K+k);
        need(other==exact&&actual==float(sycl::half(float(exact)))&&actual==cast,"onehot rounding mismatch");
      }
    }
    for(int i=0;i<64;++i) {
      const int r=i<4?std::min(i==0?0:i==1?127:i==2?128:t-1,t-1):int(mix(uint32_t(i)^0x2468u)%uint32_t(t));
      const int c=i<4?(i==0?0:i==1?127:i==2?128:n-1):int(mix(uint32_t(i)^0x1357u)%uint32_t(n));
      double dot=0.,sum=0.;
      for(int k=0;k<K;++k) { const double v=a.input(size_t(r)*K+k)*w.input(size_t(c)*K+k);dot+=v;sum+=std::abs(v); }
      const double acc=up((2*g32+g64)*up(sum/(1.-g64)));
      for(int arm=0;arm<3;++arm) {
        const double v=arm==0?ref.value(size_t(r)*ld+c,4):arm==1?direct.value(size_t(r)*ld+c,2):converted.value(size_t(r)*ld+c,2);
        const double bound=up(acc+(arm?half_round_bound(dot)+u16*acc:0.));
        need(std::abs(v-dot)<=bound,"FP64 spot diagnostic exceeded");
        if(bound>0.)maxspot=std::max(maxspot,std::abs(v-dot)/bound);
      }
    }
    std::cout<<"{\"kind\":\"validation\",\"T\":"<<t<<",\"N\":"<<n<<",\"ldy\":"<<ld
      <<",\"pattern\":\""<<pattern<<"\",\"outputs\":"<<size_t(t)*n<<",\"FP64_spots\":64"
      <<",\"max_abs_difference\":"<<maxdiff<<",\"max_comparator_bound_fraction\":"<<maxfraction
      <<",\"max_FP64_bound_fraction\":"<<maxspot<<",\"guards_inputs\":true,\"passed\":true}\n"<<std::flush;
  }
};
const char *arm_name(int arm) { return arm==0?"oneMKL32":arm==1?"oneMKL16":"oneMKL32_convert16"; }
void timed(Case &c,int arm,int count,const char *phase,int sample,bool empty=false) {
  c.rt->wait();const auto start=std::chrono::steady_clock::now();
  for(int i=0;i<count;++i) {
    if(!empty)c.launch(arm);
    if((i+1)%8==0)c.rt->wait();
  }
  c.rt->wait();
  const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::cout<<"{\"kind\":\"timing\",\"T\":"<<c.t<<",\"N\":"<<c.n<<",\"ldy\":"<<c.ld
    <<",\"arm\":\""<<(empty?"empty":arm_name(arm))<<"\",\"phase\":\""<<phase<<"\",\"sample\":"<<sample
    <<",\"calls\":"<<count<<",\"host_ms\":"<<ms<<"}\n"<<std::flush;
}
void run_case(const std::shared_ptr<s::Runtime> &rt,int t,int n,const std::string &kind,int order=-1) {
  std::cerr<<"case-start T="<<t<<" N="<<n<<" pattern="<<kind<<std::endl;
  Case c(rt,t,n,kind);CheckedDrain drain{*rt};c.upload();
  if(order<0) { for(int a=0;a<3;++a) { c.launch(a);rt->wait(); }c.validate(); }
  else {
    for(int i=0;i<3;++i)timed(c,(i+order)%3,1,"first_shape_call",0);
    c.validate();
    for(int i=0;i<6;++i) { c.launch(i%3);rt->wait(); }
    for(int i=0;i<5;++i)timed(c,0,batch,"timer_control",i,true);
    for(int j=0;j<samples;++j)for(int i=0;i<3;++i)timed(c,(i+j+order)%3,batch,"steady",j);
    c.validate();
  }
  drain.finish();std::cerr<<"case-complete"<<std::endl;
}
} // namespace
int main(int argc,char **argv) {
  try {
    std::cout<<std::setprecision(17);
    need(argc>=2,"use --smoke|--validate|--bench 0|1|2");const std::string mode=argv[1];
    need((argc==2&&(mode=="--smoke"||mode=="--validate"))||(argc==3&&mode=="--bench"&&
         (std::string(argv[2])=="0"||std::string(argv[2])=="1"||std::string(argv[2])=="2")),"invalid arguments");
    auto rt=std::make_shared<s::Runtime>(0);
    need(rt->device().is_gpu()&&rt->device().has(sycl::aspect::fp16),"GPU FP16 required");
    need(rt->compute().has_property<sycl::property::queue::in_order>(),"in-order compute required");
    if(mode=="--smoke") { run_case(rt,1,6144,"general");run_case(rt,129,10240,"general"); }
    else if(mode=="--validate") {
      for(int t:{1,127,128,129,255,256,257,8191,8192})for(int n:{6144,10240})run_case(rt,t,n,"general");
      for(int t:{1,129})for(int n:{6144,10240})run_case(rt,t,n,"cancel");
      run_case(rt,1,6144,"zero");run_case(rt,257,10240,"zero");
      run_case(rt,129,6144,"onehot");run_case(rt,8191,10240,"onehot");
    } else for(int t:{4096,8192})for(int n:{6144,10240})run_case(rt,t,n,"general",argv[2][0]-'0');
    rt->wait();std::cout<<"fixture_status=PASS;resident storage diagnostic only\n"<<std::flush;return 0;
  } catch(const std::exception &e) { std::cerr<<e.what()<<std::endl;return 1; }
}
