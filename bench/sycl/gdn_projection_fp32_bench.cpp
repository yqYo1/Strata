// Main-authored; repository LICENSE (MIT). Resident component comparison only.
// Independent general-input oracle; does not include or alter the exact fixture.
#include "native16.hpp"
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
namespace x = strata::experimental::vllm_xe2_native16;
namespace {
constexpr int K = 2560, batch = 64, samples = 5;
void need(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct CheckedDrain {
  s::Runtime &rt; sycl::queue &q; bool armed = true;
  void finish() { q.wait_and_throw(); rt.check(); armed = false; }
  ~CheckedDrain() noexcept {
    if (armed) try { finish(); } catch (...) {
      std::cerr << "checked retirement failed; owners remain unreleased\n" << std::flush;
      std::_Exit(86);
    }
  }
};
struct Buffer {
  s::Allocation allocation;
  std::vector<uint32_t> initial, observed;
  size_t begin, bytes;
  Buffer(const std::shared_ptr<s::Runtime> &rt, size_t length)
      : allocation(rt, ((length + 255) / 64) * 64, s::MemoryKind::Device),
        initial(allocation.size() / 4, 0x7fc055a5u), observed(initial.size()),
        begin(64 + (64 - uintptr_t(allocation.data()) % 64) % 64), bytes(length) {
    need(begin + bytes + 64 <= allocation.size(), "guarded buffer extent");
  }
  void *data() { return static_cast<unsigned char *>(allocation.data()) + begin; }
  void half(size_t i, uint16_t bits) {
    need(i * 2 + 2 <= bytes, "half input extent");
    std::memcpy(reinterpret_cast<unsigned char *>(initial.data()) + begin + 2*i, &bits, 2);
  }
  double half(size_t i) const {
    uint16_t bits;
    std::memcpy(&bits, reinterpret_cast<const unsigned char *>(initial.data()) + begin + 2*i, 2);
    return float(sycl::bit_cast<sycl::half>(bits));
  }
  float value(size_t i) const { return std::bit_cast<float>(observed[begin/4 + i]); }
  void upload(sycl::queue &q) { q.memcpy(allocation.data(), initial.data(), allocation.size()); }
  void download(sycl::queue &q) { q.memcpy(observed.data(), allocation.data(), allocation.size()); }
  void immutable() const { need(initial == observed, "input/guard changed"); }
  void guards(int t, int n, int ld) const {
    for (size_t i = 0; i < observed.size(); ++i) {
      const size_t pos = i >= begin/4 ? i - begin/4 : size_t(-1);
      if (pos / size_t(ld) >= size_t(t) || pos % size_t(ld) >= size_t(n))
        need(initial[i] == observed[i], "output prefix/suffix/padding changed");
    }
  }
};
uint32_t mix(uint32_t v) {
  v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; return v ^ (v >> 16);
}
uint16_t general(int row, int k, uint32_t salt) {
  const auto h = mix(uint32_t(row)*0x9e3779b9u ^ uint32_t(k)*0x85ebca6bu ^ salt);
  // Finite normal binary16: signs, fourteen exponent bins (-8..5), all mantissas.
  // Subnormal inputs, NaNs and infinities are explicitly outside this fixture.
  return uint16_t(((h >> 16) & 0x8000u) | ((7 + (h >> 24)%14) << 10) | (h & 1023u));
}
struct Stats { double sum = 0., maximum = 0.; };
double upward(double v) { return std::nextafter(v, std::numeric_limits<double>::infinity()); }
constexpr double gamma(double u) { return (K-1)*u / (1.-(K-1)*u); }
constexpr double g32 = gamma(0x1p-24), g64 = gamma(0x1p-53);

struct Case {
  std::shared_ptr<s::Runtime> rt;
  x::Gemm &native;
  int t, n, ld;
  std::string pattern;
  Buffer a, w, y, ref;
  std::vector<Stats> astats, wstats;
  Case(const std::shared_ptr<s::Runtime> &runtime, x::Gemm &g, int rows, int columns,
       const std::string &kind)
      : rt(runtime), native(g), t(rows), n(columns), ld(n + (t%2 ? 4 : 16)), pattern(kind),
        a(rt, size_t(t)*K*2), w(rt, size_t(n)*K*2),
        y(rt, size_t(t)*ld*4), ref(rt, size_t(t)*ld*4), astats(t), wstats(n) {
    for (int r = 0; r < t; ++r) for (int k = 0; k < K; ++k) {
      uint16_t bits = general(r, kind == "cancel" ? k/2 : k, 0x31415926u);
      if (kind == "zero") bits = 0;
      if (kind == "onehot") bits = k == int(mix(uint32_t(r))%K) ? uint16_t(0x3d00) : 0;
      a.half(size_t(r)*K+k, bits);
      const double v = std::abs(a.half(size_t(r)*K+k));
      astats[r].sum += v; astats[r].maximum = std::max(astats[r].maximum, v);
    }
    for (int r = 0; r < n; ++r) for (int k = 0; k < K; ++k) {
      uint16_t bits = general(r, kind == "cancel" ? k/2 : k, 0x27182818u);
      if (kind == "cancel" && k%2) bits ^= 0x8000u;
      w.half(size_t(r)*K+k, bits);
      const double v = std::abs(w.half(size_t(r)*K+k));
      wstats[r].sum += v; wstats[r].maximum = std::max(wstats[r].maximum, v);
    }
    for (auto *stats : {&astats, &wstats}) for (auto &v : *stats) v.sum = upward(v.sum/(1.-g64));
  }
  x::GdnProjectionFp32Problem problem() {
    x::GdnProjectionFp32Problem p;
    p.x = static_cast<const uint16_t *>(a.data()); p.w = static_cast<const uint16_t *>(w.data());
    p.y = static_cast<float *>(y.data()); p.t = t; p.n = n; p.k = K; p.ldy = ld; return p;
  }
  sycl::event launch(bool use_native) {
    if (use_native) return native.gdn_projection_fp32(problem());
    return oneapi::mkl::blas::row_major::gemm(rt->compute(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, t, n, K, 1.f,
        static_cast<const sycl::half *>(a.data()), K,
        static_cast<const sycl::half *>(w.data()), K, 0.f,
        static_cast<float *>(ref.data()), ld);
  }
  void upload() { auto &q=rt->compute(); a.upload(q); w.upload(q); y.upload(q); ref.upload(q); q.wait_and_throw(); rt->check(); }
  void validate() {
    auto &q=rt->compute(); a.download(q); w.download(q); y.download(q); ref.download(q);
    q.wait_and_throw(); rt->check(); a.immutable(); w.immutable(); y.guards(t,n,ld); ref.guards(t,n,ld);
    double max_difference=0., max_bound_fraction=0., max_spot_fraction=0.;
    for (int r=0;r<t;++r) for(int c=0;c<n;++c) {
      const size_t pos=size_t(r)*ld+c;
      const double actual=y.value(pos), other=ref.value(pos);
      need(std::isfinite(actual) && std::isfinite(other), "nonfinite GEMM output");
      const double sb=std::min(upward(astats[r].maximum*wstats[c].sum), upward(astats[r].sum*wstats[c].maximum));
      const double bound=upward(2*g32*sb), difference=std::abs(actual-other);
      if (difference>bound) throw std::runtime_error("oneMKL bound exceeded row="+std::to_string(r)+" col="+std::to_string(c));
      max_difference=std::max(max_difference,difference);
      if(bound>0.)max_bound_fraction=std::max(max_bound_fraction,difference/bound);
      if(pattern=="zero") need(actual==0. && other==0., "zero probe mismatch");
      if(pattern=="onehot") {
        const int k=int(mix(uint32_t(r))%K);
        const double exact=a.half(size_t(r)*K+k)*w.half(size_t(c)*K+k);
        need(actual==exact && other==exact, "onehot exact-product probe mismatch");
      }
    }
    for(int i=0;i<64;++i) {
      const int r=i<4 ? std::min(i==0 ? 0 : i==1 ? 127 : i==2 ? 128 : t-1,t-1) : int(mix(uint32_t(i)^0x2468u)%uint32_t(t));
      const int c=i<4 ? (i==0 ? 0 : i==1 ? 127 : i==2 ? 128 : n-1) : int(mix(uint32_t(i)^0x1357u)%uint32_t(n));
      double exact=0., sum=0.;
      for(int k=0;k<K;++k) { const double product=a.half(size_t(r)*K+k)*w.half(size_t(c)*K+k); exact+=product; sum+=std::abs(product); }
      const double bound=upward((2*g32+g64)*upward(sum/(1.-g64)));
      for(double actual : {double(y.value(size_t(r)*ld+c)),double(ref.value(size_t(r)*ld+c))}) {
        const double error=std::abs(actual-exact); need(error<=bound, "FP64 spot bound exceeded");
        if(bound>0.)max_spot_fraction=std::max(max_spot_fraction,error/bound);
      }
    }
    std::cout<<"{\"kind\":\"validation\",\"pattern\":\""<<pattern<<"\",\"T\":"<<t<<",\"N\":"<<n<<",\"ldy\":"<<ld
      <<",\"outputs\":"<<size_t(t)*n<<",\"FP64_spots\":64,\"max_abs_difference\":"<<max_difference
      <<",\"max_comparator_bound_fraction\":"<<max_bound_fraction<<",\"max_FP64_bound_fraction\":"<<max_spot_fraction
      <<",\"guards_inputs\":true,\"passed\":true}\n"<<std::flush;
  }
};

void timed(Case &c, bool native, int count, const char *phase, int sample, bool empty=false) {
  auto &q=c.rt->compute(); q.wait_and_throw(); c.rt->check();
  std::vector<sycl::event> events; events.reserve(count);
  const auto begin=q.ext_oneapi_submit_barrier(); begin.wait_and_throw(); c.rt->check();
  const auto start=std::chrono::steady_clock::now();
  for(int i=0;i<count;++i) {
    if(!empty) events.push_back(c.launch(native));
    if((i+1)%8==0) { q.wait_and_throw(); c.rt->check(); }
  }
  const auto end=q.ext_oneapi_submit_barrier(); end.wait_and_throw(); q.wait_and_throw(); c.rt->check();
  const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  bool queue_profile=true; double interval=0.;
  try {
    const auto a=begin.get_profiling_info<sycl::info::event_profiling::command_end>();
    const auto b=end.get_profiling_info<sycl::info::event_profiling::command_end>();
    need(b>=a, "nonmonotonic queue timestamps"); interval=double(b-a)*1e-6;
  } catch(const sycl::exception &e) { if(e.code()!=sycl::errc::invalid)throw; queue_profile=false; }
  double service=0.;
  if(native && !empty) for(const auto &e:events) {
    const auto a=e.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto b=e.get_profiling_info<sycl::info::event_profiling::command_end>();
    need(b>a,"native event profiling unavailable"); service+=double(b-a)*1e-6;
  }
  std::cout<<"{\"kind\":\"timing\",\"T\":"<<c.t<<",\"N\":"<<c.n<<",\"ldy\":"<<c.ld
    <<",\"arm\":\""<<(empty?"empty":native?"native":"oneMKL")<<"\",\"phase\":\""<<phase
    <<"\",\"sample\":"<<sample<<",\"calls\":"<<count<<",\"host_ms\":"<<wall<<",\"queue_interval_ms\":";
  if(queue_profile)std::cout<<interval;else std::cout<<"null";
  std::cout<<",\"native_event_sum_ms\":";
  if(native&&!empty)std::cout<<service;else std::cout<<"null";
  std::cout<<"}\n"<<std::flush;
}

void run_case(const std::shared_ptr<s::Runtime> &rt,x::Gemm &g,int t,int n,const std::string &pattern,int order=-1) {
  std::cerr<<"case-start T="<<t<<" N="<<n<<" pattern="<<pattern<<std::endl;
  Case c(rt,g,t,n,pattern); CheckedDrain drain{*rt,rt->compute()}; c.upload();
  if(order<0) {
    c.launch(true).wait_and_throw();rt->check();c.launch(false).wait_and_throw();rt->check();c.validate();
  } else {
    for(int i=0;i<2;++i)timed(c,((i+order)%2)==0,1,"first_shape_call",0);
    c.validate();
    for(int i=0;i<4;++i){c.launch(i%2==0).wait_and_throw();rt->check();}
    for(int i=0;i<5;++i)timed(c,false,batch,"timer_control",i,true);
    for(int j=0;j<samples;++j)for(int i=0;i<2;++i)timed(c,((i+j+order)%2)==0,batch,"steady",j);
    c.validate();
  }
  drain.finish();std::cerr<<"case-complete"<<std::endl;
}

void guards(const std::shared_ptr<s::Runtime> &rt,x::Gemm &g) {
  Case c(rt,g,1,6144,"general");CheckedDrain drain{*rt,rt->compute()};c.upload();
  std::vector<x::GdnProjectionFp32Problem> rejected;
  const auto good=c.problem(); auto add=[&](auto change){auto p=good;change(p);rejected.push_back(p);};
  add([](auto &p){p.t=0;});add([](auto &p){p.t=8193;});add([](auto &p){p.n=6143;});add([](auto &p){p.k=2559;});
  add([](auto &p){p.ldy=-1;});add([](auto &p){p.ldy=p.n-4;});add([](auto &p){p.ldy=p.n+2;});
  add([](auto &p){p.ldy=4194304;});add([](auto &p){p.t=8192;p.ldy=4194300;});
  add([](auto &p){p.x=nullptr;});add([](auto &p){p.w=nullptr;});add([](auto &p){p.y=nullptr;});
  add([](auto &p){p.x+=1;});add([](auto &p){p.w+=1;});add([](auto &p){p.y+=1;});
  add([](auto &p){p.y=reinterpret_cast<float *>(const_cast<uint16_t *>(p.x));});
  for(const auto &p:rejected){bool caught=false;try{g.gdn_projection_fp32(p);}catch(const std::invalid_argument &){caught=true;}need(caught,"invalid argument not rejected");}
  for(Buffer *p:{&c.a,&c.w,&c.y,&c.ref})p->download(rt->compute());
  drain.finish();for(Buffer *p:{&c.a,&c.w,&c.y,&c.ref})p->immutable();
  std::cout<<"{\"kind\":\"rejections\",\"count\":"<<rejected.size()<<",\"canaries_unchanged\":true,\"passed\":true}\n";
}
}
int main(int argc,char **argv) {
  try {
    std::cout<<std::setprecision(17);
    need(argc>=2,"use --guards|--smoke|--validate|--bench 0|1");const std::string mode=argv[1];
    need((argc==2&&(mode=="--guards"||mode=="--smoke"||mode=="--validate"))||
         (argc==3&&mode=="--bench"&&(std::string(argv[2])=="0"||std::string(argv[2])=="1")),"invalid fixture arguments");
    auto rt=std::make_shared<s::Runtime>(0);x::Gemm g(*rt,rt->compute());
    if(mode=="--guards")guards(rt,g);
    else if(mode=="--smoke"){run_case(rt,g,1,6144,"general");run_case(rt,g,129,10240,"general");}
    else if(mode=="--validate") {
      for(int t:{1,127,128,129,255,256,257,8191,8192})for(int n:{6144,10240})run_case(rt,g,t,n,"general");
      for(int t:{1,129})for(int n:{6144,10240})run_case(rt,g,t,n,"cancel");
      run_case(rt,g,1,6144,"zero");run_case(rt,g,257,10240,"zero");
      run_case(rt,g,129,6144,"onehot");run_case(rt,g,8191,10240,"onehot");
    } else for(int t:{4096,8192})for(int n:{6144,10240})run_case(rt,g,t,n,"general",argv[2][0]-'0');
    rt->wait();std::cout<<"fixture_status=PASS;resident component only\n"<<std::flush;return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
