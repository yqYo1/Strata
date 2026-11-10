// Main-authored compact-GU component, reusing this repository's scalar/guard fixture.
// Repository LICENSE (MIT) applies. Current public API;no GPU timing events.
#include "../../src/prefill/sycl/vllm_xe2/native16.hpp"
#include "strata/prefill/native16.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/sycl/command_profile.hpp"
#include "strata/sycl/cpu_wait_profile.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
namespace xe = strata::experimental::vllm_xe2_native16;
constexpr int N = 1280, K = 2560, H = 640;
constexpr size_t Pad = 64;
constexpr uint16_t Sentinel = 0xa55a;
std::shared_ptr<sb::Runtime> runtime;
void need(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void drain(sycl::queue &q) noexcept {
  try { q.wait_and_throw(); runtime->check(); }
  catch (...) { std::_Exit(86); }
}
void wait(sycl::queue &q, sycl::event e) {
  try {
    sb::cpu_wait_profile::blocking(sb::cpu_wait_profile::event,[&] { e.wait_and_throw(); });
    runtime->check();
  } catch (...) { drain(q); throw; }
}
template <class Submit> void submit_wait(sycl::queue &q,Submit submit) {
  // Keep local host vectors alive through synchronous enqueue failures too.
  try { wait(q,submit()); } catch (...) { drain(q); throw; }
}
struct Buffer {
  sb::Allocation owner;
  size_t n;
  sycl::queue &q;
  Buffer(size_t count, sycl::queue &queue)
      : owner(runtime, (count + 2 * Pad) * 2, sb::MemoryKind::Device), n(count), q(queue) {
    std::vector<uint16_t> all(n + 2 * Pad, Sentinel);
    submit_wait(q,[&] { return q.memcpy(owner.data(), all.data(), all.size() * 2); });
  }
  ~Buffer() { drain(q); }
  uint16_t *data() const { return owner.as<uint16_t>() + Pad; }
  void set(const std::vector<uint16_t> &v) {
    need(v.size() == n, "upload extent");
    submit_wait(q,[&] { return q.memcpy(data(), v.data(), n * 2); });
  }
  std::vector<uint16_t> get() const {
    std::vector<uint16_t> all(n + 2 * Pad);
    submit_wait(q,[&] { return q.memcpy(all.data(), owner.data(), all.size() * 2); });
    for (size_t i = 0; i < Pad; ++i)
      need(all[i] == Sentinel && all[n + Pad + i] == Sentinel, "buffer guard changed");
    return {all.begin() + Pad, all.begin() + Pad + n};
  }
  void reset() { set(std::vector<uint16_t>(n, Sentinel)); }
};
uint16_t half(float v) { return sycl::bit_cast<uint16_t>(sycl::half(v)); }
float widen(uint16_t v) { return float(sycl::bit_cast<sycl::half>(v)); }
bool nan16(uint16_t v) { return (v & 0x7c00) == 0x7c00 && (v & 0x3ff); }
int ordered(uint16_t v) { return 0x8000 + (v & 0x8000 ? -int(v & 0x7fff) : int(v)); }
int compare_hidden(const Buffer &a, const Buffer &b) {
  const auto x = a.get(), y = b.get(); int max_ulp = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    if (nan16(x[i])) need(nan16(y[i]), "hidden NaN classification mismatch");
    else {
      need(!nan16(y[i]) && std::isfinite(widen(x[i])) && std::isfinite(widen(y[i])),
           "hidden finite saturation mismatch");
      max_ulp = std::max(max_ulp, std::abs(ordered(x[i]) - ordered(y[i])));
    }
  }
  need(max_ulp <= 1, "hidden exceeds prospective one-FP16-ULP bound");
  return max_ulp;
}
struct Case {
  int rows;
  sycl::queue &q;
  xe::Gemm gemm;
  strata::prefill::Gemm production;
  Buffer x, w, gu, capture, reference, candidate;
  std::vector<uint16_t> hx, hw;
  bool edges;
  Case(int r, bool edge_inputs, sycl::queue &queue)
      : rows(r), q(queue), gemm(*runtime, q), x(size_t(r)*K,q), w(size_t(N)*K,q),
        gu(size_t(r)*N,q), capture(gu.n,q), reference(size_t(r)*H,q), candidate(reference.n,q),
        hx(x.n), hw(w.n), edges(edge_inputs) {
    for (size_t i=0;i<hx.size();++i)
      hx[i]=half(edges ? (i%K == 0 ? 1.f : 0.f) : float(int((i*13+i/K)%17)-8)/32.f);
    constexpr std::array<uint16_t,24> values{
      0x3c04,0x3c0c,0xbc04,0xbc0c,0x7bff,0x7bff,0x0001,0x0001,
      0x0000,0x8000,0x7c00,0x3c00,0xfc00,0x3c00,0x7e01,0x3c00,
      0x3c00,0x7e01,0x4c00,0x7bff,0xcc00,0x7bff,0x7bff,0xfbff};
    for (size_t i=0;i<hw.size();++i)
      hw[i]=edges ? (i%K == 0 ? values[(i/K)%values.size()] : 0) :
          half(float(int((i*7+i/K)%23)-11)/64.f);
    x.set(hx); w.set(hw);
    std::string error;
    const bool initialized=production.init_external(&q,nullptr,0,nullptr,0,error);
    need(initialized,error.c_str());
  }
  ~Case() { drain(q); }
  xe::GuSwiGLUProblem problem(bool observe=false) {
    return {x.data(),w.data(),candidate.data(),rows,observe ? capture.data() : nullptr};
  }
  sycl::event fused(xe::Tile tile, bool observe=false) {
    auto event=gemm.gu_swiglu(problem(observe),tile);
    sb::finish(&q,event);
    return event;
  }
  void separate([[maybe_unused]] xe::Tile tile) {
    try {
      // Actual production entry, including its immutable auto-tile selection,
      // binding, preflight, profiling hooks and finish; all declared rows <2334.
      need(!production.try_gu_swiglu16(x.data(),w.data(),reference.data(),rows),
           "production baseline unexpectedly selected large GU fusion");
      production.f16_out16(x.data(),w.data(),gu.data(),rows,N,K,true);
      strata::prefill::native16::swiglu(gu.data(),reference.data(),rows,&q);
    } catch (...) { drain(q); throw; }
  }
  int check(xe::Tile tile) {
    candidate.reset(); capture.reset();
    separate(tile); wait(q,fused(tile,true));
    auto a=gu.get(),b=capture.get();
    need(a==b,"BF16 capture differs from dense DPAS output");
    if (!edges && rows<=3) {
      for (int r=0;r<rows;++r) for (int c=0;c<N;++c) {
        int dot=0;
        for (int k=0;k<K;++k)
          dot+=(int((size_t(r*K+k)*13+r)%17)-8)*(int((size_t(c*K+k)*7+c)%23)-11);
        const uint32_t bits=std::bit_cast<uint32_t>(float(dot)/2048.f);
        need(a[size_t(r)*N+c]==uint16_t((bits+0x7fff+((bits>>16)&1))>>16),"independent integer dot oracle");
      }
    }
    int ulp=compare_hidden(reference,candidate);
    candidate.reset(); capture.reset(); wait(q,fused(tile));
    ulp=std::max(ulp,compare_hidden(reference,candidate));
    for (auto v:capture.get()) need(v==Sentinel,"null-capture wrote diagnostic");
    need(x.get()==hx && w.get()==hw,"input changed");
    std::cout << "{\"kind\":\"correctness\",\"rows\":" << rows << ",\"tile\":" << int(tile)
      << ",\"edges\":" << edges << ",\"max_half_ulp\":" << ulp
      << ",\"BF16_capture_exact\":true,\"guards\":true,\"inputs_immutable\":true,\"production_baseline\":true,\"status\":\"PASS\"}\n";
    return ulp;
  }
};

double batch(Case& c,bool fused,int repeat) {
  drain(c.q);
  const auto start=std::chrono::steady_clock::now();
  for (int n=0;n<repeat;++n) {
    if (fused) (void)c.fused(xe::Tile::compact_16x64);
    else c.separate(xe::Tile::compact_16x64);
  }
  drain(c.q);
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
void bench(int process,sycl::queue& q) {
  need(process>=0 && process<3,"three declared process ordinals");
  int ordinal=0;
  for (int rows:{1,16,32,64,65,96,127,128,129,256,640,1280}) {
    Case c(rows,false,q);c.check(xe::Tile::compact_16x64);
    (void)batch(c,false,8);(void)batch(c,true,8);
    int repeat=64;double first=0,second=0;
    for (;;) {
      first=batch(c,false,repeat);second=batch(c,true,repeat);
      if (first>=150 && second>=150) break;
      need(repeat<262144,"calibration budget reached before150ms");repeat*=2;
    }
    std::cout<<"{\"kind\":\"calibration\",\"rows\":"<<rows<<",\"repeat\":"<<repeat
      <<",\"separate_ms\":"<<first<<",\"fused_ms\":"<<second<<"}\n";
    for (int round=0;round<4;++round) {
      const int order=(round+process+ordinal)&1;
      for (int pass=0;pass<2;++pass) {
        const int arm=order^pass;const double ms=batch(c,arm!=0,repeat);
        need(std::isfinite(ms) && ms>=100,"sample below100ms declared eligibility floor");
        // Reset/reference/guard checks occur outside the timed batch.
        if (arm) c.separate(xe::Tile::compact_16x64);
        else (void)c.fused(xe::Tile::compact_16x64);
        drain(q);const int ulp=compare_hidden(c.reference,c.candidate);
        need(c.x.get()==c.hx && c.w.get()==c.hw,"timed inputs changed");
        (void)c.gu.get();(void)c.capture.get();
        std::cout<<"{\"kind\":\"sample\",\"process\":"<<process<<",\"rows\":"<<rows
          <<",\"round\":"<<round<<",\"order\":"<<order<<",\"arm\":"<<arm
          <<",\"repeat\":"<<repeat<<",\"batch_wall_ms\":"<<ms
          <<",\"max_half_ulp\":"<<ulp<<",\"status\":\"PASS\"}\n";
      }
    }
    ++ordinal;
  }
}
} // namespace
int main(int argc,char** argv) {
  std::cout<<std::unitbuf;std::cout.precision(17);
  try {
    need(argc>=2,"--smoke|--numerics|--bench PROCESS");
    const std::string mode=argv[1];
    need((argc==2 && (mode=="--smoke" || mode=="--numerics")) ||
      (argc==3 && mode=="--bench"),"declared mode/argument count required");
    runtime=sb::runtime_for();auto q=runtime->make_queue(false);
    if (mode=="--smoke") {
      Case c(3,false,q);c.check(xe::Tile::compact_16x64);
      Case edge(3,true,q);edge.check(xe::Tile::compact_16x64);
    } else if (mode=="--numerics") {
      for (auto tile:{xe::Tile::compact_16x64,xe::Tile::wide_128x128}) {
        for (int rows:{1,3,16,17,32,33,64,65,96,127,128,129,256,640,1280}) {Case c(rows,false,q);c.check(tile);}
        Case edge(17,true,q);edge.check(tile);
      }
    } else bench(std::stoi(argv[2]),q);
    q.wait_and_throw();runtime->check();runtime->wait();
    std::cout<<"fixture_status=PASS mode="<<mode.substr(2)<<" profiling=0\n";return 0;
  } catch (const std::exception& e) {
    if (runtime) {try {runtime->wait();} catch (...) {std::_Exit(86);}}
    std::cerr<<"fixture_status=FAIL "<<e.what()<<'\n';return 1;
  }
}
