// Main-authored GU/SwiGLU qualification; repository LICENSE (MIT) applies.
#include "native16.hpp"
#include "strata/prefill/native16.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/sycl/command_profile.hpp"
#include "strata/sycl/cpu_wait_profile.hpp"
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
  try { q.wait_and_throw(); runtime->check(); } catch (...) { std::_Exit(86); }
}
void wait(sycl::queue &q, sycl::event e) {
  try {
    sb::cpu_wait_profile::blocking(sb::cpu_wait_profile::event,[&] { e.wait_and_throw(); });
    runtime->check();
  } catch (...) { drain(q); throw; }
}
template <class Submit> void submit_wait(sycl::queue &q, Submit submit) {
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
    need(v.size() == n, "upload extent"); submit_wait(q,[&] { return q.memcpy(data(), v.data(), n * 2); });
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
void ownership() {
  for (auto layout : {xe::GuLayout::matrix, xe::GuLayout::output_copy})
    for (auto tile : {xe::Tile::compact_16x64, xe::Tile::wide_128x128})
      for (int rows : {1, 3, 17, 33, 129, 2334, 5963, 8191, 8192}) {
        const auto p = xe::inspect_gu_pair_ownership(rows, tile, layout);
        need(p.gu_values == size_t(rows) * N && p.hidden_values == size_t(rows) * H &&
             !p.gu_missing && !p.gu_multiple && !p.hidden_missing && !p.hidden_multiple &&
             p.checked_pairs == p.hidden_values, "GU ownership incomplete");
        std::cout << "case gu_ownership rows=" << rows
          << " tile=" << (tile == xe::Tile::compact_16x64 ? "compact" : "wide")
          << " layout=" << (layout == xe::GuLayout::matrix ? "matrix" : "output_copy")
          << " gu_values=" << p.gu_values << " hidden_values=" << p.hidden_values
          << " masked=" << p.masked << " checked_pairs=" << p.checked_pairs
          << " partner_mismatch=" << p.partner_mismatch
          << " packed_missing=" << p.packed_missing << " packed_multiple=" << p.packed_multiple
          << " packed_orientation_mismatch=" << p.packed_orientation_mismatch
          << " lane_xor1_same_index_eligible=" << (p.partner_mismatch == 0) << " status=PASS\n";
      }
}
class Topology;
void topology(sycl::queue &q) {
  Buffer checked(384, q); auto *p = checked.data();
  wait(q, q.parallel_for<Topology>(sycl::nd_range<1>(384, 128), [=](sycl::nd_item<1> item)
      [[sycl::reqd_sub_group_size(16)]] {
    auto sg = item.get_sub_group();
    const auto id = item.get_local_linear_id();
    p[item.get_global_linear_id()] = uint16_t(sg.get_local_linear_range() == 16 &&
        sg.get_local_linear_id() == id % 16 && sg.get_group_linear_id() == id / 16);
  }));
  for (auto value : checked.get()) need(value == 1, "TLA/local/subgroup lane mapping mismatch");
  std::cout << "{\"kind\":\"topology\",\"work_items\":384,\"status\":\"PASS\"}\n";
}
struct Case {
  int rows;
  sycl::queue &q;
  xe::Gemm gemm;
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
  }
  ~Case() { drain(q); }
  xe::GuSwiGLUProblem problem(bool observe=false) {
    return {x.data(),w.data(),candidate.data(),rows,observe ? capture.data() : nullptr};
  }
  sycl::event fused(xe::Tile tile, bool observe=false) { return gemm.gu_swiglu(problem(observe),tile); }
  void separate(xe::Tile tile) {
    try {
      xe::Problem p; p.x=x.data(); p.w=w.data(); p.y=gu.data(); p.t=rows; p.n=N; p.k=K;
      (void)gemm.beta0(p,tile);
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
      << ",\"BF16_capture_exact\":true,\"guards\":true,\"inputs_immutable\":true,\"status\":\"PASS\"}\n";
    return ulp;
  }
  void check_profile(bool fusion_enabled=true) {
    namespace cp = sb::command_profile;
    strata::prefill::Gemm routed;
    std::string error;
    if (!routed.init_external(&q,nullptr,0,nullptr,0,error)) throw std::runtime_error(error);
    candidate.reset();capture.reset();separate(xe::Tile::wide_128x128);drain(q);
    bool selected=false;
    {
      std::unique_ptr<cp::Trace> trace;
      if (cp::requested()) trace.reset(new cp::Trace(&q,1));
      // Must drain before Trace or any Case-owned device/host storage unwinds.
      struct Retirement { sycl::queue& queue; ~Retirement() { drain(queue); } } retirement{q};
      auto frame=trace ? trace->begin(0,0) : cp::Frame{};
      frame.consider(0,0,rows);
      {
        auto tag=frame.tag(cp::Kind::gu,0,0,rows,N,K);
        selected=routed.try_gu_swiglu16(x.data(),w.data(),candidate.data(),rows);
        if (!selected) routed.f16_out16(x.data(),w.data(),gu.data(),rows,N,K,true);
      }
      if (!selected) {
        auto tag=frame.tag(cp::Kind::swiglu,0,0,rows,H);
        strata::prefill::native16::swiglu(gu.data(),candidate.data(),rows,&q);
      }
      drain(q);frame.after_checked_drain();
    }
    need(selected==(fusion_enabled && rows>=2334),"profile changed production GU dispatch");
    const int ulp=compare_hidden(reference,candidate);
    need(x.get()==hx && w.get()==hw,"profiled GU changed inputs");
    for (auto v:capture.get()) need(v==Sentinel,"profiled GU wrote diagnostic storage");
    std::cout<<"{\"kind\":\"profiled_engine_gu\",\"rows\":"<<rows<<",\"selected\":"<<selected
      <<",\"expected_records\":"<<(selected ? 1 : 2)<<",\"max_half_ulp\":"<<ulp
      <<",\"guards\":true,\"inputs_immutable\":true,\"status\":\"PASS\"}\n";
  }
};
void admissions(Case &c) {
  int count=0;
  auto reject=[&](xe::GuSwiGLUProblem p, xe::Tile tile=xe::Tile::wide_128x128) {
    try { wait(c.q,c.gemm.gu_swiglu(p,tile)); }
    catch (const std::invalid_argument &) { ++count; return; }
    throw std::runtime_error("invalid GU admission submitted work");
  };
  auto p=c.problem(); p.rows=0; reject(p);
  p=c.problem(); p.rows=8193; reject(p);
  reject(c.problem(),static_cast<xe::Tile>(99));
  p=c.problem(); p.x=nullptr; reject(p);
  p=c.problem(); p.x=c.hx.data(); reject(p);
  p=c.problem(); ++p.x; reject(p);
  p=c.problem(); p.y=const_cast<uint16_t*>(p.x); reject(p);
  p=c.problem(); p.y=reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(p.y)+1); reject(p);
  Buffer tiny(1,c.q); p=c.problem(); p.x=tiny.data(); reject(p);
  p=c.problem(); p.y=tiny.data(); reject(p);
  p=c.problem(true); p.rounded_diagnostic=tiny.data(); reject(p);
  p=c.problem(true); p.rounded_diagnostic=p.y; reject(p);
  p=c.problem(true); p.rounded_diagnostic=const_cast<uint16_t*>(p.w); reject(p);
  need(count==13,"admission case count");
  std::cout << "{\"kind\":\"admissions\",\"rejected\":" << count << ",\"status\":\"PASS\"}\n";
}
class Marker;
sycl::event mark(sycl::queue &q) { return q.single_task<Marker>([] {}); }
// Exercise the public production dispatcher, including its refused-path cost.
// The process environment selects an arm before SYCL initializes; never mutate it.
struct PublicGU {
  Case &c;
  strata::prefill::Gemm routed;
  bool expected;
  PublicGU(Case &input, bool fusion) : c(input), expected(fusion && c.rows>=2334) {
    std::string error;
    if (!routed.init_external(&c.q,nullptr,0,nullptr,0,error)) throw std::runtime_error(error);
  }
  ~PublicGU() { drain(c.q); }
  template<bool Observe=false> bool call(sb::command_profile::Frame *frame=nullptr,int id=0) {
    try {
      const auto projection=[&] {
        const bool selected=routed.try_gu_swiglu16(c.x.data(),c.w.data(),c.candidate.data(),c.rows);
        if (!selected) routed.f16_out16(c.x.data(),c.w.data(),c.gu.data(),c.rows,N,K,true);
        return selected;
      };
      bool selected;
      if constexpr (Observe) {
        auto tag=frame->tag(sb::command_profile::Kind::gu,0,id,c.rows,N,K);
        selected=projection();
      } else selected=projection();
      if (!selected) {
        if constexpr (Observe) {
          auto tag=frame->tag(sb::command_profile::Kind::swiglu,0,id,c.rows,H);
          strata::prefill::native16::swiglu(c.gu.data(),c.candidate.data(),c.rows,&c.q);
        } else strata::prefill::native16::swiglu(c.gu.data(),c.candidate.data(),c.rows,&c.q);
      }
      return selected;
    } catch (...) { drain(c.q); throw; }
  }
  void prime() {
    for (int i=0;i<8;++i) need(call()==expected,"warmup public dispatch differs");
    drain(c.q);
  }
};
void public_settings(int arm) {
  const auto matches=[](const char *key,const char *wanted) {
    const char *value=std::getenv(key);
    return value && std::string(value)==wanted;
  };
  need(arm==0 || arm==1,"public arm must be0 or1");
  need(matches("STRATA_SYCL_PREFILL_NATIVE16","bf16") &&
       matches("STRATA_SYCL_PREFILL_NATIVE16_BACKEND","vllm") &&
       matches("STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE","auto") &&
       matches("STRATA_SYCL_PREFILL_GU_SWIGLU",arm ? "large" : "0"),
       "public mode requires matching immutable bf16/vllm/auto/GU startup settings");
}
template<bool Observe=false> void benchmark_public(int process,int arm,sycl::queue &q) {
  namespace cp=sb::command_profile;
  namespace cpu=sb::cpu_wait_profile;
  need(cp::requested()==Observe,"public attribution requires matching startup command-profile setting");
  need(q.has_property<sycl::property::queue::in_order>() &&
       q.has_property<sycl::property::queue::enable_profiling>(),"ordered profiling queue required");
  const auto resolution=q.get_device().get_info<sycl::info::device::profiling_timer_resolution>();
  need(resolution>0,"invalid device profiling timer resolution");
  // Prime both marker/timer queries before controls or measured arithmetic.
  for (int i=0;i<2;++i) {
    auto a=mark(q),b=mark(q);wait(q,b);
    (void)a.get_profiling_info<sycl::info::event_profiling::command_end>();
    (void)b.get_profiling_info<sycl::info::event_profiling::command_start>();
    (void)std::chrono::steady_clock::now();
  }
  const auto control=[&](int phase) {
    double maximum=0;
    for (int i=0;i<8;++i) {
      const auto wall=std::chrono::steady_clock::now();
      auto a=mark(q),b=mark(q);wait(q,b);
      const auto start=a.get_profiling_info<sycl::info::event_profiling::command_end>();
      const auto end=b.get_profiling_info<sycl::info::event_profiling::command_start>();
      const double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-wall).count();
      need(end>=start && ns>0 && std::isfinite(ns),"invalid public timing control");
      maximum=std::max(maximum,ns);
      std::cout<<"{\"kind\":\"public_control\",\"phase\":"<<phase<<",\"index\":"<<i
        <<",\"wall_ns\":"<<ns<<",\"queue_interval_ns\":"<<(end-start)
        <<",\"timer_resolution_ns\":"<<resolution<<",\"status\":\"PASS\"}\n";
    }
    return maximum;
  };
  const double initial_control=control(0);
  double minimum_batch=1e300;
  // Fixed before execution: no adaptive repeats or post-result timing threshold.
  for (int rows:{160,2333,2334}) {
    Case c(rows,false,q);PublicGU path(c,arm!=0);
    c.separate(xe::Tile::wide_128x128);drain(q);
    path.prime();need(compare_hidden(c.reference,c.candidate)<=1,"public preflight differs");
    // Longer fixed batches amortize endpoint/startup latency without relaxing
    // the existing one-percent control rule. This is a new timing envelope.
    const int repeat=rows==160 ? 2048 : 512;
    for (int round=-1;round<5;++round) {
      c.candidate.reset();path.prime();
      // The outer clock charges trace planning, queries and diagnostic output.
      // GPU samples are selected from this very warmed batch, not another call.
      const auto charged=std::chrono::steady_clock::now();
      std::unique_ptr<cp::Trace> trace;
      if constexpr (Observe) trace.reset(new cp::Trace(&q,1));
      int selected=0;
      sycl::event start,end;
      double wall_ns=0,call_cpu=-1,call_wall=-1,total_cpu=-1,total_wall=-1;
      double wait_cpu=-1,wait_wall=-1,call_wait_cpu=-1,call_wait_wall=-1,submit_cpu=-1,submit_wall=-1;
      unsigned long long submit_count=0,wait_count=0;
      {
        auto frame=trace ? trace->begin(0,0) : cp::Frame{};
        struct Retirement {
          sycl::queue &queue; bool checked=false;
          ~Retirement() { if (!checked) drain(queue); }
        } retirement{q};
        if constexpr (Observe) for (int i=0;i<repeat;++i) frame.consider(0,i,rows);
        start=mark(q);const auto wall=std::chrono::steady_clock::now();
        {
          cpu::Scope scope(Observe);
          const auto begin=Observe ? cpu::stamp() : cpu::Stamp{};
          for (int i=0;i<repeat;++i) {
            if constexpr (Observe) selected+=int(path.call<true>(&frame,i));
            else selected+=int(path.call());
          }
          const auto calls_end=Observe ? cpu::stamp() : cpu::Stamp{};
          if constexpr (Observe) {
            need(begin.valid && calls_end.valid && cpu::active,"wrapper CPU clock unavailable");
            call_cpu=calls_end.cpu-begin.cpu;call_wall=calls_end.wall-begin.wall;
            call_wait_cpu=0;call_wait_wall=0;
            for (int k=0;k<cpu::kinds;++k) {
              call_wait_cpu+=cpu::active->cpu[k];call_wait_wall+=cpu::active->wall[k];
            }
          }
          end=mark(q);wait(q,end);
          retirement.checked=true;
          wall_ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-wall).count();
          if constexpr (Observe) {
            const auto finish=cpu::stamp();const auto &s=*cpu::active;
            need(finish.valid && !s.invalid && !s.depth && !s.host_depth && !s.status_depth &&
                !s.status_part_depth,"incomplete wrapper CPU accounting");
            total_cpu=finish.cpu-begin.cpu;total_wall=finish.wall-begin.wall;
            wait_cpu=0;wait_wall=0;
            for (int k=0;k<cpu::kinds;++k) {
              need(!s.throws[k],"wrapper wait threw");
              wait_cpu+=s.cpu[k];wait_wall+=s.wall[k];wait_count+=s.count[k];
            }
            for (int k=0;k<cpu::host_kinds;++k) {
              need(!s.host_throws[k] && (k==cpu::gemm_submit || !s.host_count[k]),"unexpected wrapper host region");
            }
            submit_cpu=s.host_cpu[cpu::gemm_submit]-s.host_wait_cpu[cpu::gemm_submit];
            submit_wall=s.host_wall[cpu::gemm_submit]-s.host_wait_wall[cpu::gemm_submit];
            submit_count=s.host_count[cpu::gemm_submit];
            need(submit_count==unsigned(repeat) && wait_count>=1 &&
                cpu::duration(call_cpu-call_wait_cpu-submit_cpu) &&
                cpu::duration(call_wall-call_wait_wall-submit_wall) &&
                cpu::duration(total_cpu-call_cpu-wait_cpu+call_wait_cpu) &&
                cpu::duration(total_wall-call_wall-wait_wall+call_wait_wall),"wrapper CPU partition does not conserve");
          }
          scope.complete(true);
        }
        if constexpr (Observe) frame.after_checked_drain();
      }
      trace.reset(); // event queries and bounded profile output are charged
      const double charged_ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-charged).count();
      const auto a=start.get_profiling_info<sycl::info::event_profiling::command_end>();
      const auto b=end.get_profiling_info<sycl::info::event_profiling::command_start>();
      need(b>a && std::isfinite(wall_ns) && selected==(path.expected ? repeat : 0),
           "timed public dispatch/count/timer differs");
      const bool eligible=wall_ns>=1e8 && b-a>=100000000 &&
          initial_control<=wall_ns*.01 && double(resolution)<=double(b-a)*.001;
      minimum_batch=std::min(minimum_batch,wall_ns);
      const int ulp=compare_hidden(c.reference,c.candidate);
      need(c.x.get()==c.hx && c.w.get()==c.hw,"public timing changed inputs");
      (void)c.gu.get();for (auto v:c.capture.get()) need(v==Sentinel,"public timing wrote diagnostic capture");
      std::cout<<"{\"kind\":\"public_sample\",\"process\":"<<process<<",\"arm\":"<<arm
        <<",\"rows\":"<<rows<<",\"round\":"<<round<<",\"repeat\":"<<repeat
        <<",\"selected\":"<<selected<<",\"fallback\":"<<(repeat-selected)
        <<",\"wall_ns\":"<<wall_ns<<",\"queue_interval_ns\":"<<(b-a)
        <<",\"observed\":"<<Observe<<",\"charged_wall_ns\":"<<charged_ns
        <<",\"wrapper_cpu_ns\":"<<call_cpu*1e9<<",\"wrapper_wall_ns\":"<<call_wall*1e9
        <<",\"total_cpu_ns\":"<<total_cpu*1e9<<",\"total_wall_ns\":"<<total_wall*1e9
        <<",\"wait_cpu_ns\":"<<wait_cpu*1e9<<",\"wait_wall_ns\":"<<wait_wall*1e9
        <<",\"wrapper_wait_cpu_ns\":"<<call_wait_cpu*1e9<<",\"wrapper_wait_wall_ns\":"<<call_wait_wall*1e9
        <<",\"inner_submit_cpu_ns\":"<<submit_cpu*1e9<<",\"inner_submit_wall_ns\":"<<submit_wall*1e9
        <<",\"inner_submit_count\":"<<submit_count<<",\"wait_count\":"<<wait_count
        <<",\"max_half_ulp\":"<<ulp<<",\"eligible\":"<<eligible
        <<",\"scope\":\"public_dispatch_queue_interval_includes_host_gaps_not_GPU_service\""
        <<",\"status\":\"PASS\"}\n";
      need(eligible,"public batch below frozen duration/control/resolution eligibility");
    }
  }
  need(control(1)<=minimum_batch*.01,"final public timer/control overhead exceeds1percent");
}
void benchmark(int process,sycl::queue &q) {
  for (int rows:{64,160,2334,5963,8192}) {
    const auto tile=rows<128 ? xe::Tile::compact_16x64 : xe::Tile::wide_128x128;
    Case c(rows,false,q); c.check(tile);
    constexpr int repeat=16;
    for (int round=-1;round<5;++round) for (int order=0;order<2;++order) {
      const int arm=(process+round+2+order)%2;
      c.candidate.reset(); c.reference.reset();
      const auto start=mark(q); const auto wall=std::chrono::steady_clock::now();
      for (int i=0;i<repeat;++i) if (arm) (void)c.fused(tile); else c.separate(tile);
      const auto end=mark(q); wait(q,end);
      const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count()/repeat;
      const auto a=start.get_profiling_info<sycl::info::event_profiling::command_end>();
      const auto b=end.get_profiling_info<sycl::info::event_profiling::command_start>();
      need(b>a && wall_ms>0 && std::isfinite(wall_ms),"invalid timing");
      const double interval_ms=double(b-a)/1e6/repeat;
      if (arm) c.separate(tile); else (void)c.fused(tile);
      drain(q); const int ulp=compare_hidden(c.reference,c.candidate);
      need(c.x.get()==c.hx && c.w.get()==c.hw,"timed inputs changed");
      (void)c.gu.get(); (void)c.capture.get();
      std::cout << "{\"kind\":\"sample\",\"process\":" << process << ",\"rows\":" << rows
        << ",\"round\":" << round << ",\"order\":" << order << ",\"arm\":" << arm
        << ",\"repeat\":" << repeat << ",\"wall_ms\":" << wall_ms
        << ",\"queue_interval_ms\":" << interval_ms << ",\"max_half_ulp\":" << ulp
        << ",\"status\":\"PASS\"}\n";
    }
  }
}
} // namespace
int main(int argc,char **argv) {
  std::cout << std::unitbuf;
  std::cout.precision(17); // preserve nanosecond CPU partitions in longer batches
  try {
    need(argc>=2,"use --ownership|--smoke|--validate|--bench PROCESS");
    const std::string mode=argv[1];
    if (mode=="--ownership") {
      ownership(); std::cout << "fixture_status=PASS mode=ownership GPU_submissions=0\n"; return 0;
    }
    const bool public_attribution=mode=="--attribute-public";
    const bool public_timing=mode=="--bench-public" || public_attribution;
    const bool public_profile=mode=="--profile-public";
    const bool public_smoke=mode=="--smoke-public";
    int public_arm=0,public_process=0;
    if (public_timing || public_profile || public_smoke) {
      need(argc==(public_timing ? 4 : 3),"public mode needs PROCESS ARM or ARM");
      public_arm=std::stoi(argv[public_timing ? 3 : 2]);public_settings(public_arm);
      if (public_timing) {
        public_process=std::stoi(argv[2]);need(public_process>=0 && public_process<3,"invalid public process ordinal");
      }
    }
    need(mode=="--smoke" || mode=="--validate" || mode=="--bench" || mode=="--profile-engine-hook" ||
         public_timing || public_profile || public_smoke,"invalid mode");
    for (auto tile:{xe::Tile::compact_16x64,xe::Tile::wide_128x128}) {
      const auto p=xe::inspect_gu_pair_ownership(8192,tile,xe::GuLayout::matrix);
      need(!p.partner_mismatch && !p.packed_missing && !p.packed_multiple && !p.packed_orientation_mismatch,
           "packed partner ownership gate");
    }
    runtime=sb::runtime_for(); auto q=runtime->make_queue(true); topology(q);
    if (public_timing) {
      if (public_attribution) benchmark_public<true>(public_process,public_arm,q);
      else benchmark_public<false>(public_process,public_arm,q);
    } else if (public_profile || public_smoke) {
      for (int rows:{3,160,2333,2334}) { Case c(rows,false,q);c.check_profile(public_arm!=0); }
    } else if (mode=="--profile-engine-hook") {
      need(argc==2,"profile hook takes no timing parameters");
      for (int rows:{2333,2334,8192}) { Case c(rows,false,q);c.check_profile(); }
    } else if (mode=="--smoke") {
      for (auto tile:{xe::Tile::compact_16x64,xe::Tile::wide_128x128}) {
        Case c(3,false,q); c.check(tile); Case edge(3,true,q); edge.check(tile);
      }
      Case c(3,false,q); admissions(c);
    } else if (mode=="--validate") {
      for (auto tile:{xe::Tile::compact_16x64,xe::Tile::wide_128x128}) {
        for (int rows:{1,17,33,129,2334,5963,8191,8192}) { Case c(rows,false,q); c.check(tile); }
        Case edge(17,true,q); edge.check(tile);
      }
    } else { need(argc==3,"bench process required"); const int p=std::stoi(argv[2]);
      need(p>=0 && p<3,"invalid process ordinal"); benchmark(p,q); }
    q.wait_and_throw(); runtime->check(); runtime->wait();
    std::cout << "fixture_status=PASS mode=" << mode.substr(2) << '\n'; return 0;
  } catch (const std::exception &e) {
    if (runtime) { try { runtime->wait(); } catch (...) { std::_Exit(86); } }
    std::cerr << "fixture_status=FAIL " << e.what() << '\n'; return 1;
  }
}
