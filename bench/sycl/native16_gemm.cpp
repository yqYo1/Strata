// Authored for Strata; repository LICENSE applies.
// One process, one shape, one input format, one Runtime and one in-order queue.
#include "strata/prefill/gemm.hpp"
#include "native16.hpp"
#include "evidence.hpp"
#include "strata/sycl/launch.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using strata::prefill::Gemm;
using namespace strata::sycl_backend;
constexpr size_t Guard = 64;
constexpr uint16_t Mark16 = 0xa55a, Poison16 = 0x7fc1;
constexpr uint32_t Mark32 = 0x4a55a55a, Poison32 = 0x7fc12345;
struct Shape { int t, n, k; };
// Keep original shape IDs; add Down8192 and exact row-mass quantiles from the
// committed 78K MoE census. These are a sweep, not an interpolated model cost.
constexpr std::array<Shape, 10> Shapes{{
  {160,1280,2560}, {160,2560,640}, {8192,1280,2560}, {8192,2560,640},
  {64,1280,2560}, {64,2560,640}, {2334,1280,2560}, {2334,2560,640},
  {5963,1280,2560}, {5963,2560,640}}};
constexpr Shape Small{3,128,64}; // All routes satisfy the R810 public contract.
namespace xe = strata::experimental::vllm_xe2_native16;
enum class Route { Direct, Cast, Native, Compact, Wide };
constexpr std::array<Route,5> Routes{{Route::Direct,Route::Cast,Route::Native,Route::Compact,Route::Wide}};
const char *name(Route r) {
  switch (r) {
    case Route::Direct: return "old_f32_direct";
    case Route::Cast: return "old_f32_plus_gpu_bf16_cast";
    case Route::Native: return "native_bf16_output";
    case Route::Compact: return "vllm_compact16x64_bf16_output";
    case Route::Wide: return "vllm_wide128x128_bf16_output";
  }
  throw std::runtime_error("invalid route");
}
const char *implementation(Route r, bool bf) {
  if (r == Route::Compact) return "R810::Gemm::beta0(compact_16x64,output=bf16);local_fp32_acc";
  if (r == Route::Wide) return "R810::Gemm::beta0(wide_128x128,output=bf16);local_fp32_acc";
  if (r == Route::Native) return bf ? "Gemm::bf16_out16(configured_backend,output_bf16=true);fp32_acc" : "Gemm::f16_out16(configured_backend,output_bf16=true);fp32_acc";
  if (r == Route::Cast) return bf ? "Gemm::bf16;oneMKL_then_one_gpu_RNE_BF16_kernel" : "Gemm::f16;oneMKL_then_one_gpu_RNE_BF16_kernel";
  return bf ? "Gemm::bf16;oneMKL_FP32_output" : "Gemm::f16;oneMKL_FP32_output";
}
void order_json(int rotation) {
  std::cout << '[';
  for (int i=0; i<5; ++i) {
    if (i) std::cout << ',';
    std::cout << r816::quote(name(Routes[size_t((i+rotation)%5)]));
  }
  std::cout << ']';
}
void require(bool condition, const std::string &why) {
  if (!condition) throw std::runtime_error(why);
}
void require(bool condition, const char *why) {
  if (!condition) throw std::runtime_error(why);
}
std::string quote(const std::string &s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
    else if (c >= 32 && c < 127) out += char(c);
    else {
      constexpr char hex[] = "0123456789abcdef";
      out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
    }
  }
  return out + '"';
}
// Host-only mathematical encoder. Does not call SYCL, MKL or project conversion
// helpers. All fixture inputs are binary-exact in BOTH input storage formats.
uint16_t host16(double value, bool bf) {
  require(std::isfinite(value), "oracle encoder finite domain");
  const int fraction_bits = bf ? 7 : 10, bias = bf ? 127 : 15;
  const int top = bf ? 255 : 31;
  const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  const double magnitude = std::abs(value);
  if (magnitude == 0) return sign;
  int exponent;
  (void)std::frexp(magnitude, &exponent);
  --exponent;
  const int scale = fraction_bits - std::max(exponent, 1 - bias);
  const double scaled = std::ldexp(magnitude, scale);
  const auto lower = uint32_t(std::floor(scaled));
  const double remainder = scaled - double(lower);
  uint32_t significand = lower + (remainder > .5 || (remainder == .5 && (lower & 1)));
  if (exponent < 1 - bias) return sign | uint16_t(significand);
  if (significand == (2u << fraction_bits)) { significand >>= 1; ++exponent; }
  require(exponent + bias < top, "fixture oracle overflow");
  return sign | uint16_t(((exponent + bias) << fraction_bits) + significand - (1u << fraction_bits));
}
double decode_bf16(uint16_t bits) {
  const unsigned e = (bits >> 7) & 255, f = bits & 127;
  if (e == 255) return f ? std::numeric_limits<double>::quiet_NaN() :
      (bits & 0x8000 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity());
  const double v = e ? std::ldexp(double(128 + f), int(e) - 134) : std::ldexp(double(f), -133);
  return bits & 0x8000 ? -v : v;
}
int a(size_t r) { return int(r % 7) - 3; }
int b(size_t r) { return int((r / 7) % 5) - 2; }
int d(size_t c) { return int(c % 11) - 5; }
int e(size_t c) { return int((c / 11) % 7) - 3; }
int sign_k(size_t k) { return k % 16 < 8 ? 1 : -1; }
// Sum over the repeated 8 positive / 8 negative signs, without scanning K.
int64_t sign_sum(int k) {
  const int tail = k % 16;
  return tail <= 8 ? tail : 16 - tail;
}
double oracle(const Shape &s, size_t r, size_t c) {
  const int64_t ac = 4 * a(r) * d(c) + b(r) * e(c);
  const int64_t cross = 2 * a(r) * e(c) + 2 * b(r) * d(c);
  return std::ldexp(double(int64_t(s.k) * ac + sign_sum(s.k) * cross), -11);
}
template<class U> uint64_t fingerprint(const std::vector<U> &v) {
  // FNV-1a over a specified little-endian word stream; diagnostic, not SHA256.
  static_assert(std::is_unsigned_v<U>);
  uint64_t h = 14695981039346656037ull;
  for (U word : v) for (size_t j=0; j<sizeof(U); ++j) {
    h ^= uint8_t(word >> (j*8)); h *= 1099511628211ull;
  }
  return h;
}
template<class U> struct Buffer {
  Allocation allocation;
  size_t count, prefix, words;
  Buffer(std::shared_ptr<Runtime> rt, size_t n)
      : allocation(rt, (n + 2*Guard + 64/sizeof(U))*sizeof(U), MemoryKind::Device), count(n) {
    const auto address = reinterpret_cast<uintptr_t>(allocation.data());
    prefix = Guard + size_t((64 - (address + Guard*sizeof(U))%64)%64)/sizeof(U);
    words = allocation.size()/sizeof(U);
    require(prefix+count+Guard<=words,"aligned guard extent");
  }
  U *base() const { return allocation.as<U>(); }
  U *data() const { return static_cast<U*>(allocation.at(prefix*sizeof(U), count*sizeof(U))); }
};
template<class U> void upload(sycl::queue &q, const Buffer<U> &buf, const std::vector<U> &v) {
  require(v.size() == buf.words, "upload extent");
  validate_device_span(buf.base(), v.size()*sizeof(U), q);
  q.memcpy(buf.base(), v.data(), v.size()*sizeof(U)).wait_and_throw();
}
template<class U> std::vector<U> download(sycl::queue &q, const Buffer<U> &buf) {
  std::vector<U> host(buf.words);
  try { q.memcpy(host.data(), buf.base(), host.size()*sizeof(U)).wait_and_throw(); }
  catch (...) {
    // Do not destroy an in-flight readback's host destination after failed wait.
    try { q.wait_and_throw(); } catch (...) { std::_Exit(86); }
    throw;
  }
  return host;
}
template<class U> void guards(const std::vector<U> &v, const Buffer<U> &buf, U mark) {
  for (size_t i=0; i<buf.prefix; ++i) require(v[i]==mark,"prefix output guard changed");
  for (size_t i=buf.prefix+buf.count; i<v.size(); ++i) require(v[i]==mark,"suffix output guard changed");
}
void direct(Gemm &g, bool bf, const uint16_t *x, const uint16_t *w, float *y, Shape s) {
  if (bf) g.bf16(x,w,y,s.t,s.n,s.k,s.n,0.f);
  else g.f16(x,w,y,s.t,s.n,s.k,s.n,0.f);
}
// The true native route cannot access a global FP32 pointer or tensor. FP32
// exists only in production joint_matrix/local accumulator/epilogue arithmetic.
void native(Gemm &g, bool bf, const uint16_t *x, const uint16_t *w, uint16_t *y, Shape s) {
  if (bf) g.bf16_out16(x,w,y,s.t,s.n,s.k,true,s.n,0.f);
  else g.f16_out16(x,w,y,s.t,s.n,s.k,true,s.n,0.f);
}
// One GPU conversion submission. This is an isolated storage-only comparison,
// not a reconstruction of every operator in the former model pipeline.
void gpu_cast(sycl::queue &q, const float *x, uint16_t *y, size_t count) {
  validate_spans({{y,count*2}}, {{x,count*4}}, 4);
  validate_device_span(x,count*4,q); validate_device_span(y,count*2,q);
  q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
    const auto value = sycl::ext::oneapi::bfloat16(x[i[0]]);
    y[i[0]] = sycl::bit_cast<uint16_t>(value);
  });
}
void device_record(const Runtime &rt) {
  const auto &dev = rt.device();
  require(dev.is_gpu() && dev.get_info<sycl::info::device::vendor_id>() == 0x8086,
          "Intel GPU required, no CPU device fallback");
  std::cout << "{\"kind\":\"device\",\"name\":" << quote(dev.get_info<sycl::info::device::name>())
    << ",\"vendor\":" << quote(dev.get_info<sycl::info::device::vendor>())
    << ",\"vendor_id\":" << dev.get_info<sycl::info::device::vendor_id>()
    << ",\"driver\":" << quote(dev.get_info<sycl::info::device::driver_version>())
    << ",\"profiling_timer_resolution_ns\":" << dev.get_info<sycl::info::device::profiling_timer_resolution>()
    << ",\"platform\":" << quote(dev.get_platform().get_info<sycl::info::platform::name>())
    << ",\"platform_version\":" << quote(dev.get_platform().get_info<sycl::info::platform::version>())
    << ",\"in_order\":true,\"one_runtime_one_compute_queue\":true,\"matrix_tuples\":[";
  bool first = true;
  for (const auto &v : dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>()) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"A\":" << int(v.atype) << ",\"B\":" << int(v.btype)
      << ",\"C\":" << int(v.ctype) << ",\"D\":" << int(v.dtype)
      << ",\"exact\":[" << v.msize << ',' << v.nsize << ',' << v.ksize
      << "],\"maximum\":[" << v.max_msize << ',' << v.max_nsize << ',' << v.max_ksize << "]}";
  }
  std::cout << "]}" << std::endl;
}
void run(std::shared_ptr<Runtime> rt, Shape s, int shape_id, bool bf, bool smoke, int process_index,
         bool resident = false) {
  auto &q = rt->compute();
  require(q.has_property<sycl::property::queue::in_order>(), "in-order queue required");
  require(q.has_property<sycl::property::queue::enable_profiling>(), "profiling queue required");
  Gemm g;
  xe::Gemm vllm(*rt,q);
  std::string error;
  require(g.init_external(&q,nullptr,0,nullptr,0,error), error);
  const size_t outputs = size_t(s.t)*s.n;
  Buffer<uint16_t> x(rt,size_t(s.t)*s.k), w(rt,size_t(s.n)*s.k), cast16(rt,outputs), native16(rt,outputs), compact16(rt,outputs), wide16(rt,outputs);
  Buffer<uint32_t> y32(rt,outputs);
  const size_t bytes = x.allocation.size()+w.allocation.size()+cast16.allocation.size()+native16.allocation.size()+compact16.allocation.size()+wide16.allocation.size()+y32.allocation.size();
  require(bytes < (256u<<20), "bounded device allocation budget");
  std::vector<uint16_t> hx(x.words,Mark16), hw(w.words,Mark16);
  for (size_t r=0; r<size_t(s.t); ++r) for (size_t k=0; k<size_t(s.k); ++k)
    hx[x.prefix+r*s.k+k] = host16(std::ldexp(double(2*a(r)+b(r)*sign_k(k)),-5),bf);
  for (size_t c=0; c<size_t(s.n); ++c) for (size_t k=0; k<size_t(s.k); ++k)
    hw[w.prefix+c*s.k+k] = host16(std::ldexp(double(2*d(c)+e(c)*sign_k(k)),-6),bf);

  std::vector<uint32_t> reset32(y32.words,Mark32);
  auto reset = [](const Buffer<uint16_t> &buf) {
    std::vector<uint16_t> v(buf.words,Mark16);
    std::fill(v.begin()+buf.prefix,v.begin()+buf.prefix+buf.count,Poison16);
    return v;
  };
  const auto reset_cast=reset(cast16), reset_native=reset(native16), reset_compact=reset(compact16), reset_wide=reset(wide16);
  // All host/device owners precede the drain. On asynchronous failure no owner
  // is freed before checked completion; failed drain terminates without unwind.
  struct Drain {
    Runtime &runtime;
    ~Drain() noexcept {
      try { runtime.wait(); }
      catch (...) { std::cerr << "R816_CHECKED_DRAIN_FAILED" << std::endl; std::_Exit(86); }
    }
  } drain{*rt};
  const auto upload_start = std::chrono::steady_clock::now();
  const double upload_cpu_start = r816::thread_cpu_seconds();
  upload(q,x,hx); upload(q,w,hw);
  const double upload_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-upload_start).count();
  const double upload_cpu_seconds = r816::thread_cpu_seconds()-upload_cpu_start;
  std::fill(reset32.begin()+y32.prefix,reset32.begin()+y32.prefix+y32.count,Poison32);
  upload(q,y32,reset32); upload(q,cast16,reset_cast); upload(q,native16,reset_native);
  upload(q,compact16,reset_compact); upload(q,wide16,reset_wide);
  for (auto span : {Span{x.data(),x.count*2}, Span{w.data(),w.count*2}, Span{cast16.data(),outputs*2},
                    Span{native16.data(),outputs*2}, Span{compact16.data(),outputs*2}, Span{wide16.data(),outputs*2}, Span{y32.data(),outputs*4}})
    validate_device_span(span.pointer,span.bytes,q);
  std::cout << "{\"kind\":\"plan\",\"mode\":" << quote(smoke?"smoke":"measure")
    << ",\"resident_only\":" << (resident?"true":"false")
    << ",\"submits_per_sample\":" << (resident?8:1)
    << ",\"shape\":" << shape_id << ",\"T\":" << s.t << ",\"N\":" << s.n << ",\"K\":" << s.k
    << ",\"ldy\":" << s.n << ",\"beta\":0,\"input_type\":" << quote(bf?"bf16":"fp16")
    << ",\"device_tensor_bytes\":" << bytes << ",\"input_fnv1a64\":" << quote(std::to_string(fingerprint(hx)))
    << ",\"initial_pageable_upload_bytes\":" << (hx.size()+hw.size())*2
    << ",\"initial_pageable_upload_wall_seconds\":" << upload_seconds
    << ",\"initial_pageable_upload_thread_cpu_seconds\":" << upload_cpu_seconds
    << ",\"weight_fnv1a64\":" << quote(std::to_string(fingerprint(hw)))
    << ",\"oracle\":\"rank2_binary_exact_v1\",\"process_index\":" << process_index
    << ",\"warmups_per_route\":3,\"timed_submits_per_route\":" << (smoke?0:3)
    << ",\"pid\":" << getpid() << ",\"rotation_base\":" << process_index << ",\"route_order\":";
  if (resident) std::cout << "[\"vllm_compact16x64_bf16_output\",\"vllm_wide128x128_bf16_output\"]";
  else order_json(0);
  std::cout << "}" << std::endl;
  int ordinal = 0;
  auto one = [&](Route route, bool warmup, int sample) {
    const bool use32 = route == Route::Direct || route == Route::Cast;
    Buffer<uint16_t> &out16 = route==Route::Cast ? cast16 : route==Route::Native ? native16 : route==Route::Compact ? compact16 : wide16;
    if (use32) upload(q,y32,reset32);
    if (route == Route::Cast) upload(q,cast16,reset_cast);
    if (route == Route::Native) upload(q,native16,reset_native);
    if (route == Route::Compact) upload(q,compact16,reset_compact);
    if (route == Route::Wide) upload(q,wide16,reset_wide);
    rt->wait(); // No queued upload/reference work may enter the wall interval.
    std::cout << "{\"kind\":\"begin\",\"ordinal\":" << ordinal << ",\"route\":" << quote(name(route))
      << ",\"warmup\":" << (warmup?"true":"false") << ",\"sample\":" << sample << "}" << std::endl;
    const double cpu_start=r816::thread_cpu_seconds();
    const auto start = std::chrono::steady_clock::now();
    std::array<std::optional<sycl::event>,8> events;
    const int submits = resident ? 8 : 1;
    if (route == Route::Native) native(g,bf,x.data(),w.data(),native16.data(),s);
    else if (route == Route::Compact || route == Route::Wide) {
      const xe::Problem p{x.data(),w.data(),out16.data(),s.t,s.n,s.k,s.n,
        bf?xe::Storage::bf16:xe::Storage::fp16,xe::Storage::bf16,0.f};
      // Same resident operands and beta0 output, one in-order queue. All
      // owners remain live through completion; no transfer or host callback
      // occurs between these submissions. Validate the final output and all
      // guards after the burst, without resetting them between submissions.
      for (int i=0;i<submits;++i)
        events[size_t(i)] = vllm.beta0(p,route==Route::Compact?xe::Tile::compact_16x64:xe::Tile::wide_128x128);
    } else {
      direct(g,bf,x.data(),w.data(),reinterpret_cast<float*>(y32.data()),s);
      if (route == Route::Cast) gpu_cast(q,reinterpret_cast<const float*>(y32.data()),cast16.data(),outputs);
    }
    const double cpu_submitted=r816::thread_cpu_seconds();
    q.wait_and_throw(); rt->check();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    const double cpu_completed=r816::thread_cpu_seconds();
    require(std::isfinite(elapsed) && elapsed > 0, "invalid wall sample");
    std::array<std::array<uint64_t,3>,8> timestamps{};
    uint64_t kernel_ns = 0;
    bool contract_valid = true, cross_event_ordered = true, duration_positive = true;
    if (events[0]) for (int i=0;i<submits;++i) {
      const auto &event = *events[size_t(i)];
      auto &v = timestamps[size_t(i)];
      v = {event.get_profiling_info<sycl::info::event_profiling::command_submit>(),
           event.get_profiling_info<sycl::info::event_profiling::command_start>(),
           event.get_profiling_info<sycl::info::event_profiling::command_end>()};
      contract_valid &= v[0]<=v[1] && v[1]<=v[2];
      duration_positive &= v[1]<v[2];
      cross_event_ordered &= !i || timestamps[size_t(i-1)][2]<=v[1];
      if (v[1]<=v[2]) kernel_ns += v[2]-v[1];
    }
    // Flush the individual measurement before readback/validation so failed
    // comparisons retain their timing and explicit unvalidated status.
    std::cout << "{\"kind\":\"sample\",\"status\":\"pending_validation\",\"ordinal\":" << ordinal
      << ",\"route\":" << quote(name(route)) << ",\"implementation\":" << quote(implementation(route,bf))
      << ",\"stored_output_type\":" << quote(route==Route::Direct?"fp32":"bf16")
      << ",\"global_fp32_intermediate\":" << (route==Route::Cast?"true":"false")
      << ",\"warmup\":" << (warmup?"true":"false") << ",\"sample\":" << sample
      << ",\"submits\":" << submits
      << ",\"submit_to_completion_wall_seconds\":" << elapsed
      << ",\"thread_cpu_submit_seconds\":" << cpu_submitted-cpu_start
      << ",\"thread_cpu_wait_check_seconds\":" << cpu_completed-cpu_submitted
      << ",\"logical_flop_per_submit\":" << uint64_t(2)*s.t*s.n*s.k
      << ",\"kernel_total_seconds\":";
    if (events[0] && contract_valid && duration_positive) std::cout << double(kernel_ns)*1e-9;
    else std::cout << "null";
    std::cout << ",\"profiling_contract_valid\":" << (contract_valid?"true":"false")
      << ",\"profiling_duration_positive\":" << (duration_positive?"true":"false")
      << ",\"profiling_cross_event_ordered\":" << (cross_event_ordered?"true":"false");
    std::cout << ",\"device_timestamps_ns\":[";
    if (events[0]) for (int i=0;i<submits;++i) {
      if (i) std::cout << ',';
      const auto &v = timestamps[size_t(i)];
      std::cout << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
    }
    std::cout << "]}" << std::endl;
    // Emit every tuple before rejecting a profiling assertion. A logging
    // failure must not be mistaken for a numerical failure or hidden by the
    // normal backend-selection notice on stderr. Preserve the original gate.
    require(contract_valid && duration_positive && cross_event_ordered,
            "invalid or unordered device timestamps (see emitted tuples)");
    uint64_t hash = 0;
    if (use32) {
      const auto got = download(q,y32); guards(got,y32,Mark32);
      for (size_t i=0; i<outputs; ++i) {
        const float v = std::bit_cast<float>(got[y32.prefix+i]);
        if (!std::isfinite(v) || double(v)!=oracle(s,i/s.n,i%s.n))
          throw std::runtime_error("FP32 full finite/exact closed-form oracle mismatch at " + std::to_string(i));
      }
      hash = fingerprint(got);
    }
    if (route != Route::Direct) {
      const auto got = download(q,out16); guards(got,out16,Mark16);
      for (size_t i=0; i<outputs; ++i) {
        const double v = decode_bf16(got[out16.prefix+i]);
        const uint16_t expected = host16(oracle(s,i/s.n,i%s.n),true);
        if (!std::isfinite(v) || !(got[out16.prefix+i]==expected || (v==0 && decode_bf16(expected)==0)))
          throw std::runtime_error("BF16 full finite/RNE closed-form oracle mismatch at " + std::to_string(i));
      }
      hash = fingerprint(got);
    }
    require(download(q,x)==hx && download(q,w)==hw, "input/weight/guard immutability failure");
    std::cout << "{\"kind\":\"validation\",\"status\":\"PASS\",\"ordinal\":" << ordinal
      << ",\"route\":" << quote(name(route)) << ",\"outputs_checked\":" << outputs
      << ",\"all_finite\":true,\"closed_form_oracle\":true,\"guards\":true,\"inputs_immutable\":true"
      << ",\"output_fnv1a64\":" << quote(std::to_string(hash)) << "}" << std::endl;
    ++ordinal;
  };
  // Rotate the five-route order across warmup rounds, timed rounds and
  // independent processes. Inner submissions are never process replicates.
  const std::vector<Route> routes = resident ? std::vector<Route>{Route::Compact,Route::Wide}
                                           : std::vector<Route>(Routes.begin(),Routes.end());
  for (int round=0; round<3; ++round)
    for (size_t j=0; j<routes.size(); ++j) one(routes[(j+round+process_index)%routes.size()],true,round);
  if (!smoke) for (int round=0; round<3; ++round)
    for (size_t j=0; j<routes.size(); ++j) one(routes[(j+round+3+process_index)%routes.size()],false,round);
  rt->wait();
  std::cout << "{\"kind\":\"result\",\"status\":\"PASS\",\"validated_invocations\":" << ordinal
    << ",\"warmups_per_route\":3,\"process_index\":" << process_index
    << ",\"samples_per_route\":" << (smoke?0:3)
    << ",\"scope\":\"resident_operator_reference;not_hardware_peak_or_model_quality_or_speed;not_adversarial_accumulator_gate\"}" << std::endl;
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(17);
    const bool smoke = argc==5 && std::string(argv[1])=="--smoke";
    const bool resident = argc==6 && std::string(argv[1])=="--resident";
    require(smoke || resident || (argc==6 && std::string(argv[1])=="--measure"),
            "usage: paired_gemm --smoke fp16|bf16 --process 0|1|2 | --measure|--resident 0..9 fp16|bf16 --process 0|1|2");
    const std::string format = argv[smoke?2:3];
    require(format=="fp16" || format=="bf16", "bounded input format");
    require(std::string(argv[smoke?3:4])=="--process", "explicit process index required");
    const std::string process_token=argv[smoke?4:5];
    require(process_token=="0" || process_token=="1" || process_token=="2","bounded process index");
    const int process_index=process_token[0]-'0';
    int shape = -1;
    if (!smoke) {
      const std::string token = argv[2];
      require(token.size()==1 && token[0]>='0' && token[0]<='9', "bounded shape id");
      shape = token[0]-'0';
    }
    // Main supplies the entire fixed environment before exec. No mutation here.
    r816::libraries("before_runtime");
    auto rt = runtime_for(0); // Main explicitly qualified ordinal 0 (B570).
    device_record(*rt);
    run(rt,smoke?Small:Shapes[size_t(shape)],shape,format=="bf16",smoke,process_index,resident);
    rt->wait();
    r816::libraries("after_checked_drain");
    return 0;
  } catch (const std::exception &error) {
    std::cout << "{\"kind\":\"result\",\"status\":\"FAIL\",\"error\":" << quote(error.what()) << "}" << std::endl;
    return 1;
  }
}
