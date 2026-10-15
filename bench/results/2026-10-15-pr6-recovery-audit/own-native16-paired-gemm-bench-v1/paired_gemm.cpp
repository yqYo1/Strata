// Authored for Strata; repository LICENSE applies. Source proposal: UNRUN.
// One process, one shape, one input format, one Runtime and one in-order queue.
#include "strata/prefill/gemm.hpp"
#include "strata/sycl/launch.hpp"
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
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
constexpr std::array<Shape, 3> Shapes{{{160,1280,2560}, {160,2560,640}, {8192,1280,2560}}};
constexpr Shape Small{17,48,48}; // Same production kernels, including native tail tile.
enum class Route { Direct, Cast, Native };
constexpr std::array<Route,3> Warm{{Route::Direct,Route::Cast,Route::Native}};
constexpr std::array<std::array<Route,3>,3> Order{{
  {Route::Direct,Route::Cast,Route::Native},
  {Route::Cast,Route::Native,Route::Direct},
  {Route::Native,Route::Direct,Route::Cast}}};
const char *name(Route r) {
  return r == Route::Direct ? "old_f32_direct" : r == Route::Cast ? "old_f32_plus_gpu_bf16_cast" : "native_bf16_output";
}
const char *implementation(Route r, bool bf) {
  if (r == Route::Native) return bf ? "Gemm::bf16_out16(output_bf16=true);native_joint_matrix_fp32_acc" : "Gemm::f16_out16(output_bf16=true);native_joint_matrix_fp32_acc";
  if (r == Route::Cast) return bf ? "Gemm::bf16;oneMKL_then_one_gpu_RNE_BF16_kernel" : "Gemm::f16;oneMKL_then_one_gpu_RNE_BF16_kernel";
  return bf ? "Gemm::bf16;oneMKL_FP32_output" : "Gemm::f16;oneMKL_FP32_output";
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
  size_t count;
  Buffer(std::shared_ptr<Runtime> rt, size_t n)
      : allocation(rt, (n + 2*Guard)*sizeof(U), MemoryKind::Device), count(n) {}
  U *base() const { return allocation.as<U>(); }
  U *data() const { return static_cast<U*>(allocation.at(Guard*sizeof(U), count*sizeof(U))); }
};
template<class U> void upload(sycl::queue &q, const Buffer<U> &buf, const std::vector<U> &v) {
  require(v.size() == buf.count + 2*Guard, "upload extent");
  validate_device_span(buf.base(), v.size()*sizeof(U), q);
  q.memcpy(buf.base(), v.data(), v.size()*sizeof(U)).wait_and_throw();
}
template<class U> std::vector<U> download(sycl::queue &q, const Buffer<U> &buf) {
  std::vector<U> host(buf.count + 2*Guard);
  q.memcpy(host.data(), buf.base(), host.size()*sizeof(U)).wait_and_throw();
  return host;
}
template<class U> void guards(const std::vector<U> &v, U mark) {
  for (size_t i=0; i<Guard; ++i)
    require(v[i] == mark && v[v.size()-Guard+i] == mark, "output guard changed");
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
void run(std::shared_ptr<Runtime> rt, Shape s, int shape_id, bool bf, bool smoke) {
  auto &q = rt->compute();
  require(q.has_property<sycl::property::queue::in_order>(), "in-order queue required");
  Gemm g;
  std::string error;
  require(g.init_external(&q,nullptr,0,nullptr,0,error), error);
  const size_t outputs = size_t(s.t)*s.n;
  Buffer<uint16_t> x(rt,size_t(s.t)*s.k), w(rt,size_t(s.n)*s.k), cast16(rt,outputs), native16(rt,outputs);
  Buffer<uint32_t> y32(rt,outputs);
  const size_t bytes = x.allocation.size()+w.allocation.size()+cast16.allocation.size()+native16.allocation.size()+y32.allocation.size();
  require(bytes < (192u<<20), "bounded device allocation budget");
  std::vector<uint16_t> hx(x.count+2*Guard,Mark16), hw(w.count+2*Guard,Mark16);
  for (size_t r=0; r<size_t(s.t); ++r) for (size_t k=0; k<size_t(s.k); ++k)
    hx[Guard+r*s.k+k] = host16(std::ldexp(double(2*a(r)+b(r)*sign_k(k)),-5),bf);
  for (size_t c=0; c<size_t(s.n); ++c) for (size_t k=0; k<size_t(s.k); ++k)
    hw[Guard+c*s.k+k] = host16(std::ldexp(double(2*d(c)+e(c)*sign_k(k)),-6),bf);
  upload(q,x,hx); upload(q,w,hw);
  std::vector<uint32_t> reset32(y32.count+2*Guard,Mark32);
  std::vector<uint16_t> reset16(outputs+2*Guard,Mark16);
  std::fill(reset32.begin()+Guard,reset32.end()-Guard,Poison32);
  std::fill(reset16.begin()+Guard,reset16.end()-Guard,Poison16);
  upload(q,y32,reset32); upload(q,cast16,reset16); upload(q,native16,reset16);
  for (auto span : {Span{x.data(),x.count*2}, Span{w.data(),w.count*2}, Span{cast16.data(),outputs*2},
                    Span{native16.data(),outputs*2}, Span{y32.data(),outputs*4}})
    validate_device_span(span.pointer,span.bytes,q);
  std::cout << "{\"kind\":\"plan\",\"mode\":" << quote(smoke?"smoke":"measure")
    << ",\"shape\":" << shape_id << ",\"T\":" << s.t << ",\"N\":" << s.n << ",\"K\":" << s.k
    << ",\"ldy\":" << s.n << ",\"beta\":0,\"input_type\":" << quote(bf?"bf16":"fp16")
    << ",\"device_tensor_bytes\":" << bytes << ",\"input_fnv1a64\":" << quote(std::to_string(fingerprint(hx)))
    << ",\"weight_fnv1a64\":" << quote(std::to_string(fingerprint(hw)))
    << ",\"oracle\":\"rank2_binary_exact_v1\",\"warmup_order\":[\"old_f32_direct\",\"old_f32_plus_gpu_bf16_cast\",\"native_bf16_output\"]"
    << ",\"sample_order\":[[\"old_f32_direct\",\"old_f32_plus_gpu_bf16_cast\",\"native_bf16_output\"],[\"old_f32_plus_gpu_bf16_cast\",\"native_bf16_output\",\"old_f32_direct\"],[\"native_bf16_output\",\"old_f32_direct\",\"old_f32_plus_gpu_bf16_cast\"]]}" << std::endl;
  int ordinal = 0;
  auto one = [&](Route route, bool warmup, int sample) {
    const bool use32 = route != Route::Native;
    if (use32) upload(q,y32,reset32);
    if (route == Route::Cast) upload(q,cast16,reset16);
    if (route == Route::Native) upload(q,native16,reset16);
    rt->wait(); // No queued upload/reference work may enter the wall interval.
    std::cout << "{\"kind\":\"begin\",\"ordinal\":" << ordinal << ",\"route\":" << quote(name(route))
      << ",\"warmup\":" << (warmup?"true":"false") << ",\"sample\":" << sample << "}" << std::endl;
    const auto start = std::chrono::steady_clock::now();
    if (route == Route::Native) native(g,bf,x.data(),w.data(),native16.data(),s);
    else {
      direct(g,bf,x.data(),w.data(),reinterpret_cast<float*>(y32.data()),s);
      if (route == Route::Cast) gpu_cast(q,reinterpret_cast<const float*>(y32.data()),cast16.data(),outputs);
    }
    q.wait_and_throw(); rt->check();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    require(std::isfinite(elapsed) && elapsed > 0, "invalid wall sample");
    // Flush the individual measurement before readback/validation so failed
    // comparisons retain their timing and explicit unvalidated status.
    std::cout << "{\"kind\":\"sample\",\"status\":\"pending_validation\",\"ordinal\":" << ordinal
      << ",\"route\":" << quote(name(route)) << ",\"implementation\":" << quote(implementation(route,bf))
      << ",\"stored_output_type\":" << quote(route==Route::Direct?"fp32":"bf16")
      << ",\"global_fp32_intermediate\":" << (route==Route::Cast?"true":"false")
      << ",\"warmup\":" << (warmup?"true":"false") << ",\"sample\":" << sample
      << ",\"submit_to_completion_wall_seconds\":" << elapsed << "}" << std::endl;
    uint64_t hash = 0;
    if (use32) {
      const auto got = download(q,y32); guards(got,Mark32);
      for (size_t i=0; i<outputs; ++i) {
        const float v = std::bit_cast<float>(got[Guard+i]);
        if (!std::isfinite(v) || double(v)!=oracle(s,i/s.n,i%s.n))
          throw std::runtime_error("FP32 full finite/exact closed-form oracle mismatch at " + std::to_string(i));
      }
      hash = fingerprint(got);
    }
    if (route != Route::Direct) {
      const auto got = download(q,route==Route::Cast?cast16:native16); guards(got,Mark16);
      for (size_t i=0; i<outputs; ++i) {
        const double v = decode_bf16(got[Guard+i]);
        const uint16_t expected = host16(oracle(s,i/s.n,i%s.n),true);
        if (!std::isfinite(v) || !(got[Guard+i]==expected || (v==0 && decode_bf16(expected)==0)))
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
  for (Route r : Warm) one(r,true,-1);
  if (!smoke) for (int sample=0; sample<3; ++sample)
    for (Route r : Order[size_t(sample)]) one(r,false,sample);
  rt->wait();
  std::cout << "{\"kind\":\"result\",\"status\":\"PASS\",\"validated_invocations\":" << ordinal
    << ",\"samples_per_route\":" << (smoke?0:3)
    << ",\"scope\":\"operator_sanity_and_wall_only;not_model_quality_or_speed;not_adversarial_accumulator_gate\"}" << std::endl;
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(17);
    const bool smoke = argc==3 && std::string(argv[1])=="--smoke";
    require(smoke || (argc==4 && std::string(argv[1])=="--measure"),
            "usage: paired_gemm --smoke fp16|bf16 | --measure 0|1|2 fp16|bf16");
    const std::string format = argv[smoke?2:3];
    require(format=="fp16" || format=="bf16", "bounded input format");
    int shape = -1;
    if (!smoke) {
      const std::string token = argv[2];
      require(token=="0" || token=="1" || token=="2", "bounded shape id");
      shape = token[0]-'0';
    }
    // Main supplies the entire fixed environment before exec. No mutation here.
    auto rt = runtime_for();
    device_record(*rt);
    run(rt,smoke?Small:Shapes[size_t(shape)],shape,format=="bf16",smoke);
    return 0;
  } catch (const std::exception &error) {
    std::cout << "{\"kind\":\"result\",\"status\":\"FAIL\",\"error\":" << quote(error.what()) << "}" << std::endl;
    return 1;
  }
}
