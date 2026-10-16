// Authored for Strata; repository LICENSE (MIT) applies. SOURCE-ONLY / UNRUN.
#include "strata/prefill/gemm.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace {
using strata::prefill::Gemm;
using namespace strata::sycl_backend;
constexpr size_t Guard = 17; // Data starts at 2 mod 4; scalar global I/O permits it.
constexpr uint16_t Sentinel = 0xa55a;
void need(bool good, const std::string &why) {
  if (!good) throw std::runtime_error(why);
}
// Independent host FP64 oracle: neither SYCL conversions nor Strata's device
// conversion helpers are used. Integer fields decode by the IEEE formula.
double unpack(uint16_t bits, bool bf) {
  const int fb = bf ? 7 : 10, eb = bf ? 8 : 5, bias = bf ? 127 : 15;
  const unsigned mask = (1u << eb) - 1;
  const unsigned fraction = bits & ((1u << fb) - 1);
  const unsigned exponent = (bits >> fb) & mask;
  double v;
  if (exponent == mask)
    v = fraction ? std::numeric_limits<double>::quiet_NaN()
                 : std::numeric_limits<double>::infinity();
  else if (exponent == 0)
    v = std::ldexp(double(fraction), 1 - bias - fb);
  else
    v = std::ldexp(double((1u << fb) + fraction), int(exponent) - bias - fb);
  return bits & 0x8000 ? -v : v;
}
uint32_t round_even(double v) {
  const auto lo = uint32_t(std::floor(v));
  const double rem = v - double(lo);
  return lo + (rem > .5 || (rem == .5 && (lo & 1)));
}
uint16_t pack(double value, bool bf) {
  const int fb = bf ? 7 : 10, bias = bf ? 127 : 15, maxexp = bf ? 255 : 31;
  const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  value = std::abs(value);
  if (std::isnan(value)) return sign | uint16_t((maxexp << fb) | (1 << (fb - 1)));
  if (std::isinf(value)) return sign | uint16_t(maxexp << fb);
  if (value == 0) return sign;
  int exp;
  (void)std::frexp(value, &exp);
  --exp;
  if (exp < 1 - bias) {
    const auto fraction = round_even(std::ldexp(value, bias - 1 + fb));
    return sign | uint16_t(fraction); // A carry becomes the smallest normal.
  }
  if (exp + bias >= maxexp) return sign | uint16_t(maxexp << fb);
  auto fraction = round_even(std::ldexp(value, fb - exp));
  if (fraction == (2u << fb)) { fraction >>= 1; ++exp; }
  if (exp + bias >= maxexp) return sign | uint16_t(maxexp << fb);
  return sign | uint16_t(((exp + bias) << fb) + (fraction - (1u << fb)));
}
double half_ulp(double value, bool bf) {
  const int bias = bf ? 127 : 15, fb = bf ? 7 : 10;
  int exponent = 1 - bias;
  if (value != 0) {
    int e;
    (void)std::frexp(std::abs(value), &e);
    exponent = std::max(exponent, e - 1);
  }
  return .5 * std::ldexp(1., exponent - fb);
}
struct Buffer {
  Allocation memory;
  size_t words;
  Buffer(std::shared_ptr<Runtime> rt, size_t count)
      : memory(rt, (count + Guard * 2) * 2, MemoryKind::Device), words(count) {}
  uint16_t *base() const { return memory.as<uint16_t>(); }
  uint16_t *data() const { return base() + Guard; }
};
void write(sycl::queue &q, Buffer &b, const std::vector<uint16_t> &v) {
  need(v.size() == b.words + Guard * 2, "fixture write size");
  q.memcpy(b.base(), v.data(), v.size() * 2).wait_and_throw();
}
std::vector<uint16_t> read(sycl::queue &q, const Buffer &b) {
  std::vector<uint16_t> v(b.words + Guard * 2);
  q.memcpy(v.data(), b.base(), v.size() * 2).wait_and_throw();
  return v;
}
void call(Gemm &gemm, bool input_bf, bool output_bf, const uint16_t *x,
          const uint16_t *w, uint16_t *y, int64_t t, int64_t n, int64_t k,
          int64_t ld = 0, float beta = 0) {
  if (input_bf) gemm.bf16_out16(x, w, y, t, n, k, output_bf, ld, beta);
  else gemm.f16_out16(x, w, y, t, n, k, output_bf, ld, beta);
}
void check_number(uint16_t actual, double expected, double magnitude,
                  int k, bool outbf, const std::string &tag) {
  const double want = unpack(pack(expected, outbf), outbf), got = unpack(actual, outbf);
  if (std::isinf(want)) {
    need(got == want, tag + ": overflow must retain signed infinity");
    return;
  }
  need(std::isfinite(got), tag + ": unexpected nonfinite output");
  // FP32 reduction bound plus FP32 FMA epilogue rounding. Output storage adds
  // half an FP16 or BF16 ULP, respectively. No blanket relative 1% tolerance:
  // cancellation is governed by sum(abs(products)), not by abs(sum). The
  // bound is for these finite fixtures; not a full-model acceptance rule.
  const double u = std::ldexp(1., -24);
  const double gamma = (k + 3) * u / (1 - (k + 3) * u);
  const double bound = gamma * magnitude + half_ulp(expected, outbf);
  need(std::abs(got - expected) <= bound,
       tag + ": FP64 oracle, FP32 accumulation or output-rounding mismatch");
}
// kind 0: nontrivial projection; 1: inter-matrix-tile cancellation; 2: IEEE
// finite/subnormal/range identity. Subnormal preservation is tested explicitly,
// so an unexpected hardware flush is a failure to investigate, never a clamp.
void run_case(std::shared_ptr<Runtime> rt, Gemm &gemm, bool inbf, bool outbf,
              int t, int n, int k, int ld, float beta, int kind) {
  auto &q = rt->compute();
  const int stride = ld == 0 ? n : ld;
  Buffer dx(rt, size_t(t) * k), dw(rt, size_t(n) * k), dy(rt, size_t(t) * stride);
  std::vector<uint16_t> x(dx.words + Guard * 2, Sentinel),
      w(dw.words + Guard * 2, Sentinel), before(dy.words + Guard * 2, Sentinel);
  for (size_t i = 0; i < dx.words; ++i)
    x[Guard + i] = pack(double(int(i * 17 % 37) - 18) / 16, inbf);
  for (size_t i = 0; i < dw.words; ++i)
    w[Guard + i] = pack(double(int(i * 11 % 31) - 15) / 32, inbf);
  if (kind == 1) {
    need(k == 48, "cancellation K");
    std::fill(x.begin() + Guard, x.end() - Guard, pack(1, inbf));
    for (int c = 0; c < n; ++c)
      for (int j = 0; j < k; ++j)
        w[Guard + c * k + j] = pack(j < 16 ? 2048 : j < 32 ? 1 : -2048, inbf);
    // 16*2048 + 16*1 - 16*2048 = 16 in FP32. Rounding the accumulator to
    // FP16/BF16 after each 16-wide matrix update instead loses the middle 16.
  }
  if (kind == 2) {
    need(k == 16 && n == 16, "range identity geometry");
    std::fill(w.begin() + Guard, w.end() - Guard, pack(0, inbf));
    for (int i = 0; i < n; ++i) w[Guard + i * k + i] = pack(1, inbf);
    const std::array<uint16_t, 16> values = inbf
        ? std::array<uint16_t, 16>{0, 0x8000, 1, 0x8001, 0x007f, 0x0080,
            0x807f, 0x8080, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x3f81, 0x3f7f, 0x4000, 0x0002}
        : std::array<uint16_t, 16>{0, 0x8000, 1, 0x8001, 0x03ff, 0x0400,
            0x83ff, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x3c01, 0x3bff, 0x4000, 0x0002};
    for (size_t i = 0; i < dx.words; ++i) x[Guard + i] = values[i % values.size()];
  }
  for (int r = 0; r < t; ++r)
    for (int c = 0; c < n; ++c)
      before[Guard + r * stride + c] = beta == 0
          ? (outbf ? 0x7fc1 : 0x7e01) // beta zero must skip reading old NaNs.
          : pack(double((r + c) % 7 - 3) / 4, outbf);
  write(q, dx, x); write(q, dw, w); write(q, dy, before);
  call(gemm, inbf, outbf, dx.data(), dw.data(), dy.data(), t, n, k, ld, beta);
  rt->wait();
  const auto y = read(q, dy);
  for (size_t i = 0; i < y.size(); ++i) {
    const bool live = i >= Guard && i < Guard + dy.words && (i - Guard) % stride < size_t(n);
    if (!live) { need(y[i] == before[i], "GEMM16 sentinel/stride guard"); continue; }
    const size_t index = i - Guard, row = index / stride, col = index % stride;
    double want = beta == 0 ? 0 : double(beta) * unpack(before[i], outbf);
    double mag = std::abs(want);
    for (int j = 0; j < k; ++j) {
      const double term = unpack(x[Guard + row * k + j], inbf) *
                          unpack(w[Guard + col * k + j], inbf);
      want += term; mag += std::abs(term);
    }
    check_number(y[i], want, mag, k, outbf,
                 std::string(inbf ? "BF16" : "FP16") + "->" + (outbf ? "BF16" : "FP16") +
                 " kind=" + std::to_string(kind) + " row=" + std::to_string(row) +
                 " col=" + std::to_string(col));
  }
  need(read(q, dx) == x, "GEMM16 changed X/guards");
  need(read(q, dw) == w, "GEMM16 changed W/guards");
}
template <class F> void rejects(F function, const char *label) {
  try { function(); }
  catch (const std::invalid_argument &) { return; }
  throw std::runtime_error(std::string("missing prelaunch rejection: ") + label);
}
void rejection_cases(std::shared_ptr<Runtime> rt, Gemm &gemm) {
  auto &q = rt->compute();
  Buffer x(rt, 512), w(rt, 512), y(rt, 512);
  std::vector<uint16_t> marker(512 + Guard * 2, Sentinel);
  write(q, x, marker); write(q, w, marker); write(q, y, marker);
  Allocation small(rt, 32, MemoryKind::Device), host(rt, 1024, MemoryKind::Host);
  std::vector<uint16_t> ordinary(512, Sentinel);
  auto *foreign = sycl::malloc_device<uint16_t>(512, q);
  need(foreign != nullptr, "fixture foreign USM allocation");
  for (bool inbf : {false, true}) for (bool outbf : {false, true}) {
    auto invoke = [&](const uint16_t *a, const uint16_t *b, uint16_t *c,
                      int64_t t = 3, int64_t n = 16, int64_t k = 16,
                      int64_t ld = 0, float beta = 0) {
      call(gemm, inbf, outbf, a, b, c, t, n, k, ld, beta);
    };
    rejects([&] { invoke(x.data(), w.data(), y.data(), 0); }, "T=0");
    rejects([&] { invoke(x.data(), w.data(), y.data(), -1); }, "T<0");
    rejects([&] { invoke(x.data(), w.data(), y.data(), 3, 0); }, "N=0");
    rejects([&] { invoke(x.data(), w.data(), y.data(), 3, 15); }, "unsupported odd N");
    rejects([&] { invoke(x.data(), w.data(), y.data(), 3, 16, 17); }, "unsupported odd K");
    rejects([&] { invoke(x.data(), w.data(), y.data(), 3, 16, -16); }, "negative K");
    rejects([&] { invoke(x.data(), w.data(), y.data(), 3, 16, 16, 15); }, "short stride");
    rejects([&] { invoke(x.data(), w.data(), y.data(), 3, 16, 16, -1); }, "negative stride");
    rejects([&] { invoke(x.data(), w.data(), y.data(), INT64_MAX); }, "shape overflow");
    rejects([&] { invoke(x.data(), w.data(), y.data(), 3, 16, 16, INT64_MAX); }, "stride overflow");
    for (float bad : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                       std::numeric_limits<float>::quiet_NaN()})
      rejects([&] { invoke(x.data(), w.data(), y.data(), 3, 16, 16, 0, bad); }, "nonfinite beta");
    rejects([&] { invoke(nullptr, w.data(), y.data()); }, "null X");
    rejects([&] { invoke(x.data(), nullptr, y.data()); }, "null W");
    rejects([&] { invoke(x.data(), w.data(), nullptr); }, "null Y");
    auto misalign = [](uint16_t *v) { return reinterpret_cast<uint16_t *>(reinterpret_cast<uintptr_t>(v) + 1); };
    rejects([&] { invoke(misalign(x.data()), w.data(), y.data()); }, "X alignment");
    rejects([&] { invoke(x.data(), misalign(w.data()), y.data()); }, "W alignment");
    rejects([&] { invoke(x.data(), w.data(), misalign(y.data())); }, "Y alignment");
    rejects([&] { invoke(x.data(), w.data(), x.data()); }, "Y/X overlap");
    rejects([&] { invoke(x.data(), w.data(), w.data() + 8); }, "Y/W partial overlap");
    rejects([&] { invoke(small.as<uint16_t>(), w.data(), y.data()); }, "registered X extent");
    rejects([&] { invoke(x.data(), small.as<uint16_t>(), y.data()); }, "registered W extent");
    rejects([&] { invoke(x.data(), w.data(), small.as<uint16_t>()); }, "registered Y extent");
    rejects([&] { invoke(x.data(), w.data(), y.base() + y.words + Guard * 2 - 16); }, "Y interior range");
    rejects([&] { invoke(ordinary.data(), w.data(), y.data()); }, "ordinary host pointer");
    rejects([&] { invoke(host.as<uint16_t>(), w.data(), y.data()); }, "host USM");
    rejects([&] { invoke(foreign, w.data(), y.data()); }, "foreign unregistered device USM");
    Gemm uninitialized;
    rejects([&] { call(uninitialized, inbf, outbf, x.data(), w.data(), y.data(), 3, 16, 16); }, "uninitialized Gemm");
  }
  sycl::free(foreign, q);
  rt->wait();
  need(read(q, x) == marker && read(q, w) == marker && read(q, y) == marker,
       "rejected submission modified an operand");
  // Registry follows allocation address across moves, and forgets released ranges.
  const void *stale;
  {
    Allocation a(rt, 128, MemoryKind::Device);
    stale = a.data();
    Allocation b(std::move(a));
    validate_device_span(b.data(), 128, q);
    rejects([&] { validate_device_span(b.data(), 130, q); }, "registry bounds after move");
  }
  rejects([&] { validate_device_span(stale, 2, q); }, "retired allocation registry entry");
}
void smoke(std::shared_ptr<Runtime> rt, Gemm &gemm) {
  rejection_cases(rt, gemm);
  size_t cases = 0;
  for (bool inbf : {false, true}) for (bool outbf : {false, true}) {
    for (auto d : {std::array<int, 3>{1, 16, 16}, {3, 32, 32}, {17, 48, 48}})
      for (float beta : {0.f, 1.f, .5f, -.25f, .1f}) {
        run_case(rt, gemm, inbf, outbf, d[0], d[1], d[2], d[1] + 3, beta, 0);
        ++cases;
      }
    run_case(rt, gemm, inbf, outbf, 3, 16, 48, 0, 0.f, 1); ++cases;
    run_case(rt, gemm, inbf, outbf, 3, 16, 16, 19, 0.f, 2); ++cases;
  }
  std::cout << "{\"mode\":\"smoke\",\"status\":\"PASS\",\"numerical_cases\":" << cases << "}\n";
}
// One named shape per process; no hidden sweep or unlimited repetitions. Bench
// time includes validation/submission/wait and excludes uploads/readback. All
// output elements are verified, and every individual wall-time sample is kept.
void bench(std::shared_ptr<Runtime> rt, Gemm &gemm, int shape, bool inbf, bool outbf, int repeats) {
  constexpr std::array<std::array<int, 3>, 6> shapes{{
    {16,1280,2560}, {160,1280,2560}, {8192,1280,2560},
    {16,2560,640}, {160,2560,640}, {8192,2560,640}}};
  need(shape >= 0 && shape < 6 && repeats >= 1 && repeats <= 3, "bounded bench arguments");
  const auto d = shapes[shape];
  const int t = d[0], n = d[1], k = d[2];
  auto &q = rt->compute();
  Buffer dx(rt, size_t(t)*k), dw(rt, size_t(n)*k), dy(rt, size_t(t)*n);
  need(dx.memory.size()+dw.memory.size()+dy.memory.size() < 96u*1024*1024,
       "bench device tensor budget");
  auto upload = [&](Buffer &b, uint16_t value) {
    std::vector<uint16_t> host(b.words + Guard*2, Sentinel);
    std::fill(host.begin()+Guard, host.end()-Guard, value);
    write(q, b, host);
  };
  upload(dx, pack(1./32, inbf)); upload(dw, pack(1./32, inbf)); upload(dy, pack(0, outbf));
  call(gemm,inbf,outbf,dx.data(),dw.data(),dy.data(),t,n,k); rt->wait(); // warmup
  for (int sample = 0; sample < repeats; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    call(gemm,inbf,outbf,dx.data(),dw.data(),dy.data(),t,n,k); rt->wait();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout << "{\"mode\":\"bench\",\"shape\":" << shape << ",\"T\":" << t << ",\"N\":" << n
      << ",\"K\":" << k << ",\"input_bf16\":" << inbf << ",\"output_bf16\":" << outbf
      << ",\"sample\":" << sample << ",\"completion_wall_seconds\":" << seconds << "}\n";
  }
  const auto y = read(q,dy);
  for (size_t i=0;i<y.size();++i)
    need(y[i] == (i<Guard || i>=Guard+dy.words ? Sentinel : pack(double(k)/1024,outbf)),
         "bench output/guard mismatch");
  std::cout << "{\"mode\":\"bench\",\"status\":\"PASS\"}\n";
}
int integer(const char *s) {
  const std::string text(s);
  size_t used=0;
  const int v=std::stoi(text,&used);
  need(used==text.size(),"invalid integer argument");
  return v;
}
bool format(const char *s) {
  const std::string text(s);
  need(text=="fp16" || text=="bf16","format must be fp16 or bf16");
  return text=="bf16";
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(17);
    need((argc==2 && std::string(argv[1])=="--smoke") ||
         (argc==6 && std::string(argv[1])=="--bench"),
         "usage: sycl_gemm16_test --smoke | --bench SHAPE[0..5] INPUT[fp16|bf16] OUTPUT[fp16|bf16] REPEATS[1..3]");
    // All environment is frozen by Main before exec; no setenv/unsetenv here.
    auto rt=runtime_for();
    std::cerr << "GEMM16 device=" << rt->device().get_info<sycl::info::device::name>()
              << " driver=" << rt->device().get_info<sycl::info::device::driver_version>() << '\n';
    for (const auto& tuple : rt->device().get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
      std::cerr << "GEMM16 tuple A=" << int(tuple.atype) << " B=" << int(tuple.btype)
                << " C=" << int(tuple.ctype) << " D=" << int(tuple.dtype)
                << " exact=" << tuple.msize << ',' << tuple.nsize << ',' << tuple.ksize
                << " maximum=" << tuple.max_msize << ',' << tuple.max_nsize << ',' << tuple.max_ksize << '\n';
    Gemm gemm;
    std::string error;
    need(gemm.init_external(&rt->compute(),nullptr,0,nullptr,0,error),error);
    if (std::string(argv[1])=="--smoke") smoke(rt,gemm);
    else bench(rt,gemm,integer(argv[2]),format(argv[3]),format(argv[4]),integer(argv[5]));
  } catch (const std::exception &error) {
    std::cerr << "GEMM16 FAIL: " << error.what() << '\n';
    return 1;
  }
}
