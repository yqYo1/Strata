// Synthetic hot-operand descriptor diagnostic, not a model/native-format benchmark.
#include "strata/prefill/gemm.hpp"
#include "strata/sycl/runtime.hpp"
#include <oneapi/mkl/blas.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
using strata::sycl_backend::Runtime;
using Clock = std::chrono::steady_clock;
constexpr size_t Guard = 16;
constexpr float Sentinel = 12345.25f;
constexpr size_t ByteBudget = size_t(512) * 1024 * 1024;
constexpr std::array<int64_t, 8> Anchors = {1, 12, 428, 1490, 1792, 3245, 3855, 4096};
constexpr double PairAbs = 2e-4, PairRel = 5e-5;
constexpr double CpuFloor = 2e-5, FloatUnit = 0x1p-24;
void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
size_t product(int64_t a, int64_t b, size_t bytes) {
  if (a <= 0 || b <= 0 || uint64_t(a) > uint64_t(INT64_MAX) / uint64_t(b))
    throw std::invalid_argument("invalid/overflowing GEMM dimensions");
  const uint64_t n = uint64_t(a) * uint64_t(b);
  if (!bytes || n > (SIZE_MAX - 2 * Guard * bytes) / bytes)
    throw std::invalid_argument("GEMM byte overflow");
  return size_t(n);
}
struct Shape {
  int64_t t, n, k, ld;
  size_t x, w, y;
  Shape(int64_t rows, int64_t cols, int64_t inner, int64_t stride)
      : t(rows), n(cols), k(inner), ld(stride), x(0), w(0), y(0) {
    if (ld < n || n <= 0) throw std::invalid_argument("invalid GEMM leading dimension");
    x = product(t, k, 2); w = product(n, k, 2); y = product(t, ld, 4);
    // Device plus bounded host copies/references. Refuse before any allocation.
    const size_t xb = (x + 2 * Guard) * 2, wb = (w + 2 * Guard) * 2,
                 yb = (y + 2 * Guard) * 4;
    if (xb > ByteBudget / 4 || wb > ByteBudget / 4 || yb > ByteBudget / 6 ||
        2 * xb + 2 * wb + 6 * yb > ByteBudget)
      throw std::invalid_argument("synthetic GEMM exceeds 512 MiB live-byte budget");
  }
};
enum class Mode { Host, Validate, Benchmark, Help };
Mode parse(const std::vector<std::string> &args) {
  if (args.empty() || args == std::vector<std::string>{"--validate"}) return Mode::Validate;
  if (args == std::vector<std::string>{"--host-only"}) return Mode::Host;
  if (args == std::vector<std::string>{"--benchmark"}) return Mode::Benchmark;
  if (args == std::vector<std::string>{"--help"}) return Mode::Help;
  throw std::invalid_argument("usage: sycl_gemm_orientation_bench [--host-only|--validate|--benchmark|--help]");
}
void valid_beta(float beta) {
  if (!std::isfinite(beta)) throw std::invalid_argument("nonfinite beta");
}
void orientation(int arm) {
  if (arm != 0 && arm != 1) throw std::invalid_argument("orientation must be 0 or 1");
}
struct Descriptor {
  int64_t m, n, k, lda, ldb, ldc;
  bool trans_a, trans_b;
};
Descriptor descriptor(const Shape &s, int arm) {
  orientation(arm);
  return arm == 0 ? Descriptor{s.t, s.n, s.k, s.k, s.k, s.ld, false, true}
                  : Descriptor{s.n, s.t, s.k, s.k, s.k, s.ld, true, false};
}
void host_contracts() {
  auto reject = [](auto f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument &) { caught = true; }
    require(caught, "host refusal contract failed");
  };
  require(parse({}) == Mode::Validate && parse({"--validate"}) == Mode::Validate &&
          parse({"--host-only"}) == Mode::Host && parse({"--benchmark"}) == Mode::Benchmark,
          "CLI modes");
  reject([&] { parse({"--validate", "--benchmark"}); });
  reject([&] { parse({"--unknown"}); });
  for (int arm : {-1, 2, 99}) reject([&] { orientation(arm); });
  reject([&] { valid_beta(std::numeric_limits<float>::infinity()); });
  reject([&] { valid_beta(std::numeric_limits<float>::quiet_NaN()); });
  valid_beta(0.f); valid_beta(.25f);
  for (auto dims : {std::array<int64_t,4>{0,5,7,22}, {-1,5,7,22}, {3,0,7,22},
                   {3,5,0,22}, {3,5,7,4}, {INT64_MAX,5,7,22},
                   {1,INT64_MAX,1,INT64_MAX}, {1000000,2560,2560,2577}})
    reject([&] { Shape s(dims[0], dims[1], dims[2], dims[3]); (void)s; });
  for (auto ne : Anchors)
    for (int role = 0; role < 2; ++role) {
      Shape s(ne, role ? 2560 : 1280, role ? 640 : 2560, (role ? 2560 : 1280) + 17);
      const auto row = descriptor(s, 0), col = descriptor(s, 1);
      require(row.m == col.n && row.n == col.m && row.k == col.k &&
              row.lda == s.k && row.ldb == s.k && row.ldc == s.ld &&
              col.lda == s.k && col.ldb == s.k && col.ldc == s.ld &&
              !row.trans_a && row.trans_b && col.trans_a && !col.trans_b,
              "row/column descriptor dimensions");
    }
  // Exhaustive small descriptor address identities without a SYCL runtime.
  Shape s(3,5,7,22);
  for (int64_t t = 0; t < s.t; ++t)
    for (int64_t n = 0; n < s.n; ++n)
      for (int64_t k = 0; k < s.k; ++k) {
        const int64_t row_x = t * s.k + k, row_w = n * s.k + k, row_y = t * s.ld + n;
        const int64_t col_b = k + t * s.k, col_a = k + n * s.k, col_y = n + t * s.ld;
        require(row_x == col_b && row_w == col_a && row_y == col_y,
                "row/column transpose address identity");
      }
  std::cout << "PASS host-only CLI/shape/layout/byte-budget/overflow contracts; no runtime created" << std::endl;
}
uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
double decoded(uint16_t x) { return double(std::bit_cast<_Float16>(x)); }
template<class T> uint64_t fnv(const std::vector<T> &v) {
  uint64_t h = 14695981039346656037ULL;
  const auto *p = reinterpret_cast<const uint8_t *>(v.data());
  for (size_t i = 0; i < v.size() * sizeof(T); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
  return h;
}
template<class T> struct Buffer {
  std::shared_ptr<Runtime> rt;
  Allocation mem;
  Buffer(std::shared_ptr<Runtime> runtime, size_t count)
      : rt(runtime), mem(runtime, count * sizeof(T), MemoryKind::Device) {}
  T *base() { return mem.as<T>(); }
  T *data() { return base() + Guard; }
  void put(const std::vector<T> &v) {
    require(v.size() * sizeof(T) == mem.size(), "upload length");
    rt->wait(rt->compute().memcpy(base(), v.data(), mem.size()));
  }
  std::vector<T> get() {
    std::vector<T> v(mem.size() / sizeof(T));
    rt->wait(rt->compute().memcpy(v.data(), base(), mem.size()));
    return v;
  }
};
// Both descriptors use exactly the same USM addresses and in-order queue.
// A returned BLAS event is deliberately not treated as full GEMM service time.
void submit(sycl::queue &q, const Shape &s, int arm, const uint16_t *x,
            const uint16_t *w, float *y, float beta) {
  orientation(arm);
  valid_beta(beta);
  const auto *xh = reinterpret_cast<const sycl::half *>(x);
  const auto *wh = reinterpret_cast<const sycl::half *>(w);
  if (arm == 0)
    (void)oneapi::mkl::blas::row_major::gemm(
        q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
        s.t, s.n, s.k, 1.f, xh, s.k, wh, s.k, beta, y, s.ld);
  else
    (void)oneapi::mkl::blas::column_major::gemm(
        q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
        s.n, s.t, s.k, 1.f, wh, s.k, xh, s.k, beta, y, s.ld);
}
void guards(const Shape &s, const std::vector<float> &y) {
  for (size_t i = 0; i < Guard; ++i)
    require(y[i] == Sentinel && y[Guard + s.y + i] == Sentinel, "output end guard");
  for (int64_t t = 0; t < s.t; ++t)
    for (int64_t n = s.n; n < s.ld; ++n)
      require(y[Guard + size_t(t * s.ld + n)] == Sentinel, "output row padding");
}
struct PairError { double abs = 0, scaled = 0; };
PairError compare(const Shape &s, const std::vector<float> &a, const std::vector<float> &b,
                  const std::string &label) {
  guards(s, a); guards(s, b);
  PairError worst;
  for (int64_t t = 0; t < s.t; ++t)
    for (int64_t n = 0; n < s.n; ++n) {
      const size_t i = Guard + size_t(t * s.ld + n);
      const double av = a[i], bv = b[i], err = std::abs(av - bv);
      const double limit = PairAbs + PairRel * std::max(std::abs(av), std::abs(bv));
      require(std::isfinite(av) && std::isfinite(bv), label + " nonfinite output");
      worst.abs = std::max(worst.abs, err); worst.scaled = std::max(worst.scaled, err / limit);
      if (err > limit)
        throw std::runtime_error(label + " tolerance exceeded row=" + std::to_string(t) +
                                 " col=" + std::to_string(n) + " error=" + std::to_string(err) +
                                 " limit=" + std::to_string(limit));
    }
  return worst;
}
std::set<int64_t> selected(int64_t n) { return {0, std::min<int64_t>(1,n-1), n/2, std::max<int64_t>(0,n-2), n-1}; }
struct OracleError { size_t count = 0; double abs = 0, scaled = 0; };
OracleError oracle(const Shape &s, const std::vector<uint16_t> &x,
                   const std::vector<uint16_t> &w, const std::vector<float> &initial,
                   const std::vector<float> &y, float beta) {
  const auto rows = selected(s.t), cols = selected(s.n);
  const double gamma = (double(s.k) * FloatUnit) / (1 - double(s.k) * FloatUnit);
  OracleError worst;
  for (int64_t t = 0; t < s.t; ++t) {
    if (s.t > 12 && !rows.count(t)) continue;
    for (int64_t n = 0; n < s.n; ++n) {
      if (s.t > 12 && !cols.count(n)) continue;
      double dot = 0, magnitude = 0;
      for (int64_t k = 0; k < s.k; ++k) {
        const double term = decoded(x[Guard + size_t(t * s.k + k)]) *
                            decoded(w[Guard + size_t(n * s.k + k)]);
        dot += term; magnitude += std::abs(term);
      }
      const size_t i = Guard + size_t(t * s.ld + n);
      const double prior = double(beta) * initial[i], ref = dot + prior;
      // FP16 inputs decoded exactly. FP32 accumulation bound uses gamma_K
      // times sum(abs(products)); floor covers output/library rounding.
      const double limit = CpuFloor + gamma * magnitude + 4 * FloatUnit * (std::abs(ref) + std::abs(prior));
      const double err = std::abs(double(y[i]) - ref);
      require(std::isfinite(y[i]), "oracle nonfinite output");
      if (err > limit)
        throw std::runtime_error("CPU oracle tolerance exceeded row=" + std::to_string(t) +
                                 " col=" + std::to_string(n) + " error=" + std::to_string(err) +
                                 " limit=" + std::to_string(limit));
      ++worst.count; worst.abs = std::max(worst.abs, err);
      worst.scaled = std::max(worst.scaled, err / limit);
    }
  }
  return worst;
}
int64_t nanos(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}
void run(std::shared_ptr<Runtime> rt, int64_t ne, bool down, Mode mode) {
  Shape s(ne, down ? 2560 : 1280, down ? 640 : 2560, (down ? 2560 : 1280) + 17);
  const char *role = down ? "Down" : "GU";
  std::cout << "BEGIN role=" << role << " ne=" << s.t << " N=" << s.n << " K=" << s.k
            << " ld=" << s.ld << " input=deterministic_synthetic_fp16_hot" << std::endl;
  std::vector<uint16_t> x(s.x + 2 * Guard, 0x3555), w(s.w + 2 * Guard, 0x3555);
  std::vector<float> initial(s.y + 2 * Guard, Sentinel);
  for (size_t i = 0; i < s.x; ++i)
    x[Guard + i] = half(float(int(mix(i ^ 0x12345678ULL) % 2047) - 1023) / 2048.f);
  for (size_t i = 0; i < s.w; ++i)
    w[Guard + i] = half(float(int(mix(i ^ 0x87654321ULL) % 2047) - 1023) / 4096.f);
  for (int64_t t = 0; t < s.t; ++t)
    for (int64_t n = 0; n < s.n; ++n)
      initial[Guard + size_t(t * s.ld + n)] = float(int(mix(uint64_t(t * s.n + n)) % 127) - 63) / 256.f;
  std::cout << "input_hash_fnv1a64 x=" << std::hex << fnv(x) << " w=" << fnv(w)
            << " initial_y=" << fnv(initial) << std::dec << std::endl;
  Buffer<uint16_t> dx(rt, x.size()), dw(rt, w.size());
  Buffer<float> dy(rt, initial.size());
  dx.put(x); dw.put(w); dy.put(initial);
  auto &queue = rt->compute();
  if (mode == Mode::Validate) {
    strata::prefill::Gemm wrapper;
    std::string err;
    if (!wrapper.init_external(&queue, nullptr, 0, nullptr, 0, err))
      throw std::runtime_error("Gemm init: " + err);
    for (float beta : {0.f, .25f}) {
      std::array<std::vector<float>, 3> results;
      for (int arm = 0; arm < 3; ++arm) {
        std::cout << "validate beta=" << beta << " arm=" << arm
                  << " (0=row,1=column,2=current_Gemm_f16)" << std::endl;
        dy.put(initial);
        if (arm == 2) wrapper.f16(dx.data(), dw.data(), dy.data(), s.t, s.n, s.k, s.ld, beta);
        else submit(queue, s, arm, dx.data(), dw.data(), dy.data(), beta);
        rt->wait();
        results[arm] = dy.get();
        guards(s, results[arm]);
        const auto cpu = oracle(s, x, w, initial, results[arm], beta);
        std::cout << "oracle beta=" << beta << " arm=" << arm << " points=" << cpu.count
                  << " max_abs=" << cpu.abs << " max_error_over_limit=" << cpu.scaled << std::endl;
      }
      const auto ab = compare(s, results[0], results[1], "row/column");
      const auto wrap = compare(s, results[0], results[2], "row/current_wrapper");
      std::cout << "PASS role=" << role << " ne=" << ne << " beta=" << beta
                << " compared_outputs=" << s.t * s.n << " orientation_max_abs=" << ab.abs
                << " orientation_max_error_over_limit=" << ab.scaled
                << " wrapper_max_abs=" << wrap.abs << " wrapper_max_error_over_limit=" << wrap.scaled << std::endl;
    }
    require(dx.get() == x && dw.get() == w, "input/operand guard or content modified");
  } else {
    // First call in each shape/role/arm, not a claim that all are process JIT.
    for (int arm : {0, 1}) {
      dy.put(initial);
      const auto start = Clock::now();
      submit(queue, s, arm, dx.data(), dw.data(), dy.data(), 0.f);
      const auto submitted = Clock::now();
      rt->wait();
      const auto done = Clock::now();
      std::cout << "first_call_in_cell role=" << role << " ne=" << ne << " arm=" << arm
                << " host_submit_ns=" << nanos(start, submitted)
                << " drain_ns=" << nanos(submitted, done) << " batch_wall_ns=" << nanos(start, done) << std::endl;
    }
    constexpr std::array<int,4> ABBA = {0,1,1,0};
    for (int warm = 0; warm < 8; ++warm)
      submit(queue, s, ABBA[warm % 4], dx.data(), dw.data(), dy.data(), 0.f);
    rt->wait();
    const int repeats = ne <= 12 ? 16 : 4;
    for (int sample = 0; sample < 20; ++sample) {
      const int arm = ABBA[sample % 4];
      const auto start = Clock::now();
      for (int repeat = 0; repeat < repeats; ++repeat)
        submit(queue, s, arm, dx.data(), dw.data(), dy.data(), 0.f);
      const auto submitted = Clock::now();
      // Whole queue completion includes internal BLAS commands. No per-call
      // wait/event query/marker/trace/readback inside the timed batch.
      rt->wait();
      const auto done = Clock::now();
      std::cout << "sample role=" << role << " ne=" << ne << " N=" << s.n << " K=" << s.k
                << " ld=" << s.ld << " index=" << sample << " arm=" << arm << " repeats=" << repeats
                << " host_submit_ns=" << nanos(start, submitted)
                << " drain_ns=" << nanos(submitted, done) << " batch_wall_ns=" << nanos(start, done)
                << " wall_ns_per_call=" << double(nanos(start, done)) / repeats
                << " scope=host_submit_plus_whole_queue_drain synthetic_hot=1" << std::endl;
    }
    // Post-sample readback is outside all timed windows, with no CPU oracle.
    const auto final = dy.get();
    guards(s, final);
    for (int64_t t = 0; t < s.t; ++t)
      for (int64_t n = 0; n < s.n; ++n)
        require(std::isfinite(final[Guard + size_t(t * s.ld + n)]),
                "post-benchmark nonfinite output");
    std::cout << "post_benchmark_guard_finite PASS role=" << role << " ne=" << ne << std::endl;
  }
  rt->wait(); // Explicit checked completion while all operands/storage live.
  std::cout << "END role=" << role << " ne=" << ne << std::endl;
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    const Mode mode = parse(args);
    if (mode == Mode::Help) {
      std::cout << "usage: sycl_gemm_orientation_bench [--host-only|--validate|--benchmark|--help]\n"
                   "Default --validate: isolated parity, guards and double CPU oracle.\n"
                   "--benchmark: synthetic hot FP16 operands; 8 anchors x 2 roles x 20 ABBA samples.\n"
                   "Neither mode measures seven native formats or model rotation.\n";
      return 0;
    }
    host_contracts();
    if (mode == Mode::Host) return 0;
    // All planned shapes and resource bounds have been checked before runtime.
    auto rt = strata::sycl_backend::runtime_for();
    require(rt->compute().has_property<sycl::property::queue::in_order>(), "in-order queue required");
    std::cout << std::setprecision(12)
              << "device=" << rt->device().get_info<sycl::info::device::name>()
              << " driver=" << rt->device().get_info<sycl::info::device::driver_version>()
              << " mode=" << (mode == Mode::Validate ? "validate" : "benchmark")
              << " queue=in_order_runtime_compute profiling_property=1 alpha=1"
              << " arm0=row_major(nontrans,trans,T,N,K,X,K,W,K,Y,ld)"
              << " arm1=column_major(trans,nontrans,N,T,K,W,K,X,K,Y,ld)"
              << " pair_tolerance=" << PairAbs << "+" << PairRel << "*max_abs"
              << " oracle_tolerance=" << CpuFloor << "+gamma_K*sum_abs_products+4u*(abs_ref+abs_beta_C)"
              << " inputs=synthetic_hot_not_native_format_rotation" << std::endl;
    for (auto ne : Anchors)
      for (bool down : {false, true}) run(rt, ne, down, mode);
    rt->wait();
    std::cout << "DONE mode=" << (mode == Mode::Validate ? "validate" : "benchmark") << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << std::endl;
    return 1;
  }
}
