// Authored for Strata; repository LICENSE (MIT) applies.
// Independent CPU FP64/dyadic oracles. No Torch/TLA numeric helper is used.
// PROPOSED fixture, UNRUN. This does not replace the existing 68 GEMM cases.
#include "native16.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace s = strata::sycl_backend;
namespace x = strata::experimental::vllm_xe2_native16;
using U = uint16_t;
namespace {

void require(bool pass, const std::string &message) {
  if (!pass) throw std::runtime_error(message);
}
size_t product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::length_error("fixture size overflow");
  return a * b;
}
const char *name(x::Storage kind) {
  return kind == x::Storage::fp16 ? "fp16" : "bf16";
}
double decode(U raw, x::Storage kind) {
  const int f = kind == x::Storage::fp16 ? 10 : 7;
  const int e = kind == x::Storage::fp16 ? 5 : 8;
  const int bias = (1 << (e - 1)) - 1;
  const int frac = raw & ((1 << f) - 1);
  const int exp = (raw >> f) & ((1 << e) - 1);
  double v;
  if (exp == (1 << e) - 1)
    v = frac ? std::numeric_limits<double>::quiet_NaN()
             : std::numeric_limits<double>::infinity();
  else if (exp == 0)
    v = std::ldexp(double(frac), 1 - bias - f);
  else
    v = std::ldexp(double((1 << f) + frac), exp - bias - f);
  return std::copysign(v, raw & 0x8000 ? -1.0 : 1.0);
}
uint64_t ties_even(double positive) {
  const double floor = std::floor(positive);
  const auto n = uint64_t(floor);
  const double tail = positive - floor;
  return n + uint64_t(tail > .5 || (tail == .5 && (n & 1)));
}
U encode(double value, x::Storage kind) {
  const int f = kind == x::Storage::fp16 ? 10 : 7;
  const int eb = kind == x::Storage::fp16 ? 5 : 8;
  const int bias = (1 << (eb - 1)) - 1;
  const U sign = std::signbit(value) ? 0x8000 : 0;
  const U inf = U(((1 << eb) - 1) << f);
  if (std::isnan(value)) return U(sign | inf | (1 << (f - 1)));
  if (std::isinf(value)) return U(sign | inf);
  const double a = std::abs(value);
  if (a == 0) return sign;
  int exponent;
  (void)std::frexp(a, &exponent);
  --exponent;
  const int min_normal = 1 - bias;
  if (exponent < min_normal)
    return U(sign | ties_even(std::ldexp(a, f - min_normal)));
  uint64_t significand = ties_even(std::ldexp(a, f - exponent));
  if (significand == uint64_t(1 << (f + 1))) {
    significand >>= 1;
    ++exponent;
  }
  if (exponent > bias) return U(sign | inf);
  return U(sign | U((exponent + bias) << f) |
           U(significand - uint64_t(1 << f)));
}
const char *classification(U bits, x::Storage kind) {
  const double v = decode(bits, kind);
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return std::signbit(v) ? "-inf" : "+inf";
  if (v == 0) return std::signbit(v) ? "-zero" : "+zero";
  const U expmask = kind == x::Storage::fp16 ? 0x7c00 : 0x7f80;
  return (bits & expmask) == 0 ? "subnormal" : "normal";
}
bool matches(U observed, U expected, x::Storage kind) {
  if (std::isnan(decode(expected, kind))) return std::isnan(decode(observed, kind));
  // Dot products start at +0; do not assert a negative-zero accumulation sign.
  if ((expected & 0x7fff) == 0) return (observed & 0x7fff) == 0;
  return observed == expected;
}

// Declare AFTER every Allocation/host-vector owner used by a submit/copy.
// On any exception, this destructor checks completion before owners unwind.
// A drain failure exits without destroying potentially in-use USM/host vectors.
// Main supplies a process deadline for hangs; timeout is never cancellation.
struct CheckedDrain {
  s::Runtime &runtime;
  sycl::queue &queue;
  bool armed = true;
  void finish() {
    queue.wait_and_throw();
    runtime.check();
    armed = false;
  }
  ~CheckedDrain() noexcept {
    if (!armed) return;
    try { finish(); }
    catch (const std::exception &e) {
      std::cerr << "CHECKED_DRAIN_FAILED before owner unwind: " << e.what() << std::endl;
      std::_Exit(86);
    } catch (...) {
      std::cerr << "CHECKED_DRAIN_FAILED before owner unwind" << std::endl;
      std::_Exit(86);
    }
  }
};

struct Buffer {
  s::Allocation allocation;
  std::vector<U> initial, observed;
  size_t begin, elements;
  Buffer(const std::shared_ptr<s::Runtime> &runtime, size_t count)
      : allocation(runtime, product((product(count, 2) + 319) / 64, 64),
                   s::MemoryKind::Device),
        initial(allocation.size() / 2, U(0x55a5)), observed(initial.size()),
        begin(size_t((64 - reinterpret_cast<uintptr_t>(allocation.data()) % 64) % 64 + 64) / 2),
        elements(count) {
    require(begin + elements + 32 <= initial.size(), "fixture guard allocation extent");
  }
  U *data() { return allocation.as<U>() + begin; }
  void upload(sycl::queue &q) {
    q.memcpy(allocation.data(), initial.data(), allocation.size());
  }
  void download(sycl::queue &q) {
    q.memcpy(observed.data(), allocation.data(), allocation.size());
  }
  void immutable(const char *what) const {
    require(observed == initial, std::string(what) + " changed");
  }
  void canaries(size_t t, size_t n, size_t ld) const {
    for (size_t i = 0; i < observed.size(); ++i) {
      bool live = i >= begin && i - begin < t * ld && (i - begin) % ld < n;
      if (!live) require(observed[i] == initial[i], "output padding/guard changed");
    }
  }
};

double row_a(size_t t) { return double(int(t % 7) - 3) / 4; }
double row_b(size_t n) { return double(int(n % 9) - 4) / 4; }
double basis(size_t k) { return double(int(k % 4) - 2) / 16; }

void dyadic_case(const std::shared_ptr<s::Runtime> &runtime, x::Gemm &gemm,
                 size_t t, size_t n, size_t k, size_t padding,
                 x::Storage input, x::Storage output, x::Tile tile) {
  const size_t ld = n + padding;
  Buffer a(runtime, product(t, k)), b(runtime, product(n, k)), d(runtime, product(t, ld));
  // All source values and all FP32 intermediate sums here are exactly dyadic.
  double norm = 0;
  for (size_t z = 0; z < k; ++z) norm += basis(z) * basis(z);
  for (size_t r = 0; r < t; ++r)
    for (size_t z = 0; z < k; ++z)
      a.initial[a.begin + r * k + z] = encode(row_a(r) * basis(z), input);
  for (size_t c = 0; c < n; ++c)
    for (size_t z = 0; z < k; ++z)
      b.initial[b.begin + c * k + z] = encode(row_b(c) * basis(z), input);
  auto &q = gemm.queue();
  CheckedDrain drain{*runtime, q};
  a.upload(q); b.upload(q); d.upload(q);
  const x::Problem p{a.data(), b.data(), d.data(), int64_t(t), int64_t(n), int64_t(k),
                     int64_t(ld), input, output, 0};
  (void)gemm.beta0(p, tile);
  a.download(q); b.download(q); d.download(q);
  drain.finish();
  a.immutable("X"); b.immutable("W"); d.canaries(t, n, ld);
  for (size_t r = 0; r < t; ++r)
    for (size_t c = 0; c < n; ++c) {
      const U expected = encode(row_a(r) * row_b(c) * norm, output);
      const U actual = d.observed[d.begin + r * ld + c];
      require(matches(actual, expected, output), "dyadic exact-RNE output mismatch");
    }
  std::cout << "dyadic PASS t=" << t << " n=" << n << " k=" << k << " ld=" << ld
            << " " << name(input) << "->" << name(output) << " tile=" << int(tile) << std::endl;
}

// Every row is an independently known exact FP32 sum, independent of kernel
// reduction order: all finite partial sums fit FP32 exactly, except BF16
// subnormal behavior, which is an explicit strict-gradual qualification gate.
void edges(const std::shared_ptr<s::Runtime> &runtime, x::Gemm &gemm,
           x::Storage input, x::Storage output, x::Tile tile, size_t k) {
  constexpr size_t t = 17, n = 64, ld = 72;
  Buffer a(runtime, t * k), b(runtime, n * k), d(runtime, t * ld);
  std::vector<double> expected(t);
  std::fill_n(a.initial.begin() + a.begin, t * k, U(0));
  std::fill_n(b.initial.begin() + b.begin, n * k, encode(1, input));
  auto add = [&](size_t r, size_t z, double v) {
    a.initial[a.begin + r * k + z] = encode(v, input);
    expected[r] += decode(encode(v, input), input);
  };
  add(0, 0, 2048); add(0, 1, .25); add(0, 2, -2048); // FP16/BF16 acc loses .25
  const int f = output == x::Storage::fp16 ? 10 : 7;
  const double midpoint = std::ldexp(1.0, -f - 1);
  add(1, 0, 1); add(1, 1, midpoint);             // even lower mantissa
  add(2, 0, 1); add(2, 1, 3 * midpoint);         // odd lower mantissa
  add(3, 0, -1); add(3, 1, -midpoint);
  add(4, 0, -1); add(4, 1, -3 * midpoint);
  add(5, 0, std::numeric_limits<double>::infinity());
  add(6, 0, -std::numeric_limits<double>::infinity());
  add(7, 0, std::numeric_limits<double>::quiet_NaN());
  add(8, 0, -0.0);
  const int min_e = input == x::Storage::fp16 ? -14 : -126;
  const int in_f = input == x::Storage::fp16 ? 10 : 7;
  add(9, 0, std::ldexp(1.0, min_e));             // input min normal
  add(10, 0, std::ldexp(1.0, min_e - in_f));     // input min subnormal
  add(11, 0, std::ldexp(double((1 << in_f) - 1), min_e - in_f));
  add(12, 0, -std::ldexp(1.0, min_e - in_f));
  const int out_min_e = output == x::Storage::fp16 ? -14 : -126;
  // FP32 sums producing output subnormals where the input format permits it.
  add(13, 0, std::ldexp(1.0, out_min_e - f));
  add(14, 0, std::ldexp(3.0, out_min_e - f - 1)); // subnormal midpoint
  add(15, 0, input == x::Storage::fp16 ? 65504.0 : std::ldexp(1.0, 100));
  add(16, 0, -.5);
  auto &q = gemm.queue();
  CheckedDrain drain{*runtime, q};
  a.upload(q); b.upload(q); d.upload(q);
  (void)gemm.beta0({a.data(), b.data(), d.data(), int64_t(t), int64_t(n), int64_t(k),
                    int64_t(ld), input, output, 0}, tile);
  a.download(q); b.download(q); d.download(q);
  drain.finish();
  a.immutable("edge X"); b.immutable("edge W"); d.canaries(t, n, ld);
  bool good = true;
  for (size_t r = 0; r < t; ++r) {
    const U want = encode(expected[r], output);
    const U got = d.observed[d.begin + r * ld];
    std::cout << "edge row=" << r << " " << name(input) << "->" << name(output)
              << " expected=0x" << std::hex << want << " got=0x" << got << std::dec
              << " expected_class=" << classification(want, output)
              << " observed_class=" << classification(got, output) << std::endl;
    for (size_t c = 0; c < n; ++c)
      good &= matches(d.observed[d.begin + r * ld + c], want, output);
  }
  require(good, "edge mismatch: strict gradual-underflow target; hardware FTZ contract UNESTABLISHED");
  std::cout << "edges PASS strict gradual-underflow comparison; hardware contract still requires review" << std::endl;
}

void chain(const std::shared_ptr<s::Runtime> &runtime, x::Gemm &gemm,
           x::Storage input, x::Storage output, x::Tile tile) {
  constexpr size_t t = 3, n = 64, k = 32;
  Buffer a(runtime, t * k), w1(runtime, n * k), mid(runtime, t * n);
  Buffer w2(runtime, n * n), d(runtime, t * n);
  for (size_t r = 0; r < t; ++r)
    std::fill_n(a.initial.begin() + a.begin + r * k, k, encode(double(r + 1) / 8, input));
  std::fill_n(w1.initial.begin() + w1.begin, n * k, encode(1.0 / k, input));
  for (size_t c = 0; c < n; ++c)
    std::fill_n(w2.initial.begin() + w2.begin + c * n, n,
                encode(double(c % 4 + 1) / n, x::Storage::bf16));
  auto &q = gemm.queue();
  CheckedDrain drain{*runtime, q};
  a.upload(q); w1.upload(q); mid.upload(q); w2.upload(q); d.upload(q);
  (void)gemm.beta0({a.data(), w1.data(), mid.data(), t, n, k, n,
                    input, x::Storage::bf16, 0}, tile);
  // No wait, download, cast kernel or FP32 tensor between the two operations.
  (void)gemm.beta0({mid.data(), w2.data(), d.data(), t, n, n, n,
                    x::Storage::bf16, output, 0}, tile);
  a.download(q); w1.download(q); mid.download(q); w2.download(q); d.download(q);
  drain.finish();
  a.immutable("chain X"); w1.immutable("chain W1"); w2.immutable("chain W2");
  mid.canaries(t, n, n); d.canaries(t, n, n);
  for (size_t r = 0; r < t; ++r)
    for (size_t c = 0; c < n; ++c) {
      require(mid.observed[mid.begin + r * n + c] == encode(double(r + 1) / 8, x::Storage::bf16),
              "chain BF16 intermediate mismatch");
      require(d.observed[d.begin + r * n + c] == encode(double(r + 1) * double(c % 4 + 1) / 8, output),
              "chain final mismatch");
    }
  std::cout << "chain PASS " << name(input) << "->bf16->" << name(output)
            << " direct same-queue device intermediate" << std::endl;
}

template <class F> void rejects(F f, const char *label, size_t &count) {
  try { f(); }
  catch (const std::invalid_argument &) { ++count; return; }
  throw std::runtime_error(std::string("guard failed to reject: ") + label);
}

const U *aligned_start(const s::Allocation &allocation) {
  const auto raw = reinterpret_cast<uintptr_t>(allocation.data());
  const size_t offset = size_t((64 - raw % 64) % 64);
  return static_cast<const U *>(allocation.at(offset, 1));
}
U *aligned_last_block(const s::Allocation &allocation) {
  const auto raw = reinterpret_cast<uintptr_t>(allocation.data());
  const uintptr_t aligned = (raw + allocation.size() - 1) & ~uintptr_t(63);
  return static_cast<U *>(allocation.at(size_t(aligned - raw), 1));
}
struct UnregisteredUSM {
  sycl::context context;
  void *pointer;
  explicit UnregisteredUSM(sycl::queue &q, size_t bytes)
      : context(q.get_context()), pointer(sycl::aligned_alloc_device(64, bytes, q)) {
    if (!pointer) throw std::bad_alloc();
  }
  ~UnregisteredUSM() noexcept { sycl::free(pointer, context); }
};

void guards(const std::shared_ptr<s::Runtime> &runtime, x::Gemm &gemm, int ordinal) {
  constexpr size_t t = 3, n = 64, k = 32, ld = 72;
  Buffer a(runtime, t * k), b(runtime, n * k), d(runtime, t * ld);
  s::Allocation short_x(runtime, t * k * 2 - 2, s::MemoryKind::Device);
  s::Allocation short_d(runtime, ((t - 1) * ld + n) * 2, s::MemoryKind::Device);
  s::Allocation host(runtime, t * k * 2 + 64, s::MemoryKind::Host);
  // Independent Runtime creates an explicit second context solely for a reject.
  // No work is submitted on it. It is not hidden inside the kernel/adapter.
  auto other = std::make_shared<s::Runtime>(ordinal);
  s::Allocation foreign(other, t * k * 2 + 64, s::MemoryKind::Device);
  std::vector<U> ordinary(t * k + 64);
  auto &q = gemm.queue();
  UnregisteredUSM unregistered(q, t * k * 2);
  CheckedDrain drain{*runtime, q};
  a.upload(q); b.upload(q); d.upload(q);
  const x::Problem base{a.data(), b.data(), d.data(), t, n, k, ld,
                         x::Storage::fp16, x::Storage::bf16, 0};
  size_t rejected = 0;
  auto reject = [&](auto change, const char *label) {
    auto p = base; change(p);
    rejects([&] { (void)gemm.beta0(p); }, label, rejected);
  };
  reject([](auto &p) { p.x = nullptr; }, "null");
  reject([](auto &p) { ++p.x; }, "base alignment");
  reject([](auto &p) { p.y = const_cast<U *>(p.x); }, "output/input overlap");
  reject([](auto &p) { p.x = reinterpret_cast<const U *>(UINTPTR_MAX - 63); }, "address overflow");
  reject([&](auto &p) { p.x = aligned_last_block(short_x); }, "registered short input");
  reject([&](auto &p) { p.y = aligned_last_block(short_d); }, "registered short output");
  reject([&](auto &p) { p.x = aligned_start(foreign); }, "wrong registered context");
  reject([&](auto &p) { p.x = aligned_start(host); }, "host USM");
  reject([&](auto &p) { p.x = static_cast<const U *>(unregistered.pointer); }, "unregistered device USM");
  reject([&](auto &p) {
    const auto raw = reinterpret_cast<uintptr_t>(ordinary.data());
    p.x = ordinary.data() + size_t((64 - raw % 64) % 64) / 2;
  }, "ordinary unregistered host pointer");
  reject([](auto &p) { p.t = 0; }, "zero M");
  reject([](auto &p) { p.t = -1; }, "negative M");
  reject([](auto &p) { p.t = INT64_MAX; }, "shape/overflow limit");
  reject([](auto &p) { p.n = INT64_MAX - 63; }, "N overflow limit");
  reject([](auto &p) { p.n = 65; }, "unsupported N tail");
  reject([](auto &p) { p.k = 31; }, "unsupported K tail");
  reject([](auto &p) { p.ldy = 63; }, "stride below N");
  reject([](auto &p) { p.ldy = 65; }, "pitch alignment");
  reject([](auto &p) { p.ldy = -1; }, "negative pitch");
  reject([](auto &p) { p.beta = 1; }, "beta");
  reject([](auto &p) { p.beta = std::numeric_limits<float>::quiet_NaN(); }, "NaN beta");
  reject([](auto &p) { p.beta = std::numeric_limits<float>::infinity(); }, "infinite beta");
  reject([](auto &p) { p.input = static_cast<x::Storage>(91); }, "input enum");
  reject([](auto &p) { p.output = static_cast<x::Storage>(91); }, "output enum");
  rejects([&] { (void)gemm.beta0(base, static_cast<x::Tile>(91)); }, "tile enum", rejected);
  // A separately supplied out-of-order queue is rejected at construction.
  // Constructing an explicitly marked fixture queue does not add an adapter queue.
  {
    sycl::queue out_of_order(runtime->context(), runtime->device());
    rejects([&] { x::Gemm rejected_queue(*runtime, out_of_order); }, "out-of-order queue", rejected);
  }
  a.download(q); b.download(q); d.download(q);
  drain.finish();
  a.immutable("guard X"); b.immutable("guard W"); d.immutable("guard Y");
  std::cout << "guards PASS rejected=" << rejected << " output/inputs immutable" << std::endl;
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 5 && std::string(argv[1]) == "--device" && std::string(argv[3]) == "--suite",
            "usage: numerics_chain --device ORDINAL --suite smoke|full|edges|chain|guards");
    size_t used = 0;
    const int ordinal = std::stoi(argv[2], &used);
    require(used == std::string(argv[2]).size() && ordinal >= 0, "invalid explicit device ordinal");
    const std::string suite = argv[4];
    require(suite == "smoke" || suite == "full" || suite == "edges" || suite == "chain" || suite == "guards",
            "invalid suite");
    auto runtime = std::make_shared<s::Runtime>(ordinal);
    x::Gemm gemm(*runtime, runtime->compute());
    if (suite == "guards") { guards(runtime, gemm, ordinal); return 0; }
    for (const auto tile : {x::Tile::compact_16x64, x::Tile::wide_128x128})
      for (const auto input : {x::Storage::fp16, x::Storage::bf16})
        for (const auto output : {x::Storage::fp16, x::Storage::bf16}) {
          if (suite == "edges") {
            edges(runtime, gemm, input, output, tile, 32);
            edges(runtime, gemm, input, output, tile, 64);
          } else if (suite == "chain") {
            chain(runtime, gemm, input, output, tile);
          } else if (suite == "smoke") {
            for (const size_t t : {1, 3, 17, 160})
              dyadic_case(runtime, gemm, t, 128, 64, 8, input, output, tile);
          } else {
            for (const size_t t : {1, 3, 17, 160, 8192}) {
              dyadic_case(runtime, gemm, t, 1280, 2560, 0, input, output, tile);
              dyadic_case(runtime, gemm, t, 2560, 640, 8, input, output, tile);
              dyadic_case(runtime, gemm, t, 640, 1280, 16, input, output, tile);
            }
          }
        }
    std::cout << "COMPONENT_SUITE_PASS only; DPAS_BINARY_PROOF/model_quality/full262144/performance UNRUN" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FIXTURE_FAILED: " << e.what() << std::endl;
    return 1;
  }
}
