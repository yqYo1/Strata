// Authored for Strata; repository LICENSE (MIT) applies.
// Independent CPU FP64/dyadic oracles. No Torch/TLA numeric helper is used.
// R833 SOURCE-ONLY PUBLIC hook proposal; UNRUN, not a replacement for tests.
// No adapter header/call, numeric helper or FP32 GPU output tensor.
#include "strata/prefill/gemm.hpp"
#include "strata/sycl/runtime.hpp"
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
namespace p = strata::prefill;
// Test-local tags keep the own-repository oracle byte-identical. Not adapter.
namespace fixture_storage { enum class Storage { fp16, bf16 }; }
namespace x = fixture_storage;
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

// Reservations outlive their actual Device allocation owners. This bounds
// test-owned Device payload, including rejection-only raw USM. Runtime/driver
// internal memory is not controlled by this fixture.
struct DeviceBytes {
  static constexpr size_t limit = 128 * 1024 * 1024;
  inline static size_t live = 0, peak = 0;
  size_t bytes;
  explicit DeviceBytes(size_t count) : bytes(count) {
    require(count < limit && live < limit - count, "device payload reaches 128 MiB limit");
    live += count;
    peak = std::max(peak, live);
  }
  ~DeviceBytes() { live -= bytes; }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
};
size_t guarded_bytes(size_t count) {
  const size_t bytes = product(count, 2);
  require(bytes <= std::numeric_limits<size_t>::max() - 319, "guard extent overflow");
  return product((bytes + 319) / 64, 64);
}

struct Buffer {
  DeviceBytes reservation;
  s::Allocation allocation;
  std::vector<U> initial, observed;
  size_t begin, elements;
  Buffer(const std::shared_ptr<s::Runtime> &runtime, size_t count)
      : reservation(guarded_bytes(count)),
        allocation(runtime, reservation.bytes, s::MemoryKind::Device),
        initial(allocation.size() / 2, U(0x55a5)), observed(initial.size()),
        begin(size_t((64 - reinterpret_cast<uintptr_t>(allocation.data()) % 64) % 64 + 64) / 2),
        elements(count) {
    require(begin + elements + 32 <= initial.size(), "fixture guard allocation extent");
    require(reinterpret_cast<uintptr_t>(data()) % 64 == 0, "fixture base is not 64B aligned");
    s::validate_device_span(data(), product(count, 2), runtime->compute());
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

struct Counts {
  size_t dyadic = 0, edge = 0, cancellation = 0, chain = 0;
  size_t public_calls = 0, rejected = 0;
};
struct Problem {
  const U *x, *w;
  U *y;
  int64_t t, n, k, ldy;
  fixture_storage::Storage input, output;
  float beta = 0;
};
// The ONLY GEMM submissions here use these two PUBLIC methods.
void invoke(p::Gemm &gemm, const Problem &v, Counts *counts = nullptr) {
  if (v.input == x::Storage::fp16)
    gemm.f16_out16(v.x, v.w, v.y, v.t, v.n, v.k,
                   v.output == x::Storage::bf16, v.ldy, v.beta);
  else
    gemm.bf16_out16(v.x, v.w, v.y, v.t, v.n, v.k,
                    v.output == x::Storage::bf16, v.ldy, v.beta);
  if (counts) ++counts->public_calls;
}

double row_a(size_t t) { return double(int(t % 7) - 3) / 4; }
double row_b(size_t n) { return double(int(n % 9) - 4) / 4; }
double basis(size_t k) { return double(int(k % 4) - 2) / 16; }

void dyadic_case(const std::shared_ptr<s::Runtime> &runtime, p::Gemm &gemm,
                 size_t t, size_t n, size_t k, size_t padding,
                 x::Storage input, x::Storage output, Counts &counts) {
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
  auto &q = runtime->compute();
  CheckedDrain drain{*runtime, q};
  a.upload(q); b.upload(q); d.upload(q);
  // ldy=0 normalization and signed-zero beta acceptance through PUBLIC API.
  const Problem v{a.data(), b.data(), d.data(), int64_t(t), int64_t(n), int64_t(k),
                  padding ? int64_t(ld) : 0, input, output, padding ? -0.0f : 0.0f};
  invoke(gemm, v, &counts);
  a.download(q); b.download(q); d.download(q);
  drain.finish();
  a.immutable("X"); b.immutable("W"); d.canaries(t, n, ld);
  for (size_t r = 0; r < t; ++r)
    for (size_t c = 0; c < n; ++c) {
      const U expected = encode(row_a(r) * row_b(c) * norm, output);
      const U actual = d.observed[d.begin + r * ld + c];
      require(matches(actual, expected, output), "dyadic exact-RNE output mismatch");
    }
  ++counts.dyadic;
  std::cout << "dyadic PASS t=" << t << " n=" << n << " k=" << k << " ld=" << ld
            << " ldy_arg=" << (padding ? ld : 0) << " " << name(input) << "->" << name(output) << std::endl;
}

// Every row is an independently known exact FP32 sum, independent of kernel
// reduction order: all finite partial sums fit FP32 exactly, except BF16
// subnormal behavior, which is an explicit strict-gradual qualification gate.
void edges(const std::shared_ptr<s::Runtime> &runtime, p::Gemm &gemm,
           x::Storage input, x::Storage output, size_t k, Counts &counts) {
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
  auto &q = runtime->compute();
  CheckedDrain drain{*runtime, q};
  a.upload(q); b.upload(q); d.upload(q);
  invoke(gemm, {a.data(), b.data(), d.data(), int64_t(t), int64_t(n), int64_t(k),
                    int64_t(ld), input, output, 0}, &counts);
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
  ++counts.edge;
  std::cout << "edges PASS strict gradual-underflow comparison; hardware contract still requires review" << std::endl;
}

void chain(const std::shared_ptr<s::Runtime> &runtime, p::Gemm &gemm,
           x::Storage input, x::Storage output, Counts &counts) {
  constexpr size_t t = 3, n = 64, k = 32;
  Buffer a(runtime, t * k), w1(runtime, n * k), mid(runtime, t * n);
  Buffer w2(runtime, n * n), d(runtime, t * n);
  for (size_t r = 0; r < t; ++r)
    std::fill_n(a.initial.begin() + a.begin + r * k, k, encode(double(r + 1) / 8, input));
  std::fill_n(w1.initial.begin() + w1.begin, n * k, encode(1.0 / k, input));
  for (size_t c = 0; c < n; ++c)
    std::fill_n(w2.initial.begin() + w2.begin + c * n, n,
                encode(double(c % 4 + 1) / n, x::Storage::bf16));
  auto &q = runtime->compute();
  CheckedDrain drain{*runtime, q};
  a.upload(q); w1.upload(q); mid.upload(q); w2.upload(q); d.upload(q);
  invoke(gemm, {a.data(), w1.data(), mid.data(), t, n, k, n,
                    input, x::Storage::bf16, 0}, &counts);
  // No wait, download, cast kernel or FP32 tensor between the two operations.
  invoke(gemm, {mid.data(), w2.data(), d.data(), t, n, n, n,
                    x::Storage::bf16, output, 0}, &counts);
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
  ++counts.chain;
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
  DeviceBytes reservation;
  sycl::context context;
  void *pointer;
  explicit UnregisteredUSM(sycl::queue &q, size_t bytes)
      : reservation(bytes), context(q.get_context()), pointer(sycl::aligned_alloc_device(64, bytes, q)) {
    if (!pointer) throw std::bad_alloc();
  }
  ~UnregisteredUSM() noexcept { sycl::free(pointer, context); }
};

// Three nonzero terms occupy THREE separate K=32 tiles. All FP32
// partial sums are exact, including 2048+.25; narrowing the accumulated prefix
// after the second tile to either 16-bit type loses .25. No order tolerance.
void across_k_tiles(const std::shared_ptr<s::Runtime> &runtime, p::Gemm &gemm,
                    x::Storage input, x::Storage output, Counts &counts) {
  constexpr size_t t = 17, n = 192, k = 96, ld = 200;
  Buffer a(runtime, t * k), b(runtime, n * k), d(runtime, t * ld);
  std::fill_n(a.initial.begin() + a.begin, t * k, U(0));
  for (size_t r = 0; r < t; ++r) {
    const double sign = r & 1 ? -1.0 : 1.0;
    a.initial[a.begin + r * k] = encode(sign * 2048, input);
    a.initial[a.begin + r * k + 32] = encode(sign * .25, input);
    a.initial[a.begin + r * k + 64] = encode(sign * -2048, input);
  }
  for (size_t c = 0; c < n; ++c)
    std::fill_n(b.initial.begin() + b.begin + c * k, k,
                encode(c & 1 ? -1.0 : 1.0, input));
  auto &q = runtime->compute();
  CheckedDrain drain{*runtime, q};
  a.upload(q); b.upload(q); d.upload(q);
  invoke(gemm, {a.data(), b.data(), d.data(), t, n, k, ld, input, output, 0}, &counts);
  a.download(q); b.download(q); d.download(q);
  drain.finish();
  a.immutable("K-tile X"); b.immutable("K-tile W"); d.canaries(t, n, ld);
  for (size_t r = 0; r < t; ++r)
    for (size_t c = 0; c < n; ++c) {
      const double expected = ((r ^ c) & 1) ? -.25 : .25;
      require(matches(d.observed[d.begin + r * ld + c], encode(expected, output), output),
              "FP32 cancellation across three K tiles mismatch");
    }
  ++counts.cancellation;
  std::cout << "PUBLIC_KTILE_PASS " << name(input) << "->" << name(output)
            << " K=96 terms_at=0,32,64 expected=+/-0.25" << std::endl;
}

// No extra Runtime, context or queue for reject cases. Foreign-context and
// adapter-private enum/tile checks remain the original direct test's remit.
void guards(const std::shared_ptr<s::Runtime> &runtime, p::Gemm &gemm,
            x::Storage input, x::Storage output, Counts &counts) {
  constexpr size_t t = 3, n = 64, k = 32, ld = 72;
  // Extra W/Y capacity makes N=80, ldy=0 a genuinely valid own geometry;
  // rejection must come from vllm's stricter N64 contract, not short buffers.
  Buffer a(runtime, t * k), b(runtime, 128 * k), d(runtime, t * 128);
  DeviceBytes short_reservation(3 * 256);
  s::Allocation short_x(runtime, 256, s::MemoryKind::Device);
  s::Allocation short_w(runtime, 256, s::MemoryKind::Device);
  s::Allocation short_y(runtime, 256, s::MemoryKind::Device);
  s::Allocation host(runtime, n * k * 2 + 64, s::MemoryKind::Host);
  std::vector<U> ordinary(t * k + 64);
  auto &q = runtime->compute();
  UnregisteredUSM unregistered(q, t * k * 2);
  p::Gemm uninitialized;
  CheckedDrain drain{*runtime, q}; // AFTER every allocation/host owner
  a.upload(q); b.upload(q); d.upload(q);
  const Problem base{a.data(), b.data(), d.data(), t, n, k, ld, input, output, 0};
  size_t rejected = 0;
  auto reject = [&](auto change, const char *label) {
    auto v = base; change(v);
    rejects([&] { invoke(gemm, v); }, label, rejected);
    std::cout << "PUBLIC_REJECT_PASS " << name(input) << "->" << name(output)
              << " label=" << label << std::endl;
  };
  reject([](auto &v) { v.x = nullptr; }, "null X");
  reject([](auto &v) { v.w = nullptr; }, "null W");
  reject([](auto &v) { v.y = nullptr; }, "null Y");
  // One U preserves own's 2B alignment but violates vllm's 64B requirement.
  reject([](auto &v) { ++v.x; }, "vllm X base alignment");
  reject([](auto &v) { ++v.w; }, "vllm W base alignment");
  reject([](auto &v) { ++v.y; }, "vllm Y base alignment");
  reject([](auto &v) { v.y = const_cast<U *>(v.x); }, "Y overlaps X");
  reject([](auto &v) { v.y = const_cast<U *>(v.w); }, "Y overlaps W");
  reject([](auto &v) { v.x = reinterpret_cast<const U *>(UINTPTR_MAX - 63); }, "address overflow");
  reject([&](auto &v) { v.x = aligned_last_block(short_x); }, "registered short X");
  reject([&](auto &v) { v.w = aligned_last_block(short_w); }, "registered short W");
  reject([&](auto &v) { v.y = aligned_last_block(short_y); }, "registered short Y");
  // Distinguishes the vllm FULL rectangle from own's final-live-element span:
  // all 64 live outputs fit, while this declared pitch exceeds the allocation.
  reject([&](auto &v) {
    v.t = 1;
    v.y = const_cast<U *>(aligned_start(short_y));
    const size_t offset = reinterpret_cast<uintptr_t>(v.y) -
                          reinterpret_cast<uintptr_t>(short_y.data());
    const size_t remaining = short_y.size() - offset; // 193..256 bytes
    require(remaining >= n * 2, "short-Y fixture live row extent");
    v.ldy = int64_t((remaining / 16 + 1) * 8); // full bytes > remaining
  }, "vllm full Y rectangle including final padding");
  reject([&](auto &v) { v.x = aligned_start(host); }, "host USM");
  reject([&](auto &v) { v.x = static_cast<const U *>(unregistered.pointer); }, "unregistered device USM");
  reject([&](auto &v) {
    const auto raw = reinterpret_cast<uintptr_t>(ordinary.data());
    v.x = ordinary.data() + size_t((64 - raw % 64) % 64) / 2;
  }, "ordinary unregistered host pointer");
  reject([](auto &v) { v.t = 0; }, "zero T");
  reject([](auto &v) { v.t = -1; }, "negative T");
  reject([](auto &v) { v.n = 0; }, "zero N");
  reject([](auto &v) { v.n = -1; }, "negative N");
  reject([](auto &v) { v.k = 0; }, "zero K");
  reject([](auto &v) { v.k = -1; }, "negative K");
  reject([](auto &v) { v.t = INT64_MAX; }, "T surface/overflow limit");
  reject([](auto &v) { v.n = INT64_MAX - 63; }, "N surface/overflow limit");
  reject([](auto &v) { v.k = INT64_MAX - 31; }, "K surface/overflow limit");
  reject([](auto &v) { v.k = 0x800000; }, "block2D width bytes limit");
  reject([](auto &v) { v.t = 0xFFFFFF; }, "int coordinate byte-extent limit");
  reject([](auto &v) { v.n = 80; v.ldy = 0; }, "vllm N64 granularity (own supports N80)");
  reject([](auto &v) { v.n = 65; v.ldy = 0; }, "N tail");
  reject([](auto &v) { v.k = 16; }, "vllm K32 granularity (own supports K16)");
  reject([](auto &v) { v.k = 31; }, "K tail");
  reject([](auto &v) { v.ldy = 63; }, "stride below N");
  reject([](auto &v) { v.ldy = 65; }, "vllm pitch alignment");
  reject([](auto &v) { v.ldy = -1; }, "negative pitch");
  reject([](auto &v) { v.ldy = INT64_MAX; }, "pitch overflow limit");
  reject([](auto &v) { v.beta = 1; }, "vllm beta one (own supports finite beta)");
  reject([](auto &v) { v.beta = .5f; }, "vllm beta half");
  reject([](auto &v) { v.beta = std::numeric_limits<float>::quiet_NaN(); }, "NaN beta");
  reject([](auto &v) { v.beta = std::numeric_limits<float>::infinity(); }, "infinite beta");
  rejects([&] { invoke(uninitialized, base); }, "uninitialized public Gemm", rejected);
  a.download(q); b.download(q); d.download(q);
  drain.finish();
  a.immutable("guard X"); b.immutable("guard W"); d.immutable("guard Y");
  require(rejected == 40, "public rejection case count mismatch");
  counts.rejected += rejected;
  std::cout << "PUBLIC_GUARDS_PASS " << name(input) << "->" << name(output)
            << " rejected=" << rejected << " input/output/guards immutable" << std::endl;
}

std::string selected_tile() {
  // Read-only and before Runtime/provider initialization. Main selects each
  // tile in a separate native child; no process environment variable is set.
  const char *backend = std::getenv("STRATA_SYCL_PREFILL_NATIVE16_BACKEND");
  const char *tile = std::getenv("STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE");
  require(backend && std::string(backend) == "vllm", "child environment must select backend=vllm");
  require(tile && (std::string(tile) == "compact" || std::string(tile) == "wide"),
          "child environment must select VLLM_TILE=compact or wide");
  return tile;
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 4 && std::string(argv[1]) == "--device" &&
            (std::string(argv[3]) == "--smoke" || std::string(argv[3]) == "--full"),
            "usage: sycl_vllm_engine_native16_test --device ORDINAL --smoke|--full");
    size_t used = 0;
    const int ordinal = std::stoi(argv[2], &used);
    require(used == std::string(argv[2]).size() && ordinal >= 0, "invalid explicit device ordinal");
    const bool full = std::string(argv[3]) == "--full";
    const std::string tile = selected_tile();
    // One engine Runtime lookup. All allocations and public Gemm use this
    // existing compute queue/context/registry; no duplicate runtime.cpp linkage.
    auto runtime = s::runtime_for(ordinal);
    auto &q = runtime->compute();
    p::Gemm gemm;
    CheckedDrain lifetime_drain{*runtime, q}; // precedes Gemm/Runtime destruction
    std::string err;
    const bool initialized = gemm.init_external(static_cast<void *>(&q), nullptr, 0, nullptr, 0, err);
    require(initialized, "public Gemm::init_external failed: " + err);
    require(gemm.stream() == static_cast<void *>(&q), "public Gemm did not retain original queue");
    Counts counts;
    std::cout << "PUBLIC_HOOK_BEGIN suite=" << (full ? "full" : "smoke")
              << " backend=vllm tile=" << tile << " device=" << ordinal << std::endl;
    for (const auto input : {x::Storage::fp16, x::Storage::bf16})
      for (const auto output : {x::Storage::fp16, x::Storage::bf16}) {
        for (const size_t t : {1, 3, 17, 160}) {
          dyadic_case(runtime, gemm, t, 64, 32, 0, input, output, counts);
          // N192 exercises a 64-column tail in the wide N128 tile.
          dyadic_case(runtime, gemm, t, 192, 96, 8, input, output, counts);
        }
        if (full)
          dyadic_case(runtime, gemm, 8192, 128, 64, 8, input, output, counts);
        edges(runtime, gemm, input, output, 32, counts);
        edges(runtime, gemm, input, output, 64, counts);
        across_k_tiles(runtime, gemm, input, output, counts);
        chain(runtime, gemm, input, output, counts);
        guards(runtime, gemm, input, output, counts);
      }
    lifetime_drain.finish();
    require(counts.dyadic == (full ? 36u : 32u) && counts.edge == 8 &&
            counts.cancellation == 4 && counts.chain == 4 && counts.rejected == 160 &&
            counts.public_calls == (full ? 56u : 52u), "completion case counts mismatch");
    require(DeviceBytes::live == 0 && DeviceBytes::peak < DeviceBytes::limit,
            "device payload lifetime/budget mismatch");
    std::cout << "PUBLIC_HOOK_COUNTS positive_cases=" << (full ? 52 : 48)
              << " dyadic=" << counts.dyadic << " edge=" << counts.edge
              << " cancellation=" << counts.cancellation << " chain=" << counts.chain
              << " successful_public_calls=" << counts.public_calls
              << " rejected_calls=" << counts.rejected
              << " peak_test_device_bytes=" << DeviceBytes::peak << std::endl;
    // Success marker follows actual Gemm destruction at this scope's end.
  } catch (const std::exception &e) {
    std::cerr << "PUBLIC_HOOK_FAILED: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "PUBLIC_HOOK_FAILED: unknown exception" << std::endl;
    return 1;
  }
  std::cout << "PUBLIC_VLLM_NATIVE16_SUITE_PASS; correctness_component_only; "
               "loaded_image_ISA/model_quality/full262144/performance UNRUN" << std::endl;
  return 0;
}
