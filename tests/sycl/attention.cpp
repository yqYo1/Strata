#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
template <class T> struct Buffer {
  sycl_backend::Allocation mem;
  Buffer(size_t n)
      : mem(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device) {}
  T *data() { return mem.as<T>(); }
  void put(const std::vector<T> &v) {
    runtime->wait(
        runtime->compute().memcpy(data(), v.data(), v.size() * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> v(mem.size() / sizeof(T));
    runtime->wait(runtime->compute().memcpy(v.data(), data(), mem.size()));
    return v;
  }
};
void check(bool c, const char *what) {
  if (!c)
    throw std::runtime_error(what);
}
void close(float a, double b, double tol, const char *what) {
  if (!std::isfinite(a) || std::abs(a - b) > tol * (1 + std::abs(b)))
    throw std::runtime_error(std::string(what) + ": " + std::to_string(a) +
                             " vs " + std::to_string(b));
}
uint16_t half(float x) { return std::bit_cast<uint16_t>(_Float16(x)); }
float decode(uint16_t x) { return float(std::bit_cast<_Float16>(x)); }
void attention(int heads, int kvheads, int dim, int count, bool large) {
  const int cap = count + 17, n = heads * dim;
  QsaShapes s = qsa_real_shapes();
  s.n_head = heads;
  s.n_head_kv = kvheads;
  s.head_dim = dim;
  std::vector<float> q(n);
  std::vector<uint16_t> k(cap * kvheads * dim), v(k.size());
  for (int i = 0; i < n; ++i)
    q[i] = float(std::sin(i * .13) * (large ? 30. : 1.));
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = half(float(std::cos(i * .093) * (large ? 10. : 1.)));
    v[i] = half(float(std::sin(i * .037) + .1 * (i / dim % kvheads)));
  }
  Buffer<float> dq(n), out(n + 16), weights(heads * cap + 16), scalar(n + 16);
  Buffer<uint16_t> dk(k.size()), dv(v.size());
  Buffer<int32_t> step(kStepCount);
  dq.put(q);
  dk.put(k);
  dv.put(v);
  step.put({0, count, 0, count});
  out.put(std::vector<float>(n + 16, 1234.f));
  scalar.put(std::vector<float>(n + 16, 1234.f));
  weights.put(std::vector<float>(heads * cap + 16, 1234.f));
  qsa_attend_step(dq.data(), dk.data(), dv.data(), step.data(), cap, s,
                  out.data(), weights.data(), &runtime->compute());
  const auto result = out.get(), actualw = weights.get();
  qsa_attend(dq.data(), dk.data(), dv.data(), count, s, scalar.data(), nullptr,
             &runtime->compute());
  check(scalar.get() == result, "scalar/step attention bit parity");
  double worst = 0;
  for (int h = 0; h < heads; ++h) {
    const int kvh = h / (heads / kvheads);
    std::vector<double> p(count);
    double mx = -INFINITY, z = 0;
    for (int j = 0; j < count; ++j) {
      double dot = 0;
      for (int d = 0; d < dim; ++d)
        dot +=
            double(q[h * dim + d]) * decode(k[(j * kvheads + kvh) * dim + d]);
      p[j] = dot / std::sqrt(double(dim));
      mx = std::max(mx, p[j]);
    }
    for (auto &a : p) {
      a = std::exp(a - mx);
      z += a;
    }
    for (auto &a : p)
      a /= z;
    for (int d = 0; d < dim; ++d) {
      double ref = 0;
      for (int j = 0; j < count; ++j)
        ref += p[j] * decode(v[(j * kvheads + kvh) * dim + d]);
      worst = std::max(worst, std::abs(result[h * dim + d] - ref));
      close(result[h * dim + d], ref, large ? 3e-4 : 4e-6,
            "attention double oracle");
    }
    double sum = 0;
    for (int j = 0; j < count; ++j) {
      close(actualw[h * count + j], p[j], large ? 3e-4 : 2e-6,
            "softmax weights");
      sum += actualw[h * count + j];
    }
    if (count)
      close(float(sum), 1., 2e-6, "softmax sum");
  }
  for (size_t i = n; i < result.size(); ++i)
    check(result[i] == 1234.f, "attention guard");
  for (size_t i = heads * count; i < actualw.size(); ++i)
    check(actualw[i] == 1234.f, "weight guard");
  step.put({0, 0, 0, 0});
  qsa_attend_step(dq.data(), dk.data(), dv.data(), step.data(), cap, s,
                  out.data(), nullptr, &runtime->compute());
  const auto empty = out.get();
  for (int i = 0; i < n; ++i)
    check(empty[i] == 0, "empty attention");
  std::cout << "attention heads=" << heads << " kv=" << kvheads
            << " dim=" << dim << " cells=" << count << " large=" << large
            << " max_abs=" << worst << "\n";
}
void gate_norm(int dim, int heads) {
  QsaShapes s = qsa_real_shapes();
  s.n_head = heads;
  s.head_dim = dim;
  const int n = heads * dim;
  std::vector<float> a(n), full(2 * n), gamma(dim);
  for (int i = 0; i < n; ++i)
    a[i] = float(std::sin(i * .01));
  for (int h = 0; h < heads; ++h)
    for (int d = 0; d < dim; ++d) {
      full[h * 2 * dim + d] = 999.f;
      full[h * 2 * dim + dim + d] = float((d % 11) - 5) * 20.f;
    }
  for (int d = 0; d < dim; ++d)
    gamma[d] = .8f + .01f * (d % 13);
  Buffer<float> da(n), df(2 * n), dg(dim), out(n + 16);
  Buffer<uint16_t> oh(n + 16);
  da.put(a);
  df.put(full);
  dg.put(gamma);
  out.put(std::vector<float>(n + 16, 1234.f));
  oh.put(std::vector<uint16_t>(n + 16, 0xa5a5));
  qsa_gate_apply_f32(da.data(), df.data(), s, out.data(), &runtime->compute());
  qsa_gate_apply(da.data(), df.data(), s, oh.data(), &runtime->compute());
  auto f = out.get();
  auto bits = oh.get();
  for (int i = 0; i < n; ++i) {
    double raw = full[(i / dim) * 2 * dim + dim + i % dim];
    float ref = float(double(a[i]) / (1. + std::exp(-raw)));
    close(f[i], ref, 1e-7, "double gate");
    check(bits[i] == half(ref), "FP16 gate bytes");
  }
  native_qsa_gate_apply(da.data(), df.data(), da.data(), heads, dim,
                        &runtime->compute());
  f = da.get();
  for (int i = 0; i < n; ++i) {
    float raw = full[(i / dim) * 2 * dim + dim + i % dim];
    close(f[i], a[i] / (1.f + std::exp(-raw)), 2e-7, "native gate");
  }
  da.put(a);
  native_qsa_rms_norm_weighted(da.data(), dg.data(), da.data(), dim, heads,
                               1e-6f, &runtime->compute());
  f = da.get();
  for (int h = 0; h < heads; ++h) {
    double ss = 0;
    for (int d = 0; d < dim; ++d)
      ss += double(a[h * dim + d]) * a[h * dim + d];
    double scale = 1 / std::sqrt(ss / dim + 1e-6f);
    for (int d = 0; d < dim; ++d)
      close(f[h * dim + d], a[h * dim + d] * scale * gamma[d], 2e-6,
            "native weighted RMS");
  }
  auto guards = out.get();
  for (size_t i = n; i < guards.size(); ++i)
    check(guards[i] == 1234.f && bits[i] == 0xa5a5, "gate guard");
}
} // namespace
int main() {
  try {
    runtime = sycl_backend::runtime_for();
    for (int n : {0, 1, 7, 257, 2051}) {
      attention(24, 2, 256, n, false);
      if (n)
        attention(6, 3, 68, n, false);
    }
    attention(24, 2, 256, 257, true);
    for (int dim : {66, 128, 256, 1024, 2560})
      gate_norm(dim, 3);
    Buffer<float> a(512), b(512), w(512);
    Buffer<uint16_t> k(512), v(512);
    Buffer<int32_t> step(4);
    auto reject = [](auto f) {
      bool caught = false;
      try {
        f();
      } catch (const std::invalid_argument &) {
        caught = true;
      }
      check(caught, "invalid attention accepted");
    };
    auto s = qsa_real_shapes();
    s.n_head = 2;
    s.n_head_kv = 2;
    reject([&] {
      qsa_attend(a.data(), k.data(), v.data(), 1, s, a.data(), nullptr,
                 nullptr);
    });
    reject([&] {
      qsa_attend_step(a.data(), k.data(), v.data(), nullptr, 1, s, b.data(),
                      nullptr, nullptr);
    });
    reject([&] {
      native_qsa_rms_norm_weighted(a.data(), w.data(), b.data(), 128, 2, 1e-6f,
                                   nullptr);
    });
    reject([&] {
      native_qsa_rms_norm_weighted(a.data(), w.data(), a.data() + 1, 128, 2,
                                   1e-6f, &runtime->compute());
    });
    runtime->wait();
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
