#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> rt;
template <class T> struct Buffer {
  sycl_backend::Allocation mem;
  explicit Buffer(size_t count)
      : mem(rt, count * sizeof(T), sycl_backend::MemoryKind::Device) {}
  T *data() { return mem.as<T>(); }
  void put(const std::vector<T> &v) {
    rt->wait(rt->compute().memcpy(data(), v.data(), v.size() * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> v(mem.size() / sizeof(T));
    rt->wait(rt->compute().memcpy(v.data(), data(), mem.size()));
    return v;
  }
};
void check(bool c, const char *msg) {
  if (!c)
    throw std::runtime_error(msg);
}
uint16_t half(float v) { return std::bit_cast<uint16_t>(_Float16(v)); }
float unhalf(uint16_t b) { return float(std::bit_cast<_Float16>(b)); }
uint16_t bf(float v) {
  auto b = std::bit_cast<uint32_t>(v);
  return uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
float unbf(uint16_t v) { return std::bit_cast<float>(uint32_t(v) << 16); }
void near(float a, double b, const char *msg, double tol = 1e-5) {
  if (!std::isfinite(a) || std::abs(double(a) - b) > tol * (1 + std::abs(b)))
    throw std::runtime_error(std::string(msg) + ": " + std::to_string(a) +
                             " vs " + std::to_string(b));
}
std::vector<float> quantize(const std::vector<float> &x, bool native) {
  std::vector<float> out(x.size());
  for (size_t b = 0; b < x.size(); b += 32) {
    float a = 0;
    for (int j = 0; j < 32; ++j)
      a = std::max(a, std::abs(x[b + j]));
    const float d = a / 127.f, inv = d ? 1.f / d : 0;
    for (int j = 0; j < 32; ++j)
      out[b + j] = (native ? std::nearbyint(x[b + j] * inv)
                           : std::round(x[b + j] * inv)) *
                   unhalf(half(d));
  }
  return out;
}
void canonical() {
  constexpr int N = 256, M = 7;
  Buffer<uint16_t> xh(N);
  Buffer<uint8_t> x0(N / 32 * 34), xk(N / 256 * 292);
  std::vector<uint16_t> hv(N);
  std::vector<uint8_t> q0(N / 32 * 34), qk(292);
  std::vector<float> reference[3] = {
      std::vector<float>(N), std::vector<float>(N), std::vector<float>(N)};
  const float kd = .013f;
  std::memcpy(qk.data(), &kd, 4);
  for (int i = 0; i < N; ++i) {
    hv[i] = half(float(std::sin(i * .313)));
    reference[0][i] = unhalf(hv[i]);
    const int8_t q = int8_t((i * 17) % 255 - 127);
    const uint16_t d = half(.01f + float(i / 32) * .002f);
    q0[(i / 32) * 34] = uint8_t(d);
    q0[(i / 32) * 34 + 1] = uint8_t(d >> 8);
    q0[(i / 32) * 34 + 2 + i % 32] = uint8_t(q);
    qk[4 + i] = uint8_t(q);
    reference[1][i] = float(q) * unhalf(d);
    reference[2][i] = float(q) * kd;
  }
  xh.put(hv);
  x0.put(q0);
  xk.put(qk);
  int cases = 0;
  for (int fmt = 0; fmt < 4; ++fmt)
    for (bool offset : {false, true}) {
      SForm f;
      f.code_bits = fmt == 0 ? 2 : fmt == 2 ? 8 : 4;
      f.code_bias = fmt == 0 ? -1 : fmt == 2 ? -128 : -8;
      f.group_elems = fmt == 0 ? 64 : 16;
      f.has_offset = offset;
      f.codebook = fmt == 3 ? Codebook::Iq4Nl : Codebook::Affine;
      const int table[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                             1,    13,   25,  38,  53,  69,  89,  113};
      const int groups = N / f.group_elems, per = 8 / f.code_bits;
      std::vector<uint8_t> codes(M * N / per);
      std::vector<float> scales(M * groups), offsets(M * groups);
      for (size_t i = 0; i < codes.size(); ++i)
        codes[i] = uint8_t(i * 37 + 83);
      for (int i = 0; i < M * groups; ++i) {
        scales[i] = .001f * (1 + i % 7);
        offsets[i] = float((i % 9) - 4) * .002f;
      }
      Buffer<uint8_t> c(codes.size());
      Buffer<float> s(scales.size()), off(offsets.size()), out(M + 4);
      c.put(codes);
      s.put(scales);
      off.put(offsets);
      for (int kind = 0; kind < 3; ++kind) {
        out.put(std::vector<float>(M + 4, -777));
        if (kind == 0)
          s_gemv_split(xh.data(), c.data(), s.data(), off.data(), out.data(), N,
                       M, f, 64);
        else if (kind == 1)
          s_gemv_q8_0_split(x0.data(), c.data(), s.data(), off.data(),
                            out.data(), N, M, f, nullptr);
        else
          s_gemv_q8k_split(xk.data(), c.data(), s.data(), off.data(),
                           out.data(), N, M, f, nullptr);
        const auto actual = out.get();
        for (int row = 0; row < M; ++row) {
          double sum = 0, magnitude = 0;
          for (int j = 0; j < N; ++j) {
            const int code = (codes[row * (N / per) + j / per] >>
                              ((j % per) * f.code_bits)) &
                             ((1 << f.code_bits) - 1);
            const int group = row * groups + j / f.group_elems;
            const float w =
                std::fma(float(fmt == 3 ? table[code] : code + f.code_bias),
                         scales[group], offset ? offsets[group] : 0.f);
            const double term = double(w) * reference[kind][j];
            sum += term;
            magnitude += std::abs(term);
          }
          check(std::abs(actual[row] - sum) <= 5e-7 * (1 + magnitude),
                "canonical GEMV reference");
        }
        check(actual.back() == -777, "canonical output canary");
        ++cases;
        if (fmt == 0 && !offset && kind == 1) {
          s2_gemv_q8(x0.data(), c.data(), s.data(), out.data(), N, M, 8,
                     nullptr);
          const auto specialized = out.get();
          for (int i = 0; i < M; ++i)
            near(specialized[i], actual[i], "S2 specialized");
        }
        if (fmt == 0 && !offset && kind == 0) {
          for (bool stage : {false, true}) {
            s2_gemv_fast(xh.data(), c.data(), s.data(), out.data(), N, M, 64,
                         stage);
            check(out.get() == actual, "S2 staging changed result");
          }
        }
      }
    }
  std::cout << cases << " canonical GEMV format/activation cases passed\n";
}
struct Matrix {
  int ni, no;
  std::vector<float> values, scale;
  std::vector<uint8_t> native, codes;
  Buffer<uint8_t> nd, cd;
  Buffer<float> sd;
  Matrix(int n, int m, int salt)
      : ni(n), no(m), values(size_t(n) * m), scale(size_t(m) * (n / 32)),
        native(size_t(m) * (n / 32) * 34), codes(size_t(n) * m),
        nd(native.size()), cd(codes.size()), sd(scale.size()) {
    for (int row = 0; row < m; ++row)
      for (int b = 0; b < n / 32; ++b) {
        const float d = unhalf(half(.01f + .001f * ((b + row + salt) % 7)));
        const uint16_t bits = half(d);
        const size_t base = (size_t(row) * (n / 32) + b) * 34;
        native[base] = uint8_t(bits);
        native[base + 1] = uint8_t(bits >> 8);
        scale[row * (n / 32) + b] = d;
        for (int j = 0; j < 32; ++j) {
          const int q = ((row * 11 + b * 13 + j * 7 + salt) % 15) - 7;
          native[base + 2 + j] = uint8_t(int8_t(q));
          codes[size_t(row) * n + b * 32 + j] = uint8_t(q + 128);
          values[size_t(row) * n + b * 32 + j] = q * d;
        }
      }
    nd.put(native);
    cd.put(codes);
    sd.put(scale);
  }
  std::vector<float> dot(const std::vector<float> &x) const {
    std::vector<float> out(no);
    for (int row = 0; row < no; ++row) {
      double sum = 0;
      for (int i = 0; i < ni; ++i)
        sum += double(x[i]) * values[size_t(row) * ni + i];
      out[row] = float(sum);
    }
    return out;
  }
};
void shared() {
  constexpr int N = 256, F = 128, T = 3;
  Matrix gw(N, F, 1), uw(N, F, 5), dw(F, N, 9);
  Buffer<float> x(T * N), gate(T * F), up(T * F), g(T), out(T * N), one(N);
  Buffer<uint16_t> xb(T * N), wi(N);
  Buffer<uint8_t> q0(N / 32 * 34), qk(N / 256 * 292),
      scratch(shared_expert_scratch_bytes(F)), q81(native_q8_1_bytes(N) * T);
  std::vector<float> inputs(T * N);
  std::vector<uint16_t> bits(T * N), w(N);
  for (int i = 0; i < T * N; ++i) {
    inputs[i] = float(std::sin(i * .123) * .8);
    bits[i] = bf(inputs[i]);
  }
  for (int i = 0; i < N; ++i)
    w[i] = bf(float(std::cos(i * .71) * .02));
  x.put(inputs);
  xb.put(bits);
  wi.put(w);
  quantize_q8_0(x.data(), q0.data(), N, nullptr);
  quantize_q8_K(x.data(), qk.data(), N, nullptr);
  SForm form;
  form.code_bits = 8;
  form.code_bias = -128;
  form.group_elems = 32;
  auto *q = &rt->compute();
  for (bool ng : {false, true}) {
    shared_expert_set_native_bf16(ng);
    for (int mask = 0; mask < 8; ++mask) {
      NativeSharedWeights nw{8,
                             8,
                             8,
                             mask & 1 ? gw.nd.data() : nullptr,
                             mask & 2 ? uw.nd.data() : nullptr,
                             mask & 4 ? dw.nd.data() : nullptr,
                             q81.data()};
      shared_expert(q0.data(), qk.data(), xb.data(), form, gw.cd.data(),
                    gw.sd.data(), nullptr, form, uw.cd.data(), uw.sd.data(),
                    nullptr, form, dw.cd.data(), dw.sd.data(), nullptr,
                    wi.data(), reinterpret_cast<float *>(scratch.data()),
                    one.data(), N, F, 8, q, x.data(), &nw);
      const std::vector<float> in(inputs.begin(), inputs.begin() + N);
      const auto a = gw.dot(quantize(in, mask & 1)),
                 b = uw.dot(quantize(in, mask & 2));
      std::vector<float> h(F);
      for (int i = 0; i < F; ++i)
        h[i] = (mask ? a[i] / (1.f + std::exp(-a[i]))
                     : float(double(a[i]) / (1. + std::exp(-double(a[i]))))) *
               b[i];
      const auto ref = dw.dot(quantize(h, mask & 4));
      double dot = 0;
      for (int i = 0; i < N; ++i)
        dot += double(ng ? in[i] : unbf(bits[i])) * unbf(w[i]);
      const float factor = float(1. / (1. + std::exp(-dot)));
      const auto got = one.get();
      for (int i = 0; i < N; ++i)
        near(got[i], ref[i] * factor, "shared expert CPU reference", 3e-5);
    }
    NativeSharedWeights nw{
        8, 8, 8, gw.nd.data(), uw.nd.data(), dw.nd.data(), q81.data()};
    shared_expert_multi(T, x.data(), xb.data(), nw, wi.data(), gate.data(),
                        up.data(), g.data(), out.data(), N, F, q);
    const auto multi = out.get();
    for (int t = 0; t < T; ++t) {
      shared_expert(nullptr, nullptr, xb.data() + t * N, form, nullptr, nullptr,
                    nullptr, form, nullptr, nullptr, nullptr, form, nullptr,
                    nullptr, nullptr, wi.data(),
                    reinterpret_cast<float *>(scratch.data()), one.data(), N, F,
                    8, q, x.data() + t * N, &nw);
      const auto single = one.get();
      check(std::equal(single.begin(), single.end(), multi.begin() + t * N),
            "shared multi differs from single");
    }
  }
  std::cout << "16 partial/native shared expert cases and 2 batch comparisons "
               "passed\n";
}
void reduction() {
  constexpr int N = 259, T = 3;
  auto *q = &rt->compute();
  for (int K : {1, 2, 10, 15})
    for (bool shared : {false, true}) {
      Buffer<float> parts(T * K * N), weights(T * K), sh(T * N), out(T * N + 1),
          single(N), generic(N);
      std::vector<float> p(T * K * N), w(T * K), s(T * N);
      for (size_t i = 0; i < p.size(); ++i)
        p[i] = float(std::sin(i * .713) * (i % 19 + 1));
      for (size_t i = 0; i < w.size(); ++i)
        w[i] = float(std::cos(i * .375));
      for (size_t i = 0; i < s.size(); ++i)
        s[i] = float(std::sin(i * .11));
      parts.put(p);
      weights.put(w);
      sh.put(s);
      out.put(std::vector<float>(T * N + 1, -999));
      native_moe_combine_multi(parts.data(), weights.data(),
                               shared ? sh.data() : nullptr, out.data(), N, K,
                               T, q);
      const auto got = out.get();
      check(got.back() == -999, "MoE combine canary");
      for (int t = 0; t < T; ++t) {
        native_moe_combine(parts.data() + t * K * N, weights.data() + t * K,
                           shared ? sh.data() + t * N : nullptr, single.data(),
                           N, K, q);
        const auto one = single.get();
        check(std::equal(one.begin(), one.end(), got.begin() + t * N),
              "MoE combine multi");
        moe_combine(parts.data() + t * K * N, weights.data() + t * K,
                    shared ? sh.data() + t * N : nullptr, generic.data(), N, K,
                    q);
        const auto legacy = generic.get();
        for (int d = 0; d < N; ++d) {
          float f = p[t * K * N + d] * w[t * K];
          double sum = double(p[t * K * N + d]) * w[t * K];
          for (int e = 1; e < K; ++e) {
            f = std::fma(p[(t * K + e) * N + d], w[t * K + e], f);
            sum += double(p[(t * K + e) * N + d]) * w[t * K + e];
          }
          if (shared) {
            f += s[t * N + d];
            sum += s[t * N + d];
          }
          check(f == got[t * N + d], "native combine ordered FMA");
          near(legacy[d], sum, "legacy combine", 1e-7);
        }
      }
    }
  std::cout
      << "8 MoE reductions passed CPU ordered FMA and double references\n";
}
} // namespace
int main() {
  try {
    rt = sycl_backend::runtime_for();
    canonical();
    shared();
    reduction();
    rt->wait();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
