#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/runtime.hpp"
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata;
namespace p = strata::prefill;
std::shared_ptr<sycl_backend::Runtime> rt;
template <class T> struct Buffer {
  sycl_backend::Allocation a;
  size_t n;
  Buffer(size_t n)
      : a(rt, n * sizeof(T), sycl_backend::MemoryKind::Device), n(n) {}
  T *data() { return a.as<T>(); }
  void set(const std::vector<T> &v) {
    if (v.size() != n)
      throw std::runtime_error("upload size");
    rt->wait(rt->compute().memcpy(data(), v.data(), n * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> v(n);
    rt->wait(rt->compute().memcpy(v.data(), data(), n * sizeof(T)));
    return v;
  }
};
void need(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
void close(float a, double b, const char *s) {
  if (!std::isfinite(a) || std::abs(a - b) > 6e-6 * (1 + std::abs(b)))
    throw std::runtime_error(s);
}
uint16_t bf(float x) {
  auto u = std::bit_cast<uint32_t>(x);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float unbf(uint16_t u) { return std::bit_cast<float>(uint32_t(u) << 16); }
float sig(float x) { return 1.f / (1.f + std::exp(-x)); }
int main() {
  try {
    rt = sycl_backend::runtime_for();
    auto *s = &rt->compute();
    constexpr int T = 3, N = 2560, D = N * 4;
    std::vector<float> r(T * D), w(D), g(T * D), bo(T * N), inj(T * 7);
    for (size_t i = 0; i < r.size(); ++i) {
      r[i] = std::sin(float(i) * .017f);
      g[i] = std::cos(float(i) * .013f);
    }
    for (size_t i = 0; i < w.size(); ++i)
      w[i] = .7f + .01f * (i % 23);
    for (size_t i = 0; i < bo.size(); ++i)
      bo[i] = std::sin(float(i) * .03f) * .2f;
    for (size_t i = 0; i < inj.size(); ++i)
      inj[i] = float(int(i % 17) - 8) * .1f;
    Buffer<float> R(T * D), W(D), G(T * D), X(T * D), RS(T * 4), M(T * N),
        M2(T * N), B(T * N), I(T * 7);
    Buffer<uint16_t> hi(T * D), lo(T * D), hi2(T * D), lo2(T * D), mh(T * N),
        ml(T * N), half(T * N);
    R.set(r);
    W.set(w);
    G.set(g);
    B.set(bo);
    I.set(inj);
    p::gr_norm(R.data(), W.data(), 1e-6f, X.data(), hi.data(), T, s, lo.data());
    p::gr_norm_rs(R.data(), W.data(), 1e-6f, RS.data(), hi2.data(), T, s,
                  lo2.data());
    auto x = X.get();
    auto scale = RS.get();
    auto h = hi.get(), l = lo.get();
    need(h == hi2.get() && l == lo2.get(),
         "normalization BF16 image equivalence");
    for (int row = 0; row < T * 4; ++row) {
      double ss = 0;
      for (int d = 0; d < N; ++d)
        ss += double(r[row * N + d]) * r[row * N + d];
      double rs = 1 / std::sqrt(ss / N + 1e-6);
      close(scale[row], rs, "norm scale");
      for (int d = 0; d < N; ++d) {
        size_t i = row * N + d;
        close(x[i], r[i] * rs * w[(row % 4) * N + d], "normalized output");
        need(h[i] == bf(x[i]) && l[i] == bf(x[i] - unbf(h[i])), "hi/lo images");
      }
    }
    p::gr_mix(X.data(), G.data(), M.data(), mh.data(), T, s, half.data(),
              ml.data());
    p::gr_mix_r(R.data(), RS.data(), W.data(), G.data(), M2.data(), nullptr, T,
                s);
    auto m = M.get();
    need(m == M2.get(), "recomputed normalized mix");
    auto mi = mh.get(), mil = ml.get(), mf = half.get();
    for (int t = 0; t < T; ++t)
      for (int d = 0; d < N; ++d) {
        float v = 0;
        for (int c = 0; c < 4; ++c)
          v = std::fma(x[t * D + c * N + d], sig(g[t * D + c * N + d]), v);
        v /= 4;
        size_t i = t * N + d;
        close(m[i], v, "mixed output");
        need(mi[i] == bf(m[i]) && mil[i] == bf(m[i] - unbf(mi[i])) &&
                 mf[i] == kernels::f16_from_f32(m[i]),
             "mixed images");
      }
    p::gr_write_norm_rs(R.data(), B.data(), I.data(), 7, W.data(), 1e-6f,
                        RS.data(), hi.data(), T, s, lo.data());
    auto written = R.get();
    auto normhi = hi.get(), normlo = lo.get();
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < 4; ++c)
        for (int d = 0; d < N; ++d) {
          size_t i = t * D + c * N + d;
          close(written[i],
                std::fma(bo[t * N + d], 2 * sig(inj[t * 7 + c] / 4), r[i]),
                "residual update");
        }
    p::gr_norm_rs(R.data(), W.data(), 1e-6f, RS.data(), hi2.data(), T, s,
                  lo2.data());
    need(normhi == hi2.get() && normlo == lo2.get(), "write and next norm");
    {
      constexpr int rows = 5, cols = 259, ld = 267;
      std::vector<float> a(rows * ld, 42), weight(cols);
      for (int c = 0; c < cols; ++c)
        weight[c] = .8f + float(c % 7) * .03f;
      for (int row = 0; row < rows; ++row)
        for (int c = 0; c < cols; ++c)
          a[row * ld + c] = std::sin(float(row * cols + c) * .11f);
      Buffer<float> A(rows * ld), V(cols);
      A.set(a);
      V.set(weight);
      p::rms_rows(A.data(), V.data(), rows, cols, ld, 1e-6f, s);
      auto got = A.get();
      for (int row = 0; row < rows; ++row) {
        double ss = 0;
        for (int c = 0; c < cols; ++c)
          ss += double(a[row * ld + c]) * a[row * ld + c];
        for (int c = 0; c < ld; ++c) {
          if (c >= cols)
            need(got[row * ld + c] == 42, "RMS stride guard");
          else
            close(got[row * ld + c],
                  a[row * ld + c] / std::sqrt(ss / cols + 1e-6) * weight[c],
                  "strided RMS");
        }
      }
    }
    {
      constexpr int count = 3 * 640;
      std::vector<float> a(count), b(count), il(count * 2);
      for (int i = 0; i < count; ++i) {
        a[i] = (i % 31 - 15) * .4f;
        b[i] = (i % 19 - 9) * .3f;
        il[2 * i] = a[i];
        il[2 * i + 1] = b[i];
      }
      a[0] = il[0] = 1e5f;
      b[0] = il[1] = 1e5f;
      Buffer<float> A(count), B2(count), IL(count * 2);
      Buffer<uint16_t> Y(count), Y2(count);
      A.set(a);
      B2.set(b);
      IL.set(il);
      p::swiglu_pair(A.data(), B2.data(), Y.data(), 3, s);
      p::swiglu_interleaved(IL.data(), Y2.data(), 3, s);
      auto y = Y.get();
      need(y == Y2.get(), "interleaved SwiGLU");
      for (int i = 0; i < count; ++i) {
        float v = (a[i] / (1 + std::exp(-a[i]))) * b[i];
        v = std::clamp(v, -65504.f, 65504.f);
        close(kernels::f32_from_f16(y[i]),
              kernels::f32_from_f16(kernels::f16_from_f32(v)), "SwiGLU FP16");
      }
    }
    {
      constexpr int HV = 48;
      std::vector<float> ab(T * 96), dt(HV), a(HV);
      for (size_t i = 0; i < ab.size(); ++i)
        ab[i] = float(int(i % 101) - 50) * .6f;
      for (int i = 0; i < HV; ++i) {
        dt[i] = float(i % 7) * .2f;
        a[i] = -.1f - float(i % 5) * .1f;
      }
      Buffer<float> AB(T * 96), DT(HV), A(HV), GA(T * HV), BE(T * HV);
      AB.set(ab);
      DT.set(dt);
      A.set(a);
      p::gdn_gates(AB.data(), DT.data(), A.data(), GA.data(), BE.data(), T, s);
      auto ga = GA.get(), be = BE.get();
      for (int t = 0; t < T; ++t)
        for (int h = 0; h < HV; ++h) {
          float v = ab[t * 96 + h] + dt[h];
          close(ga[t * HV + h], (v > 20 ? v : std::log1p(std::exp(v))) * a[h],
                "GDN gate");
          close(be[t * HV + h], sig(ab[t * 96 + 48 + h]), "GDN beta");
        }
    }
    {
      std::vector<float> q(T * 24 * 512), attn(T * 24 * 256);
      for (size_t i = 0; i < q.size(); ++i)
        q[i] = std::sin(float(i) * .013f);
      for (size_t i = 0; i < attn.size(); ++i)
        attn[i] = std::cos(float(i) * .07f);
      Buffer<float> Q(q.size()), AT(attn.size()), QS(attn.size());
      Buffer<uint16_t> O(attn.size());
      Q.set(q);
      AT.set(attn);
      p::split_q(Q.data(), QS.data(), T, s);
      p::gate_attn(AT.data(), Q.data(), O.data(), T, s);
      auto split = QS.get();
      auto out = O.get();
      for (size_t i = 0; i < attn.size(); ++i) {
        size_t j =
            (i / (24 * 256)) * (24 * 512) + (i / 256 % 24) * 512 + i % 256;
        need(split[i] == q[j], "Q split");
        close(kernels::f32_from_f16(out[i]),
              kernels::f32_from_f16(
                  kernels::f16_from_f32(attn[i] * sig(q[j + 256]))),
              "attention gate");
      }
    }
    {
      constexpr int rows = 7;
      std::vector<float> d(rows * N), weight(T * 10), shared(T * N), sg(T);
      std::vector<int32_t> slot(T * 10);
      for (size_t i = 0; i < d.size(); ++i)
        d[i] = std::sin(float(i) * .031f);
      for (size_t i = 0; i < slot.size(); ++i) {
        slot[i] = int(i * 3 % rows);
        weight[i] = float(i % 9 + 1) / 71;
      }
      for (size_t i = 0; i < shared.size(); ++i)
        shared[i] = std::cos(float(i) * .011f);
      for (int t = 0; t < T; ++t)
        sg[t] = float(t) - 1;
      Buffer<float> DD(d.size()), WW(weight.size()), SH(shared.size()), SG(T),
          OUT(T * N);
      Buffer<int32_t> SL(slot.size());
      DD.set(d);
      WW.set(weight);
      SH.set(shared);
      SG.set(sg);
      SL.set(slot);
      p::moe_combine(DD.data(), SL.data(), WW.data(), SH.data(), SG.data(),
                     OUT.data(), T, s);
      auto out = OUT.get();
      for (int t = 0; t < T; ++t)
        for (int c = 0; c < N; ++c) {
          float v = 0;
          for (int k = 0; k < 10; ++k)
            v = std::fma(weight[t * 10 + k], d[slot[t * 10 + k] * N + c], v);
          close(out[t * N + c], v + shared[t * N + c] * sig(sg[t]),
                "MoE combination");
        }
    }
    std::cout << "SYCL prefill postops: normalization, mixing, residuals, "
                 "stride guards and SwiGLU passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
