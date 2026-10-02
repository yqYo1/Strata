#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/runtime.hpp"
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
      throw std::runtime_error("size");
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
int main() {
  try {
    rt = sycl_backend::runtime_for();
    auto *q = &rt->compute();
    constexpr int T = 17, C = 10240, HV = 48, V = HV * 128, ST = 128 * HV * 128;
    std::vector<float> x(T * C), hist(C * 3), w(C * 4), g(T * HV), b(T * HV),
        z(T * V), gamma(128), initial(ST);
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = std::sin(float(i) * .003f);
    for (size_t i = 0; i < hist.size(); ++i)
      hist[i] = std::cos(float(i) * .011f) * .1f;
    for (size_t i = 0; i < w.size(); ++i)
      w[i] = float(int(i % 23) - 11) * .03f;
    for (size_t i = 0; i < g.size(); ++i) {
      g[i] = -.1f - float(i % 9) * .07f;
      b[i] = .1f + float(i % 7) * .09f;
    }
    for (size_t i = 0; i < z.size(); ++i)
      z[i] = std::sin(float(i) * .017f);
    for (int i = 0; i < 128; ++i)
      gamma[i] = .8f + float(i % 7) * .04f;
    for (size_t i = 0; i < initial.size(); ++i)
      initial[i] = float(int(i % 31) - 15) * .001f;
    Buffer<float> X(x.size()), H(hist.size()), H2(hist.size()), W(w.size()),
        O(x.size()), O2(x.size()), G(g.size()), B(b.size()), Z(z.size()),
        GM(128), S(ST), S2(ST), Y(T * V), Y2(T * V);
    Buffer<uint16_t> Y16(T * V), Y216(T * V);
    X.set(x);
    H.set(hist);
    H2.set(hist);
    W.set(w);
    G.set(g);
    B.set(b);
    Z.set(z);
    GM.set(gamma);
    S.set(initial);
    S2.set(initial);
    p::gdn_conv(H.data(), X.data(), W.data(), O.data(), T, 1e-6f, q);
    for (int t = 0; t < T; ++t)
      kernels::fused_gdn_conv_l2(H2.data(), X.data() + t * C, W.data(),
                                 O2.data() + t * C, C, 32, 1e-6f, q);
    auto h = O.get();
    need(h == O2.get(), "prompt convolution versus sequential decode");
    need(H.get() == H2.get(), "convolution history");
    auto tail = H.get();
    for (int c = 0; c < C; ++c)
      for (int j = 0; j < 3; ++j)
        need(tail[c * 3 + j] == x[(T - 3 + j) * C + c], "history last inputs");
    p::gdn_recurrence(S.data(), O.data(), G.data(), B.data(), Z.data(),
                      GM.data(), 1e-6f, Y.data(), Y16.data(), T, q);
    for (int t = 0; t < T; ++t)
      kernels::fused_gdn_step_norm(
          S2.data(), O.data() + t * C, O.data() + t * C + 16 * 128,
          O.data() + t * C + 32 * 128, G.data() + t * HV, B.data() + t * HV,
          Z.data() + t * V, GM.data(), 1e-6f, Y2.data() + t * V, 16, HV, q);
    auto y = Y.get();
    need(y == Y2.get(), "prompt recurrence versus sequential decode");
    auto st = S.get();
    need(st == S2.get(), "prompt recurrent state");
    auto image = Y16.get();
    for (size_t i = 0; i < y.size(); ++i)
      need(image[i] == kernels::f16_from_f32(y[i]), "recurrence FP16 image");
    H2.set(hist);
    S2.set(initial);
    p::gdn_conv(H2.data(), X.data(), W.data(), O2.data(), 5, 1e-6f, q);
    p::gdn_conv(H2.data(), X.data() + 5 * C, W.data(), O2.data() + 5 * C, T - 5,
                1e-6f, q);
    p::gdn_recurrence(S2.data(), O2.data(), G.data(), B.data(), Z.data(),
                      GM.data(), 1e-6f, Y2.data(), Y216.data(), 5, q);
    p::gdn_recurrence(S2.data(), O2.data() + 5 * C, G.data() + 5 * HV,
                      B.data() + 5 * HV, Z.data() + 5 * V, GM.data(), 1e-6f,
                      Y2.data() + 5 * V, Y216.data() + 5 * V, T - 5, q);
    need(h == O2.get() && tail == H2.get(), "split prompt convolution");
    need(y == Y2.get() && image == Y216.get() && st == S2.get(),
         "split prompt recurrence");
    std::cout << "SYCL prompt state: 17-token real GDN geometry equals decode "
                 "and 5+12 chunks bit for bit\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
