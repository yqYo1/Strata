#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/sycl/runtime.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
using namespace strata::kernels;
using namespace strata::sycl_backend;
std::shared_ptr<Runtime> rt;
template <class T> struct B {
  Allocation a;
  size_t n;
  B(size_t count) : a(rt, count * sizeof(T), MemoryKind::Device), n(count) {}
  T *p() { return static_cast<T *>(a.data()); }
  void put(const std::vector<T> &v) {
    if (v.size() != n)
      throw std::runtime_error("size");
    rt->wait(rt->compute().memcpy(p(), v.data(), n * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> v(n);
    rt->wait(rt->compute().memcpy(v.data(), p(), n * sizeof(T)));
    return v;
  }
};
void close(float x, double ref, double tol = 2e-5) {
  if (!std::isfinite(x) || std::abs(x - ref) > tol * (1 + std::abs(ref)))
    throw std::runtime_error("CPU parity: " + std::to_string(x) +
                             " != " + std::to_string(ref));
}
void exact(const std::vector<float> &a, const std::vector<float> &b,
           const char *msg) {
  if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * 4))
    throw std::runtime_error(msg);
}
void run(int hk, int hv, int T) {
  constexpr int S = 128, N = 264;
  const int C = (hk * 2 + hv) * S, V = hv * S;
  constexpr float eps = 1e-6f;
  auto stream = &rt->compute();
  std::vector<float> hist(C * 3), x(T * C), w(C * 4), init(S * V), z(T * V),
      gamma(S), gate(T * hv), beta(T * hv);
  for (size_t i = 0; i < hist.size(); ++i)
    hist[i] = std::sin(i * .17) * .1;
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = std::sin(i * .13 + .1) * .3;
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = std::cos(i * .07) * .2;
  for (size_t i = 0; i < init.size(); ++i)
    init[i] = std::sin(i * .003) * .01;
  for (size_t i = 0; i < z.size(); ++i)
    z[i] = std::cos(i * .03);
  for (int i = 0; i < S; ++i)
    gamma[i] = .9f + .01f * (i % 17);
  for (int i = 0; i < T * hv; ++i) {
    gate[i] = -.05f * (1 + i % 7);
    beta[i] = .2f + .03f * (i % 5);
  }
  B<float> dh(hist.size()), dx(x.size()), dw(w.size()), out(x.size()),
      single(x.size());
  dh.put(hist);
  dx.put(x);
  dw.put(w);
  gdn_conv_l2_multi(dh.p(), dx.p(), dw.p(), out.p(), C, 2 * hk, eps, T, stream);
  auto h = out.get();
  exact(dh.get(), hist, "verify convolution mutated history");
  for (int t = 0; t < T; ++t)
    fused_gdn_conv_l2(dh.p(), dx.p() + t * C, dw.p(), single.p() + t * C, C,
                      2 * hk, eps, stream);
  exact(single.get(), h, "conv batch differs from single");
  if (T > 1) {
    dh.put(hist);
    single.put(std::vector<float>(T * C, 42));
    gdn_conv_l2_multi(dh.p(), dx.p(), dw.p(), single.p(), C, 2 * hk, eps, 1,
                      stream);
    gdn_conv_l2_multi(dh.p(), dx.p(), dw.p(), single.p(), C, 2 * hk, eps, T - 1,
                      stream, 1);
    exact(single.get(), h, "split convolution window");
    for (int t = 0; t < T; ++t)
      fused_gdn_conv_l2(dh.p(), dx.p() + t * C, dw.p(), single.p() + t * C, C,
                        2 * hk, eps, stream);
  }
  bool rejected = false;
  try {
    fused_gdn_conv_l2(dh.p(), dx.p(), dw.p(), dx.p(), C, 2 * hk, eps, stream);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("convolution output alias accepted");
  for (int t = 0; t < T; ++t)
    for (int head = 0; head < C / S; ++head) {
      double ref[S], ss = 0;
      for (int j = 0; j < S; ++j) {
        int c = head * S + j;
        double sum = 0;
        for (int tap = 0; tap < 4; ++tap) {
          int idx = t + tap;
          float v = idx < 3 ? hist[c * 3 + idx] : x[(idx - 3) * C + c];
          sum += double(v) * w[c * 4 + tap];
        }
        ref[j] = sum / (1 + std::exp(-sum));
        ss += ref[j] * ref[j];
      }
      for (int j = 0; j < S; ++j)
        close(h[t * C + head * S + j],
              ref[j] * (head < 2 * hk ? 1 / std::sqrt(ss + eps) : 1));
    }
  B<int32_t> keep(1);
  keep.put({T});
  auto committed_hist = dh.get();
  dh.put(hist);
  gdn_conv_commit(dh.p(), dx.p(), C, keep.p(), stream);
  exact(dh.get(), committed_hist, "conv commit");
  B<float> state(init.size()), state1(init.size()), dg(gate.size()),
      db(beta.size()), dz(z.size()), dn(S), dy(T * V), dy1(T * V);
  state.put(init);
  state1.put(init);
  dg.put(gate);
  db.put(beta);
  dz.put(z);
  dn.put(gamma);
  gdn_step_norm_multi(state.p(), out.p(), C, dg.p(), db.p(), dz.p(), dn.p(),
                      eps, dy.p(), hk, hv, T, nullptr, stream);
  auto values = dy.get();
  exact(state.get(), init, "verify recurrence mutated state");
  for (int t = 0; t < T; ++t)
    fused_gdn_step_norm(state1.p(), out.p() + t * C, out.p() + t * C + hk * S,
                        out.p() + t * C + 2 * hk * S, dg.p() + t * hv,
                        db.p() + t * hv, dz.p() + t * V, dn.p(), eps,
                        dy1.p() + t * V, hk, hv, stream);
  exact(dy1.get(), values, "step batch differs from single");
  gdn_step_norm_multi(state.p(), out.p(), C, dg.p(), db.p(), dz.p(), dn.p(),
                      eps, dy.p(), hk, hv, T, keep.p(), stream);
  exact(state.get(), state1.get(), "state commit differs from single");
  std::vector<double> ref(init.begin(), init.end());
  for (int t = 0; t < T; ++t)
    for (int head = 0; head < hv; ++head) {
      double o[S], ss = 0;
      for (int col = 0; col < S; ++col) {
        double dot = 0;
        for (int row = 0; row < S; ++row)
          dot += ref[(row * hv + head) * S + col] *
                 h[t * C + hk * S + (head % hk) * S + row];
        double decay = std::exp(double(gate[t * hv + head]));
        double delta = (h[t * C + hk * S * 2 + head * S + col] - decay * dot) *
                       beta[t * hv + head];
        o[col] = 0;
        for (int row = 0; row < S; ++row) {
          auto &v = ref[(row * hv + head) * S + col];
          v = decay * v + h[t * C + hk * S + (head % hk) * S + row] * delta;
          o[col] += v * h[t * C + (head % hk) * S + row];
        }
        o[col] /= std::sqrt(128.);
        ss += o[col] * o[col];
      }
      for (int col = 0; col < S; ++col)
        close(values[t * V + head * S + col],
              o[col] / std::sqrt(ss / 128 + eps) * gamma[col] /
                  (1 + std::exp(-double(z[t * V + head * S + col]))));
    }
  auto got = state.get();
  for (size_t i = 0; i < got.size(); ++i)
    close(got[i], ref[i], 2e-6);
  // Partial acceptance and skipped output prefixes retain the rejected suffix.
  int accepted = T / 2;
  keep.put({accepted});
  state.put(init);
  dy.put(std::vector<float>(T * V, 42));
  gdn_step_norm_multi(state.p(), out.p(), C, dg.p(), db.p(), dz.p(), dn.p(),
                      eps, dy.p(), hk, hv, T, keep.p(), stream,
                      accepted ? accepted - 1 : 0);
  state1.put(init);
  for (int t = 0; t < accepted; ++t)
    fused_gdn_step_norm(state1.p(), out.p() + t * C, out.p() + t * C + hk * S,
                        out.p() + t * C + 2 * hk * S, dg.p() + t * hv,
                        db.p() + t * hv, dz.p() + t * V, dn.p(), eps,
                        dy1.p() + t * V, hk, hv, stream);
  exact(state.get(), state1.get(), "partial state commit");
  got = dy.get();
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < V; ++i)
      close(got[t * V + i], t == accepted - 1 ? values[t * V + i] : 42, 0);
  std::vector<uint16_t> wa(N * hv), wb(N * hv);
  std::vector<float> ax(T * N), dt(hv), a(hv);
  for (int i = 0; i < N * hv; ++i) {
    wa[i] = bf16_from_f32(float(std::sin(i * .13) * .1));
    wb[i] = bf16_from_f32(float(std::cos(i * .07) * .2));
  }
  for (int i = 0; i < T * N; ++i)
    ax[i] = std::sin(i * .09);
  for (int i = 0; i < hv; ++i) {
    dt[i] = i % 2 ? 22.f : -3.f;
    a[i] = -.3f;
  }
  B<uint16_t> dwa(wa.size()), dwb(wb.size());
  B<float> dax(ax.size()), ddt(hv), da(hv), sg(T * hv), sb(T * hv);
  dwa.put(wa);
  dwb.put(wb);
  dax.put(ax);
  ddt.put(dt);
  da.put(a);
  gdn_ab_multi(dax.p(), dwa.p(), dwb.p(), ddt.p(), da.p(), dg.p(), db.p(), N,
               hv, T, stream);
  for (int t = 0; t < T; ++t)
    fused_gdn_ab(dax.p() + t * N, dwa.p(), dwb.p(), ddt.p(), da.p(),
                 sg.p() + t * hv, sb.p() + t * hv, N, hv, stream);
  exact(dg.get(), sg.get(), "gate batch");
  exact(db.get(), sb.get(), "beta batch");
  auto gg = dg.get(), bb = db.get();
  for (int t = 0; t < T; ++t)
    for (int r = 0; r < hv; ++r) {
      double al = 0, be = 0;
      for (int i = 0; i < N; ++i) {
        al += double(ax[t * N + i]) * f32_from_bf16(wa[r * N + i]);
        be += double(ax[t * N + i]) * f32_from_bf16(wb[r * N + i]);
      }
      al += dt[r];
      close(gg[t * hv + r], (al > 20 ? al : std::log1p(std::exp(al))) * a[r]);
      close(bb[t * hv + r], 1 / (1 + std::exp(-be)));
    }
}
int main() {
  try {
    rt = runtime_for();
    for (int t : {1, 3, 8}) {
      run(2, 6, t);
      run(1, 1, t);
    }
    std::cout << "SYCL fused GDN: CPU convolution, gates, recurrence and "
                 "batched commit parity passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
