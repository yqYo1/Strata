#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata::kernels;
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct B {
  T *p{};
  size_t n;
  B(size_t N) : n(N) { check(cudaMalloc(&p, n * sizeof(T))); }
  ~B() { cudaFree(p); }
  void put(const std::vector<T> &v) {
    if (v.size() != n)
      throw std::runtime_error("size");
    check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get() {
    std::vector<T> v(n);
    check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
void exact(const std::vector<float> &a, const std::vector<float> &b,
           const char *what) {
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], 4))
      throw std::runtime_error(std::string(what) + " element " +
                               std::to_string(i) + ": " + std::to_string(a[i]) +
                               " != " + std::to_string(b[i]));
}
double max_error = 0;
void near(const std::vector<float> &a, const std::vector<float> &b,
          const char *what) {
  for (size_t i = 0; i < a.size(); ++i) {
    double error = std::abs(double(a[i]) - b[i]);
    max_error = std::max(max_error, error);
    if (!std::isfinite(a[i]) || error > 3e-6 * (1 + std::abs(double(b[i]))))
      throw std::runtime_error(std::string(what) + " numerical mismatch");
  }
}
int main() {
  try {
    constexpr int N = 2560, HC = 4, LR = 320, D = N * HC, T = 3;
    constexpr float eps = 1e-6f;
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream));
    B<float> R(D * T), reference(D * T), wn(D), lo(LR * T), rs(HC * T),
        mixed(N * T), inject(HC * T), refmix(N * T), refinj(HC * T), bo(N * T),
        prev(HC * T), xn(D * T), gates(D * T), refgates(D * T);
    B<uint16_t> wd(D * LR), wu(D * LR), wi(D * HC);
    std::vector<float> residual(R.n), gamma(D), block(bo.n), ip(prev.n);
    std::vector<uint16_t> down(wd.n), up(wu.n), iw(wi.n);
    for (size_t i = 0; i < residual.size(); ++i)
      residual[i] = float(std::sin(i * .071) + .01 * (i % 17));
    for (int i = 0; i < D; ++i)
      gamma[i] = .9f + .01f * (i % 19);
    for (size_t i = 0; i < block.size(); ++i)
      block[i] = float(std::cos(i * .017) * .13);
    for (size_t i = 0; i < ip.size(); ++i)
      ip[i] = float(i) * .17f - .7f;
    for (size_t i = 0; i < down.size(); ++i) {
      down[i] = bf16_from_f32(float(.002 * std::sin(i * .031)));
      up[i] = bf16_from_f32(float(.01 * std::cos(i * .017)));
    }
    for (size_t i = 0; i < iw.size(); ++i)
      iw[i] = bf16_from_f32(float(.003 * std::sin(i * .071)));
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < HC; ++c)
        residual[t * D + c * N] = 1.f;
    for (int c = 0; c < HC; ++c)
      gamma[c * N] = 1.f;
    for (int t = 0; t < T; ++t)
      block[t * N] = 0.f;
    wn.put(gamma);
    bo.put(block);
    prev.put(ip);
    wd.put(down);
    wu.put(up);
    wi.put(iw);
    GrShapes shape{N, HC, LR};
    GrWorkspace ws;
    const size_t need = gr_workspace_init(shape, nullptr, ws);
    B<unsigned char> scratch(need);
    gr_workspace_init(shape, scratch.p, ws);
    gr_set_fp32_activations(true);
    gr_set_native_mmvf(true);
    if (!fused_gr_supported(N, HC, LR) || fused_gr_supported(N + 1, HC, LR))
      throw std::runtime_error("fused geometry gate");
    for (bool apply : {false, true})
      for (bool have_inject : {false, true}) {
        R.put(residual);
        reference.put(residual);
        inject.put(std::vector<float>(HC * T, 42));
        refinj.put(std::vector<float>(HC * T, 42));
        std::vector<float> reflo(LR * T), refrs(HC * T);
        for (int t = 0; t < T; ++t) {
          if (apply)
            native_gr_post(reference.p + t * D, bo.p + t * N, prev.p + t * HC,
                           reference.p + t * D, N, HC, stream);
          gr_read(reference.p + t * D, wn.p, wd.p, wu.p,
                  have_inject ? wi.p : nullptr, eps, shape, ws,
                  refmix.p + t * N, have_inject ? refinj.p + t * HC : nullptr,
                  stream);
          check(cudaMemcpy(reflo.data() + t * LR, ws.lo, LR * 4,
                           cudaMemcpyDeviceToHost));
          for (int c = 0; c < HC; ++c)
            check(cudaMemcpy(&refrs[t * HC + c], ws.xn + c * N, 4,
                             cudaMemcpyDeviceToHost));
        }
        FusedGrArgs args[T];
        for (int t = 0; t < T; ++t) {
          auto &a = args[t];
          a.R = R.p + t * D;
          a.R_out = R.p + t * D;
          a.apply = apply;
          a.bo_prev = bo.p + t * N;
          a.inj_prev = prev.p + t * HC;
          a.w_norm = wn.p;
          a.w_down = wd.p;
          a.w_up = wu.p;
          a.w_inject = have_inject ? wi.p : nullptr;
          a.eps = eps;
          a.lo = lo.p + t * LR;
          a.rs = rs.p + t * HC;
          a.inject_out = have_inject ? inject.p + t * HC : nullptr;
          a.mixed = mixed.p + t * N;
        }
        fused_gr_read_multi(args, T, nullptr, stream);
        exact(R.get(), reference.get(), "pending residual");
        exact(rs.get(), refrs, "normalization scale");
        near(lo.get(), reflo, "down projection");
        near(mixed.get(), refmix.get(), "mixed output");
        near(inject.get(), refinj.get(), "injection projection");

        auto batch = mixed.get();
        R.put(residual);
        for (auto &a : args)
          fused_gr_read(a, stream);
        exact(mixed.get(), batch, "single/batch GR");
        auto scales = rs.get(), updated = R.get();
        auto gpu_mixed = mixed.get(), gpu_inject = inject.get();
        auto gpu_lo = lo.get();
        bf16_gemv_fp32_mmvf_multi(lo.p, LR, wu.p, refgates.p, D,
                                  LR, D, T, stream);
        auto raw_gates = refgates.get();
        for (int t = 0; t < T; ++t) {
          std::vector<double> normalized(D), low(LR);
          for (int c = 0; c < HC; ++c) {
            double ss = 0;
            for (int d = 0; d < N; ++d) {
              double v = updated[t * D + c * N + d];
              ss += v * v;
            }
            double scale = 1 / std::sqrt(ss / N + eps);
            for (int d = 0; d < N; ++d)
              normalized[c * N + d] =
                  updated[t * D + c * N + d] * scale * gamma[c * N + d];
          }
          for (int row = 0; row < LR; ++row) {
            double dot = 0;
            for (int i = 0; i < D; ++i)
              dot += normalized[i] * f32_from_bf16(down[size_t(row) * D + i]);
            dot /= HC;
            low[row] = dot / (1 + std::exp(-dot));
          }
          for (int d = 0; d < N; ++d) {
            double value = 0;
            for (int c = 0; c < HC; ++c) {
              double gate = 0;
              for (int i = 0; i < LR; ++i)
                gate += low[i] * f32_from_bf16(up[size_t(c * N + d) * LR + i]);
              value += normalized[c * N + d] / (1 + std::exp(-gate));
            }
            value /= HC;
            if (std::abs(gpu_mixed[t * N + d] - value) >
                3e-6 * (1 + std::abs(value)))
              throw std::runtime_error("fused GR CPU mix");
          }
          if (have_inject)
            for (int c = 0; c < HC; ++c) {
              double value = 0;
              for (int i = 0; i < D; ++i)
                value += normalized[i] * f32_from_bf16(iw[c * D + i]);
              if (std::abs(gpu_inject[t * HC + c] - value) >
                  3e-6 * (1 + std::abs(value)))
                throw std::runtime_error("fused GR CPU injection");
            }
        }

        for (int t = 0; t < T; ++t)
          for (int c = 0; c < HC; ++c) {
            double sum = 0;
            for (int d = 0; d < N; ++d) {
              double x = updated[t * D + c * N + d];
              sum += x * x;
            }
            double want = 1 / std::sqrt(sum / N + eps);
            if (std::abs(scales[t * HC + c] - want) > 2e-6)
              throw std::runtime_error("RMS CPU reference");
          }
        // The workspace path must preserve every bit of the reconstructed
        // path, for both single calls and the shared multi-token scratch.
        R.put(residual);
        for (int t = 0; t < T; ++t) {
          args[t].xn = xn.p + t * D;
          fused_gr_read(args[t], stream);
        }
        exact(R.get(), updated, "workspace residual");
        exact(rs.get(), scales, "workspace normalization");
        exact(lo.get(), gpu_lo, "workspace down");
        exact(mixed.get(), gpu_mixed, "workspace mix");
        exact(inject.get(), gpu_inject, "workspace injection");
        R.put(residual);
        for (int t = 0; t < T; ++t) {
          args[t].gates = gates.p + t * D;
          fused_gr_read(args[t], stream);
        }
        exact(R.get(), updated, "up workspace residual");
        exact(rs.get(), scales, "up workspace normalization");
        exact(lo.get(), gpu_lo, "up workspace down");
        exact(mixed.get(), gpu_mixed, "up workspace mix");
        exact(inject.get(), gpu_inject, "up workspace injection");
        exact(gates.get(), raw_gates, "up workspace raw dot products");
        for (auto &a : args)
          a.xn = nullptr;
        R.put(residual);
        fused_gr_read_multi(args, T, xn.p, stream);
        exact(R.get(), updated, "batch workspace residual");
        exact(rs.get(), scales, "batch workspace normalization");
        exact(lo.get(), gpu_lo, "batch workspace down");
        exact(mixed.get(), gpu_mixed, "batch workspace mix");
        exact(inject.get(), gpu_inject, "batch workspace injection");
        exact(gates.get(), raw_gates, "batch workspace raw dot products");

        bool rejected = false;
        try {
          fused_gr_read_multi(args, T, const_cast<float *>(args[1].R), stream);
        } catch (const std::invalid_argument &) {
          rejected = true;
        }
        if (!rejected)
          throw std::runtime_error("GR batch workspace alias accepted");
        args[0].xn = xn.p;
        args[0].gates = xn.p;
        rejected = false;
        try {
          fused_gr_read(args[0], stream);
        } catch (const std::invalid_argument &) {
          rejected = true;
        }
        if (!rejected)
          throw std::runtime_error("GR projection workspace alias accepted");
        args[0].xn = nullptr;
        args[0].gates = gates.p;
        rejected = false;
        try {
          fused_gr_read(args[0], stream);
        } catch (const std::invalid_argument &) {
          rejected = true;
        }
        if (!rejected)
          throw std::runtime_error("GR up workspace without normalized input accepted");
        for (auto &a : args)
          a.gates = nullptr;
        rejected = false;
        args[0].lo = const_cast<float *>(args[0].R);
        try {
          fused_gr_read(args[0], stream);
        } catch (const std::invalid_argument &) {
          rejected = true;
        }
        if (!rejected)
          throw std::runtime_error("GR output alias accepted");
      }
    check(cudaStreamDestroy(stream));
    std::cout << "SYCL fused GR: CPU/native composed parity at 2560/4/320, "
                 "pending write, final mixer and 3-token batch passed, max "
                 "composed difference "
              << max_error << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
