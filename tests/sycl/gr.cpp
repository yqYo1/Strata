#include "strata/kernels/gr.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/sycl/runtime.hpp"
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
template <typename T> struct Buffer {
  sycl_backend::Allocation storage;
  size_t count;
  explicit Buffer(size_t n)
      : storage(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device),
        count(n) {}
  T *data() { return storage.as<T>(); }
  void upload(const std::vector<T> &v) {
    if (v.size() != count)
      throw std::logic_error("upload size");
    runtime->wait(runtime->compute().memcpy(data(), v.data(), storage.size()));
  }
  std::vector<T> read() {
    std::vector<T> v(count);
    runtime->wait(runtime->compute().memcpy(v.data(), data(), storage.size()));
    return v;
  }
};

void check(bool condition, const char *label) {
  if (!condition)
    throw std::runtime_error(label);
}
void close(float actual, double expected, double tolerance, const char *label) {
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) > tolerance * (1 + std::abs(expected)))
    throw std::runtime_error(std::string(label) + ": " +
                             std::to_string(actual) + " vs " +
                             std::to_string(expected));
}
uint16_t bf16(float value) {
  const auto u = std::bit_cast<uint32_t>(value);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float decode(uint16_t v) { return std::bit_cast<float>(uint32_t(v) << 16); }
float sigmoid(float v) { return 1.f / (1.f + std::exp(-v)); }
void composed(int n, int hc, int rank, int mode, bool head) {
  const int dim = n * hc;
  constexpr float eps = 1e-5f;
  std::vector<float> residual(dim), gamma(dim), block(n);
  std::vector<uint16_t> down(dim * rank), up(rank * dim), wi(hc * dim);
  for (int i = 0; i < dim; ++i) {
    residual[i] = float(std::sin(i * .13) + .01 * (i % hc));
    gamma[i] = .9f + .01f * (i % 17);
  }
  for (int i = 0; i < n; ++i)
    block[i] = float(std::cos(i * .37));
  for (size_t i = 0; i < down.size(); ++i) {
    down[i] = bf16(float(.02 * std::sin(i * .31)));
    up[i] = bf16(float(.03 * std::cos(i * .17)));
  }
  for (size_t i = 0; i < wi.size(); ++i)
    wi[i] = bf16(float(.01 * std::sin(i * .71)));
  std::vector<float> xn(dim), activation(dim), lo(rank), gated(dim),
      injection(hc), mixed(n);
  for (int c = 0; c < hc; ++c) {
    double sum = 0;
    for (int d = 0; d < n; ++d)
      sum += double(residual[c * n + d]) * residual[c * n + d];
    const float rs = float(1 / std::sqrt(sum / n + eps));
    for (int d = 0; d < n; ++d) {
      const int i = c * n + d;
      xn[i] = (rs * residual[i]) * gamma[i];
      activation[i] = mode == 0 ? decode(bf16(xn[i])) : xn[i];
    }
  }
  for (int k = 0; k < rank; ++k) {
    double sum = 0;
    for (int i = 0; i < dim; ++i)
      sum += double(activation[i]) * decode(down[k * dim + i]);
    const float v = float(sum) / float(hc);
    lo[k] = v / (1.f + std::exp(-v));
    if (mode == 0)
      lo[k] = decode(bf16(lo[k]));
  }
  for (int i = 0; i < dim; ++i) {
    double sum = 0;
    for (int k = 0; k < rank; ++k)
      sum += double(lo[k]) * decode(up[i * rank + k]);
    gated[i] = sigmoid(float(sum));
  }
  for (int d = 0; d < n; ++d) {
    float sum = 0;
    for (int c = 0; c < hc; ++c) {
      const int i = c * n + d;
      if (mode == 2 && !head)
        sum = std::fma(xn[i], gated[i], sum);
      else
        sum += xn[i] * gated[i];
    }
    mixed[d] = sum / float(hc);
  }
  for (int c = 0; c < hc; ++c) {
    double sum = 0;
    for (int i = 0; i < dim; ++i)
      sum += double(activation[i]) * decode(wi[c * dim + i]);
    injection[c] = float(sum);
  }
  Buffer<float> r(dim), norm_weight(dim), output(n + 4), inject(hc + 4), b(n),
      written(dim + 4);
  Buffer<uint16_t> wd(down.size()), wu(up.size()), win(wi.size());
  const GrShapes shape{n, hc, rank};
  const auto bytes = gr_workspace_bytes(shape);
  Buffer<uint8_t> workspace(bytes + 16);
  workspace.upload(std::vector<uint8_t>(bytes + 16, 0xa5));
  GrWorkspace ws;
  check(gr_workspace_init(shape, workspace.data(), ws) == bytes,
        "workspace sizing");
  r.upload(residual);
  norm_weight.upload(gamma);
  wd.upload(down);
  wu.upload(up);
  win.upload(wi);
  b.upload(block);
  output.upload(std::vector<float>(n + 4, 12345.f));
  inject.upload(std::vector<float>(hc + 4, 12345.f));
  written.upload(std::vector<float>(dim + 4, 12345.f));
  gr_set_fp32_activations(mode == 1);
  gr_set_native_mmvf(mode == 2);
  gr_read(r.data(), norm_weight.data(), wd.data(), wu.data(),
          head ? nullptr : win.data(), eps, shape, ws, output.data(),
          inject.data(), &runtime->compute());
  const auto actual = output.read(), actual_inject = inject.read();
  for (int d = 0; d < n; ++d)
    close(actual[d], mixed[d], 3e-5, "composed mixed");
  for (int i = n; i < n + 4; ++i)
    check(actual[i] == 12345.f, "mixed canary");
  for (int c = 0; c < hc; ++c) {
    if (head)
      check(actual_inject[c] == 12345.f, "head injection untouched");
    else
      close(actual_inject[c], injection[c], 3e-5, "injection");
  }
  for (int i = hc; i < hc + 4; ++i)
    check(actual_inject[i] == 12345.f, "injection canary");
  const auto storage = workspace.read();
  for (size_t i = bytes; i < storage.size(); ++i)
    check(storage[i] == 0xa5, "workspace canary");
  // Controlled injections include saturation and a zero gate; use both write
  // APIs.
  std::vector<float> gates(hc + 4, 12345.f);
  for (int c = 0; c < hc; ++c)
    gates[c] = c == 0 ? 0.f : (c & 1 ? -100.f : 100.f);
  inject.upload(gates);
  gr_write(r.data(), b.data(), inject.data(), shape, written.data(), nullptr);
  const auto result = written.read();
  gr_write(r.data(), b.data(), inject.data(), shape, r.data(), nullptr);
  const auto inplace = r.read();
  for (int i = 0; i < dim; ++i) {
    const float weight = 2.f * sigmoid(gates[i / n] / float(hc));
    close(result[i], std::fma(block[i % n], weight, residual[i]), 3e-6,
          "write");
    check(result[i] == inplace[i], "in-place write");
  }
  for (int i = dim; i < dim + 4; ++i)
    check(result[i] == 12345.f, "write canary");
}
template <typename F> void rejects(F f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid GR input accepted");
}
} // namespace
int main() {
  try {
    std::fesetenv(FE_DFL_ENV);
    runtime = sycl_backend::runtime_for();
    for (int mode = 0; mode < 3; ++mode)
      for (bool head : {false, true}) {
        composed(38, 3, 18, mode, head);
        composed(2560, 4, 320, mode, head);
      }
    gr_set_native_mmvf(false);
    gr_set_fp32_activations(false);
    GrWorkspace ws;
    rejects([&] { gr_workspace_init({-1, 4, 320}, nullptr, ws); });
    Buffer<float> x(256), gamma(256), y(256);
    rejects([&] {
      native_gr_rms_norm_weighted(x.data(), gamma.data(), y.data(), 32, 2, -1.f,
                                  nullptr);
    });
    rejects([&] {
      native_gr_pre_gated(x.data(), y.data(), y.data(), 32, 2, true, nullptr);
    });
    rejects([&] {
      native_gr_post(x.data(), gamma.data(), y.data(), x.data() + 1, 32, 2,
                     nullptr);
    });
    const GrShapes small{32, 2, 4};
    Buffer<uint8_t> scratch(gr_workspace_bytes(small));
    gr_workspace_init(small, scratch.data(), ws);
    --ws.bytes;
    rejects([&] {
      gr_read(x.data(), gamma.data(), reinterpret_cast<uint16_t *>(x.data()),
              reinterpret_cast<uint16_t *>(gamma.data()), nullptr, 1e-5f, small,
              ws, y.data(), nullptr, nullptr);
    });
    gr_workspace_init(small, nullptr, ws);
    check(!ws.xn && !ws.xq && !ws.lq && !ws.gated && !ws.lo,
          "sizing clears workspace pointers");
    runtime->wait();
    std::cout << "SYCL GR PASS: 12 composed layer/head cases, including 2560x4 "
                 "rank 320 and in-place writes\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
