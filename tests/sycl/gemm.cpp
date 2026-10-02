#include "strata/prefill/gemm.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <vector>
#define CHECK(x)                                                               \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(cudaGetErrorString(e));                         \
  } while (0)
void need(bool v, const char *why) {
  if (!v)
    throw std::runtime_error(why);
}
uint16_t encode(float x, bool bf) {
  if (!bf)
    return strata::kernels::f16_from_f32(x);
  sycl::ext::oneapi::bfloat16 b(x);
  uint16_t bits;
  std::memcpy(&bits, &b, 2);
  return bits;
}
float decode(uint16_t x, bool bf) {
  if (!bf)
    return strata::kernels::f32_from_f16(x);
  uint32_t u = uint32_t(x) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
void verify(const std::vector<uint16_t> &x, const std::vector<uint16_t> &w,
            const std::vector<float> &y, int T, int N, int K, int ld, bool bf,
            float beta) {
  for (int t = 0; t < T; ++t)
    for (int n = 0; n < ld; ++n) {
      if (n >= N) {
        need(y[t * ld + n] == 42, "output stride guard");
        continue;
      }
      double want = beta * 42, mag = std::abs(want);
      for (int k = 0; k < K; ++k) {
        double a = double(decode(x[t * K + k], bf)) * decode(w[n * K + k], bf);
        want += a;
        mag += std::abs(a);
      }
      need(std::abs(y[t * ld + n] - want) < 4e-6 * (1 + mag),
           "GEMM CPU parity");
    }
}
int main() {
  try {
    cudaStream_t q;
    CHECK(cudaStreamCreate(&q));
    int cases = 0;
    for (bool bf : {false, true})
      for (auto dims :
           {std::vector<int>{1, 17, 32}, {17, 259, 320}, {8, 640, 2560}}) {
        int T = dims[0], N = dims[1], K = dims[2], ld = N + 3;
        std::vector<uint16_t> x(T * K), w(N * K);
        for (size_t i = 0; i < x.size(); ++i)
          x[i] = encode(float(int(i * 17 % 73) - 36) / 37, bf);
        for (size_t i = 0; i < w.size(); ++i)
          w[i] = encode(float(int(i * 11 % 61) - 30) / 31, bf);
        uint16_t *dx, *dw;
        float *dy;
        CHECK(cudaMalloc(&dx, x.size() * 2));
        CHECK(cudaMalloc(&dw, w.size() * 2));
        CHECK(cudaMalloc(&dy, size_t(T) * ld * 4));
        CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
        strata::prefill::Gemm gemm;
        std::string err;
        need(gemm.init_external(q, nullptr, 0, nullptr, 0, err), err.c_str());
        for (float beta : {0.f, 0.5f, 1.f}) {
          std::vector<float> y(T * ld, 42);
          CHECK(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
          if (bf)
            gemm.bf16(dx, dw, dy, T, N, K, ld, beta);
          else
            gemm.f16(dx, dw, dy, T, N, K, ld, beta);
          CHECK(cudaStreamSynchronize(q));
          CHECK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
          verify(x, w, y, T, N, K, ld, bf, beta);
          ++cases;
        }
        CHECK(cudaFree(dx));
        CHECK(cudaFree(dw));
        CHECK(cudaFree(dy));
      }
    {
      constexpr int T = 5, N = 17, K = 256, ld = 21;
      std::vector<uint8_t> raw(N * K / 32 * 34);
      std::vector<uint16_t> w(N * K), x(T * K);
      for (int block = 0; block < N * K / 32; ++block) {
        uint16_t d = encode(0.125f, false);
        std::memcpy(raw.data() + block * 34, &d, 2);
        for (int i = 0; i < 32; ++i) {
          int8_t v = int8_t((block * 13 + i * 3) % 127 - 63);
          raw[block * 34 + 2 + i] = uint8_t(v);
          w[block * 32 + i] = encode(v * 0.125f, false);
        }
      }
      for (size_t i = 0; i < x.size(); ++i)
        x[i] = encode(float(int(i % 51) - 25) / 31, false);
      uint8_t *dw;
      uint16_t *dx;
      float *dy;
      CHECK(cudaMalloc(&dw, raw.size()));
      CHECK(cudaMalloc(&dx, x.size() * 2));
      CHECK(cudaMalloc(&dy, T * ld * 4));
      CHECK(cudaMemcpy(dw, raw.data(), raw.size(), cudaMemcpyHostToDevice));
      CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
      {
        strata::prefill::Gemm gemm;
        std::string err;
        need(gemm.init(q, 3 * K, err), err.c_str());
        std::vector<float> y(T * ld, 42);
        CHECK(cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
        gemm.native(dx, 8, dw, dy, T, N, K, ld, 0.5f);
        CHECK(cudaStreamSynchronize(q));
        CHECK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        verify(x, w, y, T, N, K, ld, false, 0.5f);
        ++cases;
      }
      CHECK(cudaFree(dw));
      CHECK(cudaFree(dx));
      CHECK(cudaFree(dy));
    }
    CHECK(cudaStreamDestroy(q));
    std::cout << "SYCL oneMKL GEMM: " << cases << " CPU parity cases passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
