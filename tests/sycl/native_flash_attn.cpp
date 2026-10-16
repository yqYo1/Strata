#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <cmath>
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
    check(
        cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get() {
    std::vector<T> v(n);
    check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
int main() {
  try {
    constexpr int H = 24, D = 256, KV = 2, C = 256, N = H * D;
    B<float> q(N), out(N + 4);
    B<uint16_t> k(C * KV * D + 1), v(C * KV * D + 1), mask(C + 1);
    B<int32_t> step(4), status(1);
    std::vector<float> query(N);
    for (int i = 0; i < N; ++i)
      query[i] = float(std::sin(i * .071) * .7);
    q.put(query);
    QsaShapes s;
    s.n_head = H;
    s.n_head_kv = KV;
    s.head_dim = D;
    s.idx_block = 4;
    s.idx_top_k = 256;
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream));
    double max_error = 0;
    int cases = 0;
    for (int width : {1, 3, 7, 31, 32, 33, 127, 128, 129, 255, 256})
      for (bool masked : {false, true}) {
        std::vector<uint16_t> keys(k.n, 0x7e00), values(v.n, 0x7e00),
            m(mask.n, 0x7e00);
        for (int i = 0; i < width * KV * D; ++i) {
          keys[i + 1] = f16_from_f32(float(std::cos(i * .041) * .8));
          values[i + 1] = f16_from_f32(float(std::sin(i * .093) * .6));
        }
        for (int i = 0; i < width; ++i)
          m[i + 1] = f16_from_f32(i && i % 3 == 0 ? -INFINITY
                                                  : float(-.125 * (i % 7)));
        k.put(keys);
        v.put(values);
        mask.put(m);
        step.put({width - 1, width, width / 4, width});
        out.put(std::vector<float>(N + 4, 42));
        native_flash_attn_short_step(q.p, k.p + 1, v.p + 1, step.p, C, C, s,
                                     out.p, status.p,
                                     masked ? mask.p + 1 : nullptr, stream);
        auto got = out.get();
        if (status.get()[0] != kNativeFlashAttnSuccess)
          throw std::runtime_error("valid short step rejected");
        for (int head = 0; head < H; ++head) {
          int kv = head / 12;
          std::vector<double> scores(width);
          double mx = -INFINITY;
          for (int cell = 0; cell < width; ++cell) {
            double dot = 0;
            for (int d = 0; d < D; ++d)
              dot += double(query[head * D + d]) *
                     f32_from_f16(keys[1 + (cell * KV + kv) * D + d]);
            scores[cell] = dot / 16 + (masked ? f32_from_f16(m[cell + 1]) : 0);
            mx = std::max(mx, scores[cell]);
          }
          double sum = 0;
          for (double &score : scores) {
            score = std::exp(score - mx);
            sum += score;
          }
          for (int d = 0; d < D; ++d) {
            double ref = 0;
            for (int cell = 0; cell < width; ++cell)
              ref += scores[cell] / sum *
                     f32_from_f16(values[1 + (cell * KV + kv) * D + d]);
            double error = std::abs(got[head * D + d] - ref);
            max_error = std::max(max_error, error);
            if (!std::isfinite(got[head * D + d]) ||
                error > 3e-5 * (1 + std::abs(ref)))
              throw std::runtime_error("short attention CPU mismatch");
          }
        }
        for (int i = N; i < N + 4; ++i)
          if (got[i] != 42)
            throw std::runtime_error("short attention output guard");
        ++cases;
      }
    for (auto rec : std::vector<std::vector<int32_t>>{{-1, 0, 0, 0},
                                                      {256, 257, 64, 257},
                                                      {2, 3, 1, 3},
                                                      {3, 3, 0, 3},
                                                      {2, 4, 0, 3}}) {
      step.put(rec);
      native_flash_attn_short_step(q.p, k.p + 1, v.p + 1, step.p, C, C, s,
                                   out.p, status.p, nullptr, stream);
      auto got = out.get();
      if (status.get()[0] != kNativeFlashAttnUnsupportedStep)
        throw std::runtime_error("invalid step accepted");
      for (int i = 0; i < N; ++i)
        if (!std::isnan(got[i]))
          throw std::runtime_error("invalid step output");
    }
    bool rejected = false;
    try {
      native_flash_attn_short_step(q.p, k.p + 1, v.p + 1, step.p, C, C, s, q.p,
                                   status.p, nullptr, stream);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("short attention alias accepted");
    check(cudaStreamDestroy(stream));
    std::cout << "SYCL native short attention: " << cases
              << " masked/unmasked CPU cases, max error " << max_error
              << ", invalid steps and guards passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
