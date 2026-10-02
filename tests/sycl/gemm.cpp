#include "strata/prefill/gemm.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <atomic>
#include <thread>
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
// Re-recorded events protect host and device rings. Keep a small two-queue
// case for adapter coverage and realistic expert matrices on the single queue
// used by SYCL prefill. The latter stalled with separate queues on B570/2026.1.
void pipeline(cudaStream_t compute, bool separate) {
  constexpr int jobs = 200, host_slots = 128, slots = 8;
  constexpr int T = 3;
  const int N = separate ? 32 : 1280, K = separate ? 640 : 2560, elements = N * K;
  cudaStream_t copy = compute;
  if (separate) CHECK(cudaStreamCreate(&copy));
  uint16_t *host, *weights, *x;
  float *y;
  CHECK(cudaMallocHost(&host, host_slots * elements * 2));
  CHECK(cudaMalloc(&weights, slots * elements * 2));
  CHECK(cudaMalloc(&x, T * K * 2));
  CHECK(cudaMalloc(&y, jobs * T * N * 4));
  std::vector<uint16_t> input(T * K, encode(1.f, false));
  CHECK(cudaMemcpy(x, input.data(), input.size() * 2, cudaMemcpyHostToDevice));
  cudaEvent_t copied[slots], used[slots], host_done[host_slots];
  for (int i = 0; i < slots; ++i) {
    CHECK(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&used[i], cudaEventDisableTiming));
  }
  for (auto &event : host_done)
    CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  strata::prefill::Gemm gemm;
  std::string err;
  need(gemm.init_external(compute, nullptr, 0, nullptr, 0, err), err.c_str());
  std::atomic<int> next{0}, issued{0};
  std::atomic<bool> ready[jobs]{};
  std::vector<std::thread> workers;
  for (int t = 0; t < 32; ++t) workers.emplace_back([&] {
    for (;;) {
      const int j = next.fetch_add(1);
      if (j >= jobs) break;
      const int h = j % host_slots;
      if (j >= host_slots) {
        while (issued.load(std::memory_order_acquire) <= j - host_slots)
          std::this_thread::yield();
        CHECK(cudaEventSynchronize(host_done[h]));
      }
      std::fill(host + h * elements, host + (h + 1) * elements,
                encode(float(j + 1) / 64, false));
      ready[j].store(true, std::memory_order_release);
    }
  });
  int staged = 0;
  for (int job = 0; job < jobs; ++job) {
    while (staged < jobs && staged < job + slots) {
      int h = staged % host_slots, d = staged % slots;
      while (!ready[staged].load(std::memory_order_acquire))
        std::this_thread::yield();
      if (staged >= slots) CHECK(cudaStreamWaitEvent(copy, used[d]));
      CHECK(cudaMemcpyAsync(weights + d * elements, host + h * elements,
                            elements * 2, cudaMemcpyHostToDevice, copy));
      CHECK(cudaEventRecord(host_done[h], copy));
      issued.store(staged + 1, std::memory_order_release);
      CHECK(cudaEventRecord(copied[d], copy));
      ++staged;
    }
    int d = job % slots;
    CHECK(cudaStreamWaitEvent(compute, copied[d]));
    gemm.f16(x, weights + d * elements, y + job * T * N, T, N, K);
    CHECK(cudaEventRecord(used[d], compute));
  }
  for (auto &worker : workers) worker.join();
  CHECK(cudaStreamSynchronize(compute));
  std::vector<float> output(jobs * T * N);
  CHECK(cudaMemcpy(output.data(), y, output.size() * 4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < jobs * T * N; ++i)
    need(output[i] == float(K * (i / (T * N) + 1)) / 64, "pipelined GEMM data");
  for (auto event : copied) CHECK(cudaEventDestroy(event));
  for (auto event : used) CHECK(cudaEventDestroy(event));
  for (auto event : host_done) CHECK(cudaEventDestroy(event));
  CHECK(cudaFreeHost(host));
  CHECK(cudaFree(weights));
  CHECK(cudaFree(x));
  CHECK(cudaFree(y));

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
    // Odd row slices must also work at the shared-expert width (640),
    // which is block-aligned but not a multiple of 256.
    for (const int K : {32, 64, 256, 640}) {
      constexpr int T = 5, N = 17, ld = 21;
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
    pipeline(q, true);
    pipeline(q, false);
    CHECK(cudaStreamDestroy(q));
    std::cout << "SYCL oneMKL GEMM: " << cases << " CPU parity cases passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
