#include "strata/kernels/cvec.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#define CHECK(x)                                                               \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(cudaGetErrorString(e));                         \
  } while (0)
void need(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
int main() {
  try {
    using namespace strata::kernels;
    constexpr int N = 259, HC = 4, T = 2, LD = N * HC + 3, BLD = N + 2, ILD = 7;
    std::vector<float> r(T * LD, 42), bo(T * BLD, 42), inj(T * ILD, 42),
        dir(3 * N, 0), scale{0, .4f, 0};
    for (int t = 0; t < T; ++t) {
      for (int i = 0; i < HC * N; ++i)
        r[t * LD + i] = std::sin(float(t * HC * N + i) * .031f);
      for (int i = 0; i < N; ++i)
        bo[t * BLD + i] = std::cos(float(t * N + i) * .017f) * .2f;
      for (int c = 0; c < HC; ++c)
        inj[t * ILD + c] = float(t + c - 2) * .2f;
    }
    double norm = 0;
    for (int d = 0; d < N; ++d) {
      dir[N + d] = std::cos(float(d) * .073f);
      norm += double(dir[N + d]) * dir[N + d];
    }
    for (int d = 0; d < N; ++d)
      dir[N + d] /= std::sqrt(norm);
    float *R, *B, *I;
    CHECK(cudaMalloc(&R, r.size() * 4));
    CHECK(cudaMalloc(&B, bo.size() * 4));
    CHECK(cudaMalloc(&I, inj.size() * 4));
    CHECK(cudaMemcpy(B, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(I, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice));
    cudaStream_t q;
    CHECK(cudaStreamCreate(&q));
    int cases = 0;
    for (int mode : {0, 1}) {
      std::string err;
      need(cvec_upload(dir, scale, mode, 1, 2, N, HC, err), err.c_str());
      need(cvec().covers(1) && !cvec().covers(0) && !cvec().covers(2),
           "steering coverage");
      need(cvec_replicate(err), "single device replication");
      for (bool write : {false, true}) {
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CHECK(cudaStreamBeginCapture(q, cudaStreamCaptureModeThreadLocal));
        cvec_apply(R, 1, T, LD, write ? B : nullptr, BLD, write ? I : nullptr,
                   ILD, write, q);
        CHECK(cudaStreamEndCapture(q, &graph));
        CHECK(cudaGraphInstantiate(&exec, graph, 0ull));
        for (bool enabled : {true, false, true}) {
          cvec_set_enabled(enabled);
          need(cvec_enabled() == enabled, "request flag");
          CHECK(cudaMemcpy(R, r.data(), r.size() * 4, cudaMemcpyHostToDevice));
          CHECK(cudaGraphLaunch(exec, q));
          CHECK(cudaStreamSynchronize(q));
          std::vector<float> got(r.size());
          CHECK(cudaMemcpy(got.data(), R, got.size() * 4,
                           cudaMemcpyDeviceToHost));
          for (int t = 0; t < T; ++t) {
            for (int c = 0; c < HC; ++c) {
              std::vector<float> values(N);
              double dot = 0;
              float gate =
                  2.f * (1.f / (1.f + std::exp(-inj[t * ILD + c] / float(HC))));
              for (int d = 0; d < N; ++d) {
                float v = r[t * LD + c * N + d];
                if (write)
                  v = std::fma(bo[t * BLD + d], gate, v);
                values[d] = v;
                dot += double(v) * dir[N + d];
              }
              for (int d = 0; d < N; ++d) {
                double expected = values[d];
                if (enabled)
                  expected = mode == 0 ? expected - scale[1] * dot * dir[N + d]
                                       : expected + dir[N + d];
                float actual = got[t * LD + c * N + d];
                need(std::isfinite(actual) &&
                         std::abs(actual - expected) <
                             3e-6 * (1 + std::abs(expected)),
                     "control vector parity");
                if (!enabled && !write)
                  need(actual == r[t * LD + c * N + d],
                       "disabled vector bit identity");
              }
            }
            for (int d = HC * N; d < LD; ++d)
              need(got[t * LD + d] == 42, "control vector stride guard");
          }
          ++cases;
        }
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
      }
    }
    CHECK(cudaStreamDestroy(q));
    CHECK(cudaFree(R));
    CHECK(cudaFree(B));
    CHECK(cudaFree(I));
    std::cout << "SYCL control vectors: " << cases
              << " add/project/write/graph-toggle cases passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
