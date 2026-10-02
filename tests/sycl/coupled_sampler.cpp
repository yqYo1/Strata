#include "strata/kernels/sampler.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace strata::kernels;
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
template <class T> struct Mem {
  T *p{};
  size_t n;
  bool host;
  Mem(size_t size, bool h = false) : n(size), host(h) {
    check(h ? cudaMallocHost(&p, size * sizeof(T))
            : cudaMalloc(&p, size * sizeof(T)));
  }
  ~Mem() {
    if (host)
      cudaFreeHost(p);
    else
      cudaFree(p);
  }
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
float draw(uint64_t seed, uint64_t counter) {
  uint32_t a = counter, b = counter >> 32, c = seed, d = seed >> 32;
  for (uint32_t i = 0; i < 10; ++i) {
    uint64_t u = uint64_t(0x9E3779B9u) * a, v = uint64_t(0xBB67AE85u) * c;
    uint32_t aa = uint32_t(v >> 32) ^ b ^ i, bb = uint32_t(v),
             cc = uint32_t(u >> 32) ^ d, dd = uint32_t(u);
    a = aa;
    b = bb;
    c = cc;
    d = dd;
  }
  return float(a >> 8) / 16777216.f;
}
std::pair<int, float> reference(std::vector<float> &l,
                                const std::vector<int> &history,
                                const std::vector<int> &map,
                                const SamplerParams &p, uint64_t counter) {
  for (size_t i = 0; i < l.size(); ++i) {
    int id = map.empty() ? int(i) : map[i];
    int count = std::count(history.begin(), history.end(), id);
    if (count) {
      l[i] = l[i] <= 0 ? l[i] * p.penalty_repeat : l[i] / p.penalty_repeat;
      l[i] -= std::fma(float(count), p.penalty_freq, p.penalty_present);
    }
  }
  std::vector<int> ids(l.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::stable_sort(ids.begin(), ids.end(),
                   [&](int a, int b) { return l[a] > l[b]; });
  if (p.greedy || p.temperature <= 0)
    return {map.empty() ? ids[0] : map[ids[0]], 1};
  int k = std::min(int(l.size()), p.top_k > 0 && p.top_k < 64 ? p.top_k : 64),
      keep = k;
  double ex[64];
  if (p.top_p < 1) {
    double sum = 0;
    for (int i = 0; i < k; ++i) {
      ex[i] = std::exp(double(l[ids[i]]) - l[ids[0]]);
      sum += ex[i];
    }
    double cum = 0;
    for (int i = 0; i < k; ++i) {
      cum += ex[i] / sum;
      if (cum >= p.top_p) {
        keep = i + 1;
        break;
      }
    }
    keep = std::max(keep, std::min(p.min_keep, k));
  }
  if (p.min_p > 0) {
    float threshold = l[ids[0]] + std::log(p.min_p);
    for (int i = 0; i < keep; ++i)
      if (l[ids[i]] < threshold) {
        keep = i;
        break;
      }
  }
  float inv = 1.f / p.temperature, mx = l[ids[0]] * inv;
  double sum = 0;
  for (int i = 0; i < keep; ++i) {
    ex[i] = std::exp(double(l[ids[i]] * inv) - mx);
    sum += ex[i];
  }
  for (int i = 0; i < keep; ++i)
    ex[i] /= sum;
  int pick = keep ? keep - 1 : 0;
  double cum = 0;
  float u = draw(p.seed, counter);
  for (int i = 0; i < keep; ++i) {
    cum += ex[i];
    if (u < cum) {
      pick = i;
      break;
    }
  }
  int id = ids[pick];
  return {map.empty() ? id : map[id], keep ? float(ex[pick]) : 1};
}
void run(bool subset) {
  constexpr int V = 257, CAP = 8, J = 3, GUARD = 4;
  const int vocab = subset ? V * 2 + 1 : V;
  Mem<SamplerParams> hp(1, true), dp(1);
  Mem<int> hh(CAP, true), ring(CAP + J + GUARD), out(J + GUARD), steps(J, true),
      map(V), inverse(vocab);
  Mem<float> hl(V * J, true), logits(V), prob(J + GUARD);
  Mem<unsigned char> scratch(coupled_draft_scratch_bytes(V));
  std::vector<int> mapping, inv(vocab, -1);
  if (subset) {
    mapping.resize(V);
    for (int i = 0; i < V; ++i) {
      mapping[i] = 2 * i + 1;
      inv[mapping[i]] = i;
    }
    map.put(mapping);
    inverse.put(inv);
  }
  cudaStream_t stream{};
  check(cudaStreamCreate(&stream));
  cudaGraph_t graph{};
  cudaGraphExec_t exec{};
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  coupled_draft_stage(hp.p, hh.p, dp.p, ring.p, CAP, stream);
  for (int j = 0; j < J; ++j) {
    check(cudaMemcpyAsync(logits.p, hl.p + j * V, V * 4, cudaMemcpyHostToDevice,
                          stream));
    coupled_draft_sample(logits.p, V, subset ? map.p : nullptr,
                         subset ? inverse.p : nullptr, vocab, dp.p, ring.p, CAP,
                         j, steps.p + j, scratch.p, out.p + j, prob.p + j,
                         stream);
  }
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&exec, graph, 0ull));
  check(cudaGraphDestroy(graph));
  for (int replay = 0; replay < 32; ++replay) {
    SamplerParams p;
    p.seed = 0xabcdef00000000ull + replay;
    p.top_k = replay % 3 ? 20 : 0;
    p.top_p = .83f;
    p.min_p = .02f;
    p.temperature = replay % 7 ? .7f : 0;
    p.penalty_last_n = replay % 5 ? 6 : 0;
    p.penalty_repeat = 1.3f;
    p.penalty_freq = .2f;
    p.penalty_present = .1f;
    *hp.p = p;
    std::vector<int> history(CAP);
    for (int i = 0; i < CAP; ++i)
      history[i] = hh.p[i] = i < 2 ? -1 : (subset ? 2 * (i / 2) + 1 : i / 2);
    for (int j = 0; j < J; ++j) {
      steps.p[j] = replay * 73 + j;
      for (int i = 0; i < V; ++i)
        hl.p[j * V + i] = float(std::sin(i * .13 + j * .7 + replay * .19) * 3);
    }
    ring.put(std::vector<int>(CAP + J + GUARD, -777));
    out.put(std::vector<int>(J + GUARD, -777));
    prob.put(std::vector<float>(J + GUARD, 42));
    check(cudaGraphLaunch(exec, stream));
    check(cudaStreamSynchronize(stream));
    auto ids = out.get();
    auto probs = prob.get();
    auto gotring = ring.get();
    std::vector<float> last;
    for (int j = 0; j < J; ++j) {
      int h = std::clamp(p.penalty_last_n, 0, CAP);
      std::vector<int> tail(history.end() - h, history.end());
      std::vector<float> l(hl.p + j * V, hl.p + (j + 1) * V);
      auto ref = reference(l, tail, mapping, p, uint64_t(steps.p[j]) + 1);
      last = l;
      if (ids[j] != ref.first || gotring[CAP + j] != ref.first ||
          std::abs(probs[j] - ref.second) > 2e-6f)
        throw std::runtime_error("coupled replay reference mismatch");
      history.push_back(ref.first);
    }
    auto penalized = logits.get();
    for (int i = 0; i < V; ++i)
      if (std::abs(penalized[i] - last[i]) > 1e-6f)
        throw std::runtime_error("in-place penalties");
    for (int i = 0; i < CAP; ++i)
      if (gotring[i] !=
          (i >= CAP - std::clamp(p.penalty_last_n, 0, CAP) ? hh.p[i] : -777))
        throw std::runtime_error("staged history range");
    for (int i = 0; i < GUARD; ++i)
      if (ids[J + i] != -777 || probs[J + i] != 42 ||
          gotring[CAP + J + i] != -777)
        throw std::runtime_error("canary");
  }
  check(cudaGraphExecDestroy(exec));
  check(cudaStreamDestroy(stream));
}
int main() {
  try {
    run(false);
    run(true);
    std::cout << "SYCL coupled sampler: 64 captured rounds, chain history, "
                 "mapped subsets and probabilities passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
