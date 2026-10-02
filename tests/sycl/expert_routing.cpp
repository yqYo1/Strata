#include "strata/kernels/s2_expert_grouped.hpp"
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
void require(bool v, const char *s) {
  if (!v)
    throw std::runtime_error(s);
}
template <class T> struct B {
  T *p{};
  size_t n;
  bool host;
  B(size_t N, bool h = false) : n(N), host(h) {
    check(h ? cudaMallocHost(&p, n * sizeof(T))
            : cudaMalloc(&p, n * sizeof(T)));
  }
  ~B() {
    if (host)
      cudaFreeHost(p);
    else
      cudaFree(p);
  }
  void put(const std::vector<T> &v) {
    require(v.size() == n, "upload size");
    check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get() {
    std::vector<T> v(n);
    check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
int main() {
  try {
    cudaStream_t st{};
    check(cudaStreamCreate(&st));
    constexpr int N = 128, E = 17, W = 37;
    B<int> ids(N), res(E), slot(N), dst(N), count(2), starts(N + 1), tok(N);
    B<unsigned long long> ptr(N);
    B<uint8_t> blobs(E * 11);
    B<float> parts(N * W), hits(N * W);
    for (int n : {1, 7, 32, 33, 127, 128}) {
      std::vector<int> iv(N, -1), rv(E), expectedSlot, expectedDst;
      for (int e = 0; e < E; ++e)
        rv[e] = e % 3 ? E - 1 - e : -1;
      for (int i = 0; i < n; ++i) {
        iv[i] = (i * 7 + 3) % (E + 2) - 1;
        if (iv[i] >= 0 && iv[i] < E && rv[iv[i]] >= 0) {
          expectedSlot.push_back(rv[iv[i]]);
          expectedDst.push_back(i);
        }
      }
      ids.put(iv);
      res.put(rv);
      slot.put(std::vector<int>(N, -99));
      dst.put(std::vector<int>(N, -99));
      if (n <= 32)
        moe_hit_select(ids.p, res.p, n, E, slot.p, dst.p, count.p, st);
      else
        moe_hit_select_multi(ids.p, res.p, n, E, slot.p, dst.p, count.p, st);
      check(cudaStreamSynchronize(st));
      auto sv = slot.get(), dv = dst.get();
      require(count.get()[0] == int(expectedDst.size()), "hit count");
      for (int i = 0; i < N; ++i) {
        require(sv[i] == (i < int(expectedSlot.size()) ? expectedSlot[i] : -99),
                "stable slot");
        require(dv[i] == (i < int(expectedDst.size()) ? expectedDst[i] : -99),
                "stable destination");
      }
      std::vector<float> pv(N * W), hv(N * W), ref;
      for (int i = 0; i < N * W; ++i) {
        pv[i] = float(i) / 4;
        hv[i] = -float(i) / 8;
      }
      ref = pv;
      for (int d : expectedDst)
        for (int j = 0; j < W; ++j)
          ref[d * W + j] += hv[d * W + j];
      parts.put(pv);
      hits.put(hv);
      moe_hit_add(parts.p, hits.p, dst.p, count.p, n, W, st);
      check(cudaStreamSynchronize(st));
      require(parts.get() == ref, "routed add");
      // Invalid device counts leave the destination untouched.
      for (int invalid : {-1, n + 1}) {
        count.put({invalid, 0});
        parts.put(pv);
        moe_hit_add(parts.p, hits.p, dst.p, count.p, n, W, st);
        check(cudaStreamSynchronize(st));
        require(parts.get() == pv, "invalid count add");
      }
      // Group valid resident ids; negative ids are skipped.
      for (int i = 0; i < n; ++i)
        if (iv[i] >= E)
          iv[i] = -1;
      ids.put(iv);
      moe_group_resident(ids.p, n, 7, blobs.p, 11, ptr.p, starts.p, count.p,
                         dst.p, tok.p, st);
      check(cudaStreamSynchronize(st));
      auto gp = ptr.get();
      auto gs = starts.get();
      dv = dst.get();
      auto tv = tok.get();
      std::vector<int> seen;
      int entries = 0;
      for (int i = 0; i < n; ++i) {
        int e = iv[i];
        bool found = false;
        for (int prior : seen)
          found |= e == prior;
        if (e < 0 || found)
          continue;
        int g = int(seen.size());
        seen.push_back(e);
        require(gp[g] == reinterpret_cast<unsigned long long>(blobs.p) + e * 11,
                "group pointer");
        require(gs[g] == entries, "group start");
        for (int j = i; j < n; ++j)
          if (iv[j] == e) {
            require(dv[entries] == j && tv[entries] == j / 7, "group entries");
            ++entries;
          }
      }
      require(gs[seen.size()] == entries &&
                  count.get() == std::vector<int>({int(seen.size()), entries}),
              "group counts");
    }
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    moe_hit_select_multi(ids.p, res.p, N, E, slot.p, dst.p, count.p, st);
    moe_hit_add(parts.p, hits.p, dst.p, count.p, N, W, st);
    check(cudaStreamEndCapture(st, &graph));
    check(cudaGraphInstantiate(&exec, graph, 0ull));
    check(cudaGraphDestroy(graph));
    for (int iteration = 0; iteration < 4; ++iteration) {
      std::vector<int> iv(N, iteration % 2 ? -1 : 2), rv(E, -1);
      rv[2] = 4;
      ids.put(iv);
      res.put(rv);
      parts.put(std::vector<float>(N * W, 1));
      hits.put(std::vector<float>(N * W, 2));
      check(cudaGraphLaunch(exec, st));
      check(cudaStreamSynchronize(st));
      require(count.get()[0] == (iteration % 2 ? 0 : N),
              "replay dynamic count");
      require(parts.get() == std::vector<float>(N * W, iteration % 2 ? 1 : 3),
              "replay routed add");
    }
    check(cudaGraphExecDestroy(exec));
    bool rejected = false;
    try {
      moe_hit_select(ids.p, res.p, 33, E, slot.p, dst.p, count.p, st);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "reject oversized single-token selection");
    check(cudaStreamDestroy(st));
    std::cout << "SYCL expert routing: stable selection, grouping and routed "
                 "add PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
