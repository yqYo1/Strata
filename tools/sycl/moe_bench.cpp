// Warm real-weight expert timings. These exclude transfers and CPU misses;
// generation throughput must be measured separately with the server harness.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace mmq = strata::prefill::mmq;
namespace {
void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
  sycl_backend::Allocation a;
  Buffer(size_t n) : a(sycl_backend::runtime_for(), n, sycl_backend::MemoryKind::Device) {}
  template<class T> T *as() { return a.as<T>(); }
  void *data() { return a.data(); }
  void put(const void *p, size_t n) { check(cudaMemcpy(data(), p, n, cudaMemcpyHostToDevice)); }
};
uint64_t hash(const std::vector<float>& v) {
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < v.size() * 4; ++i) {
    h ^= reinterpret_cast<const uint8_t*>(v.data())[i]; h *= 1099511628211ull;
  }
  return h;
}
template<class Fn>
void measure(const std::string& name, int layer, int gu, int down, int rows,
             int repeats, cudaStream_t stream, Buffer& result, size_t n, Fn run) {
  iq_set_old_kernels(true); run(); check(cudaStreamSynchronize(stream));
  std::vector<float> reference(n), output(n);
  check(cudaMemcpy(reference.data(), result.data(), n * 4, cudaMemcpyDeviceToHost));
  iq_set_old_kernels(false);
  for (int i = 0; i < 3; ++i) run();
  check(cudaStreamSynchronize(stream));
  check(cudaMemcpy(output.data(), result.data(), n * 4, cudaMemcpyDeviceToHost));
  size_t unequal = 0; double maximum = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(output[i]) || !std::isfinite(reference[i]))
      throw std::runtime_error("nonfinite expert output");
    unequal += std::memcmp(&reference[i], &output[i], 4) != 0;
    maximum = std::max(maximum, double(std::abs(reference[i] - output[i])));
  }
  for (bool old : {true, false}) {
    iq_set_old_kernels(old);
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < repeats; ++i) run();
    cudaGraph_t g{}; cudaGraphExec_t ex{};
    check(cudaStreamEndCapture(stream, &g)); check(cudaGraphInstantiate(&ex, g, 0ull));
    check(cudaGraphDestroy(g)); check(cudaGraphLaunch(ex, stream)); check(cudaStreamSynchronize(stream));
    cudaEvent_t begin{}, end{}; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    std::vector<float> samples;
    for (int i = 0; i < 5; ++i) {
      check(cudaEventRecord(begin, stream)); check(cudaGraphLaunch(ex, stream));
      check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
      float ms; check(cudaEventElapsedTime(&ms, begin, end)); samples.push_back(ms / repeats);
    }
    std::sort(samples.begin(), samples.end());
    std::cout << name << ',' << layer << ',' << gu << ',' << down << ',' << rows << ','
              << old << ',' << samples[2] << ',' << unequal << ',' << maximum << ','
              << hash(old ? reference : output) << '\n' << std::flush;
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end)); check(cudaGraphExecDestroy(ex));
  }
  iq_set_old_kernels(false);
}
}
int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 3) throw std::invalid_argument("usage: sycl_moe_bench SHARD1 [REPEATS=10]");
    const int repeats = argc == 3 ? std::stoi(argv[2]) : 10;
    if (repeats < 1 || repeats > 1000) throw std::invalid_argument("repeats outside 1..1000");
    const auto model = GgufModel::open(argv[1]);
    cudaStream_t stream{}; check(cudaStreamCreate(&stream));
    std::set<std::pair<int,int>> seen;
    std::cout << "operation,layer,gu_type,down_type,rows_per_expert,old,median_ms,unequal,max_abs,fnv1a64\n";
    constexpr int E = 8;
    for (int layer = 0; layer < 48; ++layer) {
      size_t gs, us, ds;
      const std::string prefix = "blk." + std::to_string(layer) + ".";
      const auto *g = model.find(prefix + "ffn_gate_exps.weight", &gs);
      const auto *u = model.find(prefix + "ffn_up_exps.weight", &us);
      const auto *d = model.find(prefix + "ffn_down_exps.weight", &ds);
      if (!g || !u || !d || !seen.emplace(g->type, d->type).second) continue;
      if (g->shape.size() != 3 || u->shape != g->shape || g->type != u->type ||
          d->shape.size() != 3 || d->shape[0] != g->shape[1] || d->shape[1] != g->shape[0] ||
          g->shape[2] < 64 || d->shape[2] != g->shape[2] ||
          !model.in_bounds(*g, gs) || !model.in_bounds(*u, us) || !model.in_bounds(*d, ds))
        throw std::runtime_error("unexpected expert geometry");
      const auto L = native_expert_layout(g->type, d->type, g->shape[0], g->shape[1]);
      std::vector<uint8_t> blobs(E * L.bytes), gu(E * L.down_off), down(E * (L.bytes - L.down_off));
      for (int e = 0; e < E; ++e) {
        const int id = e * 7;
        auto *b = blobs.data() + e * L.bytes;
        std::memcpy(b, model.shard(gs).tensor_data(*g) + id * L.up_off, L.up_off);
        std::memcpy(b + L.up_off, model.shard(us).tensor_data(*u) + id * L.up_off, L.up_off);
        std::memcpy(b + L.down_off, model.shard(ds).tensor_data(*d) + id * (L.bytes - L.down_off), L.bytes - L.down_off);
        std::memcpy(gu.data() + e * L.down_off, b, L.down_off);
        std::memcpy(down.data() + e * (L.bytes - L.down_off), b + L.down_off, L.bytes - L.down_off);
      }
      Buffer weights(blobs.size()), wg(gu.size()), wd(down.size()), ptr(E * 8), count(4), bounds((E + 1) * 4);
      weights.put(blobs.data(), blobs.size()); wg.put(gu.data(), gu.size()); wd.put(down.data(), down.size());
      std::vector<unsigned long long> pointers(E);
      for (int e = 0; e < E; ++e) pointers[e] = reinterpret_cast<uintptr_t>(weights.as<uint8_t>() + e * L.bytes);
      ptr.put(pointers.data(), E * 8); count.put(&E, 4);
      for (int rows : {1, 4, 16}) {
        const int total = E * rows;
        std::vector<int32_t> starts(E + 1), ids(total), tokens(total);
        for (int e = 0; e <= E; ++e) starts[e] = e * rows;
        for (int i = 0; i < total; ++i) { ids[i] = i; tokens[i] = i % rows; }
        bounds.put(starts.data(), starts.size() * 4);
        Buffer di(total * 4), dt(total * 4); di.put(ids.data(), total * 4); dt.put(tokens.data(), total * 4);
        std::vector<float> input(size_t(total) * L.n_embd);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * .13f);
        Buffer x(input.size() * 4), q(native_q8_1_bytes(L.n_embd) * total),
            scratch(native_expert_scratch_bytes(total, L.n_ff)), y(size_t(total) * L.n_embd * 4);
        x.put(input.data(), input.size() * 4);
        native_quantize_q8_1(x.as<float>(), q.data(), total * L.n_embd, 1, stream);
        if (rows <= 4)
          measure("grouped", layer, g->type, d->type, rows, repeats, stream, y, size_t(total) * L.n_embd, [&] {
            native_expert_grouped(L, ptr.as<unsigned long long>(), bounds.as<int32_t>(), count.as<int32_t>(),
                                 di.as<int32_t>(), dt.as<int32_t>(), E, total, q.data(), scratch.data(), y.as<float>(), stream);
          });
        for (bool is_down : {false, true}) {
          const int cols = is_down ? L.n_ff : L.n_embd, out = is_down ? L.n_embd : 2 * L.n_ff;
          Buffer qmm(mmq::q8_bytes(total, cols)), ym(size_t(total) * out * 4);
          mmq::quantize(x.as<float>(), nullptr, qmm.data(), is_down ? d->type : g->type, cols, L.n_embd, total, stream);
          mmq::Product p{is_down ? wd.data() : wg.data(), int(is_down ? d->type : g->type), out, cols,
                         is_down ? L.bytes - L.down_off : L.down_off, E, qmm.data(), bounds.as<int32_t>(),
                         di.as<int32_t>(), total, rows, ym.as<float>(), out};
          mmq::Context ctx;
          measure(is_down ? "mmq_down" : "mmq_gu", layer, g->type, d->type, rows, repeats, stream,
                  ym, size_t(total) * out, [&] { ctx.run(p, stream); });
        }
      }
    }
    if (seen.empty()) throw std::runtime_error("no native expert tensors found");
    check(cudaStreamDestroy(stream));
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
