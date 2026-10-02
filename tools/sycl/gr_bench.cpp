// Warm-weight timings and exact fallback/workspace comparisons for real GR weights.
// This includes norm, projections and nonlinear work; it is not generation time.
#include "strata/core/weights.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, LR = 320, D = HC * N;
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
bool gr_weight(const std::string &name) {
  return name.starts_with("output_hc_") ||
         (name.starts_with("blk.") &&
          (name.find(".hc_attn_") != std::string::npos ||
           name.find(".hc_ffn_") != std::string::npos));
}
std::vector<float> read(const float *p, int count) {
  std::vector<float> out(count);
  check(cudaMemcpy(out.data(), p, size_t(count) * 4, cudaMemcpyDeviceToHost));
  return out;
}
struct Graph {
  cudaGraphExec_t exec{};
  ~Graph() { if (exec) cudaGraphExecDestroy(exec); }
};
}
int main(int argc, char **argv) {
  try {
    using namespace strata::core;
    using namespace strata::kernels;
    using strata::sycl_backend::Allocation;
    using strata::sycl_backend::MemoryKind;
    if (argc < 2 || argc > 3)
      throw std::invalid_argument("usage: sycl_gr_bench PACK [REPEATS=50]");
    const int repeats = argc == 3 ? std::stoi(argv[2]) : 50;
    if (repeats < 1 || repeats > 1000)
      throw std::invalid_argument("repeats must be 1..1000");
    const std::string pack = argv[1];
    std::ifstream index(pack + "/index.txt");
    if (!index) throw std::runtime_error("cannot read pack index");
    std::set<std::string> skip;
    std::string line, name, err;
    while (std::getline(index, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream fields(line);
      if (!(fields >> name)) throw std::runtime_error("invalid pack index row");
      if (!gr_weight(name)) skip.insert(name);
    }
    uint64_t bytes = 0;
    if (!WeightTable::pool_bytes(pack, bytes, err, &skip))
      throw std::runtime_error(err);
    auto runtime = strata::sycl_backend::runtime_for();
    Allocation arena(runtime, bytes, MemoryKind::Device);
    WeightTable table;
    if (!table.load(pack, arena.data(), bytes, err, &skip))
      throw std::runtime_error(err);
    Allocation residual(runtime, D * 4, MemoryKind::Device);
    Allocation block(runtime, N * 4, MemoryKind::Device);
    Allocation previous(runtime, HC * 4, MemoryKind::Device);
    Allocation low(runtime, LR * 4, MemoryKind::Device);
    Allocation scale(runtime, HC * 4, MemoryKind::Device);
    Allocation mixed(runtime, N * 4, MemoryKind::Device);
    Allocation injection(runtime, HC * 4, MemoryKind::Device);
    Allocation normalized(runtime, D * 4, MemoryKind::Device);
    Allocation gates(runtime, D * 4, MemoryKind::Device);
    std::vector<float> input(D);
    for (int i = 0; i < D; ++i) input[i] = std::sin(float(i) * .13f);
    check(cudaMemcpy(residual.data(), input.data(), D * 4, cudaMemcpyHostToDevice));
    check(cudaMemset(block.data(), 0, N * 4));
    check(cudaMemset(previous.data(), 0, HC * 4));
    check(cudaMemset(injection.data(), 0, HC * 4));
    cudaStream_t stream{};
    cudaEvent_t start{}, stop{};
    check(cudaStreamCreate(&stream));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    std::cout << "name,apply,inject,spmd_ms,esimd_ms,unequal_output_bits\n";
    int count = 0;
    for (const auto &[tensor, ref] : table.all()) {
      if (!ref.resident || !gr_weight(tensor) || !tensor.ends_with("down.weight")) continue;
      const std::string prefix = tensor.substr(0, tensor.size() - 11);
      auto weight = [&](const std::string &suffix, WeightKind kind, int64_t n0, int64_t n1) {
        const auto *w = table.find(prefix + suffix);
        if (!w || !w->data || w->kind != kind || w->ne0 != n0 || w->ne1 != n1)
          throw std::runtime_error("invalid GR weight " + prefix + suffix);
        return w->data;
      };
      FusedGrArgs a;
      a.R = residual.as<float>(); a.R_out = residual.as<float>();
      a.apply = tensor.starts_with("blk.");
      // Zero block outputs keep R fixed across repeats while exercising the pending write.
      a.bo_prev = block.as<float>(); a.inj_prev = previous.as<float>();
      a.w_norm = static_cast<const float*>(weight("norm.weight", WeightKind::F32, D, 0));
      a.w_down = static_cast<const uint16_t*>(weight("down.weight", WeightKind::Bf16InF32, D, LR));
      a.w_up = static_cast<const uint16_t*>(weight("up.weight", WeightKind::Bf16InF32, LR, D));
      if (a.apply) a.w_inject = static_cast<const uint16_t*>(weight("inject.weight", WeightKind::Bf16InF32, D, HC));
      a.lo = low.as<float>(); a.rs = scale.as<float>();
      a.mixed = mixed.as<float>(); a.inject_out = injection.as<float>();
      std::vector<float> reference;
      auto outputs = [&] {
        std::vector<float> result;
        for (auto values : {read(a.R, D), read(a.rs, HC), read(a.lo, LR),
                            read(a.mixed, N), read(a.inject_out, HC)})
          result.insert(result.end(), values.begin(), values.end());
        return result;
      };
      float timing[2];
      for (int variant = 0; variant < 2; ++variant) {
        a.xn = variant ? normalized.as<float>() : nullptr;
        a.gates = variant ? gates.as<float>() : nullptr;
        fused_gr_read(a, stream);
        check(cudaStreamSynchronize(stream));
        auto result = outputs();
        if (variant == 0) reference = std::move(result);
        else if (std::memcmp(reference.data(), result.data(), result.size() * 4)) {
          size_t unequal = 0, first = result.size();
          for (size_t i = 0; i < result.size(); ++i)
            if (std::memcmp(&reference[i], &result[i], 4)) {
              ++unequal;
              first = std::min(first, i);
            }
          std::cerr << "output mismatches " << unequal << ", first index " << first
                    << ", " << reference[first] << " vs " << result[first] << '\n';
          throw std::runtime_error("GR output bits differ for " + prefix);
        }
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < repeats; ++i) fused_gr_read(a, stream);
        cudaGraph_t recorded{};
        Graph graph;
        check(cudaStreamEndCapture(stream, &recorded));
        check(cudaGraphInstantiate(&graph.exec, recorded, 0ull));
        check(cudaGraphDestroy(recorded));
        check(cudaGraphLaunch(graph.exec, stream));
        check(cudaStreamSynchronize(stream));
        std::vector<float> samples;
        for (int trial = 0; trial < 3; ++trial) {
          check(cudaEventRecord(start, stream));
          check(cudaGraphLaunch(graph.exec, stream));
          check(cudaEventRecord(stop, stream));
          check(cudaEventSynchronize(stop));
          float ms = 0;
          check(cudaEventElapsedTime(&ms, start, stop));
          samples.push_back(ms / repeats);
        }
        std::sort(samples.begin(), samples.end());
        timing[variant] = samples[1];
      }
      std::cout << prefix << ',' << a.apply << ',' << bool(a.w_inject) << ','
                << timing[0] << ',' << timing[1] << ",0\n";
      ++count;
    }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaStreamDestroy(stream));
    if (!count) throw std::runtime_error("pack contains no GR weights");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
