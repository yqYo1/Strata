// Isolated, warm-weight MMVQ timings for the real dense projection directory.
// The timed graph repeats one projection; this is not token-generation time.
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
bool projection(const std::string &name) {
  if (!name.starts_with("blk.")) return false;
  for (auto suffix : {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
                      ".attn_q.weight", ".attn_k.weight", ".attn_v.weight",
                      ".attn_output.weight", ".ffn_gate_shexp.weight",
                      ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"})
    if (name.ends_with(suffix)) return true;
  return false;
}
}
int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 5 ||
        (argc == 5 && std::string(argv[4]) != "--compare-q3" &&
         std::string(argv[4]) != "--compare-iq4xs" &&
         std::string(argv[4]) != "--spmd-iq4xs"))
      throw std::invalid_argument("usage: sycl_projection_bench SHARD1 [REPEATS=50] [COLUMNS=1] [--compare-q3|--compare-iq4xs|--spmd-iq4xs]");
    const int repeats = argc >= 3 ? std::stoi(argv[2]) : 50;
    const int columns = argc >= 4 ? std::stoi(argv[3]) : 1;
    const bool spmd_iq4xs = argc == 5 && std::string(argv[4]) == "--spmd-iq4xs";
    const bool compare = argc == 5 && !spmd_iq4xs;
    const uint32_t compare_type = compare && std::string(argv[4]) == "--compare-iq4xs" ? 23u : 11u;
    if (compare && columns != 1) throw std::invalid_argument("projection comparison requires one column");
    if (repeats < 1 || repeats > 1000 || columns < 1 || columns > 8)
      throw std::invalid_argument("repeats must be 1..1000 and columns 1..8");
    const auto model = strata::GgufModel::open(argv[1]);
    auto runtime = strata::sycl_backend::runtime_for();
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream));
    cudaEvent_t start{}, stop{};
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    std::cout << "name,type,n_in,n_out,weight_bytes,columns,device_ms\n";
    size_t count = 0;
    for (size_t shard = 0; shard < model.size(); ++shard)
      for (const auto &t : model.shard(shard).tensors()) {
        if (!projection(t.name) || t.shape.size() != 2 ||
            !strata::kernels::native_mmvq_supported(t.type)) continue;
        if (!model.in_bounds(t, shard) || t.shape[0] > INT32_MAX ||
            t.shape[1] > INT32_MAX)
          throw std::runtime_error("invalid tensor " + t.name);
        const int ni = int(t.shape[0]), no = int(t.shape[1]);
        const size_t bytes = strata::kernels::native_mmvq_weight_bytes(t.type, ni, no);
        using strata::sycl_backend::Allocation;
        using strata::sycl_backend::MemoryKind;
        Allocation weight(runtime, bytes, MemoryKind::Device);
        Allocation input(runtime, size_t(ni) * columns * 4, MemoryKind::Device);
        Allocation quant(runtime, strata::kernels::native_q8_1_bytes(ni, columns), MemoryKind::Device);
        Allocation output(runtime, size_t(no) * columns * 4, MemoryKind::Device);
        std::vector<float> host(size_t(ni) * columns);
        for (size_t i = 0; i < host.size(); ++i)
          host[i] = std::sin(float(i) * .13f);
        check(cudaMemcpy(weight.data(), model.shard(shard).tensor_data(t), bytes, cudaMemcpyHostToDevice));
        check(cudaMemcpy(input.data(), host.data(), host.size() * 4, cudaMemcpyHostToDevice));
        strata::kernels::quantize_q8_1_rows(input.as<float>(), columns, ni, quant.data(), stream);
        strata::kernels::iq_set_old_kernels(spmd_iq4xs && t.type == 23);
        auto run = [&] {
          strata::kernels::native_mmvq(t.type, weight.data(), quant.data(), output.as<float>(), ni, no, columns, stream);
        };
        for (int i = 0; i < 5; ++i) run();
        check(cudaStreamSynchronize(stream));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < repeats; ++i) run();
        cudaGraph_t graph{};
        cudaGraphExec_t exec{};
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&exec, graph, 0ull));
        check(cudaGraphDestroy(graph));
        check(cudaGraphLaunch(exec, stream));
        check(cudaStreamSynchronize(stream));
        std::vector<float> samples;
        for (int trial = 0; trial < 3; ++trial) {
          check(cudaEventRecord(start, stream));
          check(cudaGraphLaunch(exec, stream));
          check(cudaEventRecord(stop, stream));
          check(cudaEventSynchronize(stop));
          float ms = 0;
          check(cudaEventElapsedTime(&ms, start, stop));
          samples.push_back(ms / repeats);
        }
        std::sort(samples.begin(), samples.end());
        std::cout << t.name << ',' << t.type << ',' << ni << ',' << no << ','
                  << bytes << ',' << columns << ',' << samples[1] << '\n';
        check(cudaGraphExecDestroy(exec));
        if (compare && t.type == compare_type) {
          // The three-column SPMD path retains the original 128-lane
          // arithmetic. Its first column has the same inputs as this ESIMD run.
          Allocation oracle_input(runtime, quant.size() * 3, MemoryKind::Device);
          Allocation oracle_output(runtime, size_t(no) * 3 * 4, MemoryKind::Device);
          for (int c = 0; c < 3; ++c)
            check(cudaMemcpy(oracle_input.as<uint8_t>() + c * quant.size(), quant.data(), quant.size(), cudaMemcpyDeviceToDevice));
          strata::kernels::native_mmvq_set_multi_exact(false);
          strata::kernels::native_mmvq(t.type, weight.data(), oracle_input.data(), oracle_output.as<float>(), ni, no, 3, stream);
          check(cudaStreamSynchronize(stream));
          strata::kernels::native_mmvq_set_multi_exact(true);
          std::vector<float> reference(no), candidate(no);
          check(cudaMemcpy(reference.data(), oracle_output.data(), no * 4, cudaMemcpyDeviceToHost));
          check(cudaMemcpy(candidate.data(), output.data(), no * 4, cudaMemcpyDeviceToHost));
          size_t unequal = 0;
          float max_error = 0, max_value = 0;
          for (int row = 0; row < no; ++row) {
            unequal += std::memcmp(&reference[row], &candidate[row], 4) != 0;
            max_error = std::max(max_error, std::abs(reference[row] - candidate[row]));
            max_value = std::max(max_value, std::abs(reference[row]));
          }
          std::cerr << t.name << ",unequal=" << unequal << ",rows=" << no
                    << ",max_abs=" << max_error << ",max_value=" << max_value << '\n';
        }
        ++count;
      }
    if (!count) throw std::runtime_error("no supported dense projections");
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));
    check(cudaStreamDestroy(stream));
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
