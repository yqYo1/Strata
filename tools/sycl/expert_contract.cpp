// Replay recorded Q2_0 expert inputs against CPU experts and an independent
// double-precision interpretation of the GPU's actual Q8_1 blocks.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/sycl/native_q2.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata;
using namespace strata::kernels;
using namespace strata::kernels::cpu;
void require(bool value, const char *why) {
  if (!value)
    throw std::runtime_error(why);
}
void check(cudaError_t value) {
  if (value != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(value));
}
template <class T> struct Buffer {
  T *p = nullptr;
  explicit Buffer(size_t count) { check(cudaMalloc(&p, count * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
};
float half_at(const uint8_t *p) {
  return f32_from_f16(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
}
void half_put(uint8_t *p, float value) {
  uint16_t v = f16_from_f32(value);
  p[0] = uint8_t(v);
  p[1] = uint8_t(v >> 8);
}
std::vector<uint8_t> quant81(const float *x, int n) {
  std::vector<uint8_t> out(size_t(n / 32) * 36);
  for (int c = 0; c < n / 32; ++c) {
    float maximum = 0;
    std::array<float, 32> sum;
    for (int j = 0; j < 32; ++j) {
      sum[j] = x[c * 32 + j];
      maximum = std::max(maximum, std::abs(sum[j]));
    }
    for (int offset = 16; offset; offset /= 2) {
      auto before = sum;
      for (int j = 0; j < 32; ++j)
        sum[j] = before[j] + before[j ^ offset];
    }
    const float scale = maximum / 127.f;
    auto *block = out.data() + size_t(c) * 36;
    half_put(block, scale);
    half_put(block + 2, sum[0]);
    for (int j = 0; j < 32; ++j)
      block[4 + j] = uint8_t(
          int8_t(maximum == 0 ? 0 : int(std::round(x[c * 32 + j] / scale))));
  }
  return out;
}
double q2_row(const uint8_t *row, const uint8_t *x, int n) {
  double result = 0;
  for (int c = 0; c < n / 32; ++c) {
    const auto *w = row + size_t(c / 2) * 18;
    const auto *q = x + size_t(c) * 36;
    int dot = 0;
    for (int j = 0; j < 32; ++j) {
      const int element = (c % 2) * 32 + j;
      const int value =
          int((w[2 + element / 4] >> (2 * (element % 4))) & 3) - 1;
      dot += value * int(int8_t(q[4 + j]));
    }
    result += double(half_at(w)) * half_at(q) * dot;
  }
  return result;
}
void read_exact(std::ifstream &input, void *p, size_t bytes) {
  require(bool(input.read(static_cast<char *>(p), std::streamsize(bytes))),
          "truncated expert input trace");
}
int main(int argc, char **argv) {
  try {
    if ((argc != 6 && argc != 7) ||
        (argc == 7 && std::string(argv[6]) != "--bench"))
      throw std::runtime_error(
          "usage: sycl_expert_contract PACK SHARD1 TRACE FIRST_RECORD COUNT [--bench]");
    const bool bench = argc == 7;
    const int first = std::stoi(argv[4]), count = std::stoi(argv[5]);
    require(first >= 0 && count > 0 && first <= INT32_MAX - count,
            "invalid record interval");
    std::string error;
    require(expert_layout_load(argv[1], 48, 512, error), error.c_str());
    core::FileExpertSource source;
    source.set_gguf(argv[2]);
    require(source.open(argv[1], 48, 512, error), error.c_str());
    ExpertPool pool(4, false, true);
    cudaStream_t stream{};
    check(cudaStreamCreate(&stream));
    std::ifstream input(argv[3], std::ios::binary);
    require(bool(input), "cannot open trace");
    std::cout << "record,layer,experts,gpu_projection_scaled_error,cpu_gpu_max_"
                 "absolute,cpu_gpu_rmse,cpu_order_max_absolute,cpu_order_rmse,"
                 "cpu_order_up_mismatches,cpu_order_hidden_max_absolute,"
                 "cpu_order_hidden_codes_differ,cpu_order_down_mismatches";
    if (bench) std::cout << ",spmd_device_ms,esimd_device_ms";
    std::cout << '\n';
    for (int record = 0; record < first + count; ++record) {
      int32_t header[3];
      read_exact(input, header, sizeof(header));
      const int layer = header[0], width = header[1], k = header[2];
      require(layer >= 0 && layer < 48 && width == H && k > 0 && k <= 15,
              "invalid expert trace geometry");
      std::vector<int32_t> ids(k);
      std::vector<float> weights(k), x(width);
      read_exact(input, ids.data(), k * 4);
      read_exact(input, weights.data(), k * 4);
      read_exact(input, x.data(), width * 4);
      for (int id : ids)
        require(id >= 0 && id < 512, "invalid expert id");
      for (float value : x)
        require(std::isfinite(value), "nonfinite expert activation");
      if (record < first)
        continue;
      const auto &fmt = expert_layout().fmt.at(size_t(layer));
      require(fmt.gu_type == 42 && fmt.d_type == 42 && fmt.n_ff == FF,
              "diagnostic supports Q2_0/Q2_0 experts");
      const auto layout = native_expert_layout(42, 42, H, FF);
      std::vector<float> cpu(size_t(k) * H), gpu(cpu.size());
      core::ExpertDispatch dispatch;
      dispatch.pool = &pool;
      dispatch.src = &source;
      dispatch.layers = layer;
      core::expert_pool_dispatch(&dispatch, x.data(), ids.data(),
                                 weights.data(), H, k, cpu.data());
      require(!dispatch.failed,
              dispatch.fail ? dispatch.fail : "CPU dispatch failed");
      std::vector<uint8_t> blobs(size_t(k) * layout.bytes);
      for (int e = 0; e < k; ++e) {
        const auto *b = source.blob(layer, ids[e]);
        require(b != nullptr, "missing expert blob");
        std::memcpy(blobs.data() + size_t(e) * layout.bytes, b, layout.bytes);
      }
      Buffer<uint8_t> db(blobs.size()), qx(H / 32 * 36),
          scratch(native_expert_scratch_bytes(k, FF));
      Buffer<float> dx(H), out(gpu.size());
      Buffer<int32_t> starts(k + 1), tok(k), dst(k), groups(1);
      Buffer<unsigned long long> ptr(k);
      std::vector<unsigned long long> addresses(k);
      std::vector<int32_t> start(k + 1), zeros(k, 0), destinations(k);
      for (int e = 0; e < k; ++e) {
        addresses[e] = (unsigned long long)(db.p + size_t(e) * layout.bytes);
        start[e] = destinations[e] = e;
      }
      start[k] = k;
      check(
          cudaMemcpy(db.p, blobs.data(), blobs.size(), cudaMemcpyHostToDevice));
      check(cudaMemcpy(dx.p, x.data(), H * 4, cudaMemcpyHostToDevice));
      check(cudaMemcpy(ptr.p, addresses.data(), k * 8, cudaMemcpyHostToDevice));
      check(cudaMemcpy(starts.p, start.data(), (k + 1) * 4,
                       cudaMemcpyHostToDevice));
      check(cudaMemcpy(tok.p, zeros.data(), k * 4, cudaMemcpyHostToDevice));
      check(cudaMemcpy(dst.p, destinations.data(), k * 4,
                       cudaMemcpyHostToDevice));
      check(cudaMemcpy(groups.p, &k, 4, cudaMemcpyHostToDevice));
      native_quantize_q8_1(dx.p, qx.p, H, 1, stream);
      native_expert_grouped(layout, ptr.p, starts.p, groups.p, dst.p, tok.p, k,
                            k, qx.p, scratch.p, out.p, stream);
      check(cudaStreamSynchronize(stream));
      std::vector<uint8_t> activation(H / 32 * 36),
          workspace(native_expert_scratch_bytes(k, FF));
      check(cudaMemcpy(activation.data(), qx.p, activation.size(),
                       cudaMemcpyDeviceToHost));
      check(cudaMemcpy(workspace.data(), scratch.p, workspace.size(),
                       cudaMemcpyDeviceToHost));
      check(cudaMemcpy(gpu.data(), out.p, gpu.size() * 4,
                       cudaMemcpyDeviceToHost));
      std::array<float, 2> device_ms{};
      if (bench) {
        // The input Q8_1 blocks and real expert weights stay resident.
        // Time full gate/up, activation, hidden quantization and down graphs.
        // Every recorded expert has one entry, so SPMD tile size is immaterial.
        cudaEvent_t start{}, stop{};
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&stop));
        for (int mode = 0; mode < 2; ++mode) {
          iq_set_old_kernels(mode == 0);
          auto run = [&] {
            native_expert_grouped(layout, ptr.p, starts.p, groups.p, dst.p,
                                  tok.p, k, k, qx.p, scratch.p, out.p, stream);
          };
          for (int i = 0; i < 5; ++i) run();
          check(cudaStreamSynchronize(stream));
          std::vector<float> result(gpu.size());
          check(cudaMemcpy(result.data(), out.p, result.size() * 4,
                           cudaMemcpyDeviceToHost));
          require(std::memcmp(result.data(), gpu.data(), gpu.size() * 4) == 0,
                  "SPMD/ESIMD real grouped output bits differ");
          check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
          constexpr int repeats = 50;
          for (int i = 0; i < repeats; ++i) run();
          cudaGraph_t graph{};
          cudaGraphExec_t exec{};
          check(cudaStreamEndCapture(stream, &graph));
          check(cudaGraphInstantiate(&exec, graph, 0ull));
          check(cudaGraphDestroy(graph));
          check(cudaGraphLaunch(exec, stream));
          check(cudaStreamSynchronize(stream));
          std::array<float, 3> samples{};
          for (float &ms : samples) {
            check(cudaEventRecord(start, stream));
            check(cudaGraphLaunch(exec, stream));
            check(cudaEventRecord(stop, stream));
            check(cudaEventSynchronize(stop));
            check(cudaEventElapsedTime(&ms, start, stop));
            ms /= repeats;
          }
          std::sort(samples.begin(), samples.end());
          device_ms[mode] = samples[1];
          check(cudaGraphExecDestroy(exec));
        }
        iq_set_old_kernels(false);
        check(cudaEventDestroy(start));
        check(cudaEventDestroy(stop));
      }
      require(activation == quant81(x.data(), H),
              "input Q8_1 differs from independent quantization");
      const size_t plane = (size_t(k) * FF * 4 + 255) & ~size_t(255);
      auto at = [&](int p, size_t i) {
        float value;
        std::memcpy(&value, workspace.data() + size_t(p) * plane + i * 4, 4);
        return value;
      };
      std::vector<float> hidden(size_t(k) * FF);
      for (size_t j = 0; j < hidden.size(); ++j) {
        const double g = at(0, j), u = at(1, j);
        hidden[j] = at(2, j);
        const double expected = g / (1 + std::exp(-g)) * u;
        require(std::isfinite(hidden[j]) && std::abs(hidden[j] - expected) <=
                                                1e-5 * (1 + std::abs(expected)),
                "SwiGLU reference mismatch");
      }
      auto hq = quant81(hidden.data(), int(hidden.size()));
      require(std::memcmp(hq.data(), workspace.data() + 3 * plane, hq.size()) ==
                  0,
              "hidden Q8_1 differs from independent quantization");
      double projection_error = 0, cpu_error = 0, square = 0;
      auto compare = [&](double expected, double actual) {
        require(std::isfinite(actual), "nonfinite GPU output");
        const double scaled =
            std::abs(expected - actual) / (1 + std::abs(expected));
        projection_error = std::max(projection_error, scaled);
        require(scaled <= 1e-4,
                "GPU projection differs from independent Q8_1 dot reference");
      };
      for (int e = 0; e < k; ++e) {
        const auto *b = blobs.data() + size_t(e) * layout.bytes;
        for (int row = 0; row < FF; ++row) {
          compare(q2_row(b + size_t(row) * layout.gu_row, activation.data(), H),
                  at(0, size_t(e) * FF + row));
          compare(q2_row(b + layout.up_off + size_t(row) * layout.gu_row,
                         activation.data(), H),
                  at(1, size_t(e) * FF + row));
        }
        for (int row = 0; row < H; ++row) {
          const size_t j = size_t(e) * H + row;
          compare(q2_row(b + layout.down_off + size_t(row) * layout.d_row,
                         hq.data() + size_t(e) * (FF / 32) * 36, FF),
                  gpu[j]);
          const double diff = double(gpu[j]) - cpu[j];
          cpu_error = std::max(cpu_error, std::abs(diff));
          square += diff * diff;
        }
      }
      Buffer<float> fp32_scales(H / 32);
      native_q2_quantize_cpu_order(dx.p, qx.p, fp32_scales.p, H, stream);
      native_q2_expert_cpu_order(ptr.p, dst.p, groups.p, k, qx.p, fp32_scales.p,
                                 scratch.p, out.p, stream);
      check(cudaStreamSynchronize(stream));
      std::vector<uint8_t> q80(H / 32 * 34);
      std::vector<float> host_scales(H / 32), cpu_order(gpu.size());
      check(cudaMemcpy(q80.data(), qx.p, q80.size(), cudaMemcpyDeviceToHost));
      check(cudaMemcpy(host_scales.data(), fp32_scales.p,
                       host_scales.size() * 4, cudaMemcpyDeviceToHost));
      check(cudaMemcpy(cpu_order.data(), out.p, cpu_order.size() * 4,
                       cudaMemcpyDeviceToHost));
      ActQ host_quant;
      act_quant_any(x.data(), H, host_quant);
      for (int c = 0; c < H / 32; ++c) {
        require(host_scales[c] == host_quant.scale[c],
                "CPU-order input scales differ from CPU ActQ");
        require(std::memcmp(q80.data() + c * 34 + 2, host_quant.q + c * 32,
                            32) == 0,
                "CPU-order input codes differ from CPU ActQ");
      }
      double order_error = 0, order_square = 0;
      check(cudaMemcpy(workspace.data(), scratch.p, workspace.size(),
                       cudaMemcpyDeviceToHost));
      const size_t pairs = size_t(k) * FF;
      const auto *order_gu = reinterpret_cast<const float *>(workspace.data());
      const auto *order_hq = workspace.data() + pairs * 8;
      const auto *order_hs = reinterpret_cast<const float *>(
          order_hq + ((pairs / 32 * 34 + 15) & ~size_t(15)));
      int up_mismatches = 0, hidden_codes_differ = 0, down_mismatches = 0;
      double hidden_error = 0;
      for (int e = 0; e < k; ++e) {
        std::vector<float> gate(FF), up(FF), hidden_cpu(FF), down(H);
        const ActQ *a[] = {&host_quant};
        float *g[] = {gate.data()}, *u[] = {up.data()}, *y[] = {down.data()};
        const uint8_t *blob = blobs.data() + size_t(e) * layout.bytes;
        q2_rows_any(blob, layout.gu_row, H / 64, a, 1, g, 0, FF);
        q2_rows_any(blob + layout.up_off, layout.gu_row, H / 64, a, 1, u, 0,
                    FF);
        for (int j = 0; j < FF; ++j) {
          up_mismatches += up[j] != order_gu[pairs + size_t(e) * FF + j];
          hidden_cpu[j] = gate[j] / (1.f + std::exp(-gate[j])) * up[j];
          hidden_error =
              std::max(hidden_error, std::abs(double(hidden_cpu[j]) -
                                              order_gu[size_t(e) * FF + j]));
        }
        ActQ hidden_quant, actual_quant;
        act_quant_any(hidden_cpu.data(), FF, hidden_quant);
        actual_quant.nchunks = FF / 32;
        for (int b = 0; b < FF / 32; ++b) {
          const size_t block = size_t(e) * (FF / 32) + b;
          const auto *codes = order_hq + block * 34 + 2;
          std::memcpy(actual_quant.q + b * 32, codes, 32);
          actual_quant.scale[b] = order_hs[block];
          int sum = 0;
          for (int j = 0; j < 32; ++j) {
            sum += actual_quant.q[b * 32 + j];
            hidden_codes_differ +=
                actual_quant.q[b * 32 + j] != hidden_quant.q[b * 32 + j];
          }
          actual_quant.sum[b] = sum;
          actual_quant.hx[b] = actual_quant.scale[b] * float(sum);
        }
        const ActQ *aq[] = {&actual_quant};
        q2_rows_any(blob + layout.down_off, layout.d_row, FF / 64, aq, 1, y, 0,
                    H);
        for (int j = 0; j < H; ++j)
          down_mismatches += down[j] != cpu_order[size_t(e) * H + j];
      }
      for (size_t j = 0; j < cpu.size(); ++j) {
        const double diff = double(cpu_order[j]) - cpu[j];
        require(std::isfinite(cpu_order[j]), "nonfinite CPU-order output");
        order_error = std::max(order_error, std::abs(diff));
        order_square += diff * diff;
      }
      std::cout << record << ',' << layer << ',' << k << ',' << projection_error
                << ',' << cpu_error << ',' << std::sqrt(square / gpu.size())
                << ',' << order_error << ','
                << std::sqrt(order_square / gpu.size()) << ',' << up_mismatches
                << ',' << hidden_error << ',' << hidden_codes_differ << ','
                << down_mismatches;
      if (bench) std::cout << ',' << device_ms[0] << ',' << device_ms[1];
      std::cout << '\n';
    }
    check(cudaStreamDestroy(stream));
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
