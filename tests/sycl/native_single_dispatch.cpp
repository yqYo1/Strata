#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/sycl/native_q2.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata::core;
using namespace strata::kernels::cpu;
void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}
struct Fixture : ExpertSource {
  std::filesystem::path path;
  std::vector<std::vector<uint8_t>> blobs;
  NativeFmt fmt;
  explicit Fixture(int type = 8) {
    std::string err;
    require(native_fmt(type, type, H, FF, fmt, err), err); // Q8_0 or Q2_0
    path = std::filesystem::temp_directory_path() /
           ("strata-native-single-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
    {
      std::ofstream meta(path / "native_experts.txt");
      meta << "0 " << type << " " << type << " 0 " << fmt.bytes << "\n1 "
           << type << " " << type << " " << 3 * fmt.bytes << " " << fmt.bytes
           << "\n";
      require(bool(meta), "write layout");
    }
    require(expert_layout_load(path.string(), 2, 3, err), err);
    blobs.resize(6, std::vector<uint8_t>(fmt.bytes));
    const size_t block_bytes = type == 8 ? 34 : 18;
    for (int e = 0; e < 6; ++e)
      for (size_t b = 0; b < fmt.bytes / block_bytes; ++b) {
        auto *p = blobs[e].data() + b * block_bytes;
        p[0] = 0;
        p[1] = 0x18; // binary16 1/512
        for (size_t j = 0; j < block_bytes - 2; ++j)
          p[2 + j] = type == 8
                         ? uint8_t(int((b * 17 + j * 11 + e * 5) % 31) - 15)
                         : uint8_t(b * 17 + j * 11 + e * 5);
      }
  }
  ~Fixture() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  const uint8_t *blob(int64_t l, int64_t e) override {
    return l >= 0 && l < 2 && e >= 0 && e < 3 ? blobs[l * 3 + e].data()
                                              : nullptr;
  }
};
void check(cudaError_t status) {
  require(status == cudaSuccess, cudaGetErrorString(status));
}
template <class T> struct DeviceBuffer {
  T *p = nullptr;
  explicit DeviceBuffer(size_t count) {
    check(cudaMalloc(&p, count * sizeof(T)));
  }
  ~DeviceBuffer() { cudaFree(p); }
};
void quantize_contract() {
  cudaStream_t stream{};
  check(cudaStreamCreate(&stream));
  DeviceBuffer<float> input(H), scales(H / 32);
  DeviceBuffer<uint8_t> blocks(H / 32 * 34);
  std::vector<float> values(H), got_scales(H / 32);
  std::vector<uint8_t> got(H / 32 * 34);
  for (int i = 0; i < H; ++i)
    values[i] = float(std::sin(i * .713) * (1 + i % 113));
  for (int i = 0; i < 32; ++i)
    values[i] = 0;
  const float ties[] = {-127.f, -126.5f, -.5f, 0.f, .5f, 126.5f, 127.f};
  for (int i = 0; i < 32; ++i)
    values[32 + i] = ties[i % 7];
  ActQ reference;
  act_quant_any(values.data(), H, reference);
  check(cudaMemcpy(input.p, values.data(), H * 4, cudaMemcpyHostToDevice));
  strata::kernels::native_q2_quantize_cpu_order(input.p, blocks.p, scales.p, H,
                                                stream);
  check(cudaStreamSynchronize(stream));
  check(cudaMemcpy(got.data(), blocks.p, got.size(), cudaMemcpyDeviceToHost));
  check(cudaMemcpy(got_scales.data(), scales.p, got_scales.size() * 4,
                   cudaMemcpyDeviceToHost));
  for (int b = 0; b < H / 32; ++b) {
    require(got_scales[b] == reference.scale[b],
            "native Q2 FP32 quantization scale");
    require(std::memcmp(got.data() + b * 34 + 2, reference.q + b * 32, 32) == 0,
            "native Q2 activation codes, including ties and zeros");
    const int sum =
        int(int16_t(uint16_t(got[b * 34]) | (uint16_t(got[b * 34 + 1]) << 8)));
    require(sum == reference.sum[b], "native Q2 stored activation code sum");
  }
  check(cudaStreamDestroy(stream));
}
std::vector<float> gpu_hits(Fixture &f, ExpertPool &pool,
                            bool cpu_order = false, bool capture = true) {
  constexpr int K = 3;
  cudaStream_t stream{};
  check(cudaStreamCreate(&stream));
  ExpertCache cache;
  std::string error;
  require(cache.open_sized({int64_t(f.fmt.bytes + 256), int64_t(f.fmt.bytes)},
                           2, 3, error),
          error);
  DeviceBuffer<float> mixed(H), parts(K * H), hit_out(K * H), scales(H / 32);
  DeviceBuffer<uint8_t> xq(H / 32 * 36),
      scratch(strata::kernels::native_expert_scratch_bytes(K, FF));
  DeviceBuffer<int32_t> slots(K), dst(K), meta(2 * K + 2);
  DeviceBuffer<unsigned long long> ptr(K);
  ExpertDispatch d;
  d.pool = &pool;
  d.src = &f;
  d.n_expert = 3;
  d.cache = &cache;
  d.cache_base = cache.device_slot(0);
  d.cache_blob = f.fmt.bytes;
  d.hit_scratch = scratch.p;
  d.parts_out = parts.p;
  d.hit_out = hit_out.p;
  d.parts_elems = K * H;
  d.mixed = mixed.p;
  d.x_q8_0_hit = xq.p;
  d.x_q8_0_hit_scale = scales.p;
  d.d_slot = slots.p;
  d.d_dst = dst.p;
  d.h_slot.resize(K);
  d.h_dst.resize(K);
  d.hit_native = true;
  d.hit_cpu_order = cpu_order;
  d.native_hit_ptr = ptr.p;
  d.native_hit_meta = meta.p;
  d.h_native_hit_ptr.resize(K);
  d.h_native_hit_meta.resize(2 * K + 2);
  if (capture) {
    require(expert_hit_prepare_graphs(d, stream, error), error);
    require(bool(d.native_hit_graphs) == !cpu_order, "native hit graph mode");
    if (!cpu_order)
      require(!expert_hit_prepare_graphs(d, stream, error),
              "duplicate native hit setup must be refused");
  }
  std::vector<float> snapshots;
  float max_error = 0;
  for (int iteration = 0; iteration < 4; ++iteration) {
    const int layer = iteration == 2 ? 1 : 0;
    int32_t ids[K] = {iteration % 2 ? 1 : 2, iteration % 2 ? 2 : 0,
                      iteration % 2 ? 0 : 1};
    float weights[K] = {.1f, .3f, .6f};
    std::vector<float> x(H), cpu(K * H), reference(K * H), actual(K * H),
        hit(K * H);
    for (int i = 0; i < H; ++i)
      x[i] = std::sin(float(i) * .03f + iteration) * .7f;
    ExpertDispatch reference_dispatch;
    reference_dispatch.pool = &pool;
    reference_dispatch.src = &f;
    reference_dispatch.n_expert = 3;
    reference_dispatch.layers = layer;
    expert_pool_dispatch(&reference_dispatch, x.data(), ids, weights, H, K,
                         reference.data());
    require(!reference_dispatch.failed, "native CPU reference dispatch");
    float reference_max = 0;
    for (float value : reference)
      reference_max = std::max(reference_max, std::abs(value));
    require(reference_max > 1e-5f, "nonzero expert reference fixture");
    check(cudaMemcpy(mixed.p, x.data(), H * 4, cudaMemcpyHostToDevice));
    d.layers = layer;
    expert_hit_run(&d, stream, HitPhase::Launch, ids, K);
    require(!d.failed, d.fail ? d.fail : "GPU hit launch");
    require(d.n_hits == (layer == 0 ? 2 : 0), "expected GPU hit count");
    expert_pool_dispatch(&d, x.data(), ids, weights, H, K, cpu.data());
    require(!d.failed, d.fail ? d.fail : "GPU/CPU dispatch");
    for (int e = 0; e < K; ++e)
      for (int j = 0; j < H; ++j)
        require(cpu[e * H + j] == (d.is_hit[e] ? 0 : reference[e * H + j]),
                "CPU hit/miss ownership");
    check(cudaMemcpyAsync(parts.p, cpu.data(), cpu.size() * 4,
                          cudaMemcpyHostToDevice, stream));
    expert_hit_run(&d, stream, HitPhase::Combine, ids, K);
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(actual.data(), parts.p, actual.size() * 4,
                     cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < actual.size(); ++j) {
      max_error = std::max(max_error, std::abs(actual[j] - reference[j]));
      require(std::isfinite(actual[j]) &&
                  std::abs(actual[j] - reference[j]) <=
                      1e-4f * (1 + std::abs(reference[j])),
              "native GPU/CPU expert result tolerance");
    }
    // A second Combine must not add the same hits twice.
    expert_hit_run(&d, stream, HitPhase::Combine, ids, K);
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(hit.data(), parts.p, hit.size() * 4,
                     cudaMemcpyDeviceToHost));
    require(hit == actual, "duplicate hit combination");
    snapshots.insert(snapshots.end(), actual.begin(), actual.end());
  }
  require(d.cache_admitted == 2 && d.cache_hits == 4 && d.cache_refused == 6,
          "cache ownership counters");
  check(cudaStreamDestroy(stream));
  std::cout << "Native GPU hit handoff: mixed/miss-only layers, sized slots, "
               "reordered hits; max CPU difference "
            << max_error << "\n";
  return snapshots;
}
int main() {
  try {
    Fixture f;
    ExpertPool pool(2, false, true);
    ExpertDispatch d;
    d.pool = &pool;
    d.src = &f;
    d.n_expert = 3;
    std::vector<float> x(H), out(3 * H), ref(3 * H), ff(FF);
    std::vector<uint8_t> act(kNativeActBytes), hq(kNativeHBytes);
    int32_t ids[3] = {2, 0, 1};
    float weights[3] = {.1f, .3f, .6f};
    for (int l = 0; l < 2; ++l) {
      for (int i = 0; i < H; ++i)
        x[i] = std::sin(float(i) * .07f + l) * .7f;
      native_quant_act(f.fmt, x.data(), act.data());
      for (int e = 0; e < 3; ++e) {
        const void *a = act.data();
        float *fp = ff.data();
        native_gu_rows(f.fmt, f.blob(l, ids[e]), &a, 1, &fp, 0, FF);
        native_quant_h(f.fmt, ff.data(), hq.data());
        const void *q = hq.data();
        float *op = ref.data() + e * H;
        native_down_rows(f.fmt, f.blob(l, ids[e]), &q, 1, &op, 0, H);
      }
      expert_pool_dispatch(&d, x.data(), ids, weights, H, 3, out.data());
      require(!d.failed, d.fail ? d.fail : "dispatch failed");
      require(out == ref,
              "single-token dispatch differs from direct native rows");
      require(d.layers == l + 1 && d.experts == 3 * (l + 1),
              "dispatch counters");
    }
    d.layers = 0;
    ids[1] = 3;
    expert_pool_dispatch(&d, x.data(), ids, weights, H, 3, out.data());
    require(d.failed && d.fail_expert == 3 && d.layers == 0,
            "invalid routed id");
    d.failed = false;
    d.fail = nullptr;
    d.host_res = ids;
    expert_pool_dispatch(&d, x.data(), ids, weights, H, 3, out.data());
    require(d.failed &&
                std::string(d.fail).find("matching native GPU hit decisions") !=
                    std::string::npos,
            "native per-layer GPU residency must be rejected");
    auto graph_bits = [&](Fixture& fixture) {
      const auto plain = gpu_hits(fixture, pool, false, false);
      const auto replay = gpu_hits(fixture, pool);
      require(plain.size() == replay.size() &&
                  std::memcmp(plain.data(), replay.data(), plain.size() * 4) == 0,
              "native hit graph/uncaptured output bits");
    };
    graph_bits(f);
    Fixture q2(42);
    graph_bits(q2);
    quantize_contract();
    gpu_hits(q2, pool, true);
    std::cout << "Native single dispatch: changed activations/layers, routing "
                 "order, raw outputs and invalid id PASS\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
