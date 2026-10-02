#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <chrono>
#include <cmath>
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
  Fixture() {
    std::string err;
    require(native_fmt(8, 8, H, FF, fmt, err), err); // Q8_0 gate/up/down
    path = std::filesystem::temp_directory_path() /
           ("strata-native-single-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
    {
      std::ofstream meta(path / "native_experts.txt");
      meta << "0 8 8 0 " << fmt.bytes << "\n1 8 8 " << 3 * fmt.bytes << " "
           << fmt.bytes << "\n";
      require(bool(meta), "write layout");
    }
    require(expert_layout_load(path.string(), 2, 3, err), err);
    blobs.resize(6, std::vector<uint8_t>(fmt.bytes));
    for (int e = 0; e < 6; ++e)
      for (size_t b = 0; b < fmt.bytes / 34; ++b) {
        auto *p = blobs[e].data() + b * 34;
        p[0] = 0;
        p[1] = 0x18; // binary16 1/512
        for (int j = 0; j < 32; ++j)
          p[2 + j] = uint8_t(int((b * 17 + j * 11 + e * 5) % 31) - 15);
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
    require(d.failed && std::string(d.fail).find("without a GPU hit hook") !=
                            std::string::npos,
            "native per-layer GPU residency must be rejected");
    std::cout << "Native single dispatch: changed activations/layers, routing "
                 "order, raw outputs and invalid id PASS\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
