#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
template <class T> struct Buffer {
  sycl_backend::Allocation mem;
  Buffer(size_t n)
      : mem(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device) {}
  T *data() { return mem.as<T>(); }
  void put(const std::vector<T> &v) {
    runtime->wait(
        runtime->compute().memcpy(data(), v.data(), v.size() * sizeof(T)));
  }
  std::vector<T> get() {
    std::vector<T> v(mem.size() / sizeof(T));
    runtime->wait(runtime->compute().memcpy(v.data(), data(), mem.size()));
    return v;
  }
};
void check(bool ok, const char *msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
void run(RopeScaling sc, int width, bool multi, bool inplace, int use_table) {
  constexpr int maxpos = 262145, rows = 9, nr = 64;
  std::vector<float> c(size_t(maxpos) * 32), s(c.size());
  build_rope_table(nr, sc, maxpos, c.data(), s.data());
  // Independently construct sample float64 angles including the last context
  // position.
  for (int p : {0, 1, 8191, 32768, 262144})
    for (int i = 0; i < 32; ++i) {
      double a = p * std::pow(sc.freq_base, -2. * i / nr), ms = 1.;
      if (sc.type != RopeScalingType::None) {
        const double fs = sc.freq_scale();
        double cd[2];
        sc.corr_dims(nr, cd);
        const double mix =
            rope_yarn_ramp(float(cd[0]), float(cd[1]), i) * sc.ext_factor;
        a = (fs * a) * (1. - mix) + a * mix;
        ms = sc.mscale();
      }
      check(c[size_t(p) * 32 + i] == float(std::cos(a) * ms),
            "host cosine table");
      check(s[size_t(p) * 32 + i] == float(std::sin(a) * ms),
            "host sine table");
    }
  std::vector<int> pos = {0, 1, 17, 8191, 32767, 32768, 65535, 131071, 262144};
  std::vector<int32_t> m(rows * 3);
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < 3; ++j)
      m[r * 3 + j] = std::max(0, pos[r] - j * 7);
  if (multi)
    for (int r = 0; r < rows; ++r)
      pos[r] = r;
  std::vector<float> x(rows * width + 16, 1234.f);
  for (int i = 0; i < rows * width; ++i)
    x[i] = float(std::sin(.031 * i));
  Buffer<float> dc(c.size()), ds(s.size()), dx(x.size()), dy(x.size());
  Buffer<int> dp(pos.size());
  Buffer<int32_t> dm(m.size());
  dc.put(c);
  ds.put(s);
  dx.put(x);
  dy.put(std::vector<float>(x.size(), 1234.f));
  dp.put(pos);
  dm.put(m);
  mrope_table_set(multi ? dm.data() : nullptr);
  if (use_table)
    rope_table_set(dc.data(), ds.data(), (use_table == 2 ? 1024 : maxpos), sc);
  auto *out = inplace ? dx.data() : dy.data();
  rope_neox_apply(dx.data(), out, rows, width, nr, dc.data(), ds.data(),
                  dp.data(), &runtime->compute());
  auto y = inplace ? dx.get() : dy.get();
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < width; ++i) {
      float ref = x[r * width + i];
      if (i < nr) {
        int pair = i % 32, p = multi ? m[r * 3 + pair % 3] : pos[r];
        float a = x[r * width + pair], b = x[r * width + pair + 32];
        ref = i < 32
                  ? a * c[size_t(p) * 32 + pair] - b * s[size_t(p) * 32 + pair]
                  : a * s[size_t(p) * 32 + pair] + b * c[size_t(p) * 32 + pair];
      }
      check(y[r * width + i] == ref, "generic/table bit parity");
    }
  dx.put(x);
  native_rope_apply(dx.data(), out, rows, width, nr, sc, dp.data(),
                    &runtime->compute());
  y = inplace ? dx.get() : dy.get();
  double maxerr = 0.;
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < width; ++i) {
      float ref = x[r * width + i];
      if (i < nr) {
        int pair = i % 32, p = multi ? m[r * 3 + pair % 3] : pos[r];
        float co, si;
        if (use_table && (use_table == 1 || p < 1024)) {
          co = c[size_t(p) * 32 + pair];
          si = s[size_t(p) * 32 + pair];
        } else {
          const auto k = sc.kernel_args(nr);
          float ts = std::pow(float(sc.freq_base), -2.f / nr);
          float extrap = float(p) * std::pow(ts, float(pair));
          rope_scaled_angle(extrap, k.freq_scale, k.corr_low, k.corr_high,
                            k.ext_factor, k.attn_factor, pair, co, si);
        }
        float a = x[r * width + pair], b = x[r * width + pair + 32];
        ref = i < 32 ? a * co - b * si : a * si + b * co;
      }
      maxerr = std::max(maxerr, double(std::abs(y[r * width + i] - ref)));
      check(std::isfinite(y[r * width + i]), "native finite");
    }
  std::cout << "rope mode=" << int(sc.type) << " width=" << width
            << " mrope=" << multi << " inplace=" << inplace
            << " table=" << use_table << " max_abs=" << maxerr << "\n";
  check(maxerr <= (use_table == 1 ? 0. : 2e-6), "analytic float32 parity");
  for (size_t i = rows * width; i < y.size(); ++i)
    check(y[i] == 1234.f, "guard");
  mrope_table_set(nullptr);
  rope_table_release(dc.data());
  check(rope_table_for(sc).cos == nullptr, "release");
}
} // namespace
int main(int argc, char **) {
  try {
    const bool table = argc > 1;
    setenv("STRATA_ROPE_TABLE", table ? "1" : "0", 1);
    runtime = sycl_backend::runtime_for();
    for (int mode = 0; mode < 3; ++mode) {
      RopeScaling sc;
      sc.type = RopeScalingType(mode);
      if (mode) {
        sc.factor = 8;
        sc.attn_factor = 1.1;
        sc.ext_factor = mode == 2 ? 1. : 0.;
      }
      for (int width : {128, 256})
        for (bool m : {false, true}) {
          run(sc, width, m, m, table);
          if (table)
            run(sc, width, m, !m, 2);
        }
    }
    std::vector<float> a(256), b(256);
    auto rejects = [](auto f) {
      bool caught = false;
      try {
        f();
      } catch (const std::invalid_argument &) {
        caught = true;
      }
      check(caught, "invalid argument accepted");
    };
    rejects([&] { build_rope_table(63, 1e7, 1, a.data(), b.data()); });
    rejects([&] { build_rope_table(64, 1e7, 1, a.data(), a.data()); });
    RopeScaling invalid;
    invalid.type = RopeScalingType::Linear;
    invalid.factor = 0;
    rejects([&] { build_rope_table(64, invalid, 1, a.data(), b.data()); });
    Buffer<float> dx(512), dy(512);
    Buffer<int> dp(2);
    rejects([&] {
      rope_neox_apply(dx.data(), dx.data() + 1, 1, 128, 64, dy.data(),
                      dy.data() + 64, dp.data(), &runtime->compute());
    });
    rejects([&] {
      native_rope_apply(dx.data(), dy.data(), 1, 128, 64, RopeScaling{},
                        dp.data(), nullptr);
    });
    rejects([&] {
      native_rope_apply(dx.data(), dy.data(), 1, 129, 64, RopeScaling{},
                        dp.data(), &runtime->compute());
    });
    rejects([&] {
      native_rope_apply(dx.data(), dy.data(), 1, 128, 64, invalid, dp.data(),
                        &runtime->compute());
    });
    runtime->wait();
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
