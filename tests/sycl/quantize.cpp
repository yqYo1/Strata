#include "strata/kernels/quantize_act.hpp"
#include "strata/sycl/runtime.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace strata;
using namespace strata::kernels;
namespace {
std::shared_ptr<sycl_backend::Runtime> runtime;
template <typename T> struct Buffer {
  sycl_backend::Allocation storage;
  size_t count;
  explicit Buffer(size_t n)
      : storage(runtime, n * sizeof(T), sycl_backend::MemoryKind::Device),
        count(n) {}
  T *data() { return storage.as<T>(); }
  void upload(const std::vector<T> &v) {
    if (v.size() != count)
      throw std::logic_error("upload size");
    runtime->wait(runtime->compute().memcpy(data(), v.data(), storage.size()));
  }
  std::vector<T> read() {
    std::vector<T> v(count);
    runtime->wait(runtime->compute().memcpy(v.data(), data(), storage.size()));
    return v;
  }
};
uint16_t half(float value) {
  const _Float16 h = _Float16(value);
  uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}
float unhalf(const uint8_t *p) {
  _Float16 h;
  std::memcpy(&h, p, 2);
  return float(h);
}
void check(bool condition, const char *label) {
  if (!condition)
    throw std::runtime_error(label);
}
void run(int width, bool scaled) {
  constexpr int nb = 2048;
  const int bytes = width == 32 ? 34 : 292;
  const size_t n = size_t(width) * nb;
  std::vector<float> x(n), expected_scales(nb);
  std::mt19937 random(719);
  for (int b = 0; b < nb; ++b) {
    const float magnitude = std::ldexp(1.f, (b % 25) - 16);
    for (int i = 0; i < width; ++i)
      x[b * width + i] =
          float(int(random() % 2000001) - 1000000) * (magnitude / 1000000.f);
  }
  std::fill_n(x.begin(), width, 0.f);
  // Exactly representable scales and ties; strict-first signed maximum in Q8_K.
  for (int b = 1; b <= 4; ++b) {
    for (int i = 0; i < width; ++i)
      x[b * width + i] = float(i % 31 - 15) + .5f;
    x[b * width] = b & 1 ? 127.f : -127.f;
    x[b * width + 1] = -x[b * width];
  }
  std::vector<uint8_t> expected(nb * bytes + 16, 0xa5);
  for (int b = 0; b < nb; ++b) {
    auto *out = expected.data() + b * bytes;
    const auto *in = x.data() + b * width;
    float maximum = 0, signed_max = 0;
    for (int i = 0; i < width; ++i) {
      if (std::abs(in[i]) > maximum) {
        maximum = std::abs(in[i]);
        signed_max = in[i];
      }
    }
    if (width == 32) {
      const float scale = maximum / 127.f;
      expected_scales[b] = scale;
      const auto bits = half(scale);
      std::memcpy(out, &bits, 2);
      const float inv = scale > 0 ? 1.f / scale : 0;
      for (int i = 0; i < 32; ++i) {
        int q;
        if (scaled) {
          const float t = in[i] * inv;
          // FP32 addition is part of the CPU expert contract.
          q = int(t + std::copysign(.5f, t));
          q = std::clamp(q, -127, 127);
        } else {
          q = maximum == 0 ? 0
                           : int(std::nearbyint(double(in[i]) / double(scale)));
          q = std::clamp(q, -128, 127);
        }
        out[2 + i] = uint8_t(int8_t(q));
      }
    } else {
      const float inv = maximum == 0 ? 0 : -127.f / signed_max;
      const float scale = maximum == 0 ? 0 : 1.f / inv;
      std::memcpy(out, &scale, 4);
      // An independent ties-to-even operation replaces the GPU's bit trick.
      for (int i = 0; i < 256; ++i) {
        const float product = in[i] * inv;
        out[4 + i] =
            uint8_t(int8_t(std::min(127, int(std::nearbyint(product)))));
      }
      for (int g = 0; g < 16; ++g) {
        int16_t sum = 0;
        for (int i = 0; i < 16; ++i)
          sum += int8_t(out[4 + 16 * g + i]);
        std::memcpy(out + 260 + 2 * g, &sum, 2);
      }
    }
  }
  Buffer<float> input(n), scales(nb + 4), decoded(n + 4);
  Buffer<uint8_t> blocks(expected.size());
  input.upload(x);
  blocks.upload(std::vector<uint8_t>(expected.size(), 0xa5));
  scales.upload(std::vector<float>(nb + 4, 12345.f));
  decoded.upload(std::vector<float>(n + 4, 12345.f));
  auto *stream = &runtime->compute();
  if (width == 256)
    quantize_q8_K(input.data(), blocks.data(), n, stream);
  else if (scaled)
    quantize_q8_0_scaled(input.data(), blocks.data(), scales.data(), n, stream);
  else
    quantize_q8_0(input.data(), blocks.data(), n, stream);
  const auto actual = blocks.read();
  for (size_t i = 0; i < expected.size(); ++i)
    if (actual[i] != expected[i])
      throw std::runtime_error(
          "quantization byte mismatch width=" + std::to_string(width) +
          " scaled=" + std::to_string(scaled) + " byte=" + std::to_string(i) +
          " actual=" + std::to_string(actual[i]) +
          " expected=" + std::to_string(expected[i]));
  if (scaled) {
    const auto actual_scales = scales.read();
    for (int b = 0; b < nb; ++b) {
      if (actual_scales[b] != expected_scales[b]) {
        std::cerr << "Scale block " << b << std::hexfloat
                  << " actual=" << actual_scales[b]
                  << " expected=" << expected_scales[b] << std::defaultfloat
                  << '\n';
        throw std::runtime_error("FP32 scales");
      }
    }
    for (int i = nb; i < nb + 4; ++i)
      check(actual_scales[i] == 12345.f, "scale canary");
  }
  // Also exercise null-stream completion.
  if (width == 256)
    dequant_q8_K(blocks.data(), decoded.data(), n, nullptr);
  else
    dequant_q8_0(blocks.data(), decoded.data(), n, nullptr);
  const auto result = decoded.read();
  for (int b = 0; b < nb; ++b) {
    const auto *p = expected.data() + b * bytes;
    float scale;
    if (width == 256)
      std::memcpy(&scale, p, 4);
    else
      scale = unhalf(p);
    for (int i = 0; i < width; ++i)
      check(result[b * width + i] ==
                float(int8_t(p[(width == 256 ? 4 : 2) + i])) * scale,
            "dequantization");
  }
  for (size_t i = n; i < n + 4; ++i)
    check(result[i] == 12345.f, "dequant canary");
  std::cout << "Q8 width=" << width << " scaled=" << scaled << ": " << nb
            << " blocks byte-exact\n";
}
template <typename F> void rejects(F f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid shape/span accepted");
}
} // namespace
int main() {
  try {
    std::fesetenv(FE_DFL_ENV);
    runtime = sycl_backend::runtime_for();
    run(32, false);
    run(32, true);
    run(256, false);
    Buffer<float> x(512);
    Buffer<uint8_t> blocks(1024);
    rejects([&] { quantize_q8_0(x.data(), blocks.data(), -32, nullptr); });
    rejects([&] { dequant_q8_0(blocks.data(), x.data(), 33, nullptr); });
    rejects([&] { dequant_q8_K(blocks.data(), x.data(), 255, nullptr); });
    rejects([&] {
      quantize_q8_0_scaled(x.data(), blocks.data(), nullptr, 32, nullptr);
    });
    rejects([&] {
      quantize_q8_K(x.data(), reinterpret_cast<uint8_t *>(x.data()), 256,
                    nullptr);
    });
    rejects([&] {
      quantize_q8_0_scaled(x.data(), blocks.data(), x.data() + 1, 32, nullptr);
    });
    quantize_q8_0(nullptr, nullptr, 0, nullptr);
    quantize_q8_K(nullptr, nullptr, 0, nullptr);
    runtime->wait();
    std::cout << "SYCL activation quantization PASS\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
