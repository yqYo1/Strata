#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels {
namespace {
using namespace sycl_backend;

size_t validate(const float *x, const uint8_t *blocks, int64_t n, int width,
                int bytes, bool inverse, float *scales = nullptr) {
  if (n < 0 || n % width)
    throw std::invalid_argument("SYCL quantization requires whole blocks");
  const auto count = checked_count(n / width, bytes);
  const auto floats = checked_count(n, sizeof(float));
  if (n) {
    if (inverse)
      validate_spans({{x, floats}}, {{blocks, count}});
    else if (scales)
      validate_spans({{blocks, count}, {scales, checked_count(n / width, 4)}},
                     {{x, floats}});
    else
      validate_spans({{blocks, count}}, {{x, floats}});
  }
  return size_t(n / width);
}
void put_half(uint8_t *p, float value) {
  const auto bits = f16_from_f32(value);
  p[0] = uint8_t(bits);
  p[1] = uint8_t(bits >> 8);
}
float get_half(const uint8_t *p) {
  return f32_from_f16(uint16_t(p[0]) | uint16_t(p[1]) << 8);
}
} // namespace

void quantize_q8_0(const float *x, uint8_t *blocks, int64_t n, void *stream) {
  const auto count = validate(x, blocks, n, 32, 34, false);
  for_each(count, stream, [=](size_t b) {
    auto *out = blocks + b * 34;
    const auto *in = x + b * 32;
    float maximum = 0;
    for (int i = 0; i < 32; ++i)
      maximum = sycl::fmax(maximum, sycl::fabs(in[i]));
    const float scale = sycl::ext::intel::math::fdiv_rn(maximum, 127.f);
    put_half(out, scale);
    for (int i = 0; i < 32; ++i) {
      // Canonical Q8_0 divides by the FP32 scale in FP64, then ties to even.
      const double q = maximum == 0
                           ? 0
                           : sycl::rint(sycl::ext::intel::math::ddiv_rn(
                                 double(in[i]), double(scale)));
      out[i + 2] = uint8_t(int8_t(sycl::clamp(q, -128., 127.)));
    }
  });
}

void quantize_q8_0_scaled(const float *x, uint8_t *blocks, float *scales,
                          int64_t n, void *stream) {
  if (n && !scales)
    throw std::invalid_argument("SYCL scaled Q8_0 requires scales");
  const auto count = validate(x, blocks, n, 32, 34, false, scales);
  for_each(count, stream, [=](size_t b) {
    auto *out = blocks + b * 34;
    const auto *in = x + b * 32;
    float maximum = 0;
    for (int i = 0; i < 32; ++i)
      maximum = sycl::fmax(maximum, sycl::fabs(in[i]));
    const float scale =
        maximum > 0 ? sycl::ext::intel::math::fdiv_rn(maximum, 127.f) : 0;
    const float inv =
        scale > 0 ? sycl::ext::intel::math::fdiv_rn(1.f, scale) : 0;
    scales[b] = scale;
    put_half(out, scale);
    for (int i = 0; i < 32; ++i) {
      // Match the CPU expert's reciprocal multiply and half-away rule.
      const float t = in[i] * inv;
      const float rounded = t + (t >= 0 ? .5f : -.5f);
      out[i + 2] = uint8_t(int8_t(sycl::clamp(int(rounded), -127, 127)));
    }
  });
}

void dequant_q8_0(const uint8_t *blocks, float *x, int64_t n, void *stream) {
  const auto count = validate(x, blocks, n, 32, 34, true);
  for_each(count, stream, [=](size_t b) {
    const auto *in = blocks + b * 34;
    const float scale = get_half(in);
    for (int i = 0; i < 32; ++i)
      x[b * 32 + i] = float(int8_t(in[i + 2])) * scale;
  });
}

void quantize_q8_K(const float *x, uint8_t *blocks, int64_t n, void *stream) {
  const auto count = validate(x, blocks, n, 256, 292, false);
  for_each(count, stream, [=](size_t b) {
    const auto *in = x + b * 256;
    auto *out = blocks + b * 292;
    float maximum = 0, signed_maximum = 0;
    for (int i = 0; i < 256; ++i) {
      const float magnitude = sycl::fabs(in[i]);
      if (magnitude > maximum) {
        maximum = magnitude;
        signed_maximum = in[i];
      }
    }
    if (maximum == 0) {
      for (int i = 0; i < 292; ++i)
        out[i] = 0;
      return;
    }
    const float inv = sycl::ext::intel::math::fdiv_rn(-127.f, signed_maximum);
    const float scale = sycl::ext::intel::math::fdiv_rn(1.f, inv);
    const auto bits = sycl::bit_cast<uint32_t>(scale);
    for (int i = 0; i < 4; ++i)
      out[i] = uint8_t(bits >> (8 * i));
    for (int group = 0; group < 16; ++group) {
      int sum = 0;
      for (int i = 0; i < 16; ++i) {
        // Separate FP32 product and addition are required (contraction off).
        const float product = inv * in[group * 16 + i];
        const float rounded = product + 12582912.f;
        const int nearest =
            int(sycl::bit_cast<uint32_t>(rounded) & 0x7fffff) - 0x400000;
        const int q = sycl::min(127, nearest);
        out[4 + group * 16 + i] = uint8_t(int8_t(q));
        sum += q;
      }
      const uint16_t s = uint16_t(int16_t(sum));
      out[260 + 2 * group] = uint8_t(s);
      out[261 + 2 * group] = uint8_t(s >> 8);
    }
  });
}

void dequant_q8_K(const uint8_t *blocks, float *x, int64_t n, void *stream) {
  const auto count = validate(x, blocks, n, 256, 292, true);
  for_each(count, stream, [=](size_t b) {
    const auto *in = blocks + b * 292;
    uint32_t bits = 0;
    for (int i = 0; i < 4; ++i)
      bits |= uint32_t(in[i]) << (8 * i);
    const float scale = sycl::bit_cast<float>(bits);
    for (int i = 0; i < 256; ++i)
      x[b * 256 + i] = float(int8_t(in[4 + i])) * scale;
  });
}
} // namespace strata::kernels
