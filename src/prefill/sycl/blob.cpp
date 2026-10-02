#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl/launch.hpp"
namespace strata::prefill {
namespace {
template <bool half>
void expand(const uint8_t *blob, uint16_t *gu, uint16_t *down, void *stream) {
  constexpr size_t gu_codes = 1280 * 640, down_codes = 2560 * 160;
  constexpr size_t gu_scale = gu_codes + down_codes,
                   down_scale = gu_scale + 1280 * 40 * 2;
  strata::sycl_backend::validate_spans(
      {{gu, 1280 * 2560 * 2}, {down, 2560 * 640 * 2}},
      {{blob, down_scale + 2560 * 10 * 2}});
  strata::sycl_backend::for_each(gu_codes + down_codes, stream, [=](size_t i) {
    const bool gate = i < gu_codes;
    const size_t index = gate ? i : i - gu_codes, width = gate ? 640 : 160,
                 row = index / width, byte = index % width;
    const size_t so =
        (gate ? gu_scale : down_scale) + (row * (width / 16) + byte / 16) * 2;
    const float scale = strata::kernels::f32_from_f16(
        uint16_t(blob[so]) | (uint16_t(blob[so + 1]) << 8));
    const uint8_t packed = blob[i];
    for (int j = 0; j < 4; ++j) {
      float v = float(int((packed >> (2 * j)) & 3) - 1) * scale;
      uint16_t bits;
      if constexpr (half)
        bits = strata::kernels::f16_from_f32(v);
      else {
        auto u = sycl::bit_cast<uint32_t>(v);
        bits = uint16_t((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
      }
      (gate ? gu : down)[index * 4 + j] = bits;
    }
  });
}
} // namespace
void blob_dequant(const uint8_t *b, uint16_t *g, uint16_t *d, void *s) {
  expand<false>(b, g, d, s);
}
void blob_dequant_f16(const uint8_t *b, uint16_t *g, uint16_t *d, void *s) {
  expand<true>(b, g, d, s);
}
} // namespace strata::prefill
