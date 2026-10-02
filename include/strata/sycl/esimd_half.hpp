#pragma once

#include <sycl/ext/intel/esimd.hpp>

namespace strata::sycl_backend {

// Native conversion preserves finite FP16 values, including signed zero and
// subnormals. Keep the original bit expansion for infinities and NaN payloads;
// the hardware conversion otherwise quiets signaling NaNs.
template <int N>
sycl::ext::intel::esimd::simd<float, N> esimd_half_decode(
    sycl::ext::intel::esimd::simd<uint32_t, N> bits) SYCL_ESIMD_FUNCTION {
  namespace e = sycl::ext::intel::esimd;
  auto words = e::convert<uint16_t>(bits);
  const e::simd<sycl::half, N> halves =
      words.template bit_cast_view<sycl::half>();
  auto result = e::convert<float>(halves);
  e::simd<uint32_t, N> special =
      ((bits & 0x8000u) << 16) | 0x7f800000u | ((bits & 1023u) << 13);
  result.merge(special.template bit_cast_view<float>(),
               (bits & 0x7c00u) == 0x7c00u);
  return result;
}

} // namespace strata::sycl_backend
