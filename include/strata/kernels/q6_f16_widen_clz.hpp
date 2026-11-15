// Main-authored binary16 widening experiment; repository LICENSE.
// No production caller. Qualify host/device semantics and Q6 timing first.
#pragma once

#include <climits>
#include <cstdint>

namespace strata::kernels {

// Widen raw binary16 to raw binary32, preserving zero signs and NaN payloads.
// Derivation uses the binary16 definition, not another engine's converter.
// A nonzero ten-bit subnormal needs 1..10 shifts to put its leading bit at10;
// clz(man|1)-21 gives that count without a data-dependent normalization loop.
// The |1 makes the builtin's argument nonzero even for signed-zero inputs.
inline std::uint32_t q6_f32_bits_from_f16_clz(std::uint16_t h) {
  static_assert(UINT_MAX == 0xffffffffu, "requires a32-bit clz argument");
  const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
  const std::uint32_t exponent = (h >> 10) & 31u;
  const std::uint32_t mantissa = h & 1023u;
  const std::uint32_t shifts =
      std::uint32_t(__builtin_clz(static_cast<unsigned>(mantissa | 1u))) - 21u;
  const std::uint32_t subnormal = ((113u - shifts) << 23) |
      (((mantissa << shifts) & 1023u) << 13);
  const std::uint32_t normal = ((exponent + 112u) << 23) | (mantissa << 13);
  const std::uint32_t special = 0x7f800000u | (mantissa << 13);
  const std::uint32_t magnitude = exponent == 0
      ? (mantissa == 0 ? 0u : subnormal)
      : (exponent == 31 ? special : normal);
  return sign | magnitude;
}

} // namespace strata::kernels
