// Host semantic screen for the Main-authored Q6 widening proposal; repository LICENSE.
// No GPU initialization, timing, production integration or device qualification.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/q6_f16_widen_clz.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {
std::uint32_t bits(float value) {
  std::uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

// Independent finite-value oracle: exact dyadic arithmetic in double, followed
// by an exact conversion to float. No normalization shifts or clz are shared.
std::uint32_t oracle(std::uint16_t h) {
  const int exponent = (h >> 10) & 31;
  const unsigned mantissa = h & 1023u;
  if (exponent == 31) {
    // Special-value payload placement is the existing raw-bit contract.
    return (std::uint32_t(h & 0x8000u) << 16) | 0x7f800000u | (mantissa << 13);
  }
  const double magnitude = std::ldexp(
      double(mantissa + (exponent == 0 ? 0u : 1024u)),
      exponent == 0 ? -24 : exponent - 25);
  return bits(static_cast<float>((h & 0x8000u) ? -magnitude : magnitude));
}
} // namespace

int main() {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "requires IEEE binary32");
  static_assert(std::numeric_limits<double>::digits >= 53,
                "oracle requires at least binary64 precision");
  unsigned reference_mismatches = 0, candidate_mismatches = 0, shown = 0;
  unsigned zeros = 0, subnormals = 0, normals = 0, infinities = 0, nans = 0;
  std::uint64_t checksum = 14695981039346656037ull;
  for (unsigned value = 0; value < 65536u; ++value) {
    const auto h = static_cast<std::uint16_t>(value);
    const auto expected = oracle(h);
    const auto reference = bits(strata::kernels::f32_from_f16(h));
    const auto candidate = strata::kernels::q6_f32_bits_from_f16_clz(h);
    reference_mismatches += reference != expected;
    candidate_mismatches += candidate != expected;
    if ((reference != expected || candidate != expected) && shown++ < 8) {
      std::fprintf(stderr, "half=%04x expected=%08x reference=%08x candidate=%08x\n",
                   value, unsigned(expected), unsigned(reference), unsigned(candidate));
    }
    const unsigned exponent = (value >> 10) & 31u, mantissa = value & 1023u;
    if (exponent == 0) { if (mantissa == 0) ++zeros; else ++subnormals; }
    else if (exponent == 31) { if (mantissa == 0) ++infinities; else ++nans; }
    else ++normals;
    // Identity only; this loop is not a performance measurement.
    checksum = (checksum ^ candidate) * 1099511628211ull;
  }
  const bool passed = reference_mismatches == 0 && candidate_mismatches == 0 &&
      zeros == 2 && subnormals == 2046 && normals == 61440 &&
      infinities == 2 && nans == 2046;
  std::printf("f16_widen_clz_host_status=%s patterns=65536 zeros=%u subnormals=%u "
              "normals=%u infinities=%u nans=%u reference_mismatches=%u "
              "candidate_mismatches=%u checksum=%016llx\n",
              passed ? "PASS" : "FAIL", zeros, subnormals, normals, infinities, nans,
              reference_mismatches, candidate_mismatches,
              static_cast<unsigned long long>(checksum));
  return passed ? 0 : 1;
}
