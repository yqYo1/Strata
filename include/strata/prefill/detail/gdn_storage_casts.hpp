// Main-authored; repository LICENSE (MIT). Dedicated prefill storage only.
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace strata::prefill::detail {

inline float gdn_half_load(uint16_t bits) {
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

inline uint16_t gdn_half_store(float value) {
  return sycl::bit_cast<uint16_t>(static_cast<sycl::half>(value));
}

} // namespace strata::prefill::detail
