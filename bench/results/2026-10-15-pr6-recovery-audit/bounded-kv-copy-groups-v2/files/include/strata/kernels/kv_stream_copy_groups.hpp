#pragma once
// Host-only policy parsing; no environment mutation, SYCL headers, or runtime.
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
inline void validate_kv_stream_copy_group_cap(uint32_t cap) {
  if (cap != 0 && cap != 128 && cap != 256 && cap != 512 && cap != 1024)
    throw std::invalid_argument("KV copy group cap must be 0, 128, 256, 512, or 1024");
}
inline uint32_t kv_stream_copy_group_cap_from_text(const char *value) {
  if (!value || !*value || std::strcmp(value, "0") == 0) return 0;
  if (std::strcmp(value, "128") == 0) return 128;
  if (std::strcmp(value, "256") == 0) return 256;
  if (std::strcmp(value, "512") == 0) return 512;
  if (std::strcmp(value, "1024") == 0) return 1024;
  throw std::invalid_argument(
      "STRATA_SYCL_KV_COPY_GROUPS must be 0, 128, 256, 512, or 1024");
}
} // namespace strata::kernels
