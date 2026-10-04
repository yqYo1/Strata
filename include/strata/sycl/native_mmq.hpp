#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::kernels {
// Arbitrary routed prompt rows. Input weights use GGUF blocks; activation
// rows use Q8_1 with the reduction dimension padded to 512 values. Bounds are
// device-resident and may change between launches. No allocation or host wait.
struct NativeMmq {
  const void *weights = nullptr, *activation = nullptr;
  int type = -1, experts = 0;
  int64_t rows = 0, cols = 0, total_rows = 0, max_rows = 0, ld_output = 0;
  size_t expert_bytes = 0;
  const int32_t *bounds = nullptr, *destinations = nullptr;
  float *output = nullptr;
  // The caller permits in-place rearrangement and refills before the next run.
  bool scratch_weights = false;
  // Optional matching host bounds, read synchronously during submission only.
  // A captured compact grid must be recaptured if these bounds change.
  const int32_t *host_bounds = nullptr;
};
void native_mmq(const NativeMmq &p, void *stream);
} // namespace strata::kernels
