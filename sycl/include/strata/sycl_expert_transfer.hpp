#pragma once

namespace strata::core {
// GpuPlanSink's SYCL-only DMA mode: stable pageable arena -> host USM -> device staging.
// Positive modes retain the original direct/device-copy-kernel meanings; zero is ordinary DMA.
inline constexpr int kSyclHostStagingDma = -1;
}
