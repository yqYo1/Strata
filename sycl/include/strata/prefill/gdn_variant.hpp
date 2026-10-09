// SYCL prompt recurrence variant for the existing Qwen GDN shapes.
#pragma once
#include <cstdint>
#include <cstddef>

namespace strata::prefill {
// Original key-head arithmetic with SG16 and 256 GRFs. Returns false without
// launching for fewer than 256 tokens; the caller keeps the original recurrence.
bool gdn_recurrence_keyhead_variant(float *state, const float *h, const float *gate, const float *beta, const float *z,
                                    const float *gamma, float eps, float *y, uint16_t *y16, int64_t T, void *stream);

struct GdnQuadReport {
    enum class Status { Denied, Empty, Submitted };
    Status status = Status::Denied;
    const char *reason = "not_checked";
    uint32_t compiled_subgroup = 0;
    size_t device_max_workgroup = 0, kernel_max_workgroup = 0;
    size_t private_bytes = 0, spill_bytes = 0;
    bool private_known = false, spill_known = false;
};
// Explicit candidate, independent of environment selectors. False means denied
// before any candidate/norm submission and leaves all state/output untouched.
// T==0 returns true/Empty before queue lookup; negative T throws. Once submission
// is attempted, exceptions propagate: never retry legacy on the same state.
// diagnostic_deny is a pre-submit parity seam only, never a hardware support claim.
bool gdn_recurrence_quad_variant(float *state, const float *h, const float *gate, const float *beta, const float *z,
                                const float *gamma, float eps, float *y, uint16_t *y16, int64_t T, void *stream,
                                int64_t ld16 = 0, GdnQuadReport *report = nullptr, bool diagnostic_deny = false);
// Explicit frozen pipeline body and launch geometry, bypassing all selectors.
// Empty is a no-submit no-op. This is a correctness reference, not a new default.
void gdn_recurrence_pipeline_reference(float *state, const float *h, const float *gate, const float *beta,
                                     const float *z, const float *gamma, float eps, float *y, uint16_t *y16,
                                     int64_t T, void *stream, int64_t ld16 = 0);
} // namespace strata::prefill
