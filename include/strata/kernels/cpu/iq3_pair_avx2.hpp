// Paired IQ3_S NT1 gate/up for this model's H2560/FF640; repository LICENSE.
#pragma once
#include <cstdint>

namespace strata::kernels::cpu {
struct NativeFmt;
// Returns false without writes for unsupported geometry or invalid/overlapping
// spans. The caller owns immutable weights/activation and writable640-row ff
// storage for the synchronous call. Does not access environment or device state.
bool iq3_pair_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* act,
                     float* ff, int r0, int r1);

// Optional production-dispatch evidence, fixed before exec with
// STRATA_IQ3_PAIR_AUDIT=1. Read after synchronous GU/pool completion.
struct Iq3PairAudit {
    bool enabled = false;
    uint64_t calls = 0, rows = 0;
};
Iq3PairAudit iq3_pair_audit_snapshot() noexcept;
void iq3_pair_note_dispatch(uint64_t rows) noexcept;
} // namespace strata::kernels::cpu
