// Exact IQ2_S NT1 fused gate/up, specialized for this model's H2560/FF640.
// Main-authored integration; repository MIT license.
#pragma once
#include <array>
#include <cstdint>
namespace strata::kernels::cpu {
struct NativeFmt;
// False without writing for unsupported build/format/shape, invalid byte spans,
// alignment, overlapping storage or row range. Inputs are immutable native blobs/Q8_K activations.
// The caller owns their storage for the complete synchronous call.
// No environment, pool, Down, NT2+ or device state is accessed here.
bool iq2_pair_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* act,
                       float* ff, int r0, int r1);

// Optional host-only dispatch evidence, fixed before exec with
// STRATA_IQ2_PAIR_AUDIT=1. Counters identify threads, never requests.
// Read after the synchronous pool has drained for complete per-phase deltas.
struct Iq2GuThreadAudit {
    uint64_t tid=0, calls=0, row_ops=0, nt1_rows=0, pair_rows=0;
    int cpu=-1;
    bool pinned=false;
};
struct Iq2GuAudit {
    bool enabled=false, overflow=false;
    unsigned size=0;
    std::array<Iq2GuThreadAudit,64> threads{};
};
Iq2GuAudit iq2_gu_audit_snapshot() noexcept;
} // namespace strata::kernels::cpu
