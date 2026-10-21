// Exact IQ2_S NT1 fused gate/up, specialized for this model's H2560/FF640.
// Main-authored integration; repository MIT license.
#pragma once
#include <cstdint>
namespace strata::kernels::cpu {
struct NativeFmt;
// False without writing for unsupported build/format/shape, invalid byte spans,
// alignment or row range. Inputs are immutable native blobs/Q8_K activations.
// The caller owns their storage for the complete synchronous call.
// No environment, pool, Down, NT2+ or device state is accessed here.
bool iq2_pair_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* act,
                       float* ff, int r0, int r1);
} // namespace strata::kernels::cpu
