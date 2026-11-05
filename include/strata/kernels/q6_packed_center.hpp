// SPDX-License-Identifier: MIT
// Main-authored Q6 component candidate; not selected by production dispatch.
#pragma once
#include <cstdint>

namespace strata::kernels::component {

// Precondition: each byte is in [0,63], as guaranteed by Q6's 0x0f/0x30
// reconstruction masks. Return four packed signed bytes representing value-32.
// Each byte plus0x60 is in [0x60,0x9f], so addition cannot carry into another
// byte. XOR0x80 then maps that interval to [-32,31] in two's-complement bytes.
// Unsigned arithmetic is defined for every input; inputs outside the domain
// are not the Q6 centering operation. No memory, queue or global state here.
inline constexpr uint32_t q6_center_bytes(uint32_t six_bit_bytes) noexcept {
    return (six_bit_bytes + 0x60606060u) ^ 0x80808080u;
}

} // namespace strata::kernels::component
