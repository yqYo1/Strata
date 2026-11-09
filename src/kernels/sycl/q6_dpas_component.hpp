// Main-authored compressed-Q6 component; repository LICENSE.
// Opt-in decode C4 proposal. Only the measured full target head is eligible
// for model trials; default adoption and model-level gain remain unqualified.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace strata::kernels::component {

// Sixteen rows interleaved at uint16_t granularity. Retain all 210 bytes of
// each Q6_K block; pad only the final output-row tile. No FP16 weight expansion.
inline size_t q6_dpas_packed_bytes(int k, int n) {
    if (k <= 0 || k % 256 || n <= 0)
        throw std::invalid_argument("Q6 DPAS dimensions");
    const uint64_t bytes = ((uint64_t(n) + 15) / 16) * 16 * (k / 256) * 210;
    if (bytes > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Q6 DPAS 32-bit gather extent");
    return size_t(bytes);
}

inline bool q6_dpas_c4_shape(int k, int n, int columns) noexcept {
    return columns == 4 && k > 0 && k % 256 == 0 && n > 0 &&
        ((uint64_t(n) + 15) / 16) * 16 * (k / 256) * 210 <= std::numeric_limits<uint32_t>::max();
}

inline void q6_dpas_pack(std::span<const uint8_t> source,
                         std::span<uint16_t> destination, int k, int n) {
    const size_t bytes = q6_dpas_packed_bytes(k, n);
    const size_t blocks = size_t(k / 256);
    if (source.size() != size_t(n) * blocks * 210 || destination.size() != bytes / 2)
        throw std::invalid_argument("Q6 DPAS pack spans");
    std::fill(destination.begin(), destination.end(), uint16_t(0));
    for (size_t row = 0; row < size_t(n); ++row)
        for (size_t b = 0; b < blocks; ++b) {
            const size_t src = (row * blocks + b) * 210;
            const size_t dst = (row / 16 * blocks + b) * (16 * 105) + row % 16;
            for (size_t word = 0; word < 105; ++word)
                destination[dst + word * 16] = uint16_t(source[src + 2 * word]) |
                    uint16_t(uint16_t(source[src + 2 * word + 1]) << 8);
        }
}

// All spans must be distinct live Device Allocations in the explicit queue's
// context. The Q8_1 pitch is (k/32)*36; all four columns are present. Ownership,
// upload completion and retirement are the caller's responsibility. C1 fails
// before any runtime/device access. The reference C1 route stays unchanged.
void q6_dpas_c4(const uint16_t* packed, const void* q8, float* output,
                int k, int n, int columns, void* stream);

// Sole-packed-owner consumers: exact1..4 Q8_1 columns,with matching input and
// output spans. No padding buffers or writes to inactive columns. C4 reuses
// the original measured kernel;other widths require their own qualification.
void q6_dpas_columns(const uint16_t* packed, const void* q8, float* output,
                     int k, int n, int columns, void* stream);

// Prospective component: read the canonical row-major Q6_K blocks directly.
// It owns no additional weight storage and does not alter C1 or prompt routes.
// The source has exactly n*(k/256)*210 bytes; masked loads cover an unpadded
// final output tile. Same explicit queue, disjoint-span and lifetime contract.
// No production dispatcher selects this entry before component/model checks.
void q6_dpas_c4_canonical(const void* canonical, const void* q8, float* output,
                         int k, int n, int columns, void* stream);
} // namespace strata::kernels::component
