#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::prefill::detail {

inline uint64_t layer_major_mul(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("layer-major byte multiplication overflow");
    const uint64_t bytes = a * b;
    // Pointer arithmetic and the existing cache API use signed differences.
    if (bytes > uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes > uint64_t(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("layer-major span exceeds addressable bytes");
    return bytes;
}

inline uint64_t layer_major_add(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
        throw std::overflow_error("layer-major byte addition overflow");
    const uint64_t bytes = a + b;
    if (bytes > uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes > uint64_t(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("layer-major span exceeds addressable bytes");
    return bytes;
}

// Missing/auto selects the largest complete chunk prefix allowed by the budget.
// A decimal value is a maximum row count, rounded down to complete chunks;
// exactly n includes its partial final chunk. No signs, whitespace or suffixes.
inline int64_t layer_major_requested_rows(const char* value) {
    if (!value || std::strcmp(value, "auto") == 0) return -1;
    uint64_t rows = 0;
    const char* end = value + std::strlen(value);
    const auto parsed = std::from_chars(value, end, rows);
    if (value == end || parsed.ec != std::errc{} || parsed.ptr != end ||
        rows > uint64_t(std::numeric_limits<int64_t>::max()))
        throw std::invalid_argument("STRATA_PREFILL_LAYER_MAJOR_R_GPU must be auto or decimal rows");
    return int64_t(rows);
}

struct LayerMajorSpan {
    bool device = false;
    uint64_t offset_bytes = 0, bytes = 0;
};

struct LayerMajorResidualPlan {
    int64_t rows = 0, chunk = 0, gpu_rows = 0;
    uint64_t row_bytes = 0, total_bytes = 0, gpu_bytes = 0, host_bytes = 0;

    static LayerMajorResidualPlan make(int64_t n, int64_t c, int64_t width,
                                       uint64_t gpu_budget, int64_t requested) {
        if (n <= 0 || c <= 0 || width <= 0 || requested < -1 || requested > n)
            throw std::invalid_argument("invalid layer-major residual geometry or requested GPU rows");
        LayerMajorResidualPlan p;
        p.rows = n; p.chunk = c;
        p.row_bytes = layer_major_mul(uint64_t(width), sizeof(float));
        p.total_bytes = layer_major_mul(uint64_t(n), p.row_bytes);
        int64_t want = requested < 0 ? int64_t(std::min<uint64_t>(uint64_t(n), gpu_budget / p.row_bytes))
                                     : requested;
        p.gpu_rows = want == n ? n : (want / c) * c;
        p.gpu_bytes = layer_major_mul(uint64_t(p.gpu_rows), p.row_bytes);
        if (p.gpu_bytes > gpu_budget)
            throw std::invalid_argument("requested layer-major GPU residual prefix exceeds free VRAM with headroom");
        p.host_bytes = p.total_bytes - p.gpu_bytes;
        return p;
    }

    LayerMajorSpan span(int64_t row, int64_t count) const {
        if (row < 0 || row >= rows || row % chunk || count <= 0 ||
            count != std::min(chunk, rows - row))
            throw std::invalid_argument("invalid layer-major chunk bounds");
        const bool device = row < gpu_rows;
        if (device && count > gpu_rows - row)
            throw std::invalid_argument("layer-major chunk crosses residual stores");
        const uint64_t offset = layer_major_mul(uint64_t(device ? row : row - gpu_rows), row_bytes);
        const uint64_t bytes = layer_major_mul(uint64_t(count), row_bytes);
        const uint64_t limit = device ? gpu_bytes : host_bytes;
        if (offset > limit || bytes > limit - offset)
            throw std::invalid_argument("layer-major residual offset exceeds store");
        return {device, offset, bytes};
    }
};

// Records delivery order independently for every layer. Used before invoking a
// caller callback, so duplicate/out-of-order chunks cannot be published.
struct LayerMajorChunkCursor {
    int64_t rows, chunk, next = 0, calls = 0;
    void accept(int64_t row, int64_t count) {
        if (row != next || row < 0 || row >= rows || count <= 0 ||
            count != std::min(chunk, rows - row))
            throw std::invalid_argument("layer-major callback order differs from ascending chunks");
        next += count;
        ++calls;
    }
    bool complete() const { return next == rows; }
};

} // namespace strata::prefill::detail
