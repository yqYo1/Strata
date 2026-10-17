#pragma once

// Original Strata source; repository LICENSE applies. Capacity arithmetic only.
// This plan submits nothing and does not prove pointer identity, last readers,
// route uniqueness, numeric equivalence, or GPU capability. The owning caller
// must establish those before borrowing any of the existing allocations.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::prefill::detail {

struct MoeAccumPlan {
    static constexpr int64_t width = 2560, experts = 512, top_k = 10, hc = 4;
    static constexpr int64_t ff = 640, chunk = 8192, maximum_rows = 81920;
    enum class Plane : size_t {
        accumulator, weights, slots, source, job_weights, injection,
        gathered, gu, hidden, down_job, shared_chunk, shared_gate,
        logits, shared_gate_input, shared_up_input, shared_hidden, status, count
    };
    struct Span {
        uint64_t offset = 0, bytes = 0, reserved = 0;
    };
    struct MixedSpan {
        bool tail = false;
        uint64_t offset = 0, bytes = 0;
    };
    int64_t rows = 0, primary_rows = 0, tail_rows = 0;
    uint64_t primary_bytes = 0, tail_bytes = 0, region_bytes = 0;
    std::array<Span, size_t(Plane::count)> planes{};

    static uint64_t multiply(uint64_t a, uint64_t b) {
        if (b && a > uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) / b)
            throw std::overflow_error("MoE accumulator span exceeds pointer arithmetic");
        return a * b;
    }

    static MoeAccumPlan make(int64_t n, uint64_t legacy_r_bytes,
                              uint64_t embedding_bytes, uint64_t available_region_bytes) {
        if (n <= chunk || n > maximum_rows)
            throw std::invalid_argument("MoE accumulator window must contain 8193..81920 rows");
        MoeAccumPlan plan;
        plan.rows = n;
        const uint64_t row_bytes = uint64_t(width) * 2;
        if (multiply(uint64_t(n), row_bytes) <= legacy_r_bytes) {
            plan.primary_rows = n;
        } else {
            const uint64_t capacity = legacy_r_bytes / row_bytes;
            plan.primary_rows = int64_t(std::min<uint64_t>(uint64_t(n), capacity / chunk * chunk));
        }
        plan.tail_rows = n - plan.primary_rows;
        plan.primary_bytes = multiply(uint64_t(plan.primary_rows), row_bytes);
        plan.tail_bytes = multiply(uint64_t(plan.tail_rows), row_bytes);
        if (plan.primary_bytes > legacy_r_bytes || plan.tail_bytes > embedding_bytes)
            throw std::invalid_argument("MoE mixed banks exceed existing R/embedding allocations");
        const uint64_t w = uint64_t(n), j = uint64_t(chunk);
        const std::array<uint64_t, size_t(Plane::count)> payloads = {
            multiply(w, width * 4), multiply(w, top_k * 4), multiply(w, top_k * 4),
            multiply(w, top_k * 4), multiply(w, top_k * 4), multiply(w, hc * 4),
            multiply(j, width * 2), multiply(j, 2 * ff * 2), multiply(j, ff * 2),
            multiply(j, width * 2), multiply(j, width * 2), multiply(w, 4),
            multiply(j, experts * 4), multiply(j, ff * 2), multiply(j, ff * 2),
            multiply(j, ff * 2), 4
        };
        for (size_t i = 0; i < payloads.size(); ++i) {
            // Reserve alignment and 256 bytes between views; this is not a
            // claim that a runtime canary has been initialized or checked.
            const uint64_t payload = payloads[i];
            if (payload > uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) - 511)
                throw std::overflow_error("MoE accumulator alignment overflow");
            const uint64_t reserved = (payload + 255) / 256 * 256 + 256;
            if (plan.region_bytes > available_region_bytes ||
                reserved > available_region_bytes - plan.region_bytes)
                throw std::invalid_argument("MoE accumulator views exceed existing shared region");
            plan.planes[i] = {plan.region_bytes, payload, reserved};
            plan.region_bytes += reserved;
        }
        return plan;
    }

    // The first GPU candidate owns a separate contiguous FP16 mixed allocation.
    // This preserves the current mixed_h producer without borrowing typed R.
    static MoeAccumPlan make_contiguous(int64_t n, uint64_t mixed_bytes,
                                        uint64_t available_region_bytes) {
        return make(n, mixed_bytes, 0, available_region_bytes);
    }

    const Span& span(Plane plane) const {
        return planes.at(size_t(plane));
    }

    // Preparation and final shared projections consume one contiguous chunk.
    // A routed gather must instead select the appropriate bank per source row.
    MixedSpan mixed_piece(int64_t first, int64_t count) const {
        if (first < 0 || first >= rows || first % chunk || count <= 0 ||
            count != std::min(chunk, rows - first))
            throw std::invalid_argument("MoE mixed piece is not a complete logical chunk");
        const bool tail = first >= primary_rows;
        if (!tail && count > primary_rows - first)
            throw std::invalid_argument("MoE mixed piece crosses banks");
        const uint64_t offset = multiply(uint64_t(tail ? first - primary_rows : first), width * 2);
        const uint64_t bytes = multiply(uint64_t(count), width * 2);
        const uint64_t capacity = tail ? tail_bytes : primary_bytes;
        if (offset > capacity || bytes > capacity - offset)
            throw std::invalid_argument("MoE mixed piece exceeds its logical bank");
        return {tail, offset, bytes};
    }
};

} // namespace strata::prefill::detail
