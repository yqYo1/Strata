#pragma once
// Main-authored; repository LICENSE applies. Host metadata only: no SYCL,
// GPU events, extra device readback, or change to the expert execution order.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace strata::prefill::detail {

inline bool moe_upload_census_requested() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_SYCL_PREFILL_MOE_CENSUS");
        if (!value || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        throw std::invalid_argument("STRATA_SYCL_PREFILL_MOE_CENSUS must be0 or1");
    }();
    return enabled;
}

struct MoeUploadCensus {
    std::array<uint32_t, 8193> uploads{}, hits{};
    uint32_t upload_count = 0, hit_count = 0;

    // Called exactly once on the first job of each nonempty expert, after its
    // upload was accepted or its private-bank cache hit was established.
    void record(int64_t rows, bool upload) {
        if (rows < 1 || rows > 8192 || upload_count + hit_count >= 512)
            throw std::invalid_argument("MoE census: invalid first-job geometry");
        ++(upload ? uploads : hits)[size_t(rows)];
        ++(upload ? upload_count : hit_count);
    }

    // The caller MUST have completed the existing checked HC-write drain.
    // These counts describe accepted uploads, not measured physical bus bytes.
    bool emit_after_checked_drain(FILE* out, int64_t layer, int64_t first,
            int64_t rows, int gu_type, int down_type, uint64_t blob_bytes) const noexcept {
        if (!out || layer < 0 || layer >= 48 || first < 0 || rows < 1 ||
            rows > 262144 || first > 262144 - rows || !blob_bytes ||
            blob_bytes > (16u << 20) || upload_count + hit_count == 0)
            return false;
        if (std::fprintf(out, "strata MoE upload census: {\"version\":1,\"layer\":%lld,\"first_row\":%lld,\"rows\":%lld,\"gu_type\":%d,\"down_type\":%d,\"blob_bytes\":%llu,\"accepted_uploads\":%u,\"cache_hits\":%u,\"requested_upload_bytes\":%llu,\"final_drain\":\"checked\",\"accepted_upload_rows_histogram\":[",
                (long long)layer, (long long)first, (long long)rows, gu_type, down_type,
                (unsigned long long)blob_bytes, upload_count, hit_count,
                (unsigned long long)(blob_bytes * upload_count)) < 0)
            return false;
        const auto histogram = [out](const auto& values) noexcept {
            bool comma = false;
            for (size_t n = 1; n < values.size(); ++n) if (values[n]) {
                if (std::fprintf(out, "%s[%zu,%u]", comma ? "," : "", n, values[n]) < 0)
                    return false;
                comma = true;
            }
            return true;
        };
        return histogram(uploads) && std::fputs("],\"cache_hit_rows_histogram\":[", out) >= 0 &&
            histogram(hits) && std::fputs("]}\n", out) >= 0;
    }
};
} // namespace strata::prefill::detail
