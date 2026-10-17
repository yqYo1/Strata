#pragma once

// Original Strata implementation under the repository LICENSE. Host metadata
// only: the caller builds this AFTER router readback has completed, and keeps a
// shared owner until every asynchronous source upload has been checked/drained.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata::prefill::detail {

class MoeWindowRoutes final {
public:
    static constexpr int32_t experts = 512;
    static constexpr int32_t top_k = 10;
    static constexpr int32_t max_rows = 16384;
    static constexpr int32_t max_accum_rows = 81920;
    static constexpr int32_t max_job_rows = 8192;

    struct Job {
        int32_t expert;
        int32_t expert_row;
        int32_t rows;
        int32_t global_row;   // host source offset AND Down destination row
        // Each source upload targets device_src[0], never device_src+global_row.
    };

    // A const shared owner keeps all vector addresses stable through queued DMA.
    // No mutation, resize, or move operation is exposed on a published plan.
    static std::shared_ptr<const MoeWindowRoutes> build(
            int64_t rows, const int32_t* ids, size_t id_count) {
        if (rows < 1 || rows > max_rows)
            throw std::invalid_argument("MoE window rows outside [1,16384]");
        const size_t routes = size_t(rows) * top_k;
        if (!ids || id_count != routes)
            throw std::invalid_argument("MoE window ID span mismatch");
        return std::shared_ptr<const MoeWindowRoutes>(
            new MoeWindowRoutes(int32_t(rows), ids, routes, false));
    }

    // Separate factory keeps the original 16K API and its rejection contract.
    static std::shared_ptr<const MoeWindowRoutes> build_accum(
            int64_t rows, const int32_t* ids, size_t id_count) {
        if (rows < 1 || rows > max_accum_rows)
            throw std::invalid_argument("MoE accumulator rows outside [1,81920]");
        const size_t routes = size_t(rows) * top_k;
        if (!ids || id_count != routes)
            throw std::invalid_argument("MoE accumulator ID span mismatch");
        return std::shared_ptr<const MoeWindowRoutes>(
            new MoeWindowRoutes(int32_t(rows), ids, routes, true));
    }

    MoeWindowRoutes(const MoeWindowRoutes&) = delete;
    MoeWindowRoutes& operator=(const MoeWindowRoutes&) = delete;
    MoeWindowRoutes(MoeWindowRoutes&&) = delete;
    MoeWindowRoutes& operator=(MoeWindowRoutes&&) = delete;

    int32_t rows() const noexcept { return rows_; }
    const std::array<int32_t, experts>& counts() const noexcept { return counts_; }
    const std::array<int32_t, experts + 1>& offsets() const noexcept { return offsets_; }
    const std::vector<int32_t>& source_rows() const noexcept { return source_; }
    const std::vector<int32_t>& slot_map() const noexcept { return slots_; }
    const std::vector<Job>& jobs() const noexcept { return jobs_; }
    const std::vector<int32_t>& original_slots() const noexcept { return original_slots_; }

    const int32_t* job_source(size_t index) const {
        return source_.data() + jobs_.at(index).global_row;
    }

private:
    MoeWindowRoutes(int32_t rows, const int32_t* ids, size_t routes, bool inverse) : rows_(rows) {
        // Validate the COMPLETE router output before publishing any upload view.
        for (int32_t t = 0; t < rows; ++t) {
            for (int32_t k = 0; k < top_k; ++k) {
                const int32_t e = ids[size_t(t) * top_k + k];
                if (e < 0 || e >= experts)
                    throw std::invalid_argument("MoE window expert ID outside [0,511]");
                for (int32_t j = 0; j < k; ++j)
                    if (ids[size_t(t) * top_k + j] == e)
                        throw std::invalid_argument("MoE window repeated expert within token");
                // Check BEFORE increment: exactly rows occurrences are valid.
                if (counts_[size_t(e)] >= rows)
                    throw std::invalid_argument("MoE window expert rows exceed window");
                ++counts_[size_t(e)];
            }
        }
        for (int32_t e = 0; e < experts; ++e)
            offsets_[size_t(e) + 1] = offsets_[size_t(e)] + counts_[size_t(e)];
        if (offsets_.back() != int32_t(routes))
            throw std::logic_error("MoE window route mass mismatch");

        source_.resize(routes);
        slots_.resize(routes);
        if (inverse) original_slots_.resize(routes);
        auto cursor = offsets_;
        // Stable in original token-slot order; combine reads slots in that order.
        for (size_t i = 0; i < routes; ++i) {
            const int32_t p = cursor[size_t(ids[i])]++;
            slots_[i] = p;
            source_[size_t(p)] = int32_t(i / top_k);
            if (inverse) original_slots_[size_t(p)] = int32_t(i);
        }
        jobs_.reserve(experts + experts * size_t(rows > max_job_rows));
        for (int32_t e = 0; e < experts; ++e)
            for (int32_t at = 0; at < counts_[size_t(e)]; at += max_job_rows) {
                const int32_t count = std::min(max_job_rows, counts_[size_t(e)] - at);
                jobs_.push_back({e, at, count, offsets_[size_t(e)] + at});
            }
    }

    int32_t rows_;
    std::array<int32_t, experts> counts_{};
    std::array<int32_t, experts + 1> offsets_{};
    std::vector<int32_t> source_;
    std::vector<int32_t> slots_;
    std::vector<int32_t> original_slots_;
    std::vector<Job> jobs_;
};

} // namespace strata::prefill::detail
