#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::prefill::detail {

// Upload sources are immutable until the caller has drained the draft stream.
// Keep every batch's bounds, including the final partial batch, in live storage.
class DraftHostUploads {
public:
    DraftHostUploads(const int32_t* tokens, int64_t n, int64_t first, int64_t batch,
                     int64_t hc, bool with_bounds) : first_(first), batch_(batch) {
        if (!tokens || first < 0 || n <= first || batch <= 0 || hc <= 0 ||
            n - first > std::numeric_limits<int32_t>::max() ||
            std::min(batch, n - first) > std::numeric_limits<int32_t>::max() / hc)
            throw std::invalid_argument("invalid draft host upload geometry");
        tokens_.assign(tokens + first, tokens + n);
        if (with_bounds) {
            const int64_t batches = 1 + (n - first - 1) / batch;
            bounds_.resize(size_t(batches) * 4);
            for (int64_t b = 0; b < batches; ++b) {
                const int64_t count = std::min(batch, n - first - b * batch);
                bounds_[size_t(b) * 4 + 0] = 0;
                bounds_[size_t(b) * 4 + 1] = int32_t(count);
                bounds_[size_t(b) * 4 + 2] = 0;
                bounds_[size_t(b) * 4 + 3] = int32_t(count * hc);
            }
        }
    }

    const int32_t* tokens(int64_t position) const {
        check_position(position);
        return tokens_.data() + (position - first_);
    }
    const int32_t* bounds(int64_t position) const {
        check_position(position);
        if (bounds_.empty()) throw std::invalid_argument("draft bounds were not requested");
        return bounds_.data() + size_t((position - first_) / batch_) * 4;
    }

private:
    void check_position(int64_t position) const {
        if (position < first_ || uint64_t(position - first_) >= tokens_.size() ||
            (position - first_) % batch_ != 0)
            throw std::invalid_argument("invalid draft upload batch position");
    }
    int64_t first_, batch_;
    std::vector<int32_t> tokens_, bounds_;
};

// Construct after the host storage, so this guard retires its lease first.
// Successful callers use their existing final stream wait, then retire().
// An early return/exception must drain before destroying the upload sources.
// If completion cannot be proved, stop the native process with the sources live.
template<class Drain, class Fatal>
class DraftUploadGuard {
public:
    DraftUploadGuard(Drain drain, Fatal fatal) : drain_(std::move(drain)), fatal_(std::move(fatal)) {}
    DraftUploadGuard(const DraftUploadGuard&) = delete;
    DraftUploadGuard& operator=(const DraftUploadGuard&) = delete;
    ~DraftUploadGuard() noexcept {
        if (!active_) return;
        try { if (drain_()) return; } catch (...) {}
        fatal_();
        std::abort(); // A fatal callback must not permit host-storage destruction.
    }
    void arm() noexcept { active_ = true; }
    void retire() noexcept { active_ = false; }
private:
    Drain drain_;
    Fatal fatal_;
    bool active_ = false;
};

} // namespace strata::prefill::detail
