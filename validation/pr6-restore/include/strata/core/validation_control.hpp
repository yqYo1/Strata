#pragma once
// Independently authored project code under repository LICENSE.
#include <algorithm>
#include <cstdint>
#include <initializer_list>

namespace strata::core::detail {
// No environment switch can enable the legacy mixed volatile/device-atomic
// handshake. Its source remains only for historical comparison.
inline constexpr bool concurrent_doorbell_admitted() noexcept { return false; }
inline constexpr const char* doorbell_exclusion =
    "PR6 validation: concurrent mapped doorbell unsupported (D3); decode host loops, "
    "token graphs and every verifier mode are excluded; ordinary Prefill remains source-available";

// End is a count, so a final cell at limit-1 is legal. Empty prefill is a
// no-op at any admitted endpoint, without requiring token/residual pointers.
inline bool position_span(int64_t first, int64_t count,
                          std::initializer_list<int64_t> limits,
                          int64_t lookahead = 0) noexcept {
    if (first < 0 || count < 0 || lookahead < 0) return false;
    int64_t cap = 262144;
    for (const auto limit : limits) {
        if (limit <= 0) return false;
        cap = std::min(cap, limit);
    }
    if (first > cap || count > cap - first) return false;
    return lookahead <= cap - first - count;
}
inline bool prefill_admission(const void* residual, const void* tokens,
                             int64_t first, int64_t count,
                             std::initializer_list<int64_t> limits) noexcept {
    return position_span(first, count, limits) &&
           (count == 0 || (residual && tokens));
}
// Catch-up rows and the draft chain have different endpoints. Check each
// subtraction before forming p+a or any sampled counter/position cell+1.
inline bool draft_admission(const void* residual, const void* tokens, const void* output,
                            int64_t p, int T, int a, int max_t, int max_drafts,
                            std::initializer_list<int64_t> limits) noexcept {
    if (!residual || !tokens || !output || max_t < 2 || max_t > 8 ||
        T < 1 || T > max_t || a < 0 || a >= T ||
        !position_span(p, T, limits)) return false;
    const int chain = std::max(1, std::min(max_t - 1, max_drafts));
    // a < T; chain <= 7. Reserve a cells before the chain, then the
    // chain and one verifying token, without forming p+a at admission.
    return position_span(p, a, limits, int64_t(chain) + 1);
}

// False completion and provider errors must never be conflated. The caller
// consumes errors even when the queue query reports pending; delivery uses 87.
enum class RingProgress { pending, reached, complete_without_ring, query_error };
template<class Empty, class Errors, class Ring>
inline RingProgress ring_progress(Empty empty, Errors errors, Ring reached) noexcept {
    try {
        const bool complete = empty();
        errors();
        if (reached()) return RingProgress::reached;
        return complete ? RingProgress::complete_without_ring : RingProgress::pending;
    } catch (...) { return RingProgress::query_error; }
}

// One attempt, including dimension/model/IO/allocation failures. The gate is
// taken before metadata mutation; repeated calls preserve the prior state.
class OneAttemptLoad {
    bool attempted_ = false;
public:
    bool begin() noexcept {
        if (attempted_) return false;
        attempted_ = true;
        return true;
    }
};
} // namespace strata::core::detail
