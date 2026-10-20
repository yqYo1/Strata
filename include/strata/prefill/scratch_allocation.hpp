// Original Strata orchestration helpers; repository LICENSE applies.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::prefill::detail {

// Existing routing/page tables use signed32 indices. Validate before callers
// form T*expert_count or T*page_count, not only inside the final byte take.
inline uint64_t scratch_rows(int64_t chunk, int64_t experts, int64_t cells, int64_t routes) {
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    if (routes <= 0 || routes > limit || chunk <= 0 || chunk > limit / routes ||
        experts <= 0 || experts > limit || cells <= 0 || cells > limit)
        throw std::invalid_argument("prefill scratch: shape exceeds signed32 routing/page bounds");
    return uint64_t(chunk);
}

struct ScratchReservation {
    uint64_t bytes;
    uint64_t next;
};

// Preserve Alloc's 256-byte guard and alignment, including a zero-sized take.
// Do not let an overflow turn an impossible allocation into a small one.
inline ScratchReservation reserve_scratch(uint64_t used, uint64_t count,
        uint64_t element_bytes, uint64_t capacity = std::numeric_limits<uint64_t>::max()) {
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (!element_bytes || count > maximum / element_bytes)
        throw std::overflow_error("prefill scratch: element byte overflow");
    const uint64_t payload = count * element_bytes;
    if (payload > maximum - 511)
        throw std::overflow_error("prefill scratch: guard/alignment overflow");
    const uint64_t bytes = (payload + 511) & ~uint64_t(255);
    if (used > capacity || bytes > capacity - used)
        throw std::length_error("prefill scratch: allocation exceeds capacity");
    return {bytes, used + bytes};
}

inline uint64_t scratch_with_slack(uint64_t used, uint64_t slack) {
    if (slack > std::numeric_limits<uint64_t>::max() - used)
        throw std::overflow_error("prefill scratch: sizing slack overflow");
    return used + slack;
}

// A failed carve/rebind can have changed pointers and T. Keep that object
// unusable until Prefill::reset constructs an empty state; no pointer rollback
// or successful queue retirement is inferred from this host state alone.
class ScratchLayoutState {
public:
    bool ready() const noexcept { return phase_ == Phase::ready; }
    uint64_t generation() const noexcept { return generation_; }
    ScratchLayoutState() = default;
    ScratchLayoutState(const ScratchLayoutState&) = delete;
    ScratchLayoutState& operator=(const ScratchLayoutState&) = delete;

    class Update {
    public:
        Update(ScratchLayoutState& state, bool initial) noexcept : state_(state) {
            if (state.phase_ == (initial ? Phase::empty : Phase::ready) &&
                state.generation_ != std::numeric_limits<uint64_t>::max()) {
                state.phase_ = Phase::changing;
                active_ = true;
            }
        }
        ~Update() noexcept { if (active_) state_.phase_ = Phase::poisoned; }
        Update(const Update&) = delete;
        Update& operator=(const Update&) = delete;
        bool started() const noexcept { return active_; }
        void commit() noexcept {
            if (active_) {
                ++state_.generation_;
                state_.phase_ = Phase::ready;
                active_ = false;
            }
        }
    private:
        ScratchLayoutState& state_;
        bool active_ = false;
    };
private:
    enum class Phase { empty, changing, ready, poisoned };
    Phase phase_ = Phase::empty;
    uint64_t generation_ = 0;
};

} // namespace strata::prefill::detail
