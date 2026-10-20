// Original Strata orchestration helpers; repository LICENSE applies.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::prefill::detail {

inline uint64_t scratch_product(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("prefill scratch: dimension product overflow");
    return a * b;
}

inline int64_t scratch_qsa_primary(int64_t layers, int64_t interval,
        bool has_states, int64_t first, int64_t count) {
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    if (layers <= 0 || layers > limit || interval <= 0 || interval > layers)
        throw std::invalid_argument("prefill scratch: invalid QSA layer geometry");
    const int64_t total = layers / interval;
    // first is GLOBAL, whereas count is the initialized range's length.
    if (!has_states || first < 0 || first >= total || count <= 0 || count > total - first)
        throw std::invalid_argument("prefill scratch: invalid primary QSA range");
    return first;
}

struct ScratchQsaPlan {
    uint64_t max_blocks, rows, values, scales, q4_bytes;
};

inline ScratchQsaPlan scratch_qsa_plan(int64_t cells, int64_t state_cells,
        int64_t pages, int64_t slots, int mode, int64_t heads, int64_t dimension,
        int64_t page_size, int64_t index_block, bool q8, bool q4, bool hybrid) {
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    if (cells <= 0 || cells > limit || state_cells != cells || heads <= 0 || heads > limit ||
        dimension <= 0 || dimension > limit || dimension % 4 || page_size <= 0 ||
        index_block <= 0 || pages <= 0 || pages > limit || slots <= 0 || slots > pages ||
        mode < 0 || mode > 2 || int(q8) + int(q4) + int(hybrid) > 1)
        throw std::invalid_argument("prefill scratch: invalid QSA size metadata");
    const int64_t expected_pages = (cells - 1) / page_size + 1;
    if (pages != expected_pages || (mode == 0 ? slots != pages : slots >= pages) ||
        (hybrid && mode != 0) || ((q8 || hybrid) && dimension % 64) ||
        ((q4 || hybrid) && dimension % 128))
        throw std::invalid_argument("prefill scratch: inconsistent QSA storage geometry");
    const uint64_t blocks = uint64_t(cells / index_block) + 2;
    if (blocks > uint64_t(limit))
        throw std::overflow_error("prefill scratch: QSA index block count overflow");
    const uint64_t rows = scratch_product(scratch_product(uint64_t(pages), uint64_t(heads)), uint64_t(page_size));
    const uint64_t values = scratch_product(rows, uint64_t(dimension));
    (void)scratch_product(values, 2); // FP16 byte payload before Alloc's take
    return {blocks, rows, values, scratch_product(rows, uint64_t(dimension / 64)),
        scratch_product(rows, scratch_product(uint64_t(dimension / 128), 144))};
}

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
