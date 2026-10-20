// Original Strata cache/scratch ownership contract; repository LICENSE applies.
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace strata::core {
class ExpertCache;
namespace detail { class CacheScratchState; }

// This retains the ownership record, not the allocation. The cache must refuse
// close/reopen while a loan or reader is live. Only the actual cache can mint a
// ticket. A registered USM pointer by itself is not a ticket.
class CacheScratchLease {
public:
    CacheScratchLease() = default;
    explicit operator bool() const noexcept { return bool(state_); }
    uint64_t generation() const noexcept { return generation_; }
    bool contains(const void* owner, const void* pointer, uint64_t bytes,
                  bool require_exclusive = true) const;
private:
    friend class ExpertCache;
    friend class detail::CacheScratchState;
    CacheScratchLease(std::shared_ptr<detail::CacheScratchState> state, uint64_t generation)
        : state_(std::move(state)), generation_(generation) {}
    std::shared_ptr<detail::CacheScratchState> state_;
    uint64_t generation_ = 0;
};

namespace detail {

// Host admission only. activate()'s callback MUST actually retire all GPU,
// graph, callback and producer readers; Reader only counts host entry scopes.
// Likewise refill/publish callbacks must check transfer completion and every
// residency publication. None of these phases infer successful device work.
class CacheScratchState : public std::enable_shared_from_this<CacheScratchState> {
public:
    enum class Phase { ready, prepared, masked, exclusive, used, refilling, poisoned, closed };
    CacheScratchState(const void* owner, const void* base, uint64_t bytes)
        : owner_(owner), base_(reinterpret_cast<uintptr_t>(base)), bytes_(bytes) {
        if (!owner || !base_ || !bytes || bytes > std::numeric_limits<uintptr_t>::max() - base_)
            throw std::invalid_argument("cache scratch: invalid parent interval");
    }
    CacheScratchState(const CacheScratchState&) = delete;
    CacheScratchState& operator=(const CacheScratchState&) = delete;

    class Reader {
    public:
        Reader() = default;
        Reader(Reader&& other) noexcept : state_(std::move(other.state_)) {}
        Reader& operator=(Reader&&) = delete;
        Reader(const Reader&) = delete;
        Reader& operator=(const Reader&) = delete;
        ~Reader() { if (state_) state_->end_read(); }
    private:
        friend class CacheScratchState;
        explicit Reader(std::shared_ptr<CacheScratchState> state) : state_(std::move(state)) {}
        std::shared_ptr<CacheScratchState> state_;
    };
    Reader read() {
        auto self = shared_from_this();
        std::lock_guard lock(mutex_);
        if (phase_ != Phase::ready || readers_ == std::numeric_limits<uint64_t>::max())
            throw std::logic_error("cache scratch: decode reader prohibited until refill publication");
        ++readers_;
        return Reader(std::move(self));
    }
    bool readable() const { std::lock_guard lock(mutex_); return phase_ == Phase::ready; }
    Phase phase() const { std::lock_guard lock(mutex_); return phase_; }

    uint64_t prepare(uint64_t offset, uint64_t bytes) {
        std::lock_guard lock(mutex_);
        if (phase_ != Phase::ready || readers_ || generation_ == std::numeric_limits<uint64_t>::max())
            throw std::logic_error("cache scratch: parent busy or generation exhausted");
        if (!bytes || offset > bytes_ || bytes > bytes_ - offset || (base_ + offset) % 256)
            throw std::invalid_argument("cache scratch: invalid/alignment-exceeding loan interval");
        offset_ = offset; length_ = bytes; ++generation_; phase_ = Phase::prepared;
        return generation_;
    }

    template<class Publish> bool mask(uint64_t generation, Publish&& publish) {
        return transition(generation, Phase::prepared, Phase::masked, std::forward<Publish>(publish));
    }
    template<class Retire> bool activate(uint64_t generation, Retire&& retire) {
        // Reusing the same masked range for another prompt segment is allowed
        // after its previous use retired, without publishing it to decode.
        Phase before;
        { std::lock_guard lock(mutex_); require(generation);
          before = phase_;
          if (before != Phase::masked && before != Phase::used)
              throw std::logic_error("cache scratch: activation before mask or after refill began"); }
        return transition(generation, before, Phase::exclusive, std::forward<Retire>(retire));
    }
    template<class Drain> bool retire(uint64_t generation, Drain&& drain) {
        return transition(generation, Phase::exclusive, Phase::used, std::forward<Drain>(drain));
    }
    void begin_refill(uint64_t generation) {
        std::lock_guard lock(mutex_); require(generation);
        if (phase_ != Phase::used || transition_active_) throw std::logic_error("cache scratch: refill before checked retirement");
        phase_ = Phase::refilling;
    }
    template<class Complete> bool publish_refilled(uint64_t generation, Complete&& complete) {
        return transition(generation, Phase::refilling, Phase::ready, std::forward<Complete>(complete));
    }
    void poison() { std::lock_guard lock(mutex_); if (phase_ != Phase::closed) phase_ = Phase::poisoned; }
    void close() {
        std::lock_guard lock(mutex_);
        if (phase_ != Phase::ready || readers_)
            throw std::logic_error("cache scratch: parent close before reader/loan retirement and refill");
        phase_ = Phase::closed;
    }
    bool fill_allowed() const {
        std::lock_guard lock(mutex_); return phase_ == Phase::ready || phase_ == Phase::refilling;
    }

private:
    friend class ::strata::core::CacheScratchLease;
    const void* const owner_;
    const uintptr_t base_;
    const uint64_t bytes_;
    mutable std::mutex mutex_;
    Phase phase_ = Phase::ready;
    uint64_t generation_ = 0, offset_ = 0, length_ = 0, readers_ = 0;
    bool transition_active_ = false;
    void require(uint64_t generation) const {
        if (!generation || generation != generation_)
            throw std::logic_error("cache scratch: foreign or stale parent generation");
    }
    void end_read() noexcept {
        std::lock_guard lock(mutex_);
        if (!readers_) std::terminate();
        --readers_;
    }
    template<class Callback> bool transition(uint64_t generation, Phase from, Phase to,
                                            Callback&& callback) {
        { std::lock_guard lock(mutex_); require(generation);
          if (phase_ != from || transition_active_)
              throw std::logic_error("cache scratch: invalid or nested transition");
          transition_active_ = true; }
        // Do not hold mutex_ around GPU waits or callbacks. Admission stays
        // closed throughout. Reentrant/parallel transitions fail, never wait.
        bool ok = false;
        try { ok = bool(std::forward<Callback>(callback)()); }
        catch (...) { std::lock_guard lock(mutex_); phase_ = Phase::poisoned; transition_active_ = false; throw; }
        std::lock_guard lock(mutex_);
        if (phase_ != from || !transition_active_) {
            phase_ = Phase::poisoned; transition_active_ = false;
            throw std::logic_error("cache scratch: ownership changed during transition");
        }
        transition_active_ = false; phase_ = ok ? to : Phase::poisoned;
        return ok;
    }
    bool contains(const CacheScratchLease& lease, const void* owner, const void* pointer,
                  uint64_t bytes, bool exclusive) const {
        std::lock_guard lock(mutex_);
        if (lease.state_.get() != this || lease.generation_ != generation_ || owner != owner_ ||
            transition_active_ || (exclusive ? phase_ != Phase::exclusive :
                (phase_ != Phase::prepared && phase_ != Phase::masked && phase_ != Phase::exclusive && phase_ != Phase::used)))
            return false;
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        const auto begin = base_ + offset_;
        return bytes && address >= begin && address - begin <= length_ && bytes <= length_ - (address - begin);
    }
};
} // namespace detail

inline bool CacheScratchLease::contains(const void* owner, const void* pointer, uint64_t bytes,
                                       bool require_exclusive) const {
    return state_ && state_->contains(*this, owner, pointer, bytes, require_exclusive);
}
} // namespace strata::core
