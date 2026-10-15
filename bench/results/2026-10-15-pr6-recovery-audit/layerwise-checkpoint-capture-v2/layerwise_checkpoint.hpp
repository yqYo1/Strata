#pragma once
// Proposal only. Serialized serve issuer; snapshots running state, never K/V.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/layout.hpp"
#include "strata/program/conv_cache.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace strata::program {
inline int64_t layerwise_next(int64_t pos, int64_t interval) {
    if (pos < 0 || interval < 0) throw std::invalid_argument("negative checkpoint position/interval");
    return interval > INT64_MAX - pos ? INT64_MAX : pos + interval;
}

// Shared by the real serve call site and the CPU contract fixture. Reread
// replays from zero while valid old checks remain beyond zero; do not send that
// request through the new append-only capture transaction, including later runs.
inline bool layerwise_capture_enabled(bool periodic, bool windows, int64_t reread_to,
                                      int64_t begin, int64_t end, int64_t chunk) noexcept {
    return periodic && !windows && reread_to <= 0 && begin >= 0 &&
           end > begin && chunk > 0 && end - begin > chunk;
}

// Borrow a callback slot only inside a synchronous run's lexical scope. Create
// the guard BEFORE assigning the borrowed callback, so assignment exceptions
// restore the previous callback too. The transaction owner must be outer scope.
template<class Callback> class ScopedLayerwiseCallback {
    Callback* slot_ = nullptr;
    Callback saved_;
public:
    ScopedLayerwiseCallback(Callback& slot, bool enabled) noexcept {
        static_assert(std::is_nothrow_default_constructible_v<Callback>);
        static_assert(std::is_nothrow_move_assignable_v<Callback>);
        if (enabled) { slot_ = &slot; saved_ = std::move(slot); }
    }
    ~ScopedLayerwiseCallback() noexcept { if (slot_) *slot_ = std::move(saved_); }
    ScopedLayerwiseCallback(const ScopedLayerwiseCallback&) = delete;
    ScopedLayerwiseCallback& operator=(const ScopedLayerwiseCallback&) = delete;
};

class LayerwiseCheckpoints {
    using Check = core::ConversationCheckpoint;
    struct Meta { int64_t pos; uint64_t used; int64_t old; };
    struct Pending { Check state; uint64_t mask = 0; bool ple = false, prompt = false; };
    std::vector<Meta> old_, final_;
    std::vector<Pending> pending_;
    std::vector<Check> publication_;
    std::array<int64_t, 48> reached_{};
    int64_t begin_ = 0, end_ = 0, chunk_ = 0, layer_ = 0;
    uint64_t serial_ = 0, identity_ = 0, old_clock_ = 0, clock_ = 0;
    bool active_ = false;
    static uint64_t add(uint64_t a, uint64_t b) {
        if (b > uint64_t(INT64_MAX) || a > uint64_t(INT64_MAX) - b)
            throw std::overflow_error("layer checkpoint byte count overflow");
        return a + b;
    }
    static uint64_t mul(uint64_t a, uint64_t b) {
        if (b && a > uint64_t(INT64_MAX) / b)
            throw std::overflow_error("layer checkpoint byte count overflow");
        return a * b;
    }
    static uint64_t bytes(const Check& c) {
        uint64_t n = mul(c.ids.capacity(), sizeof(int32_t));
        n = add(n, mul(c.imgs.capacity(), sizeof(core::ConversationImageKey)));
        for (auto p : {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos}) n = add(n, p->capacity());
        if (!c.stage_parts.empty()) throw std::invalid_argument("layer checkpoint stage parts unsupported");
        return n;
    }
    static bool fail(std::string& e, const char* text) { e = text; return false; }
public:
    // Copy must finish its queue work, including error retirement, before returning.
    using Copy = std::function<bool(void*, const void*, size_t, std::string&)>;
    bool active() const { return active_; }
    uint64_t identity() const { return identity_; }
    void discard() noexcept {
        active_ = false; old_.clear(); final_.clear(); pending_.clear(); publication_.clear();
        // No callback retains a DMA destination: every Copy is checked/blocking.
    }
    bool begin(const std::vector<Check>& checks, uint64_t clock, int64_t cap,
               const std::vector<int64_t>& ids, const std::vector<core::ConversationImageKey>& images,
               const core::SessionState& ss, const core::ModelGeometry& g,
               int64_t begin, int64_t end, int64_t chunk, int64_t threshold, int64_t interval,
               bool boundary_event, std::string& e) try {
        discard();
        if (cap <= 0 || cap > 262144 || checks.size() > uint64_t(cap) || g.n_layers != 48 || g.qsa_interval != 4 ||
            g.n_qsa_layers() != 12 || g.n_gdn_layers() != 36 || ss.layer_lo != 0 || ss.layer_hi != 48 ||
            ss.qsa_ord0 || ss.gdn_ord0 || ss.qsa_alloc != 12 || ss.gdn_alloc != 36 ||
            ids.size() > 262144 || begin < 0 || end <= begin || uint64_t(end) > ids.size() ||
            end > ss.max_cells || chunk <= 0 || interval <= 0 || threshold < 0)
            return fail(e, "layer checkpoint invalid geometry, cache or schedule");
        if (std::any_of(ids.begin(), ids.end(), [](int64_t id) { return id < 0 || id > INT32_MAX; }))
            return fail(e, "layer checkpoint token outside native int32 range");
        if (serial_ == UINT64_MAX) return fail(e, "layer checkpoint run identity exhausted");
        identity_ = ++serial_; begin_ = begin; end_ = end; chunk_ = chunk;
        old_clock_ = clock_ = clock; layer_ = 0; reached_.fill(begin);
        // Metadata-only preview is bounded too. This is before any preview
        // vector grows; resident capacities from a previous transaction are
        // already reflected by the available-memory sample.
        const uint64_t chunks = uint64_t(1 + (end - begin - 1) / chunk);
        const uint64_t max_meta = std::min(uint64_t(cap) + 1, add(add(checks.size(), chunks), 1));
        const uint64_t preview = mul(add(add(checks.size(), max_meta), 4), 512);
        if (!core::conversation_memory_admit(core::conversation_available_memory(), preview, 8ull << 30))
            return fail(e, "layer checkpoint metadata exceeds RAM reserve");
        uint64_t old_bytes = 0;
        for (size_t i = 0; i < checks.size(); ++i) {
            const auto& c = checks[i];
            if (c.ids.empty() || c.ids.size() > uint64_t(begin) ||
                !std::equal(c.ids.begin(), c.ids.end(), ids.begin()))
                return fail(e, "layer checkpoint old chain is not an input prefix");
            old_.push_back({int64_t(c.ids.size()), c.used, int64_t(i)});
            old_bytes = add(old_bytes, bytes(c));
        }
        final_ = old_;
        auto event = [&](int64_t position) {
            if (clock_ == UINT64_MAX) throw std::overflow_error("checkpoint stamp exhausted");
            const uint64_t stamp = ++clock_;
            auto it = std::find_if(final_.begin(), final_.end(), [&](const Meta& m) { return m.pos == position; });
            if (it != final_.end()) it->used = stamp;
            else final_.push_back({position, stamp, -1});
            if (final_.size() > uint64_t(cap)) {
                std::vector<uint64_t> stamps;
                for (const auto& m : final_) stamps.push_back(m.used);
                const size_t victim = conv_cache::eviction_victim(stamps.data(), stamps.size(), cap);
                final_.erase(final_.begin() + std::ptrdiff_t(victim));
            }
        };
        int64_t at = begin;
        while (at < end) {
            const int64_t done = at + std::min(chunk, end - at); // bounded by end
            if (done >= threshold) { event(done); threshold = layerwise_next(done, interval); }
            at = done;
        }
        if (boundary_event) event(end); // root/turn de-dup updates its stamp a second time
        core::ConversationStateSizes z;
        if (!core::conversation_session_sizes(g, ss, z, e)) return false;
        uint64_t payload = add(z.gdn, ss.ple_hist ? z.ple : 0);
        payload = add(payload, mul(12, add(add(z.tail, z.dead), z.block_pos)));
        uint64_t new_bytes = 0, fresh = 0;
        for (const auto& m : final_) if (m.old < 0) {
            uint64_t n = add(payload, mul(uint64_t(m.pos), sizeof(int32_t)));
            for (const auto& im : images) if (im.start < m.pos) n = add(n, sizeof(im));
            new_bytes = add(new_bytes, n); ++fresh;
        }
        // Conservative 2x vector capacity allowance. Old owners remain resident and
        // are already charged in MemAvailable. Count bounded working metadata too.
        uint64_t metadata = mul(add(add(checks.size(), final_.size()), 4), 256);
        metadata = add(metadata, mul(fresh, sizeof(Pending)));
        metadata = add(metadata, mul(final_.size(), sizeof(Check)));
        const uint64_t need = mul(2, add(new_bytes, metadata));
        const uint64_t live_peak = add(old_bytes, need); // checked old + new, not a KV snapshot estimate
        (void) live_peak;
        if (!core::conversation_memory_admit(core::conversation_available_memory(), need, 8ull << 30))
            return fail(e, "layer checkpoint new+old payload exceeds RAM admission with 8 GiB reserve");
        pending_.reserve(size_t(fresh)); publication_.resize(final_.size());
        for (const auto& m : final_) if (m.old < 0) {
            Pending p;
            p.state.ids.assign(ids.begin(), ids.begin() + m.pos);
            for (const auto& im : images) if (im.start < m.pos) p.state.imgs.push_back(im);
            p.state.gdn.resize(z.gdn); p.state.ple.resize(ss.ple_hist ? z.ple : 0);
            p.state.tails.resize(size_t(mul(12, z.tail)));
            p.state.dead.resize(size_t(mul(12, z.dead)));
            p.state.block_pos.resize(size_t(mul(12, z.block_pos)));
            if (!core::conversation_checkpoint_validate(p.state, ss, g, e)) return false;
            p.ple = !ss.ple_hist;
            pending_.push_back(std::move(p));
        }
        uint64_t actual = metadata;
        for (const auto& p : pending_) actual = add(actual, bytes(p.state));
        if (actual > need || !core::conversation_memory_admit(core::conversation_available_memory(), 0, 8ull << 30))
            return fail(e, "layer checkpoint actual vector capacity exceeded reservation");
        active_ = true; return true;
    } catch (const std::exception& ex) { discard(); e = ex.what(); return false; }

    bool layer(uint64_t run, int64_t l, int64_t done, const core::SessionState& ss,
               const core::ModelGeometry& g, const Copy& copy, std::string& e) {
        if (!active_ || run != identity_ || l != layer_ || l < 0 || l >= 48 ||
            reached_[size_t(l)] >= end_ ||
            done != reached_[size_t(l)] + std::min(chunk_, end_ - reached_[size_t(l)]))
            return fail(e, "layer checkpoint run/layer/prefix order mismatch");
        for (auto& p : pending_) if (int64_t(p.state.ids.size()) == done) {
            const uint64_t bit = uint64_t(1) << l;
            if (p.mask & bit) return fail(e, "duplicate layer checkpoint piece");
            core::ConversationStateSizes z;
            if (!core::conversation_session_sizes(g, ss, z, e)) return false;
            int64_t qsa = 0, gdn = 0;
            for (int64_t prior = 0; prior < l; ++prior) (core::is_qsa_layer(g, prior) ? qsa : gdn)++;
            if (core::is_qsa_layer(g, l)) {
                const auto& q = ss.qsa_states[qsa];
                if (!copy(p.state.tails.data() + size_t(qsa) * z.tail, q.idx_tail, z.tail, e) ||
                    !copy(p.state.dead.data() + size_t(qsa) * z.dead, q.idx_dead, z.dead, e) ||
                    !copy(p.state.block_pos.data() + size_t(qsa) * z.block_pos, q.idx_block_pos, z.block_pos, e)) return false;
            } else {
                const size_t row = z.gdn / 36;
                if (!copy(p.state.gdn.data() + size_t(gdn) * row,
                          reinterpret_cast<const uint8_t*>(ss.gdn_state) + size_t(gdn) * row, row, e)) return false;
            }
            // PLE history is advanced ONLY by l==1 in run_impl, never by l==47.
            if (l == 1) {
                if (!p.state.ple.empty() && !copy(p.state.ple.data(), ss.ple_hist, z.ple, e)) return false;
                p.ple = true;
            }
            p.mask |= bit; // mark only after every checked copy succeeds
        }
        reached_[size_t(l)] = done;
        if (done == end_) ++layer_;
        return true;
    }
    bool prompt(int64_t done, std::string& e) {
        if (!active_ || layer_ < 47 || done < begin_ || done > end_)
            return fail(e, "layer checkpoint target/MTP completion order mismatch");
        for (auto& p : pending_) if (int64_t(p.state.ids.size()) == done) {
            if (p.mask != ((uint64_t(1) << 48) - 1) || p.prompt)
                return fail(e, "layer checkpoint target/MTP pieces incomplete/duplicate");
            p.prompt = true;
        }
        return true;
    }
    // Called ONLY after Prefill::run, MTP expert reattachment and loan refill.
    // All validation precedes noexcept moves; no partially published chain.
    bool commit(std::vector<Check>& checks, uint64_t& clock, std::string& e) {
        if (!active_ || layer_ != 48 || clock != old_clock_ || checks.size() != old_.size())
            return fail(e, "layer checkpoint transaction boundary mismatch");
        for (size_t i = 0; i < old_.size(); ++i)
            if (checks[i].ids.size() != uint64_t(old_[i].pos) || checks[i].used != old_[i].used)
                return fail(e, "layer checkpoint published chain mutated during run");
        for (const auto& p : pending_)
            if (p.mask != ((uint64_t(1) << 48) - 1) || !p.ple || !p.prompt)
                return fail(e, "layer checkpoint incomplete pending state");
        if (publication_.size() != final_.size())
            return fail(e, "layer checkpoint publication storage mismatch");
        for (const auto& m : final_) {
            if (m.old >= 0) {
                if (uint64_t(m.old) >= checks.size() || checks[size_t(m.old)].ids.size() != uint64_t(m.pos))
                    return fail(e, "layer checkpoint old publication mapping mismatch");
            } else if (std::count_if(pending_.begin(), pending_.end(), [&](const Pending& p) {
                           return p.state.ids.size() == uint64_t(m.pos);
                       }) != 1) return fail(e, "layer checkpoint fresh publication mapping mismatch");
        }
        static_assert(std::is_nothrow_move_assignable_v<Check>);
        for (size_t i = 0; i < final_.size(); ++i) {
            const auto& m = final_[i];
            if (m.old >= 0) publication_[i] = std::move(checks[size_t(m.old)]);
            else {
                auto p = std::find_if(pending_.begin(), pending_.end(), [&](const Pending& p) {
                    return p.state.ids.size() == uint64_t(m.pos);
                });
                // begin constructed exactly one owner for every fresh final key.
                publication_[i] = std::move(p->state);
            }
            publication_[i].used = m.used;
        }
        checks.swap(publication_); clock = clock_;
        discard(); return true;
    }
};
} // namespace strata::program
