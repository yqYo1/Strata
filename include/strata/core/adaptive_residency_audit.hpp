// Main-authored diagnostic contract; repository LICENSE.
#pragma once

#include "strata/core/expert_cache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::core {

// Host-side assertions for one real adaptive replacement group. This owns no
// GPU work or payloads: the caller owns readers/queues and supplies synchronous
// device-table readbacks, the real private slot lookup and exact byte checks.
// Use outside a prefill scratch lease; its temporary masks have another owner.
// The initial claimed/layer-compatible slot ownership comes from the qualified
// ExpertCache. This checker preserves it; it does not discover claim metadata.
// A failure is sticky. The caller must stop dependent work and drain while all
// native owners remain live, rather than retry or reset this object.
class AdaptiveResidencyAudit {
public:
    struct Swap {
        std::size_t incoming;             // layer * n_expert + expert
        std::size_t outgoing;
        int32_t slot;
    };

    // Finite diagnostic limits, independent of token/context capacity.
    static constexpr std::size_t max_entries = 1u << 20;
    static constexpr std::size_t max_cycle_payload_bytes = 64u << 20;

    AdaptiveResidencyAudit(std::vector<std::size_t> layer_bytes,
                           std::size_t experts, std::size_t slots,
                           std::size_t swap_limit)
        : experts_(experts), slots_(slots), swap_limit_(swap_limit),
          layer_bytes_(std::move(layer_bytes)) {
        if (layer_bytes_.empty() || !experts_ || experts_ > max_entries ||
            layer_bytes_.size() > max_entries / experts_ || !slots_ ||
            slots_ > max_entries || slots_ > std::size_t(std::numeric_limits<int32_t>::max()) ||
            !swap_limit_ || swap_limit_ > slots_)
            throw std::invalid_argument("adaptive cache audit: invalid geometry");
        for (const auto bytes : layer_bytes_)
            if (!bytes || bytes > max_cycle_payload_bytes)
                throw std::invalid_argument("adaptive cache audit: invalid payload extent");
        const auto rows = layer_bytes_.size() * experts_;
        masked_.resize(rows);
        published_.resize(rows);
        seen_.resize(slots_);
        swaps_.reserve(swap_limit_);
    }

    AdaptiveResidencyAudit(const AdaptiveResidencyAudit&) = delete;
    AdaptiveResidencyAudit& operator=(const AdaptiveResidencyAudit&) = delete;

    bool idle() const { return stage_ == Stage::idle; }
    bool failed() const { return stage_ == Stage::failed; }
    uint64_t generation() const { return generation_; }
    uint64_t completed() const { return completed_; }
    const std::vector<Swap>& swaps() const { return swaps_; }

    // All bookkeeping is allocated before the caller submits any refill.
    // CheckVictim(swap, bytes, err) must compare the outgoing GPU slot with
    // its immutable NativeExpertSource RAM bytes. RAM-complement exchange
    // sources need a separate saved-victim contract; this alone cannot qualify it.
    template<class PrivateSlot, class CheckVictim>
    [[nodiscard]] bool prepare(const std::vector<int32_t>& host, const std::vector<int32_t>& device,
                 const std::vector<Swap>& swaps, PrivateSlot&& private_slot,
                 CheckVictim&& check_victim, std::string& err) {
        err.clear();
        if (!idle()) return fail(err, "prepare requires an idle auditor");
        stage_ = Stage::failed;            // exceptions cannot leave a reusable cycle
        if (generation_ == std::numeric_limits<uint64_t>::max())
            return fail(err, "generation overflow");
        if (host.size() != masked_.size() || swaps.empty() || swaps.size() > swap_limit_)
            return fail(err, "wrong map or positive-swap extent");
        std::fill(seen_.begin(), seen_.end(), uint8_t(0));
        for (const auto slot : host) {
            if (slot == kNotResident) continue;
            if (slot < 0 || std::size_t(slot) >= slots_ || seen_[std::size_t(slot)])
                return fail(err, "before map aliases or names an invalid slot");
            seen_[std::size_t(slot)] = 1;
        }
        if (!views(host, host, device, private_slot, err)) return false;
        std::copy(host.begin(), host.end(), masked_.begin());
        std::copy(host.begin(), host.end(), published_.begin());
        std::size_t payload = 0;
        for (const auto& swap : swaps) {
            if (swap.incoming >= host.size() || swap.outgoing >= host.size() ||
                swap.incoming / experts_ != swap.outgoing / experts_ ||
                swap.slot < 0 || std::size_t(swap.slot) >= slots_ ||
                host[swap.incoming] != kNotResident ||
                masked_[swap.outgoing] != swap.slot ||
                published_[swap.incoming] != kNotResident)
                return fail(err, "invalid,duplicate or cross-layer replacement");
            const auto bytes = layer_bytes_[swap.incoming / experts_];
            if (payload > max_cycle_payload_bytes - bytes)
                return fail(err, "replacement payload budget");
            payload += bytes;
            masked_[swap.outgoing] = kNotResident;
            published_[swap.outgoing] = kNotResident;
            published_[swap.incoming] = swap.slot;
        }
        swaps_.assign(swaps.begin(), swaps.end());
        for (const auto& swap : swaps_)
            if (!check_victim(swap, layer_bytes_[swap.outgoing / experts_], err))
                return fail_if_empty(err, "outgoing payload differs");
        ++generation_;
        stage_ = Stage::prepared;
        return true;
    }

    template<class PrivateSlot>
    [[nodiscard]] bool masked(const std::vector<int32_t>& host, const std::vector<int32_t>& device,
                PrivateSlot&& private_slot, std::string& err) {
        err.clear();
        if (stage_ != Stage::prepared) return fail(err, "mask before prepare or after failure");
        stage_ = Stage::failed;
        if (!views(masked_, host, device, private_slot, err)) return false;
        stage_ = Stage::masked;
        return true;
    }

    // Wait returns true only after the actual refill completion was checked.
    // CheckIncoming(swap, bytes, err) then performs the full-byte comparison.
    // Invoke before RAM-complement source commits or residency publication.
    template<class Wait, class CheckIncoming>
    [[nodiscard]] bool complete(Wait&& wait, CheckIncoming&& check_incoming, std::string& err) {
        err.clear();
        if (stage_ != Stage::masked) return fail(err, "completion requires the checked mask");
        stage_ = Stage::failed;
        if (!wait()) return fail(err, "refill completion failed");
        for (const auto& swap : swaps_)
            if (!check_incoming(swap, layer_bytes_[swap.incoming / experts_], err))
                return fail_if_empty(err, "incoming payload differs");
        stage_ = Stage::payload_checked;
        return true;
    }

    // A partial host/private/device publication never closes the audit cycle.
    template<class PrivateSlot>
    [[nodiscard]] bool published(const std::vector<int32_t>& host, const std::vector<int32_t>& device,
                   PrivateSlot&& private_slot, std::string& err) {
        err.clear();
        if (stage_ != Stage::payload_checked)
            return fail(err, "publication before checked completion and payload");
        stage_ = Stage::failed;
        if (!views(published_, host, device, private_slot, err)) return false;
        ++completed_;
        stage_ = Stage::idle;
        return true;
    }

private:
    enum class Stage { idle, prepared, masked, payload_checked, failed };

    bool fail(std::string& err, const char* message) {
        stage_ = Stage::failed;
        err = std::string("adaptive cache audit: ") + message;
        return false;
    }

    bool fail_if_empty(std::string& err, const char* message) {
        stage_ = Stage::failed;
        if (err.empty()) err = std::string("adaptive cache audit: ") + message;
        return false;
    }

    template<class PrivateSlot>
    bool views(const std::vector<int32_t>& expected, const std::vector<int32_t>& host,
               const std::vector<int32_t>& device, PrivateSlot& private_slot,
               std::string& err) {
        if (host.size() != expected.size() || device.size() != expected.size())
            return fail(err, "snapshot extent");
        for (std::size_t row = 0; row < expected.size(); ++row) {
            if (host[row] != expected[row] || device[row] != expected[row] ||
                private_slot(row) != expected[row]) {
                stage_ = Stage::failed;
                err = "adaptive cache audit: host/device/private identity differs at row " +
                      std::to_string(row);
                return false;
            }
        }
        return true;
    }

    std::size_t experts_, slots_, swap_limit_;
    std::vector<std::size_t> layer_bytes_;
    std::vector<int32_t> masked_, published_;
    std::vector<uint8_t> seen_;
    std::vector<Swap> swaps_;
    uint64_t generation_ = 0, completed_ = 0;
    Stage stage_ = Stage::idle;
};

} // namespace strata::core
