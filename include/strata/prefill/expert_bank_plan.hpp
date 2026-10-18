#pragma once

// Authored for Strata; repository LICENSE applies. CPU planning only.
// These decisions establish no USM owner, queue ordering or completed transfer.
#include "strata/prefill/layer_major_plan.hpp"
#include <array>

namespace strata::prefill::detail {

struct ExpertBankPlan {
    static constexpr int64_t experts = 512;
    enum class Selection { original, unsupported, explicit_prefix, insufficient, staged };
    Selection selection = Selection::original;
    int64_t slots = experts;
    uint64_t stride = 0, device_bytes = 0, host_upload_bytes = 0;
    LayerMajorResidualPlan residual;

    // A smaller bank is admissible ONLY with a fully GPU-resident residual.
    // Otherwise preserve the existing full layer cache and host-tail scheduler.
    // Host source storage always holds all512 experts, regardless of GPU slots.
    static ExpertBankPlan make(int64_t n, int64_t chunk, int64_t width,
            uint64_t blob_stride, uint64_t free_gpu, uint64_t other_reserved,
            uint64_t headroom, int64_t requested_rows, LayerMajorResidualDtype dtype,
            int64_t requested_slots, bool supported) {
        if (!blob_stride || blob_stride % 256 || requested_slots < 0 || requested_slots > experts)
            throw std::invalid_argument("invalid expert bank stride or slot count");
        ExpertBankPlan p;
        p.stride = blob_stride;
        p.host_upload_bytes = layer_major_mul(experts, blob_stride);
        const uint64_t fixed = layer_major_add(other_reserved, headroom);
        if (requested_slots) {
            p.selection = Selection::unsupported;
            if (supported && dtype == LayerMajorResidualDtype::bf16 && chunk == 8192 &&
                width == 10240 && n > chunk) {
                p.selection = Selection::explicit_prefix;
                if (requested_rows == -1 || requested_rows == n) {
                    p.selection = Selection::insufficient;
                    const uint64_t bank = layer_major_mul(uint64_t(requested_slots), blob_stride);
                    const uint64_t need = layer_major_add(fixed, bank);
                    if (need <= free_gpu) {
                        auto residual = LayerMajorResidualPlan::make(n, chunk, width,
                            free_gpu - need, -1, dtype);
                        if (residual.gpu_rows == n) {
                            p.selection = Selection::staged;
                            p.slots = requested_slots;
                            p.device_bytes = bank;
                            p.residual = residual;
                            return p;
                        }
                    }
                }
            }
        }
        p.device_bytes = p.host_upload_bytes;
        const uint64_t need = layer_major_add(fixed, p.device_bytes);
        if (need > free_gpu)
            throw std::invalid_argument("original expert bank exceeds free GPU budget with headroom");
        p.residual = LayerMajorResidualPlan::make(n, chunk, width,
            free_gpu - need, requested_rows, dtype);
        return p;
    }

    bool staged() const { return selection == Selection::staged; }
};

// Fixed first-demand admission plus ONE reusable scratch slot. Permanent slots
// are never evicted; a repeated ascending expert scan cannot evict its future
// cache hits. Private mappings may be published after successful enqueue only
// when every reader uses the same in-order queue and the slab survives its drain.
class ExpertBankSlots {
public:
    struct Ticket {
        int32_t expert = -1, slot = -1;
        uint64_t generation = 0;
        bool upload = false, permanent = false;
    };
    explicit ExpertBankSlots(int32_t slots) : slots_(slots) {
        if (slots < 1 || slots > ExpertBankPlan::experts)
            throw std::invalid_argument("expert bank needs1..512 slots");
        residency_.fill(-1);
    }
    ExpertBankSlots(const ExpertBankSlots&) = delete;
    ExpertBankSlots& operator=(const ExpertBankSlots&) = delete;

    Ticket begin(int32_t expert) {
        if (expert < 0 || expert >= ExpertBankPlan::experts || pending_.upload)
            throw std::invalid_argument("invalid expert or unqueued bank ticket");
        const int32_t hit = residency_[size_t(expert)];
        if (hit >= 0) return {expert, hit, 0, false, true};
        if (generation_ == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("expert bank ticket generation exhausted");
        const bool permanent = used_ < slots_ - 1;
        pending_ = {expert, permanent ? used_ : slots_ - 1, ++generation_, true, permanent};
        return pending_;
    }

    // This marks accepted enqueue, not DMA completion. On failed/ambiguous
    // enqueue the caller exits through its checked owner lease, without retry.
    void queued(const Ticket& t) {
        if (!pending_.upload || !t.upload || t.expert != pending_.expert ||
            t.slot != pending_.slot || t.generation != pending_.generation ||
            t.permanent != pending_.permanent)
            throw std::invalid_argument("stale or mismatched expert bank ticket");
        if (t.permanent) {
            residency_[size_t(t.expert)] = t.slot;
            ++used_;
        }
        pending_ = {};
    }

    // Caller must have checked retirement of the previous layer's readers.
    void reset() {
        if (pending_.upload)
            throw std::invalid_argument("reset before expert upload enqueue");
        residency_.fill(-1);
        used_ = 0;
    }
    int32_t permanent_used() const { return used_; }
    int32_t slots() const { return slots_; }

private:
    int32_t slots_, used_ = 0;
    uint64_t generation_ = 0;
    Ticket pending_;
    std::array<int32_t, size_t(ExpertBankPlan::experts)> residency_{};
};

} // namespace strata::prefill::detail
