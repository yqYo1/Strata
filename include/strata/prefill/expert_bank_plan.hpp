#pragma once

// Authored for Strata; repository LICENSE applies. CPU planning only.
// These decisions establish no USM owner, queue ordering or completed transfer.
#include "strata/prefill/layer_major_plan.hpp"
#include <array>

namespace strata::prefill::detail {

struct ExpertBankPlan {
    static constexpr int64_t experts = 512;
    static constexpr int64_t window_rows = 81920; // mode10, chunk8192
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
            uint64_t headroom, uint64_t max_device_alloc_bytes,
            int64_t requested_rows, LayerMajorResidualDtype dtype,
            int64_t requested_slots, bool supported) {
        if (!blob_stride || blob_stride % 256 || !max_device_alloc_bytes ||
            requested_slots < 0 || requested_slots > experts)
            throw std::invalid_argument("invalid expert bank stride or slot count");
        ExpertBankPlan p;
        p.stride = blob_stride;
        p.host_upload_bytes = layer_major_mul(experts, blob_stride);
        const uint64_t fixed = layer_major_add(other_reserved, headroom);
        if (requested_slots) {
            p.selection = Selection::unsupported;
            // An all-GPU residual is insufficient when its final window is
            // <=one chunk: that scheduler branch uses normal FFN, whose
            // expert pointers still require the full512-slot cache.
            const int64_t tail = n > 0 ? n % window_rows : 0;
            const bool short_tail = n > window_rows && tail > 0 && tail <= chunk;
            if (supported && !short_tail && dtype == LayerMajorResidualDtype::bf16 &&
                chunk == 8192 && width == 10240 && n > chunk) {
                p.selection = Selection::explicit_prefix;
                if (requested_rows == -1 || requested_rows == n) {
                    p.selection = Selection::insufficient;
                    const uint64_t bank = layer_major_mul(uint64_t(requested_slots), blob_stride);
                    const uint64_t need = layer_major_add(fixed, bank);
                    if (need <= free_gpu && bank <= max_device_alloc_bytes) {
                        auto residual = LayerMajorResidualPlan::make(n, chunk, width,
                            std::min(free_gpu - need, max_device_alloc_bytes), -1, dtype);
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
        if (need > free_gpu || p.device_bytes > max_device_alloc_bytes)
            throw std::invalid_argument("original expert bank exceeds free GPU budget with headroom");
        p.residual = LayerMajorResidualPlan::make(n, chunk, width,
            std::min(free_gpu - need, max_device_alloc_bytes), requested_rows, dtype);
        return p;
    }

    bool staged() const { return selection == Selection::staged; }
};

// A private view of an existing allocation; no allocation, ownership or drain.
// Rebind only after all readers of the preceding layer have checked retirement.
struct ExpertBankLayout {
    uint64_t backing_bytes = 0, stride = 0, used_bytes = 0;
    int32_t slots = 0;
    static ExpertBankLayout packed(uint64_t backing, uint64_t layer_stride) {
        if (!backing || !layer_stride || layer_stride % 256 || backing < layer_stride)
            throw std::invalid_argument("invalid private expert bank extent");
        const auto count = std::min<uint64_t>(ExpertBankPlan::experts, backing / layer_stride);
        return {backing, layer_stride, layer_major_mul(count, layer_stride), int32_t(count)};
    }
    uint64_t offset(int32_t slot) const {
        if (slot < 0 || slot >= slots)
            throw std::invalid_argument("private expert bank slot out of range");
        return layer_major_mul(uint64_t(slot), stride);
    }
    bool full_fit() const { return slots == ExpertBankPlan::experts; }
};

// Fixed first-demand admission plus ONE reusable scratch slot. Permanent slots
// are never evicted; a repeated ascending expert scan cannot evict its future
// cache hits. Private mappings may be published after successful enqueue only
// when every reader uses the same in-order queue and the slab survives its drain.
// An explicitly full-fit512 view needs no scratch: all experts can be permanent.
class ExpertBankSlots {
public:
    struct Ticket {
        int32_t expert = -1, slot = -1;
        uint64_t generation = 0;
        bool upload = false, permanent = false;
    };
    explicit ExpertBankSlots(int32_t slots, bool full_fit = false) : slots_(slots), full_fit_(full_fit) {
        if (slots < 1 || slots > ExpertBankPlan::experts || (full_fit && slots != ExpertBankPlan::experts))
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
        const bool permanent = used_ < slots_ - (full_fit_ ? 0 : 1);
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
    bool full_fit() const { return full_fit_; }

private:
    int32_t slots_, used_ = 0;
    bool full_fit_ = false;
    uint64_t generation_ = 0;
    Ticket pending_;
    std::array<int32_t, size_t(ExpertBankPlan::experts)> residency_{};
};

} // namespace strata::prefill::detail
