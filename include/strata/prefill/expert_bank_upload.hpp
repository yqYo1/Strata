#pragma once

// Authored for Strata; repository LICENSE applies. The caller owns the slab,
// cache and private map through a checked queue drain, including error exits.
#include "strata/core/expert_cache.hpp"
#include "strata/prefill/expert_bank_plan.hpp"
#include "strata/prefill/fullfit_upload_plan.hpp"
#include "strata/prefill/expert_upload_staging.hpp"
#include "strata/sycl/launch.hpp"
#include "strata/sycl/compat/cuda_runtime.h"

namespace strata::prefill::detail {

// This view never changes ExpertCache's general slot layout/residency. It may
// only borrow a private, untracked allocation owned by the outer checked lease;
// tracked decode caches must continue through their existing writer protocol.
class PrivateExpertBank {
public:
    PrivateExpertBank(core::ExpertCache& cache, uint64_t stride)
        : cache_(&cache), base_(cache.device_slot(0)),
          layout_(ExpertBankLayout::packed(uint64_t(cache.bytes()), stride)) {
        check_owner();
    }
    const ExpertBankLayout& layout() const { return layout_; }
    const core::ExpertCache* owner() const { return cache_; }
    uint8_t* device_slot(int32_t slot) const {
        check_owner();
        return base_ + size_t(layout_.offset(slot));
    }
private:
    void check_owner() const {
        if (!cache_->valid() || !base_ || cache_->device_slot(0) != base_ ||
            cache_->slot_offsets() || cache_->scratch_tracking_enabled() ||
            cache_->bytes() <= 0 || uint64_t(cache_->bytes()) != layout_.backing_bytes)
            throw std::invalid_argument("private expert bank owner changed or is not private");
    }
    core::ExpertCache* cache_;
    uint8_t* base_;
    ExpertBankLayout layout_;
};

inline bool prefill_fullfit_eager_requested() {
    static const bool selected=[] {
        const char* value=std::getenv("STRATA_SYCL_PREFILL_FULLFIT_UPLOAD");
        if (!value || !std::strcmp(value,"jit")) return false;
        if (!std::strcmp(value,"eager")) return true;
        throw std::invalid_argument("prefill full-fit upload: expected jit or eager");
    }();
    return selected;
}

struct PrefillFullfitUploadReceipt {
    std::array<bool,512> first_upload{};
    uint64_t experts=0,copy_commands=0,payload_bytes=0;
};

// U1 changes only the position of the existing individual copies. Source and
// bank owners belong to the outer checked lease; no wait or queue is added.
inline PrefillFullfitUploadReceipt stage_prefill_fullfit_eager(ExpertBankSlots& slots,
        const PrivateExpertBank& bank,const uint8_t* slab,uint64_t slab_bytes,
        uint64_t source_stride,const std::vector<int32_t>& experts,void* stream,
        std::initializer_list<sycl_backend::Span> live_scratch) {
    if (!stream || !slab || slab_bytes!=layer_major_mul(512,source_stride))
        throw std::invalid_argument("prefill eager upload source/queue extent");
    auto& queue=sycl_backend::queue_for(stream);
    const auto& layout=bank.layout();
    auto* base=bank.device_slot(0);
    sycl_backend::validate_device_span(base,size_t(layout.backing_bytes),queue);
    sycl_backend::validate_spans({{base,size_t(layout.backing_bytes)}},
        {{slab,size_t(slab_bytes)}},1);
    sycl_backend::validate_spans({{base,size_t(layout.backing_bytes)}},live_scratch,1);
    const auto plan=PrefillFullfitUploadPlan::make(experts,layout,source_stride);
    PrefillFullfitUploadReceipt result;
    enqueue_prefill_fullfit_upload(plan,slots,[&](const PrefillFullfitCopy& copy) {
        if (cudaMemcpyAsync(base+size_t(copy.destination_offset),slab+size_t(copy.source_offset),
                size_t(copy.bytes),cudaMemcpyHostToDevice,static_cast<cudaStream_t>(stream))!=cudaSuccess)
            return false;
        ++result.copy_commands;
        return true;
    });
    result.experts=plan.experts().size();result.payload_bytes=plan.payload_bytes();
    for (int32_t expert:plan.experts()) result.first_upload[size_t(expert)]=true;
    return result;
}

template<class Enqueue>
inline ExpertBankSlots::Ticket stage_private_expert_bank_enqueue(ExpertBankSlots& slots,
        const PrivateExpertBank& bank, const uint8_t* slab, uint64_t slab_bytes,
        uint64_t blob_bytes, int32_t expert, void* stream, uint64_t source_stride, Enqueue&& enqueue) {
    const auto& layout = bank.layout();
    if (!stream || !slab || !blob_bytes || blob_bytes > layout.stride ||
        blob_bytes > source_stride || slots.slots() != layout.slots ||
        slots.full_fit() != layout.full_fit() ||
        slab_bytes != layer_major_mul(ExpertBankPlan::experts, source_stride))
        throw std::invalid_argument("private expert bank source/slot extent mismatch");
    auto& queue = sycl_backend::queue_for(stream);
    auto* base = bank.device_slot(0); // checks frozen backing identity
    sycl_backend::validate_device_span(base, size_t(layout.backing_bytes), queue);
    sycl_backend::validate_spans({{base, size_t(layout.backing_bytes)}},
                                {{slab, size_t(slab_bytes)}}, 1);
    const auto ticket = slots.begin(expert);
    if (ticket.upload) {
        const auto source_offset = layer_major_mul(uint64_t(expert), source_stride);
        if (source_offset > slab_bytes || blob_bytes > slab_bytes - source_offset)
            throw std::invalid_argument("private expert bank source offset mismatch");
        if (!enqueue(bank.device_slot(ticket.slot), slab + size_t(source_offset), size_t(blob_bytes), stream))
            throw std::runtime_error("private expert bank upload enqueue failed");
        slots.queued(ticket); // accepted enqueue; same-queue readers precede any overwrite
    }
    return ticket;
}

inline ExpertBankSlots::Ticket stage_private_expert_bank(ExpertBankSlots& slots,
        const PrivateExpertBank& bank, const uint8_t* slab, uint64_t slab_bytes,
        uint64_t blob_bytes, int32_t expert, void* stream, uint64_t source_stride,
        ExpertUploadStaging* staging = nullptr) {
    return stage_private_expert_bank_enqueue(slots,bank,slab,slab_bytes,blob_bytes,expert,stream,source_stride,
        [staging](void* dst, const void* src, size_t bytes, void* queue) {
            if (staging) return staging->enqueue(dst, src, bytes, queue);
#if defined(STRATA_SYCL_EXPERT_UPLOAD_ATTRIBUTION)
            sycl_backend::cpu_wait_profile::Host submission(sycl_backend::cpu_wait_profile::upload_submit);
#endif
            return cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,static_cast<cudaStream_t>(queue)) == cudaSuccess;
        });
}

inline ExpertBankSlots::Ticket stage_expert_bank(ExpertBankSlots& slots,
        core::ExpertCache& cache, const uint8_t* slab, uint64_t slab_bytes,
        uint64_t stride, uint64_t blob_bytes, int32_t expert, void* stream,
        uint64_t source_stride = 0) {
    if (!source_stride) source_stride = stride;
    if (!stream || !slab || !stride || stride % 256 || !blob_bytes || blob_bytes > stride ||
        blob_bytes > source_stride ||
        cache.slot_offsets() || cache.slots() != slots.slots() ||
        uint64_t(cache.bytes()) != layer_major_mul(uint64_t(slots.slots()), stride) ||
        slab_bytes != layer_major_mul(ExpertBankPlan::experts, source_stride))
        throw std::invalid_argument("prefill expert bank: owner/stride extent mismatch");
    auto& queue = sycl_backend::queue_for(stream); // requires the in-order queue
    sycl_backend::validate_device_span(cache.device_slot(0), size_t(cache.bytes()), queue);
    sycl_backend::validate_spans({{cache.device_slot(0), size_t(cache.bytes())}},
                                {{slab, size_t(slab_bytes)}}, 1);
    const auto ticket = slots.begin(expert);
    if (ticket.upload) {
        const auto offset = layer_major_mul(uint64_t(expert), source_stride);
        if (offset > slab_bytes || blob_bytes > slab_bytes - offset)
            throw std::invalid_argument("prefill expert bank: source extent mismatch");
        std::string error;
        if (!cache.fill_slot(ticket.slot, slab + size_t(offset), stream, error, int64_t(blob_bytes)))
            throw std::runtime_error(error);
        // Accepted enqueue, not completion. Every consumer and next scratch
        // overwrite uses this same queue. Scratch is never a permanent mapping.
        slots.queued(ticket);
    }
    return ticket;
}

} // namespace strata::prefill::detail
