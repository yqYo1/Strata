#pragma once

// Authored for Strata; repository LICENSE applies. The caller owns the slab,
// cache and private map through a checked queue drain, including error exits.
#include "strata/core/expert_cache.hpp"
#include "strata/prefill/expert_bank_plan.hpp"
#include "strata/prefill/fullfit_upload_plan.hpp"
#include "strata/prefill/expert_upload_staging.hpp"
#include "strata/sycl/launch.hpp"
#include "strata/sycl/compat/cuda_runtime.h"
#if defined(STRATA_SYCL_PREFILL_NATIVE_COPY)
#include "strata/sycl/native_copy_list.hpp"
#else
namespace strata::sycl_backend::native_copy { class NativeCopyList; }
#endif
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

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

class PrefillSplitExpertBatch {
public:
    static constexpr int32_t capacity_limit = 16;

    template<size_t Experts>
    static PrefillSplitExpertBatch make(const std::array<int32_t, Experts>& counts, int32_t first,
                                       int32_t capacity, bool coalesced) {
        static_assert(Experts <= size_t(std::numeric_limits<int32_t>::max()));
        if (first < 0 || size_t(first) >= Experts || counts[size_t(first)] <= 0 ||
            capacity < 1 || capacity > capacity_limit)
            throw std::invalid_argument("split prefill expert upload: invalid first expert or batch capacity");
        const int32_t limit = first + std::min(capacity, int32_t(Experts) - first);
        int32_t end = first + 1;
        while (end < limit && counts[size_t(end)] > 0) ++end;
        return PrefillSplitExpertBatch(first, end - first, coalesced);
    }

    int32_t end() const { return first_ + count_; }
    int32_t count() const { return count_; }
    int32_t copies() const { return coalesced_ ? 1 : count_; }

    // The caller validates the extents and retains the source and private
    // bank through the checked drain of this same in-order queue.
    cudaError_t enqueue(uint8_t* bank, const uint8_t* slab, size_t stride, cudaStream_t stream) const {
        const uint8_t* src = slab + size_t(first_) * stride;
        const int32_t per_copy = coalesced_ ? count_ : 1;
        for (int32_t i = 0; i < count_; i += per_copy) {
            const size_t offset = size_t(i) * stride;
            const auto status = cudaMemcpyAsync(bank + offset, src + offset, size_t(per_copy) * stride,
                                                cudaMemcpyHostToDevice, stream);
            if (status != cudaSuccess) return status;
        }
        return cudaSuccess;
    }

private:
    PrefillSplitExpertBatch(int32_t first, int32_t count, bool coalesced)
        : first_(first), count_(count), coalesced_(coalesced) {}
    int32_t first_;
    int32_t count_;
    bool coalesced_;
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

enum class PrefillHostStageOrder { all_before_copy, span_before_copy };

struct PrefillFullfitUploadReceipt {
    std::array<bool,512> first_upload{};
    uint64_t experts=0,copy_commands=0,payload_bytes=0;
    uint64_t host_staged_bytes=0;
};

// The original same-queue route publishes after submission. The fenced
// controls wait at this call site before publishing either transport's map.
// Source and bank owners live through the outer checked lease on every exit.
inline PrefillFullfitUploadReceipt stage_prefill_fullfit_eager(ExpertBankSlots& slots,
        const PrivateExpertBank& bank,const uint8_t* slab,uint64_t slab_bytes,
        uint64_t source_stride,const std::vector<int32_t>& experts,void* stream,
        std::initializer_list<sycl_backend::Span> live_scratch,
        sycl_backend::Allocation* host_staging = nullptr,
        sycl_backend::native_copy::NativeCopyList* native_copy = nullptr,
        bool wait_for_copies = false,
        std::optional<PrefillFullfitUploadPlan>* deferred_publication = nullptr,
        PrefillHostStageOrder stage_order = PrefillHostStageOrder::all_before_copy) {
    const bool stage_per_span = stage_order == PrefillHostStageOrder::span_before_copy;
    if (stage_per_span && (!native_copy || !host_staging))
        throw std::invalid_argument("prefill per-span staging requires native copy and host USM");
    if (!stream || !slab || slab_bytes!=layer_major_mul(512,source_stride))
        throw std::invalid_argument("prefill eager upload source/queue extent");
    if (native_copy && (!host_staging || !wait_for_copies))
        throw std::invalid_argument("native prefill copies require host staging and checked completion");
    if (deferred_publication && (!native_copy || deferred_publication->has_value()))
        throw std::invalid_argument("prefill deferred publication requires an unused native copy plan");
#if !defined(STRATA_SYCL_PREFILL_NATIVE_COPY)
    if (native_copy) throw std::logic_error("native prefill copies require the native-copy build");
#endif
    auto& queue=sycl_backend::queue_for(stream);
    const auto& layout=bank.layout();
    auto* base=bank.device_slot(0);
    sycl_backend::validate_device_span(base,size_t(layout.backing_bytes),queue);
    sycl_backend::validate_spans({{base,size_t(layout.backing_bytes)}},
        {{slab,size_t(slab_bytes)}},1);
    sycl_backend::validate_spans({{base,size_t(layout.backing_bytes)}},live_scratch,1);
    auto plan=PrefillFullfitUploadPlan::make(experts,layout,source_stride);
    PrefillFullfitUploadReceipt result;
    if (wait_for_copies && cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) != cudaSuccess)
        throw std::runtime_error("prefill full-fit upload preceding compute drain failed");
    // Keep source offsets and individual DMA commands identical. The caller
    // drains the prior layer before rewriting these CPU-writable USM pages.
    const uint8_t* upload_source = slab;
    uint8_t* staged = nullptr;
    const auto stage = [&](const PrefillFullfitCopy& copy) {
        std::memcpy(staged + size_t(copy.source_offset), slab + size_t(copy.source_offset),
                    size_t(copy.bytes));
    };
    if (host_staging) {
        require_empty_prefill_fullfit_bank(slots);
        staged = host_staging->as<uint8_t>();
        if (!staged || host_staging->size() < slab_bytes ||
            sycl::get_pointer_type(staged, queue.get_context()) != sycl::usm::alloc::host ||
            sycl::get_pointer_type(slab, queue.get_context()) == sycl::usm::alloc::device)
            throw std::invalid_argument("prefill eager upload host staging source or extent");
        sycl_backend::validate_spans({{staged, size_t(slab_bytes)}},
            {{slab, size_t(slab_bytes)}, {base, size_t(layout.backing_bytes)}}, 1);
        sycl_backend::validate_spans({{staged, size_t(slab_bytes)}}, live_scratch, 1);
#if defined(STRATA_SYCL_PREFILL_NATIVE_COPY)
        if (native_copy) native_copy->validate_endpoints(queue, base, staged);
#endif
        if (stage_per_span) {
            // The sorted unique source offsets and padded strides in the
            // plan keep CPU writes off the already submitted source lines.
            if (reinterpret_cast<uintptr_t>(staged) % 256)
                throw std::invalid_argument("prefill per-span staging requires a 256-byte-aligned host base");
        } else {
            for (const auto& copy : plan.copies()) stage(copy);
        }
        upload_source = staged;
        result.host_staged_bytes = plan.payload_bytes();
    }
    auto submit = [&](const PrefillFullfitCopy& copy) {
        if (stage_per_span) stage(copy);
        auto* destination = base + size_t(copy.destination_offset);
        const auto* source = upload_source + size_t(copy.source_offset);
#if defined(STRATA_SYCL_PREFILL_NATIVE_COPY)
        if (native_copy) native_copy->append(destination, source, size_t(copy.bytes));
        else
#endif
        if (cudaMemcpyAsync(destination, source, size_t(copy.bytes), cudaMemcpyHostToDevice,
                            static_cast<cudaStream_t>(stream)) != cudaSuccess) return false;
        ++result.copy_commands;
        return true;
    };
    if (wait_for_copies) {
        require_empty_prefill_fullfit_bank(slots);
        for (const auto& copy : plan.copies())
            if (!submit(copy)) throw std::runtime_error("prefill full-fit upload enqueue failed");
        if (!deferred_publication) {
#if defined(STRATA_SYCL_PREFILL_NATIVE_COPY)
            if (native_copy) native_copy->wait();
            else
#endif
            if (cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) != cudaSuccess)
                throw std::runtime_error("prefill full-fit upload copy completion failed");
            for (const auto& copy : plan.copies()) publish_prefill_fullfit_copy(plan, slots, copy);
        }
    } else enqueue_prefill_fullfit_upload(plan, slots, submit);
    result.experts=plan.experts().size();result.payload_bytes=plan.payload_bytes();
    for (int32_t expert:plan.experts()) result.first_upload[size_t(expert)]=true;
    if (deferred_publication) deferred_publication->emplace(std::move(plan));
    return result;
}

#if defined(STRATA_SYCL_PREFILL_NATIVE_COPY)
// The request's checked lease holds the copy list, source, bank, and slots
// until this publication boundary or a checked failure drain.
inline void finish_prefill_fullfit_deferred(ExpertBankSlots& slots,
        sycl_backend::native_copy::NativeCopyList& native_copy,
        std::optional<PrefillFullfitUploadPlan>& pending) {
    if (!pending) throw std::logic_error("prefill deferred upload has no accepted plan");
    require_empty_prefill_fullfit_bank(slots);
    native_copy.wait();
    for (const auto& copy : pending->copies()) publish_prefill_fullfit_copy(*pending, slots, copy);
    pending.reset();
}
#endif

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
