#pragma once

// Authored for Strata; repository LICENSE applies. The caller owns the slab,
// cache and private map through a checked queue drain, including error exits.
#include "strata/core/expert_cache.hpp"
#include "strata/prefill/expert_bank_plan.hpp"
#include "strata/sycl/launch.hpp"

namespace strata::prefill::detail {

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
