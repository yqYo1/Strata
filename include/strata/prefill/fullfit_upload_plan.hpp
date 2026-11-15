#pragma once
// Main-authored, repository LICENSE. CPU planning/publication only.
// The native caller must retain source/bank owners through a checked drain and
// put every enqueue and reader on the same in-order queue. No DMA completion is
// established here. No decode path calls this experimental prefill helper.
#include "strata/prefill/expert_bank_plan.hpp"
#include <vector>

namespace strata::prefill::detail {

struct PrefillFullfitCopy {
    uint64_t source_offset=0, destination_offset=0, bytes=0;
    size_t first=0, experts=0;
};

class PrefillFullfitUploadPlan {
    std::vector<int32_t> experts_;
    std::vector<PrefillFullfitCopy> copies_;
    uint64_t payload_bytes_=0;
    PrefillFullfitUploadPlan()=default;
public:
    const std::vector<int32_t>& experts() const { return experts_; }
    const std::vector<PrefillFullfitCopy>& copies() const { return copies_; }
    uint64_t payload_bytes() const { return payload_bytes_; }

    static PrefillFullfitUploadPlan make(const std::vector<int32_t>& ids,
            const ExpertBankLayout& bank, uint64_t source_stride, bool coalesce) {
        const auto checked=ExpertBankLayout::packed(bank.backing_bytes,bank.stride);
        if (!checked.full_fit() || checked.slots!=bank.slots ||
            checked.used_bytes!=bank.used_bytes || source_stride<bank.stride ||
            source_stride%256 || ids.size()>size_t(ExpertBankPlan::experts))
            throw std::invalid_argument("prefill full-fit upload geometry");
        const auto source_bytes=layer_major_mul(ExpertBankPlan::experts,source_stride);
        PrefillFullfitUploadPlan plan;
        plan.experts_=ids;
        for (size_t i=0;i<ids.size();++i)
            if (ids[i]<0 || ids[i]>=ExpertBankPlan::experts || (i && ids[i]<=ids[i-1]))
                throw std::invalid_argument("prefill full-fit upload expert order/range");
        plan.payload_bytes_=layer_major_mul(ids.size(),bank.stride);
        for (size_t first=0;first<ids.size();) {
            size_t end=first+1;
            if (coalesce && source_stride==bank.stride)
                while (end<ids.size() && ids[end]==ids[end-1]+1) ++end;
            const auto source=layer_major_mul(uint64_t(ids[first]),source_stride);
            const auto destination=layer_major_mul(first,bank.stride);
            const auto bytes=layer_major_mul(end-first,bank.stride);
            if (source>source_bytes || bytes>source_bytes-source ||
                destination>bank.backing_bytes || bytes>bank.backing_bytes-destination)
                throw std::invalid_argument("prefill full-fit upload copy bounds");
            plan.copies_.push_back({source,destination,bytes,first,end-first});
            first=end;
        }
        return plan;
    }
};

// A copy must be accepted before any of its mappings are published. Failed or
// ambiguous enqueue exits to the native caller's checked owner lease, without
// retry. Earlier accepted copies/maps survive until that same checked drain.
template<class Enqueue>
inline void enqueue_prefill_fullfit_upload(const PrefillFullfitUploadPlan& plan,
        ExpertBankSlots& slots, Enqueue&& enqueue) {
    if (!slots.full_fit() || slots.slots()!=ExpertBankPlan::experts || slots.permanent_used())
        throw std::invalid_argument("prefill full-fit upload requires empty private bank");
    for (const auto& copy:plan.copies()) {
        if (!enqueue(copy)) throw std::runtime_error("prefill full-fit upload enqueue failed");
        for (size_t i=copy.first;i<copy.first+copy.experts;++i) {
            const auto ticket=slots.begin(plan.experts()[i]);
            if (!ticket.upload || !ticket.permanent || ticket.slot!=int32_t(i))
                throw std::runtime_error("prefill full-fit upload publication mismatch");
            slots.queued(ticket);
        }
    }
}

} // namespace strata::prefill::detail
