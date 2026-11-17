#pragma once
// Main-authored from the owned PrefillSyclCopyGroups contract; repository LICENSE.
// Unbuilt, unselected proposal for a fresh, full-fit prefill weight bank.
#include "strata/prefill/sycl_copy_groups.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace strata::prefill::detail {

// One host submitter. Each destination slot is written exactly once by this
// owner, in increasing slot order. All sources, the exclusive destination bank
// and consumer buffers remain alive through terminal drain(), including errors.
// Declare this owner after those borrowed owners. For a cache/arena loan, drain
// and destroy it while the checked lease is still live: member destruction alone
// does not establish that ordering. Accepted enqueue is not completion. Cache
// publication must carry readiness to every consumer and invalidate partial
// state on failure; never consume a slot without its readiness dependency.
//
// Unlike a recycled bank, these slots need no per-expert reader marker: none
// can be overwritten before the terminal drain. Enforcing that restriction is
// part of the same coherent unit as omitting the markers. This does not qualify
// cache publication, model handoff, asynchronous failure handling or performance.
class PrefillSyclUniqueSlotCopies {
    using Groups = PrefillSyclCopyGroups;

public:
    using Placement = Groups::Placement;
    using Ready = Groups::Ready;

    PrefillSyclUniqueSlotCopies(std::shared_ptr<sycl_backend::Runtime> runtime,
                               void* slots, std::size_t expert_bytes,
                               std::size_t slot_count, Placement placement)
        : slot_count_(prefill_checked_slot_count(slot_count)),
          groups_(std::move(runtime), slots, {expert_bytes, 1, slot_count_},
                  placement, slot_count_) {}

    PrefillSyclUniqueSlotCopies(const PrefillSyclUniqueSlotCopies&) = delete;
    PrefillSyclUniqueSlotCopies& operator=(const PrefillSyclUniqueSlotCopies&) = delete;

    Ready prefill_copy_next_slot(std::size_t expected_slot,
                                std::span<const std::byte> source) {
        if (expected_slot != submitted_ || submitted_ >= slot_count_)
            throw std::invalid_argument("prefill unique slots: nonsequential or reused slot");
        const std::array<std::span<const std::byte>, 1> sources{source};
        auto ticket = groups_.prefill_copy_group(sources);
        ++submitted_; // No slot advance/publication after a rejected submission.
        return ticket;
    }

    void prefill_wait_ready(Ready ticket) { groups_.prefill_wait_ready(ticket); }
    void prefill_wait_ready(std::span<const Ready> tickets) {
        groups_.prefill_wait_ready(tickets);
    }

    std::span<const std::byte> prefill_expert_bytes(Ready ticket) {
        return groups_.prefill_expert(ticket, 0);
    }

    std::size_t prefill_submitted_slots() const noexcept { return submitted_; }
    void drain() { groups_.drain(); }

private:
    const std::size_t slot_count_;
    Groups groups_; // Its destructor drains before any borrowed owner may retire.
    std::size_t submitted_ = 0;

    static std::size_t prefill_checked_slot_count(std::size_t count) {
        if (!count || count > 512)
            throw std::invalid_argument("prefill unique slots: capacity outside 1..512");
        return count;
    }
};
} // namespace strata::prefill::detail
