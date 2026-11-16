#pragma once
// Main-authored standard-SYCL grouping proposal; repository LICENSE.
// Uses only this project's owned copy-lane/Runtime contracts. No external
// implementation. Unbuilt and unselected until its separate comparison closes.
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/command_profile_hook.hpp"
#include "strata/sycl/runtime.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::prefill::detail {

// One host submitter, immutable ordinary RAM or same-context host-USM sources.
// The caller retains readable sources, the entire device slot allocation and
// all math allocations until terminal drain(), including exceptional paths.
// Declare this owner last so its destructor drains before borrowed owners die.
// A returned device span must only be read by the ordered compute queue, after
// prefill_wait_ready() and before prefill_record_readers() for that group.
//
// Groups are submitted/consumed in sequence, using a fixed number of rotating
// banks. Copying the next use of a bank requires its previous readers marker.
// Free banks can be filled ahead of compute. Copies remain separate commands;
// this class changes readiness/retirement granularity, not payload or kernels.
// Keep total slots fixed when comparing group width independently of capacity.
class PrefillSyclCopyGroups {
public:
    enum class Placement { Compute, Separate };
    struct Layout {
        std::size_t expert_bytes;
        std::size_t group_width;
        std::size_t bank_count;
    };
    class Ready {
        friend class PrefillSyclCopyGroups;
        Ready(std::uint64_t owner, std::size_t index) : owner_(owner), index_(index) {}
        std::uint64_t owner_;
        std::size_t index_;
    };

    PrefillSyclCopyGroups(std::shared_ptr<sycl_backend::Runtime> runtime,
                         void* slots, Layout layout, Placement placement,
                         std::size_t capacity = 1024)
        : runtime_(std::move(runtime)), slots_(static_cast<std::byte*>(slots)),
          layout_(layout), bytes_(checked_bytes(layout)),
          identity_(next_identity_.fetch_add(1)), capacity_(capacity) {
        require(runtime_ && slots_ && identity_ && capacity_ && capacity_ <= 4096 &&
                bytes_ <= std::numeric_limits<std::uintptr_t>::max() -
                    reinterpret_cast<std::uintptr_t>(slots_),
                "prefill copy groups owner/range/capacity");
        require(placement == Placement::Compute || placement == Placement::Separate,
                "prefill copy groups placement");
        auto& compute = runtime_->compute();
        require(compute.has_property<sycl::property::queue::in_order>(),
                "prefill copy groups require in-order compute");
        sycl_backend::validate_device_span(slots_, bytes_, compute);
        states_.reserve(capacity_);
        bank_owners_.resize(layout_.bank_count);
        if (placement == Placement::Separate) {
            separate_.emplace(runtime_->make_queue(
                compute.has_property<sycl::property::queue::enable_profiling>()));
            require(separate_->get_context() == compute.get_context() &&
                    separate_->get_device() == compute.get_device() &&
                    separate_->has_property<sycl::property::queue::in_order>() &&
                    separate_->has_property<sycl::property::queue::enable_profiling>() ==
                        compute.has_property<sycl::property::queue::enable_profiling>(),
                    "prefill copy groups queue property/context mismatch");
        }
    }
    PrefillSyclCopyGroups(const PrefillSyclCopyGroups&) = delete;
    PrefillSyclCopyGroups& operator=(const PrefillSyclCopyGroups&) = delete;
    ~PrefillSyclCopyGroups() noexcept {
        sycl_backend::checked_cleanup([&] { drain(); }, [] {},
                                     "prefill copy groups retirement");
    }

    Ready prefill_copy_group(std::span<const std::span<const std::byte>> sources) {
        require(!sealed_ && !poisoned_ && !sources.empty() &&
                sources.size() <= layout_.group_width && states_.size() < capacity_,
                "prefill copy groups source count/capacity/state");
        const auto bank = states_.size() % layout_.bank_count;
        const auto previous = bank_owners_[bank];
        require(!previous || states_[*previous].readers.has_value(),
                "prefill copy groups bank still has unordered readers");
        const auto begin = reinterpret_cast<std::uintptr_t>(slots_);
        const auto end = begin + bytes_;
        // Reject every malformed span before the first command is accepted.
        // Readability/source kind is a caller precondition, as for queue.memcpy.
        for (const auto source : sources) {
            const auto address = reinterpret_cast<std::uintptr_t>(source.data());
            require(source.data() && source.size() == layout_.expert_bytes &&
                    source.size() <= std::numeric_limits<std::uintptr_t>::max() - address &&
                    (address + source.size() <= begin || address >= end),
                    "prefill copy groups source span/alias");
        }
        auto& queue = separate_ ? *separate_ : runtime_->compute();
        try {
            std::optional<sycl::event> last;
            for (std::size_t i = 0; i < sources.size(); ++i) {
                runtime_->check(); // Retain the existing per-payload error check.
                auto* destination = slots_ +
                    (bank * layout_.group_width + i) * layout_.expert_bytes;
                // The first copy depends only on this bank's already-submitted
                // readers. In-order copies make the last event cover the group;
                // no additional copy-queue barrier is needed.
                last = i == 0 && previous
                    ? queue.memcpy(destination, sources[i].data(), layout_.expert_bytes,
                                   *states_[*previous].readers)
                    : queue.memcpy(destination, sources[i].data(), layout_.expert_bytes);
                sycl_backend::command_profile::observe(&queue, *last);
            }
            states_.push_back(State{bank, sources.size(), *last, {}, {}});
            bank_owners_[bank] = states_.size() - 1;
        } catch (...) {
            // A later error check/copy can throw after earlier copies were
            // accepted. No ticket is published; terminal drain still owns both
            // queues before any borrowed source or destination can be released.
            poisoned_ = true;
            throw;
        }
        return Ready(identity_, states_.size() - 1);
    }

    void prefill_wait_ready(Ready ticket) {
        auto& state = current(ticket);
        require(ticket.index_ == next_consumer_ && !state.wait && !state.readers,
                "prefill copy groups consumption order/readiness");
        try {
            runtime_->check();
            state.wait = runtime_->compute().ext_oneapi_submit_barrier({state.copy});
        } catch (...) { poisoned_ = true; throw; }
    }

    std::span<const std::byte> prefill_expert(Ready ticket, std::size_t slot) {
        const auto& state = current(ticket);
        require(ticket.index_ == next_consumer_ && state.wait && !state.readers &&
                slot < state.count, "prefill copy groups expert outside active readers");
        return {slots_ + (state.bank * layout_.group_width + slot) * layout_.expert_bytes,
                layout_.expert_bytes};
    }

    void prefill_record_readers(Ready ticket) {
        auto& state = current(ticket);
        require(ticket.index_ == next_consumer_ && state.wait && !state.readers,
                "prefill copy groups readers/readiness order");
        try {
            runtime_->check();
            // Caller has submitted EVERY quantized reader for this group.
            // Separate GU/Down weights, activations and accumulators remain
            // owned and ordered by their own compute consumers.
            state.readers = runtime_->compute().ext_oneapi_submit_barrier();
            ++next_consumer_;
        } catch (...) { poisoned_ = true; throw; }
    }

    void drain() {
        sealed_ = true;
        if (drained_) return;
        std::exception_ptr error;
        try { runtime_->wait(); } catch (...) { error = std::current_exception(); }
        try { if (separate_) separate_->wait_and_throw(); }
        catch (...) { if (!error) error = std::current_exception(); }
        try { runtime_->check(); }
        catch (...) { if (!error) error = std::current_exception(); }
        if (error) { poisoned_ = true; std::rethrow_exception(error); }
        drained_ = true;
    }

private:
    struct State {
        std::size_t bank, count;
        sycl::event copy;
        std::optional<sycl::event> wait, readers;
    };
    std::shared_ptr<sycl_backend::Runtime> runtime_;
    std::byte* const slots_;
    const Layout layout_;
    const std::size_t bytes_;
    const std::uint64_t identity_;
    const std::size_t capacity_;
    std::optional<sycl::queue> separate_;
    std::vector<State> states_;
    std::vector<std::optional<std::size_t>> bank_owners_;
    std::size_t next_consumer_ = 0;
    bool poisoned_ = false, sealed_ = false, drained_ = false;
    inline static std::atomic<std::uint64_t> next_identity_{1};

    static void require(bool value, const char* message) {
        if (!value) throw std::invalid_argument(message);
    }
    static std::size_t checked_bytes(Layout layout) {
        require(layout.expert_bytes && layout.group_width && layout.group_width <= 512 &&
                layout.bank_count && layout.bank_count <= 4096 / layout.group_width,
                "prefill copy groups layout");
        const auto slots = layout.bank_count * layout.group_width;
        require(layout.expert_bytes <= std::numeric_limits<std::size_t>::max() / slots,
                "prefill copy groups byte overflow");
        return slots * layout.expert_bytes;
    }
    State& current(Ready ticket) {
        require(!sealed_ && !poisoned_ && ticket.owner_ == identity_ &&
                ticket.index_ < states_.size(), "prefill copy groups stale/foreign readiness");
        auto& state = states_[ticket.index_];
        require(bank_owners_[state.bank] == ticket.index_,
                "prefill copy groups bank was recycled");
        return state;
    }
};
} // namespace strata::prefill::detail
