#pragma once
// Main-authored standard-SYCL dependency experiment; repository LICENSE.
// Uses this project's Runtime/checked-cleanup contracts and installed SYCL
// queue barriers. No external implementation; not selected by the engine.
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

// One quantized scratch bank, submitted by one host thread. Sources are
// borrowed ordinary readable host memory (or same-context host USM), never
// passed to a kernel. The caller keeps them immutable and keeps the bank and
// all consumer allocations alive through drain(), including exceptional paths.
// drain() permanently seals the lane; repeated drains are harmless and no
// submission or consumer token can be used afterward.
// Declare the lane after those owners so its destructor drains first.
//
// Q1 changes only compute versus a distinct SYCL queue with matching properties.
// The latter does not establish which physical engine the driver selects.
// O changes only the position of prefill_record_readers(): after both IQ
// readers, or after all math. GU/Down dequantized weights have separate owners;
// their overwrite remains ordered after their own consumers on compute.
class PrefillSyclCopyLane {
public:
    enum class Placement { Compute, Separate };
    class Ready {
        friend class PrefillSyclCopyLane;
        Ready(std::uint64_t owner, std::size_t index) : owner_(owner), index_(index) {}
        std::uint64_t owner_;
        std::size_t index_;
    };

    PrefillSyclCopyLane(std::shared_ptr<sycl_backend::Runtime> runtime,
                       void* bank, std::size_t bytes, Placement placement,
                       std::size_t capacity = 1024)
        : runtime_(std::move(runtime)), bank_(bank), bytes_(bytes),
          identity_(next_identity_.fetch_add(1)), capacity_(capacity) {
        require(runtime_ && identity_ && bank_ && bytes_ && capacity_ && capacity_ <= 4096,
                "prefill SYCL copy lane owner/range/capacity");
        auto& compute = runtime_->compute();
        require(compute.has_property<sycl::property::queue::in_order>(),
                "prefill SYCL copy lane requires in-order compute");
        sycl_backend::validate_device_span(bank_, bytes_, compute);
        states_.reserve(capacity_); // Allocate bookkeeping before any submission.
        require(placement == Placement::Compute || placement == Placement::Separate,
                "prefill SYCL copy lane placement");
        if (placement == Placement::Separate) {
            separate_.emplace(runtime_->make_queue(
                compute.has_property<sycl::property::queue::enable_profiling>()));
            require(separate_->get_context() == compute.get_context() &&
                    separate_->get_device() == compute.get_device() &&
                    separate_->has_property<sycl::property::queue::in_order>() &&
                    separate_->has_property<sycl::property::queue::enable_profiling>() ==
                        compute.has_property<sycl::property::queue::enable_profiling>(),
                    "prefill SYCL copy queue property/context mismatch");
        }
    }
    PrefillSyclCopyLane(const PrefillSyclCopyLane&) = delete;
    PrefillSyclCopyLane& operator=(const PrefillSyclCopyLane&) = delete;
    ~PrefillSyclCopyLane() noexcept {
        sycl_backend::checked_cleanup([&] { drain(); }, [] {},
                                     "prefill SYCL copy lane retirement");
    }

    Ready prefill_copy(std::span<const std::byte> source) {
        require(!sealed_ && !poisoned_ && source.data() && source.size() == bytes_ &&
                bytes_ <= std::numeric_limits<std::uintptr_t>::max() -
                    reinterpret_cast<std::uintptr_t>(source.data()) &&
                states_.size() < capacity_ &&
                (states_.empty() || states_.back().readers.has_value()),
                "prefill SYCL copy source/capacity/previous readers");
        runtime_->check();
        auto& queue = separate_ ? *separate_ : runtime_->compute();
        try {
            // No external events: dependencies can only name an already
            // submitted predecessor from this lane, so no future-event cycle.
            const auto after = states_.empty() ? std::optional<sycl::event>{}
                                               : states_.back().readers;
            auto event = after ? queue.memcpy(bank_, source.data(), bytes_, *after)
                               : queue.memcpy(bank_, source.data(), bytes_);
            states_.push_back(State{event, {}, {}});
            sycl_backend::command_profile::observe(&queue, event);
        } catch (...) { poisoned_ = true; throw; }
        return Ready(identity_, states_.size() - 1);
    }

    void prefill_wait_ready(Ready ticket) {
        auto& state = current(ticket);
        require(!state.wait && !state.readers, "prefill SYCL copy readiness already consumed");
        runtime_->check();
        try { state.wait = runtime_->compute().ext_oneapi_submit_barrier({state.copy}); }
        catch (...) { poisoned_ = true; throw; }
    }

    void prefill_record_readers(Ready ticket) {
        auto& state = current(ticket);
        require(state.wait && !state.readers, "prefill SYCL copy readers/readiness order");
        runtime_->check();
        // Invoke after BOTH quantized-bank IQ readers are submitted. On the
        // in-order queue one marker then covers both. This does not retire
        // their separate dequantized weights, activations or accumulators.
        try { state.readers = runtime_->compute().ext_oneapi_submit_barrier(); }
        catch (...) { poisoned_ = true; throw; }
    }

    void drain() {
        sealed_ = true;
        if (drained_) return;
        // Always attempt both drains. An uncertain completion fails before
        // any borrowed allocation/event/queue owner may be destroyed.
        std::exception_ptr error;
        try { runtime_->wait(); } catch (...) { error = std::current_exception(); }
        try { if (separate_) separate_->wait_and_throw(); }
        catch (...) { if (!error) error = std::current_exception(); }
        runtime_->check();
        if (error) std::rethrow_exception(error);
        drained_ = true;
    }

private:
    struct State {
        sycl::event copy;
        std::optional<sycl::event> wait, readers;
    };
    std::shared_ptr<sycl_backend::Runtime> runtime_;
    void* const bank_;
    const std::size_t bytes_;
    const std::uint64_t identity_;
    const std::size_t capacity_;
    std::optional<sycl::queue> separate_;
    std::vector<State> states_;
    bool poisoned_ = false, sealed_ = false, drained_ = false;
    inline static std::atomic<std::uint64_t> next_identity_{1};

    static void require(bool value, const char* message) {
        if (!value) throw std::invalid_argument(message);
    }
    State& current(Ready ticket) {
        require(!sealed_ && !poisoned_ && ticket.owner_ == identity_ &&
                ticket.index_ < states_.size() && ticket.index_ + 1 == states_.size(),
                "prefill SYCL copy stale/foreign readiness");
        return states_[ticket.index_];
    }
};
} // namespace strata::prefill::detail
