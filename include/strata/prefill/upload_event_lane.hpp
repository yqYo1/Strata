#pragma once
// Main-authored protocol experiment; repository LICENSE. API design reference:
// Intel SYCL Level Zero interop specification and installed Level Zero1.17.
// No external implementation. Not selected by the production engine.
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "strata/sycl/checked_cleanup.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace strata::prefill::detail {

class PrefillMappedHostSource;

// The caller retains every source/destination and all GPU readers through
// drain(). Tokens are local to this nonmovable lane and never recycled.
// Reader fences are minted on this lane's compute queue, not imported from
// arbitrary contexts. A scratch overwrite must name its last reader fence.
class PrefillUploadEventLane {
public:
    struct Region { void* data; std::size_t bytes; };
    struct Source {
        const void* data;
        std::size_t bytes;
        Source(const void* pointer, std::size_t size) : data(pointer), bytes(size) {}
    private:
        struct Imported {
            ze_context_handle_t context = nullptr;
            std::uint64_t owner = 0;
            std::uintptr_t base = 0;
            std::size_t bytes = 0;
            ze_memory_type_t allocation_type = ZE_MEMORY_TYPE_UNKNOWN;
            std::uint64_t allocation_id = 0;
            std::atomic<bool> live{false};
        };
        // A view keeps only this tombstone alive,not the mapping or heap owner.
        // Reimporting the same address creates a different live state.
        std::shared_ptr<const Imported> imported_;
        friend class PrefillMappedHostSource;
        friend class PrefillUploadEventLane;
    };
    struct Ready { std::uint64_t owner; std::size_t index; };
    struct Readers { std::uint64_t owner; std::size_t index; };

    PrefillUploadEventLane(sycl::queue compute, std::function<void()> check,
                          std::size_t capacity = 1024)
        : compute_(std::move(compute)), check_(std::move(check)),
          identity_(next_identity_.fetch_add(1)), capacity_(capacity) {
        need(bool(check_) && identity_ && capacity_ > 0 && capacity_ <= 1024,
             "prefill upload lane owner/capacity");
        need(compute_.get_backend() == sycl::backend::ext_oneapi_level_zero &&
             compute_.has_property<sycl::property::queue::in_order>(),
             "prefill upload lane requires in-order LevelZero compute");
        ready_.reserve(capacity_); readers_.reserve(capacity_);
        waits_.reserve(2 * capacity_);
        auto device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(compute_.get_device());
        const auto context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(compute_.get_context());
        auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(compute_);
        need(std::holds_alternative<ze_command_list_handle_t>(native), "prefill upload compute must be immediate");
        std::uint32_t ordinal = ~0u, index = ~0u, count = 0;
        ze(zeCommandListGetOrdinal(std::get<ze_command_list_handle_t>(native), &ordinal));
        ze(zeCommandListImmediateGetIndex(std::get<ze_command_list_handle_t>(native), &index));
        need(ordinal == 0 && index == 0, "prefill upload compute ordinal/index");
        ze(zeDeviceGetCommandQueueGroupProperties(device, &count, nullptr));
        need(count >= 2 && count < 64, "prefill upload queue groups");
        std::vector<ze_command_queue_group_properties_t> groups(count);
        for (auto& group : groups) group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        ze(zeDeviceGetCommandQueueGroupProperties(device, &count, groups.data()));
        need(groups[1].flags == ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY && groups[1].numQueues == 1,
             "prefill upload copy-only ordinal1/index0 required");
        try {
            ze_command_queue_desc_t desc{};
            desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
            desc.ordinal = 1;
            desc.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER | ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
            desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
            ze(zeCommandListCreateImmediate(context, device, &desc, &copy_));
            ze(zeCommandListGetOrdinal(copy_, &ordinal));
            ze(zeCommandListImmediateGetIndex(copy_, &index));
            need(ordinal == 1 && index == 0, "prefill upload actual copy ordinal/index");
            ze_event_pool_desc_t pool{};
            pool.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
            pool.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
            pool.count = std::uint32_t(capacity_);
            ze(zeEventPoolCreate(context, &pool, 1, &device, &pool_));
        } catch (...) { release(); throw; } // No command has been appended.
    }
    PrefillUploadEventLane(const PrefillUploadEventLane&) = delete;
    PrefillUploadEventLane& operator=(const PrefillUploadEventLane&) = delete;
    ~PrefillUploadEventLane() noexcept {
        sycl_backend::checked_cleanup([&] { drain(); }, [&] { release(); },
                                     "prefill upload event lane retirement");
    }

    Ready copy_after(Region dst, Source src, std::size_t bytes,
                     std::optional<Readers> after = {}) {
        need(!poisoned_ && dst.data && src.data && bytes && bytes <= dst.bytes && bytes <= src.bytes &&
             ready_.size() < capacity_, "prefill upload copy range/capacity");
        need(sycl::get_pointer_type(dst.data, compute_.get_context()) == sycl::usm::alloc::device,
             "prefill upload requires same-context device destination");
        const auto context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(compute_.get_context());
        // On the explicit pinned NEO property contract an imported heap reports
        // HOST,just like host USM. Query and reject active mapped ranges under
        // the same registry lock used for import/unimport; a bare Source must
        // not borrow a mapping while bypassing its lane/lifetime proof.
        std::unique_lock source_guard(source_mutex_, std::defer_lock);
        if (!src.imported_) source_guard.lock();
        if (src.imported_) {
            const auto& lease = *src.imported_;
            const auto pointer = reinterpret_cast<std::uintptr_t>(src.data);
            need(lease.live.load(std::memory_order_acquire) && lease.context == context &&
                 lease.owner == identity_ && pointer >= lease.base && pointer - lease.base <= lease.bytes &&
                 src.bytes <= lease.bytes - (pointer - lease.base),
                 "prefill upload stale/foreign imported-source lease/range");
        }
        ze_memory_allocation_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
        ze(zeMemGetAllocProperties(context, src.data, &props, nullptr));
        if (src.imported_) {
            need(props.type == src.imported_->allocation_type && props.id == src.imported_->allocation_id,
                 "prefill upload source import identity no longer live");
            void* base = nullptr; std::size_t extent = 0;
            ze(zeMemGetAddressRange(context, src.data, &base, &extent));
            const auto pointer = reinterpret_cast<std::uintptr_t>(src.data);
            const auto start = reinterpret_cast<std::uintptr_t>(base);
            need(pointer >= start && pointer - start <= extent && src.bytes <= extent - (pointer - start),
                 "prefill upload imported-source range");
        } else {
            const auto pointer = reinterpret_cast<std::uintptr_t>(src.data);
            need(bytes <= std::numeric_limits<std::uintptr_t>::max() - pointer,
                 "prefill upload source span overflow");
            const auto next = imported_sources_.lower_bound(pointer);
            need(next == imported_sources_.end() || pointer + bytes <= next->first,
                 "prefill upload mapped source requires its owned lease");
            if (next != imported_sources_.begin()) {
                const auto prior = std::prev(next);
                need(prior->first + prior->second.bytes <= pointer,
                     "prefill upload mapped source requires its owned lease");
            }
            need(props.type == ZE_MEMORY_TYPE_HOST &&
                 sycl::get_pointer_type(src.data, compute_.get_context()) == sycl::usm::alloc::host,
                 "prefill upload requires same-context host USM or a checked imported-source lease");
        }
        if (source_guard.owns_lock()) source_guard.unlock();
        ze_event_handle_t dependency = nullptr;
        if (after) { validate(*after); dependency = readers_[after->index].native; }
        ReadyState state;
        ze_event_desc_t desc{};
        desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
        desc.index = std::uint32_t(ready_.size());
        desc.signal = desc.wait = ZE_EVENT_SCOPE_FLAG_HOST | ZE_EVENT_SCOPE_FLAG_DEVICE;
        ze(zeEventCreate(pool_, &desc, &state.native));
        ready_.push_back(std::move(state)); // Reserved before any native work.
        const Ready ticket{identity_, ready_.size() - 1};
        try {
            // Publish only accepted transfer + successfully imported readiness.
            // A failed/ambiguous append poisons this lane; checked drain owns it.
            ze(zeCommandListAppendMemoryCopy(copy_, dst.data, src.data, bytes,
                ready_.back().native, dependency ? 1 : 0, dependency ? &dependency : nullptr));
            ready_.back().imported = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
                {ready_.back().native, sycl::ext::oneapi::level_zero::ownership::keep}, compute_.get_context());
            need(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*ready_.back().imported) ==
                 ready_.back().native, "prefill upload imported event identity");
        } catch (...) { poisoned_ = true; throw; }
        return ticket;
    }

    void wait_ready(Ready ticket) {
        validate(ticket);
        need(!poisoned_ && waits_.size() < 2 * capacity_, "prefill upload wait capacity/poison");
        try { waits_.push_back(compute_.ext_oneapi_submit_barrier({*ready_[ticket.index].imported})); }
        catch (...) { poisoned_ = true; throw; }
    }

    Readers readers_done() {
        need(!poisoned_ && readers_.size() < capacity_, "prefill upload reader capacity/poison");
        ReaderState state;
        try {
            state.event = compute_.ext_oneapi_submit_barrier();
            state.native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(state.event);
            need(state.native != nullptr, "prefill upload native reader event");
            readers_.push_back(std::move(state));
        } catch (...) { poisoned_ = true; throw; }
        return {identity_, readers_.size() - 1};
    }

    bool ready(Ready ticket) const { validate(ticket); return status(ready_[ticket.index].native); }
    bool ready(Readers ticket) const { validate(ticket); return status(readers_[ticket.index].native); }
    void drain() {
        // The fixture's external host gate must release BEFORE this call.
        // No owner may be released after an uncertain/failed queue completion.
        if (copy_ && zeCommandListHostSynchronize(copy_, 5000000000ull) != ZE_RESULT_SUCCESS)
            sycl_backend::cleanup_fail_stop("prefill upload native drain failed/timeout");
        compute_.wait_and_throw(); check_();
    }
private:
    friend class PrefillMappedHostSource;
    struct SourceRange { std::size_t bytes; const PrefillMappedHostSource* owner; };
    inline static std::mutex source_mutex_;
    inline static std::map<std::uintptr_t, SourceRange> imported_sources_;
    struct ReadyState { ze_event_handle_t native = nullptr; std::optional<sycl::event> imported; };
    struct ReaderState { sycl::event event; ze_event_handle_t native = nullptr; };
    inline static std::atomic<std::uint64_t> next_identity_{1};
    sycl::queue compute_;
    std::function<void()> check_;
    const std::uint64_t identity_;
    const std::size_t capacity_;
    ze_command_list_handle_t copy_ = nullptr;
    ze_event_pool_handle_t pool_ = nullptr;
    std::vector<ReadyState> ready_;
    std::vector<ReaderState> readers_;
    std::vector<sycl::event> waits_;
    bool poisoned_ = false;
    static void need(bool value, const char* message) {
        if (!value) throw std::invalid_argument(message);
    }
    static void ze(ze_result_t value) {
        if (value != ZE_RESULT_SUCCESS) throw std::runtime_error("prefill upload LevelZero result=" +
                                                                std::to_string(unsigned(value)));
    }
    static bool status(ze_event_handle_t event) {
        const auto result = zeEventQueryStatus(event);
        if (result == ZE_RESULT_SUCCESS) return true;
        if (result == ZE_RESULT_NOT_READY) return false;
        ze(result); return false;
    }
    void validate(Ready ticket) const {
        need(ticket.owner == identity_ && ticket.index < ready_.size() &&
             ready_[ticket.index].imported.has_value(), "prefill upload stale/foreign ready token");
    }
    void validate(Readers ticket) const {
        need(ticket.owner == identity_ && ticket.index < readers_.size(),
             "prefill upload stale/foreign reader token");
    }
    void release() noexcept {
        if (copy_ && zeCommandListDestroy(copy_) != ZE_RESULT_SUCCESS)
            sycl_backend::cleanup_fail_stop("prefill upload copy list destruction");
        copy_ = nullptr;
        waits_.clear(); readers_.clear();
        for (auto& state : ready_) state.imported.reset();
        for (auto& state : ready_)
            if (state.native && zeEventDestroy(state.native) != ZE_RESULT_SUCCESS)
                sycl_backend::cleanup_fail_stop("prefill upload event destruction");
        ready_.clear();
        if (pool_ && zeEventPoolDestroy(pool_) != ZE_RESULT_SUCCESS)
            sycl_backend::cleanup_fail_stop("prefill upload pool destruction");
        pool_ = nullptr;
    }
};
} // namespace strata::prefill::detail
