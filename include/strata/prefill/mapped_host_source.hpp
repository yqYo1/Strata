#pragma once
// Main-authored source-admission experiment; repository LICENSE.
// API contract: oneapi-src/level-zero-spec, EXT_ExternalMemMap, API1.16.
// No external implementation or XeStrata reference. No production selection.
#include "strata/prefill/upload_event_lane.hpp"
#include <cstring>
#include <limits>
#include <iterator>
#include <map>
#include <mutex>
#include <unistd.h>

namespace strata::prefill::detail {

// The application retains an entire page-aligned heap allocation through this
// lease, and declares the lease AFTER its lane (so the lease retires first).
// All copies borrowing a mapped view must use this exact lane. Retirement drains
// that lane before unimporting; zeMemFree does not free the original heap storage.
// This owner rejects overlapping imports made through this interface. Other
// components must not independently import any part of the supplied allocation.
class PrefillMappedHostSource {
    PrefillUploadEventLane& lane_;
    ze_context_handle_t context_ = nullptr;
    void* mapped_ = nullptr;
    std::size_t bytes_ = 0;
    std::shared_ptr<PrefillUploadEventLane::Source::Imported> lease_;
    struct Range { std::size_t bytes; const PrefillMappedHostSource* owner; };
    inline static std::mutex mutex_;
    inline static std::map<std::uintptr_t, Range> active_;

    static void need(bool ok, const char* message) {
        if (!ok) throw std::invalid_argument(message);
    }
    static void ze(ze_result_t code) {
        if (code != ZE_RESULT_SUCCESS)
            throw std::runtime_error("prefill host mapping LevelZero result=" + std::to_string(unsigned(code)));
    }
public:
    static bool supported(ze_driver_handle_t driver) {
        ze_api_version_t version{};
        ze(zeDriverGetApiVersion(driver, &version));
        if (version < ZE_MAKE_VERSION(1, 16)) return false; // Read-only flag, not just the1.14extension.
        std::uint32_t count = 0;
        ze(zeDriverGetExtensionProperties(driver, &count, nullptr));
        need(count > 0 && count <= 1024, "prefill host mapping bounded extension list");
        std::vector<ze_driver_extension_properties_t> properties(count);
        ze(zeDriverGetExtensionProperties(driver, &count, properties.data()));
        need(count <= properties.size(), "prefill host mapping extension count grew");
        for (std::uint32_t i = 0; i < count; ++i)
            if (std::strncmp(properties[i].name, ZE_EXTERNAL_MEMORY_MAPPING_EXT_NAME,
                             ZE_MAX_EXTENSION_NAME) == 0)
                return properties[i].version >= ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION_1_0;
        return false;
    }

    PrefillMappedHostSource(PrefillUploadEventLane& lane, void* heap, std::size_t bytes)
        : lane_(lane), context_(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
              lane.compute_.get_context())), bytes_(bytes) {
        const long page = ::sysconf(_SC_PAGESIZE);
        const auto start = reinterpret_cast<std::uintptr_t>(heap);
        need(heap && bytes && page > 0 && start % std::size_t(page) == 0 && bytes % std::size_t(page) == 0 &&
             bytes <= std::numeric_limits<std::uintptr_t>::max() - start,
             "prefill host mapping requires a bounded whole-page heap allocation");
        auto driver = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
            lane.compute_.get_device().get_platform());
        need(supported(driver), "prefill read-only external host mapping unsupported");
        const std::lock_guard lock(mutex_);
        auto next = active_.lower_bound(start);
        need(next == active_.end() || start + bytes <= next->first,
             "prefill host mapping overlaps an existing import");
        if (next != active_.begin()) {
            const auto prior = std::prev(next);
            need(prior->first + prior->second.bytes <= start, "prefill host mapping overlaps an existing import");
        }
        need(sycl::get_pointer_type(heap, lane.compute_.get_context()) == sycl::usm::alloc::unknown,
             "prefill host mapping requires ordinary heap storage");
        lease_ = std::make_shared<PrefillUploadEventLane::Source::Imported>();
        lease_->context = context_; lease_->owner = lane_.identity_;
        lease_->base = start; lease_->bytes = bytes;
        ze_external_memmap_sysmem_ext_desc_t external{};
        external.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC;
        external.pSystemMemory = heap;
        external.size = bytes;
        ze_host_mem_alloc_desc_t host{};
        host.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        host.pNext = &external;
        host.flags = ZE_HOST_MEM_ALLOC_FLAG_MEM_READ_ONLY;
        ze(zeMemAllocHost(context_, &host, bytes, 0, &mapped_));
        try {
            need(mapped_ == heap, "prefill host mapping changed the virtual address");
            ze_memory_allocation_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
            ze(zeMemGetAllocProperties(context_, mapped_, &props, nullptr));
            need(props.type == ZE_MEMORY_TYPE_HOST_IMPORTED, "prefill host mapping allocation type");
            void* base = nullptr; std::size_t extent = 0;
            ze(zeMemGetAddressRange(context_, mapped_, &base, &extent));
            need(base == mapped_ && extent >= bytes, "prefill host mapping allocation extent");
            need(active_.emplace(start, Range{bytes, this}).second, "prefill host mapping duplicate owner");
            lease_->live.store(true, std::memory_order_release);
        } catch (...) {
            if (mapped_ && zeMemFree(context_, mapped_) != ZE_RESULT_SUCCESS)
                sycl_backend::cleanup_fail_stop("prefill rejected host mapping unimport");
            mapped_ = nullptr; throw;
        }
    }
    PrefillMappedHostSource(const PrefillMappedHostSource&) = delete;
    PrefillMappedHostSource& operator=(const PrefillMappedHostSource&) = delete;
    ~PrefillMappedHostSource() noexcept {
        // Main serializes admissions and retirement on each lane. Concurrent
        // copy_after/retirement on the same lane is outside this interface.
        lease_->live.store(false, std::memory_order_release);
        sycl_backend::checked_cleanup([&] { lane_.drain(); }, [&] {
            const std::lock_guard lock(mutex_);
            const auto key = reinterpret_cast<std::uintptr_t>(mapped_);
            const auto found = active_.find(key);
            if (!mapped_ || found == active_.end() || found->second.owner != this)
                sycl_backend::cleanup_fail_stop("prefill host mapping retirement owner");
            if (zeMemFree(context_, mapped_) != ZE_RESULT_SUCCESS)
                sycl_backend::cleanup_fail_stop("prefill host mapping unimport");
            active_.erase(found); mapped_ = nullptr;
        }, "prefill mapped host source retirement");
    }

    PrefillUploadEventLane::Source view(std::size_t offset, std::size_t bytes) const {
        need(mapped_ && bytes && offset <= bytes_ && bytes <= bytes_ - offset,
             "prefill host mapping view bounds");
        PrefillUploadEventLane::Source source{static_cast<const std::byte*>(mapped_) + offset, bytes};
        source.imported_ = lease_;
        return source;
    }
};
} // namespace strata::prefill::detail
