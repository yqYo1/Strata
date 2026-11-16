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

    static void need(bool ok, const char* message) {
        if (!ok) throw std::invalid_argument(message);
    }
    static void ze(ze_result_t code) {
        if (code != ZE_RESULT_SUCCESS)
            throw std::runtime_error("prefill host mapping LevelZero result=" + std::to_string(unsigned(code)));
    }
public:
    enum class PropertyContract { StandardImported, PinnedNeoHost };
    struct PropertyProof {
        ze_memory_type_t type = ZE_MEMORY_TYPE_UNKNOWN;
        std::uint64_t allocation_id = 0;
    };
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

    PrefillMappedHostSource(PrefillUploadEventLane& lane, void* heap, std::size_t bytes,
                           PropertyContract contract = PropertyContract::StandardImported)
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
        need(contract == PropertyContract::StandardImported || contract == PropertyContract::PinnedNeoHost,
             "prefill host mapping explicit property contract");
        if (contract == PropertyContract::PinnedNeoHost) {
            // Experimental opt-in only. Main must also pin the exact provider
            // image/environment to the37017156e native-property observation.
            // No generic conformance or production qualification is implied.
            ze_api_version_t version{}; ze(zeDriverGetApiVersion(driver, &version));
            ze_device_properties_t device{}; device.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze(zeDeviceGetProperties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                lane.compute_.get_device()), &device));
            need(version == ZE_MAKE_VERSION(1, 17) && device.vendorId == 0x8086 && device.deviceId == 0xe20c,
                 "prefill pinned NEO property contract requires the qualified API1.17 B570");
        }
        const std::lock_guard lock(PrefillUploadEventLane::source_mutex_);
        auto& active = PrefillUploadEventLane::imported_sources_;
        auto next = active.lower_bound(start);
        need(next == active.end() || start + bytes <= next->first,
             "prefill host mapping overlaps an existing import");
        if (next != active.begin()) {
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
            const auto expected = contract == PropertyContract::StandardImported ?
                ZE_MEMORY_TYPE_HOST_IMPORTED : ZE_MEMORY_TYPE_HOST;
            need(props.type == expected && (contract != PropertyContract::PinnedNeoHost || props.id != 0),
                 "prefill host mapping allocation type/id");
            void* base = nullptr; std::size_t extent = 0;
            ze(zeMemGetAddressRange(context_, mapped_, &base, &extent));
            need(base == mapped_ && extent >= bytes, "prefill host mapping allocation extent");
            need(active.emplace(start, PrefillUploadEventLane::SourceRange{bytes, this}).second,
                 "prefill host mapping duplicate owner");
            lease_->allocation_type = props.type; lease_->allocation_id = props.id;
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
            const std::lock_guard lock(PrefillUploadEventLane::source_mutex_);
            auto& active = PrefillUploadEventLane::imported_sources_;
            const auto key = reinterpret_cast<std::uintptr_t>(mapped_);
            const auto found = active.find(key);
            if (!mapped_ || found == active.end() || found->second.owner != this)
                sycl_backend::cleanup_fail_stop("prefill host mapping retirement owner");
            if (zeMemFree(context_, mapped_) != ZE_RESULT_SUCCESS)
                sycl_backend::cleanup_fail_stop("prefill host mapping unimport");
            active.erase(found); mapped_ = nullptr;
        }, "prefill mapped host source retirement");
    }

    PropertyProof property_proof() const {
        need(mapped_ && lease_->live.load(std::memory_order_acquire), "prefill host mapping retired proof");
        return {lease_->allocation_type, lease_->allocation_id};
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
