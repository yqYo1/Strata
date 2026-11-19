#pragma once

// Authored for Strata under the repository LICENSE. The allocation owners
// must outlive the copy list and its checked drain.
#include "strata/sycl/checked_cleanup.hpp"
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace strata::sycl_backend::native_copy {

inline constexpr auto backend = sycl::backend::ext_oneapi_level_zero;

inline void check_ze(ze_result_t result, const char* operation) {
    if (result != ZE_RESULT_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with Level Zero status " +
                                 std::to_string(static_cast<unsigned int>(result)));
}

inline ze_command_list_handle_t immediate_list(const sycl::queue& queue) {
    if (queue.get_backend() != backend || !queue.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("copy list requires an in-order Level Zero queue");
    auto handle = sycl::get_native<backend>(queue);
    if (!std::holds_alternative<ze_command_list_handle_t>(handle))
        throw std::invalid_argument("copy list requires immediate Level Zero compute");
    return std::get<ze_command_list_handle_t>(handle);
}

inline ze_command_queue_desc_t list_config(ze_command_list_handle_t list,
        ze_context_handle_t context, ze_device_handle_t device, uint32_t expected_ordinal) {
    ze_command_queue_desc_t config{};
    config.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    ze_context_handle_t actual_context{};
    ze_device_handle_t actual_device{};
    check_ze(zeCommandListGetContextHandle(list, &actual_context), "read list context");
    check_ze(zeCommandListGetDeviceHandle(list, &actual_device), "read list device");
    if (actual_context != context || actual_device != device)
        throw std::invalid_argument("copy list context/device differs");
    check_ze(zeCommandListGetOrdinal(list, &config.ordinal), "read list ordinal");
    check_ze(zeCommandListImmediateGetIndex(list, &config.index), "read list index");
    check_ze(zeCommandListImmediateGetMode(list, &config.mode), "read list mode");
    check_ze(zeCommandListImmediateGetFlags(list, &config.flags), "read list flags");
    check_ze(zeCommandListImmediateGetPriority(list, &config.priority), "read list priority");
    if (config.ordinal != expected_ordinal || config.index != 0 ||
        config.mode != ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS ||
        !(config.flags & ZE_COMMAND_QUEUE_FLAG_IN_ORDER) ||
        (config.flags & ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT))
        throw std::invalid_argument("copy list requires the requested asynchronous in-order queue with copy offload disabled");
    return config;
}

inline ze_command_queue_desc_t copy_config(const sycl::queue& compute, uint32_t ordinal) {
    const auto list = immediate_list(compute);
    const auto context = sycl::get_native<backend>(compute.get_context());
    const auto device = sycl::get_native<backend>(compute.get_device());
    auto config = list_config(list, context, device, 0);
    uint32_t count = 0;
    check_ze(zeDeviceGetCommandQueueGroupProperties(device, &count, nullptr), "count queue groups");
    if (ordinal > 1 || count < 2 || count >= 64)
        throw std::invalid_argument("unexpected Level Zero copy queue groups");
    std::vector<ze_command_queue_group_properties_t> groups(count);
    for (auto& group : groups) group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    check_ze(zeDeviceGetCommandQueueGroupProperties(device, &count, groups.data()), "read queue groups");
    constexpr auto compute_copy = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE |
                                  ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    if (count < 2 || (groups[0].flags & compute_copy) != compute_copy || groups[0].numQueues != 1 ||
        groups[1].flags != ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY || groups[1].numQueues != 1)
        throw std::invalid_argument("copy list requires single compute/copy and copy-only queue groups");
    config.ordinal = ordinal;
    return config;
}

class NativeCopyList {
public:
    NativeCopyList(const sycl::queue& compute, uint32_t ordinal)
        : NativeCopyList(sycl::get_native<backend>(compute.get_context()),
                         sycl::get_native<backend>(compute.get_device()), copy_config(compute, ordinal)) {}

    NativeCopyList(ze_context_handle_t context, ze_device_handle_t device,
                   const ze_command_queue_desc_t& reference) : context_(context), device_(device) {
        check_ze(zeCommandListCreateImmediate(context, device, &reference, &list_), "create copy list");
        if (!list_) throw std::runtime_error("Level Zero returned no copy list");
        try {
            const auto actual = list_config(list_, context, device, reference.ordinal);
            if (actual.index != reference.index || actual.flags != reference.flags ||
                actual.mode != reference.mode || actual.priority != reference.priority)
                throw std::runtime_error("native copy list differs from its descriptor");
        } catch (...) {
            close();
            throw;
        }
    }
    NativeCopyList(const NativeCopyList&) = delete;
    NativeCopyList& operator=(const NativeCopyList&) = delete;
    ~NativeCopyList() noexcept { close(); }

    void validate_endpoints(const sycl::queue& compute, void* destination, const void* source) const {
        if (pending_ || compute.get_backend() != backend ||
            sycl::get_native<backend>(compute.get_context()) != context_ ||
            sycl::get_native<backend>(compute.get_device()) != device_)
            throw std::invalid_argument("copy list belongs to a different domain or has pending work");
        ze_memory_allocation_properties_t properties{};
        properties.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
        check_ze(zeMemGetAllocProperties(context_, source, &properties, nullptr), "check native copy source");
        if (properties.type != ZE_MEMORY_TYPE_HOST)
            throw std::invalid_argument("native copy requires admitted host memory");
        ze_device_handle_t destination_device{};
        check_ze(zeMemGetAllocProperties(context_, destination, &properties, &destination_device),
                 "check native copy destination");
        if (properties.type != ZE_MEMORY_TYPE_DEVICE || destination_device != device_)
            throw std::invalid_argument("native copy requires local device memory");
    }

    void append(void* destination, const void* source, size_t bytes) {
        pending_ = true; // An ambiguous append failure still requires a drain.
        check_ze(zeCommandListAppendMemoryCopy(list_, destination, source, bytes, nullptr, 0, nullptr),
                 "append copy");
    }

    void wait() {
        if (!pending_) return;
        check_ze(zeCommandListHostSynchronize(list_, 5'000'000'000ull), "drain copy list");
        pending_ = false;
    }

private:
    void close() noexcept {
        checked_cleanup([&] { wait(); }, [&] {
            check_ze(zeCommandListDestroy(list_), "destroy copy list");
        }, "native copy list retirement");
    }
    ze_context_handle_t context_;
    ze_device_handle_t device_;
    ze_command_list_handle_t list_{};
    bool pending_ = false;
};

} // namespace strata::sycl_backend::native_copy
