#pragma once

// Authored for Strata under the repository LICENSE. Hold the prefix fixture
// and host USM payload fixed when comparing two native queue groups.
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <time.h>
#include <variant>
#include <vector>

namespace strata::bench {
namespace fullfit_overlap {

inline void require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}

inline void check_ze(ze_result_t result, const char *operation) {
    if (result != ZE_RESULT_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with Level Zero status " +
                                 std::to_string(static_cast<unsigned int>(result)));
}

inline constexpr auto backend = sycl::backend::ext_oneapi_level_zero;

inline ze_command_list_handle_t immediate_list(sycl::queue &queue) {
    require(queue.get_backend() == backend &&
            queue.has_property<sycl::property::queue::in_order>(),
            "fixture queue must be in-order Level Zero");
    auto handle = sycl::get_native<backend>(queue);
    require(std::holds_alternative<ze_command_list_handle_t>(handle),
            "fixture queue must return an immediate list");
    return std::get<ze_command_list_handle_t>(handle);
}

inline ze_command_queue_desc_t list_config(ze_command_list_handle_t list,
                                           ze_context_handle_t context, ze_device_handle_t device,
                                           uint32_t expected_ordinal) {
    ze_command_queue_desc_t config{};
    config.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    ze_context_handle_t actual_context{};
    ze_device_handle_t actual_device{};
    check_ze(zeCommandListGetContextHandle(list, &actual_context), "read list context");
    check_ze(zeCommandListGetDeviceHandle(list, &actual_device), "read list device");
    require(actual_context == context && actual_device == device, "list context/device differs");
    check_ze(zeCommandListGetOrdinal(list, &config.ordinal), "read list ordinal");
    check_ze(zeCommandListImmediateGetIndex(list, &config.index), "read list index");
    check_ze(zeCommandListImmediateGetMode(list, &config.mode), "read list mode");
    check_ze(zeCommandListImmediateGetFlags(list, &config.flags), "read list flags");
    check_ze(zeCommandListImmediateGetPriority(list, &config.priority), "read list priority");
    require(config.ordinal == expected_ordinal && config.index == 0 &&
            config.mode == ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS &&
            (config.flags & ZE_COMMAND_QUEUE_FLAG_IN_ORDER) &&
            !(config.flags & ZE_COMMAND_QUEUE_FLAG_COPY_OFFLOAD_HINT),
            "fixture requires the requested asynchronous ordinal, index 0 with copy offload disabled");
    return config;
}

class NativeCopyList {
public:
    NativeCopyList(ze_context_handle_t context, ze_device_handle_t device,
                   const ze_command_queue_desc_t &reference) {
        check_ze(zeCommandListCreateImmediate(context, device, &reference, &list_), "create copy list");
        require(list_ != nullptr, "Level Zero returned no copy list");
        try {
            const auto actual = list_config(list_, context, device, reference.ordinal);
            require(actual.ordinal == reference.ordinal && actual.index == reference.index &&
                    actual.flags == reference.flags && actual.mode == reference.mode &&
                    actual.priority == reference.priority, "native copy list differs from its descriptor");
        } catch (...) {
            close(); // No accepted command is possible during construction.
            throw;
        }
    }
    NativeCopyList(const NativeCopyList &) = delete;
    NativeCopyList &operator=(const NativeCopyList &) = delete;
    ~NativeCopyList() noexcept { close(); }

    void append(void *destination, const void *source, size_t bytes) {
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
        sycl_backend::checked_cleanup([&] { wait(); }, [&] {
            check_ze(zeCommandListDestroy(list_), "destroy copy list");
        }, "native copy list retirement");
    }
    ze_command_list_handle_t list_{};
    bool pending_ = false;
};

template<class WaitCopy>
struct CopyDrain {
    sycl_backend::Runtime &runtime;
    WaitCopy &wait_copy;
    ~CopyDrain() noexcept {
        sycl_backend::checked_cleanup([&] {
            // Runtime::wait does not own a queue returned by make_queue.
            std::exception_ptr error;
            try {
                wait_copy();
            } catch (...) {
                error = std::current_exception();
            }
            runtime.wait();
            if (error) std::rethrow_exception(error);
        }, [] {}, "attention overlap fixture drain");
    }
};

inline uint8_t payload_byte(size_t index) {
    return uint8_t(1 + (index ^ (index >> 16)) % 251);
}

inline uint64_t raw_clock_ns() {
    timespec ts{};
    require(clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0, "read observer clock");
    return uint64_t(ts.tv_sec) * 1000000000 + uint64_t(ts.tv_nsec);
}

enum class Arm { compute_only, copy_only, serial_separate, deferred_wait };
inline constexpr std::array arm_names{"compute_only", "copy_only", "serial_separate", "deferred_wait"};
inline constexpr std::array arms{Arm::compute_only, Arm::copy_only, Arm::serial_separate, Arm::deferred_wait};

template<class ResetAttention, class SubmitAttention, class CheckAttention>
int compare(const std::shared_ptr<sycl_backend::Runtime> &runtime, bool smoke, int process,
            ResetAttention reset_attention, SubmitAttention submit_attention,
            CheckAttention check_attention) {
    const size_t experts = smoke ? 4 : 512;
    const size_t stride = smoke ? 256 : 1971200;
    const size_t payload_bytes = experts * stride;
    constexpr size_t guard_bytes = 256;
    constexpr uint8_t guard = 0xa7;
    const size_t bank_bytes = payload_bytes + 2 * guard_bytes;
    const char *clock_setting = std::getenv("STRATA_FULLFIT_OVERLAP_OBSERVE_CLOCK");
    const bool observe_clock = clock_setting && std::string_view(clock_setting) == "1";
    const char *copy_setting = std::getenv("STRATA_FULLFIT_OVERLAP_COPY_ORDINAL");
    const std::string_view copy_ordinal = copy_setting ? copy_setting : "";
    require(copy_ordinal == "0" || copy_ordinal == "1", "invalid copy ordinal");
    require(bank_bytes <= runtime->device().get_info<sycl::info::device::max_mem_alloc_size>(),
            "full-fit fixture exceeds maximum single device allocation");

    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    const auto source_begin = Clock::now();
    sycl_backend::Allocation host_source(runtime, payload_bytes, sycl_backend::MemoryKind::Host);
    auto *source = host_source.as<uint8_t>();
    const auto source_allocated = Clock::now();
    for (size_t i = 0; i < payload_bytes; ++i) source[i] = payload_byte(i);
    const auto source_initialized = Clock::now();
    require(sycl::get_pointer_type(source, runtime->context()) == sycl::usm::alloc::host &&
            sycl::get_pointer_type(source + payload_bytes - 1, runtime->context()) == sycl::usm::alloc::host,
            "copy source type or extent differs");
    std::cout << "fullfit_attention_overlap source=host_usm"
              << " bytes=" << payload_bytes
              << " alloc_ms=" << std::setprecision(12) << Milliseconds(source_allocated - source_begin).count()
              << " fill_ms=" << Milliseconds(source_initialized - source_allocated).count()
              << " type=PASS\n";
    std::vector<uint8_t> observed(bank_bytes);
    sycl_backend::Allocation bank(runtime, bank_bytes, sycl_backend::MemoryKind::Device);
    auto &compute = runtime->compute();
    auto copy = runtime->make_queue(true);
    auto *base = bank.as<uint8_t>();
    auto *destination = base + guard_bytes;
    sycl_backend::validate_device_span(base, bank_bytes, compute);
    sycl_backend::validate_device_span(base, bank_bytes, copy);
    sycl_backend::validate_spans({{base, bank_bytes}}, {{source, payload_bytes}}, 1);
    sycl_backend::validate_host_span(source, payload_bytes, copy);
    require(copy.get_context() == compute.get_context() && copy.get_device() == compute.get_device(),
            "unexpected copy queue device or context");

    const auto device = sycl::get_native<backend>(compute.get_device());
    const auto context = sycl::get_native<backend>(compute.get_context());
    const auto compute_list = immediate_list(compute);
    const auto copy_list = immediate_list(copy);
    list_config(compute_list, context, device, 0);
    auto copy_config = list_config(copy_list, context, device, 0);
    require(compute_list != copy_list, "compute and copy must use distinct command lists");
    uint32_t groups_count = 0;
    check_ze(zeDeviceGetCommandQueueGroupProperties(device, &groups_count, nullptr), "count queue groups");
    require(groups_count >= 2 && groups_count < 64, "unexpected queue group count");
    std::vector<ze_command_queue_group_properties_t> groups(groups_count);
    for (auto &group : groups) group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    check_ze(zeDeviceGetCommandQueueGroupProperties(device, &groups_count, groups.data()), "read queue groups");
    constexpr auto compute_copy = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE |
                                  ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
    require(groups_count >= 2 && (groups[0].flags & compute_copy) == compute_copy &&
            groups[0].numQueues == 1 && groups[1].flags == ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY &&
            groups[1].numQueues == 1, "fixture requires single compute/copy and copy-only queue groups");
    copy_config.ordinal = copy_ordinal == "1" ? 1 : 0;
    ze_memory_allocation_properties_t memory{};
    memory.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    check_ze(zeMemGetAllocProperties(context, source, &memory, nullptr), "check native copy source");
    require(memory.type == ZE_MEMORY_TYPE_HOST, "native source must be host USM");
    ze_device_handle_t bank_device{};
    check_ze(zeMemGetAllocProperties(context, base, &memory, &bank_device), "check native destination");
    require(memory.type == ZE_MEMORY_TYPE_DEVICE && bank_device == device,
            "native destination must be local device USM");
    NativeCopyList native(context, device, copy_config);
    std::cout << "fullfit_attention_overlap copy_api=native"
              << " ordinal=" << copy_config.ordinal << " index=" << copy_config.index
              << " mode=" << copy_config.mode << " flags=" << copy_config.flags
              << " priority=" << copy_config.priority << " domain=PASS\n";
    // This guard dies before either allocation and before either transfer list.
    auto wait_copy = [&] { native.wait(); };
    CopyDrain pending{*runtime, wait_copy};

    auto reset_bank = [&] {
        compute.memset(base, guard, bank_bytes);
        compute.memset(destination, 0, payload_bytes);
        compute.wait_and_throw();
        runtime->check();
    };
    auto submit_copies = [&](size_t count) {
        for (size_t expert = 0; expert < count; ++expert) {
            const size_t offset = expert * stride;
            native.append(destination + offset, source + offset, stride);
        }
    };
    auto verify_bank = [&](size_t copied_experts) {
        compute.memcpy(observed.data(), base, bank_bytes).wait_and_throw();
        runtime->check();
        const auto first = observed.begin() + guard_bytes;
        const auto copied = first + copied_experts * stride;
        const auto last = first + payload_bytes;
        require(std::all_of(observed.begin(), first, [](uint8_t value) { return value == guard; }) &&
                std::all_of(last, observed.end(), [](uint8_t value) { return value == guard; }) &&
                std::equal(first, copied, source) &&
                std::all_of(copied, last, [](uint8_t value) { return value == 0; }),
                "expert payload or guard mismatch");
    };

    if (smoke) {
        struct Cancelled {};
        reset_bank();
        try {
            CopyDrain cancel{*runtime, wait_copy};
            submit_copies(1);
            throw Cancelled{};
        } catch (const Cancelled &) {
            verify_bank(1);
        }
        std::cout << "fullfit_attention_overlap cancel_after_copy=PASS\n";
    }

    using Seconds = std::chrono::duration<double>;
    Seconds clock_pair_max{};
    for (int i = 0; i < 128; ++i) {
        const auto first = Clock::now();
        clock_pair_max = std::max(clock_pair_max, Seconds(Clock::now() - first));
    }
    std::cout << "fullfit_attention_overlap clock_pair_max_s=" << std::setprecision(12)
              << clock_pair_max.count() << '\n';
    const int repeats = smoke ? 1 : 5;
    for (int round = 0; round < repeats; ++round) {
        for (size_t index = 0; index < arms.size(); ++index) {
            const size_t arm_index = (index + round + 2 * process) % arms.size();
            const Arm arm = arms[arm_index];
            const bool copies = arm != Arm::compute_only;
            const bool attention = arm != Arm::copy_only;
            reset_bank();
            reset_attention();
            const uint64_t observer_start = observe_clock ? raw_clock_ns() : 0;
            const auto start = Clock::now();
            if (copies) {
                submit_copies(experts);
                if (arm != Arm::deferred_wait) {
                    wait_copy();
                    runtime->check();
                }
            }
            if (attention) submit_attention();
            if (arm == Arm::deferred_wait) wait_copy();
            if (attention) compute.wait_and_throw();
            runtime->check();
            const std::chrono::duration<double, std::milli> elapsed =
                Clock::now() - start;
            const uint64_t observer_end = observe_clock ? raw_clock_ns() : 0;

            verify_bank(copies ? experts : 0);
            check_attention(attention);
            std::cout << "fullfit_attention_overlap process=" << process << " round=" << round
                      << " order=" << index << " arm=" << arm_names[arm_index]
                      << " copies=" << (copies ? experts : 0)
                      << " requested_bytes=" << (copies ? payload_bytes : 0)
                      << " attention_calls=" << int(attention)
                      << " milliseconds=" << std::setprecision(12) << elapsed.count()
                      << " raw_start=" << observer_start << " raw_end=" << observer_end
                      << " payload=PASS guards=PASS output=PASS\n";
        }
    }
    for (size_t i = 0; i < payload_bytes; ++i)
        require(source[i] == payload_byte(i), "host copy source changed");
    return repeats * int(arms.size());
}

} // namespace fullfit_overlap
} // namespace strata::bench
