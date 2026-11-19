#pragma once

// Authored for Strata under the repository LICENSE. Hold the prefix fixture
// and native copy queue fixed when comparing sources of actual weights.
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/launch.hpp"
#include "strata/sycl/native_copy_list.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <time.h>
#include <unistd.h>
#include <variant>
#include <vector>

namespace strata::bench {
namespace fullfit_overlap {

inline void require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}

using sycl_backend::native_copy::backend;
using sycl_backend::native_copy::check_ze;
using sycl_backend::native_copy::immediate_list;
using sycl_backend::native_copy::list_config;
using sycl_backend::native_copy::NativeCopyList;
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;
inline constexpr size_t expert_stride = 1971200;

enum class SourceKind { Allocated, Staged, Borrowed };

inline SourceKind source_kind() {
    const char *setting = std::getenv("STRATA_FULLFIT_OVERLAP_SOURCE");
    const std::string_view kind = setting ? setting : "";
    if (kind == "allocated") return SourceKind::Allocated;
    if (kind == "staged") return SourceKind::Staged;
    if (kind == "borrowed") return SourceKind::Borrowed;
    throw std::runtime_error("missing actual layer source");
}

class FullArena {
public:
    FullArena() {
        const auto started = Clock::now();
        const char *shard = std::getenv("STRATA_FULLFIT_WEIGHT_SHARD");
        const char *pack = std::getenv("STRATA_FULLFIT_WEIGHT_PACK");
        require(shard && pack, "missing actual model paths");
        std::string error;
        if (!kernels::cpu::expert_layout_load(pack, 48, 512, error))
            throw std::runtime_error(error);
        const auto &layout = kernels::cpu::expert_layout();
        require(layout.native && layout.n_layers == 48 && layout.n_expert == 512 &&
                layout.total == 50292326400ull && layout.blob_bytes(2) == expert_stride,
                "actual arena geometry differs");
        arena_.set_gguf(shard);
        if (!arena_.open(pack, 48, 512, 6, error))
            throw std::runtime_error(error);
        const auto base = reinterpret_cast<uintptr_t>(arena_.resident_layer(0).data);
        require(arena_.mapped() && arena_.blobs() == 48 * 512 && base &&
                base <= UINTPTR_MAX - layout.total && !arena_.resident_layer(-1).data &&
                !arena_.resident_layer(48).data, "invalid full arena ownership");
        for (int64_t layer = 0; layer < 48; ++layer) {
            const auto view = arena_.resident_layer(layer);
            require(view.data && view.experts == 512 && view.stride == layout.blob_bytes(layer) &&
                    view.bytes == 512 * view.stride && view.bytes % 4096 == 0 &&
                    reinterpret_cast<uintptr_t>(view.data) == base + layout.layer_offset(layer) &&
                    reinterpret_cast<uintptr_t>(view.data) % 4096 == 0 &&
                    arena_.blob(layer, 0) == view.data &&
                    arena_.blob(layer, 511) == view.data + 511 * view.stride &&
                    !arena_.pinned(layer, 0) && !arena_.device_alias(layer, 0),
                    "ordinary expert arena view does not match its owner");
        }
        std::cout << "fullfit_attention_overlap arena=production layers=48 bytes=" << layout.total
                  << " load_ms=" << std::setprecision(12) << Milliseconds(Clock::now() - started).count()
                  << " layout=PASS\n";
    }

    core::ResidentExpertLayer layer(int64_t index, size_t experts = 512) const {
        const auto view = arena_.resident_layer(index);
        const auto prefix = core::ResidentExpertLayer::make(view.data, view.bytes, 0, experts, view.stride);
        require(prefix.data, "invalid actual arena layer prefix");
        return prefix;
    }

private:
    core::ArenaExpertSource arena_;
};

class LayerSource {
public:
    LayerSource(const std::shared_ptr<sycl_backend::Runtime> &runtime,
                core::ResidentExpertLayer layer, SourceKind kind, sycl_backend::Allocation *staging,
                bool full_check = true)
        : context_(sycl::get_native<backend>(runtime->context())), layer_(layer),
          staging_(staging), full_check_(full_check) {
        const long page = sysconf(_SC_PAGESIZE);
        require(page == 4096 && layer_.data && layer_.bytes >= 8192 &&
                reinterpret_cast<uintptr_t>(layer_.data) % page == 0 && layer_.bytes % page == 0,
                "whole arena pages required");
        fingerprint_ = fingerprint();
        const auto preparing = Clock::now();
        if (kind == SourceKind::Allocated) {
            owned_staging_ = sycl_backend::Allocation(runtime, layer_.bytes, sycl_backend::MemoryKind::Host);
            staging_ = &owned_staging_;
        }
        if (kind == SourceKind::Borrowed) {
            admit_borrowed(runtime);
        } else {
            require(staging_ && staging_->size() >= layer_.bytes, "missing host USM staging extent");
            sycl_backend::validate_host_span(staging_->data(), layer_.bytes, runtime->compute());
            std::memcpy(staging_->data(), layer_.data, layer_.bytes);
        }
        const auto prepared = Clock::now();
        const char *name = kind == SourceKind::Allocated ? "allocated" :
                           kind == SourceKind::Staged ? "staged" : "borrowed";
        std::cout << "fullfit_attention_overlap source=" << name << " bytes=" << layer_.bytes
                  << " prepare_ms=" << std::setprecision(12) << Milliseconds(prepared - preparing).count()
                  << " allocation_id=" << allocation_id_ << " type=PASS\n";
    }
    LayerSource(const LayerSource &) = delete;
    LayerSource &operator=(const LayerSource &) = delete;
    ~LayerSource() noexcept { release(); }

    const uint8_t *data() const { return staging_ ? staging_->as<uint8_t>() : layer_.data; }
    const core::ResidentExpertLayer &layer() const { return layer_; }

    // Called only after the fixture's native copy list and GPU readers drain.
    // Scope unwinding destroys those owners before this source.
    void finish() {
        if (staging_) require(std::memcmp(layer_.data, data(), layer_.bytes) == 0, "staged payload changed");
        const bool borrowed = imported_ != nullptr;
        const bool allocated = owned_staging_.data() != nullptr;
        const auto retiring = Clock::now();
        release();
        if (allocated) {
            owned_staging_ = {};
            staging_ = nullptr;
        }
        const double retire_ms = borrowed || allocated ? Milliseconds(Clock::now() - retiring).count() : 0.0;
        if (borrowed) {
            ze_memory_allocation_properties_t memory{};
            memory.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
            check_ze(zeMemGetAllocProperties(context_, layer_.data, &memory, nullptr), "check retired layer");
            require(memory.type == ZE_MEMORY_TYPE_UNKNOWN, "borrowed layer remains imported");
        }
        require(fingerprint_ == fingerprint(), "ordinary arena payload changed");
        std::cout << "fullfit_attention_overlap retire_ms=" << std::setprecision(12)
                  << retire_ms << " source_unchanged=PASS\n";
    }

private:
    uint64_t fingerprint() const {
        if (full_check_) return core::fnv1a64(layer_.data, layer_.bytes);
        return core::fnv1a64(layer_.data + layer_.bytes - 4096, 4096,
                             core::fnv1a64(layer_.data, 4096));
    }

    void admit_borrowed(const std::shared_ptr<sycl_backend::Runtime> &runtime) {
        const auto driver = sycl::get_native<backend>(runtime->device().get_platform());
        const auto device = sycl::get_native<backend>(runtime->device());
        ze_api_version_t version{};
        ze_device_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        check_ze(zeDriverGetApiVersion(driver, &version), "read driver API version");
        check_ze(zeDeviceGetProperties(device, &props), "read device properties");
        require(version == ZE_MAKE_VERSION(1, 17) && props.vendorId == 0x8086 && props.deviceId == 0xe20c &&
                layer_.bytes <= props.maxMemAllocSize &&
                sycl::get_pointer_type(layer_.data, runtime->context()) == sycl::usm::alloc::unknown,
                "bounded B570 API 1.17 ordinary memory required");
        uint32_t count = 0;
        check_ze(zeDriverGetExtensionProperties(driver, &count, nullptr), "count memory extensions");
        require(count > 0 && count <= 1024, "unexpected driver extension count");
        std::vector<ze_driver_extension_properties_t> extensions(count);
        check_ze(zeDriverGetExtensionProperties(driver, &count, extensions.data()), "read memory extensions");
        require(count <= extensions.size() &&
                std::any_of(extensions.begin(), extensions.begin() + count, [](const auto &ext) {
                    return std::strncmp(ext.name, ZE_EXTERNAL_MEMORY_MAPPING_EXT_NAME, ZE_MAX_EXTENSION_NAME) == 0 &&
                           ext.version >= ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION_1_0;
                }), "external memory mapping extension unavailable");
        ze_external_memmap_sysmem_ext_desc_t external{};
        external.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMMAP_SYSMEM_EXT_DESC;
        external.pSystemMemory = const_cast<uint8_t *>(layer_.data);
        external.size = layer_.bytes;
        ze_host_mem_alloc_desc_t host{};
        host.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
        host.pNext = &external;
        host.flags = ZE_HOST_MEM_ALLOC_FLAG_MEM_READ_ONLY;
        check_ze(zeMemAllocHost(context_, &host, layer_.bytes, 0, &imported_), "import read-only arena range");
        try {
            require(imported_ == layer_.data, "import returned a different address");
            ze_memory_allocation_properties_t memory{};
            memory.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
            check_ze(zeMemGetAllocProperties(context_, imported_, &memory, nullptr), "read imported source type");
            void *base = nullptr;
            size_t bytes = 0;
            check_ze(zeMemGetAddressRange(context_, imported_, &base, &bytes), "read imported source extent");
            require(memory.type == ZE_MEMORY_TYPE_HOST && memory.id && base == imported_ && bytes >= layer_.bytes,
                    "pinned NEO source type, lease or extent differs");
            allocation_id_ = memory.id;
        } catch (...) {
            release();
            throw;
        }
    }

    void release() noexcept {
        if (imported_) {
            if (zeMemFree(context_, imported_) != ZE_RESULT_SUCCESS)
                sycl_backend::cleanup_fail_stop("read-only layer unimport failed");
            imported_ = nullptr;
        }
    }

    ze_context_handle_t context_;
    core::ResidentExpertLayer layer_;
    sycl_backend::Allocation owned_staging_;
    sycl_backend::Allocation *staging_;
    bool full_check_;
    void *imported_ = nullptr;
    uint64_t allocation_id_ = 0;
    uint64_t fingerprint_ = 0;
};

inline void source_admission(const std::shared_ptr<sycl_backend::Runtime> &runtime) {
    require(source_kind() == SourceKind::Borrowed, "actual full-arena metadata must use read-only mappings");
    FullArena arena;
    for (int64_t layer = 0; layer < 48; ++layer) {
        std::cout << "fullfit_attention_overlap admission_layer=" << layer << '\n';
        LayerSource source(runtime, arena.layer(layer), SourceKind::Borrowed, nullptr, false);
        source.finish();
    }
}

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
    const size_t stride = expert_stride;
    const size_t payload_bytes = experts * stride;
    constexpr size_t guard_bytes = 256;
    constexpr uint8_t guard = 0xa7;
    const size_t bank_bytes = payload_bytes + 2 * guard_bytes;
    const char *clock_setting = std::getenv("STRATA_FULLFIT_OVERLAP_OBSERVE_CLOCK");
    const bool observe_clock = clock_setting && std::string_view(clock_setting) == "1";
    require(bank_bytes <= runtime->device().get_info<sycl::info::device::max_mem_alloc_size>(),
            "full-fit fixture exceeds maximum single device allocation");

    const auto kind = source_kind();
    FullArena arena;
    sycl_backend::Allocation staging;
    double staging_startup_ms = 0;
    if (kind == SourceKind::Staged) {
        const auto started = Clock::now();
        staging = sycl_backend::Allocation(runtime, payload_bytes, sycl_backend::MemoryKind::Host);
        std::memset(staging.data(), 0, payload_bytes);
        const auto prior_layer = arena.layer(1);
        std::memcpy(staging.data(), prior_layer.data, std::min<uint64_t>(payload_bytes, prior_layer.bytes));
        staging_startup_ms = Milliseconds(Clock::now() - started).count();
    }
    std::cout << "fullfit_attention_overlap staging_startup_ms=" << std::setprecision(12)
              << staging_startup_ms << '\n';
    LayerSource source(runtime, arena.layer(2, experts), kind, kind == SourceKind::Staged ? &staging : nullptr);
    std::vector<uint8_t> observed(bank_bytes);
    sycl_backend::Allocation bank(runtime, bank_bytes, sycl_backend::MemoryKind::Device);
    auto &compute = runtime->compute();
    auto copy = runtime->make_queue(true);
    auto *base = bank.as<uint8_t>();
    auto *destination = base + guard_bytes;
    sycl_backend::validate_device_span(base, bank_bytes, compute);
    sycl_backend::validate_device_span(base, bank_bytes, copy);
    sycl_backend::validate_spans({{base, bank_bytes}}, {{source.data(), payload_bytes}}, 1);
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
    copy_config.ordinal = 1;
    ze_memory_allocation_properties_t memory{};
    memory.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    check_ze(zeMemGetAllocProperties(context, source.data(), &memory, nullptr), "check native copy source");
    require(memory.type == ZE_MEMORY_TYPE_HOST, "native source must be admitted host memory");
    ze_device_handle_t bank_device{};
    check_ze(zeMemGetAllocProperties(context, base, &memory, &bank_device), "check native destination");
    require(memory.type == ZE_MEMORY_TYPE_DEVICE && bank_device == device,
            "native destination must be local device USM");
    NativeCopyList native(context, device, copy_config);
    std::cout << "fullfit_attention_overlap copy_api=native"
              << " ordinal=" << copy_config.ordinal << " index=" << copy_config.index
              << " mode=" << copy_config.mode << " flags=" << copy_config.flags
              << " priority=" << copy_config.priority << " domain=PASS\n";
    // This guard dies before the native copy list and either buffer owner.
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
            native.append(destination + offset, source.data() + offset, stride);
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
                std::equal(first, copied, source.layer().data) &&
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
    wait_copy();
    runtime->wait();
    source.finish();
    return repeats * int(arms.size());
}

} // namespace fullfit_overlap
} // namespace strata::bench
