#pragma once

// Authored for Strata under the repository LICENSE. Use the existing prefix
// fixture and copy wrapper with synthetic sources at the layer-2 stride.
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/compat/cuda_runtime.h"
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
#include <string_view>
#include <time.h>
#include <vector>

namespace strata::bench {
namespace fullfit_overlap {

inline void require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}

struct CopyDrain {
    sycl_backend::Runtime &runtime;
    sycl::queue &copy;
    ~CopyDrain() noexcept {
        sycl_backend::checked_cleanup([&] {
            // Runtime::wait does not own a queue returned by make_queue.
            std::exception_ptr error;
            try {
                copy.wait_and_throw();
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

enum class Arm { compute_only, copy_only, serial_shared, serial_separate, deferred_wait };
inline constexpr std::array arm_names{"compute_only", "copy_only", "serial_shared",
                                       "serial_separate", "deferred_wait"};
inline constexpr std::array arms{Arm::compute_only, Arm::copy_only, Arm::serial_shared,
                                  Arm::serial_separate, Arm::deferred_wait};

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
    const char *source_setting = std::getenv("STRATA_FULLFIT_OVERLAP_SOURCE");
    const std::string_view source_kind = source_setting ? source_setting : "";
    require(source_kind == "ordinary" || source_kind == "host_usm", "invalid copy source kind");
    const bool use_host_usm = source_kind == "host_usm";
    require(bank_bytes <= runtime->device().get_info<sycl::info::device::max_mem_alloc_size>(),
            "full-fit fixture exceeds maximum single device allocation");

    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    const auto source_begin = Clock::now();
    std::vector<uint8_t> ordinary_source(use_host_usm ? 0 : payload_bytes);
    sycl_backend::Allocation host_source;
    if (use_host_usm)
        host_source = sycl_backend::Allocation(runtime, payload_bytes, sycl_backend::MemoryKind::Host);
    auto *source = use_host_usm ? host_source.as<uint8_t>() : ordinary_source.data();
    const auto source_allocated = Clock::now();
    for (size_t i = 0; i < payload_bytes; ++i) source[i] = payload_byte(i);
    const auto source_initialized = Clock::now();
    const auto pointer_type = use_host_usm ? sycl::usm::alloc::host : sycl::usm::alloc::unknown;
    require(sycl::get_pointer_type(source, runtime->context()) == pointer_type &&
            sycl::get_pointer_type(source + payload_bytes - 1, runtime->context()) == pointer_type,
            "copy source type or extent differs");
    std::cout << "fullfit_attention_overlap source=" << source_kind
              << " bytes=" << payload_bytes
              << " alloc_ms=" << std::setprecision(12) << Milliseconds(source_allocated - source_begin).count()
              << " fill_ms=" << Milliseconds(source_initialized - source_allocated).count()
              << " type=PASS\n";
    std::vector<uint8_t> observed(bank_bytes);
    sycl_backend::Allocation bank(runtime, bank_bytes, sycl_backend::MemoryKind::Device);
    auto &compute = runtime->compute();
    auto copy = runtime->make_queue(true);
    CopyDrain pending{*runtime, copy};
    auto *base = bank.as<uint8_t>();
    auto *destination = base + guard_bytes;
    sycl_backend::validate_device_span(base, bank_bytes, compute);
    sycl_backend::validate_device_span(base, bank_bytes, copy);
    sycl_backend::validate_spans({{base, bank_bytes}}, {{source, payload_bytes}}, 1);
    if (use_host_usm) sycl_backend::validate_host_span(source, payload_bytes, copy);
    require(copy.get_context() == compute.get_context() && copy.get_device() == compute.get_device(),
            "unexpected copy queue device or context");

    auto reset_bank = [&] {
        compute.memset(base, guard, bank_bytes);
        compute.memset(destination, 0, payload_bytes);
        compute.wait_and_throw();
        runtime->check();
    };
    auto submit_copies = [&](sycl::queue &queue, size_t count) {
        for (size_t expert = 0; expert < count; ++expert) {
            const size_t offset = expert * stride;
            require(cudaMemcpyAsync(destination + offset, source + offset, stride,
                                    cudaMemcpyHostToDevice, &queue) == cudaSuccess,
                    "expert copy refused");
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
            CopyDrain cancel{*runtime, copy};
            submit_copies(copy, 1);
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
    const int repeats = smoke ? 1 : int(arms.size());
    for (int round = 0; round < repeats; ++round) {
        for (size_t index = 0; index < arms.size(); ++index) {
            const size_t arm_index = (index + round + 2 * process) % arms.size();
            const Arm arm = arms[arm_index];
            const bool copies = arm != Arm::compute_only;
            const bool attention = arm != Arm::copy_only;
            auto &copy_queue = arm == Arm::serial_shared ? compute : copy;
            reset_bank();
            reset_attention();
            const uint64_t observer_start = observe_clock ? raw_clock_ns() : 0;
            const auto start = Clock::now();
            if (copies) {
                submit_copies(copy_queue, experts);
                if (arm != Arm::deferred_wait) {
                    copy_queue.wait_and_throw();
                    runtime->check();
                }
            }
            if (attention) submit_attention();
            if (arm == Arm::deferred_wait) copy_queue.wait_and_throw();
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
