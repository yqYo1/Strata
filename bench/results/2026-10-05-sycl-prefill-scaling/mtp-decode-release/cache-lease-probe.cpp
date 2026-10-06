// Exercise Strata's actual VMM cache through captured-kernel replay. This is
// a storage/graph check, not a model-output or full-context validation.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/expert_cache.hpp"
#include "strata/kernels/cpu/expert.hpp"

#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(int64_t slots, int64_t slot_bytes) {
    namespace graph = sycl::ext::oneapi::experimental;
    auto& q = dpct::get_in_order_queue();
    strata::core::ExpertCache storage;
    storage.set_segment_bytes(8ll << 20);
    std::string error;
    if (!storage.open(slots, 1, slots, slot_bytes, error)) throw std::runtime_error(error);
    const int64_t bytes = storage.full_bytes(), physical = storage.mapped_bytes();
    const int64_t expected_physical = ((bytes + (8ll << 20) - 1) / (8ll << 20)) * (8ll << 20);
    if (physical != expected_physical || bytes % 4) throw std::runtime_error("unexpected segment geometry");
    const size_t words = static_cast<size_t>(bytes / 4);
    auto* address = reinterpret_cast<uint32_t*>(storage.device_slot(0));
    auto* result = sycl::malloc_device<uint32_t>(words, q);
    if (!result) throw std::runtime_error("result allocation failed");
    auto release_result = [&](uint32_t* value) { q.wait_and_throw(); sycl::free(value, q); };
    std::unique_ptr<uint32_t, decltype(release_result)> result_guard(result, release_result);
    std::vector<uint32_t> source(words), actual(words);
    for (size_t i = 0; i < words; ++i) source[i] = static_cast<uint32_t>(i * 2654435761ull) ^ 0x713af0c5u;
    q.memcpy(address, source.data(), static_cast<size_t>(bytes)).wait_and_throw();

    graph::command_graph captured(q.get_context(), q.get_device());
    captured.add([=](sycl::handler& handler) {
        handler.parallel_for<class LeasedCacheRead>(sycl::range<1>(words), [=](sycl::id<1> index) {
            result[index] = address[index] ^ 0xa5a5a5a5u;
        });
    });
    auto executable = captured.finalize();
    auto replay = [&] {
        q.ext_oneapi_graph(executable).wait_and_throw();
        q.memcpy(actual.data(), result, static_cast<size_t>(bytes)).wait_and_throw();
        for (size_t i = 0; i < words; ++i)
            if (actual[i] != (source[i] ^ 0xa5a5a5a5u)) throw std::runtime_error("captured kernel read the wrong weight");
    };
    replay();
    for (int round = 0; round < 3; ++round) {
        const auto begin = Clock::now();
        if (!storage.shrink(0, error)) throw std::runtime_error(error);
        const double unmap_ms = milliseconds(begin);
        if (storage.mapped_bytes() != 0 || storage.slots() != 0 || storage.full_slots() != slots)
            throw std::runtime_error("cache remained physically backed");
        const auto map_start = Clock::now();
        if (!storage.grow(bytes, error)) throw std::runtime_error(error);
        const double map_ms = milliseconds(map_start);
        if (storage.device_slot(0) != reinterpret_cast<uint8_t*>(address) ||
            storage.slots() != slots || storage.full_slots() != slots || storage.mapped_bytes() != physical)
            throw std::runtime_error("cache restoration changed its geometry or address");
        const auto copy_start = Clock::now();
        q.memcpy(address, source.data(), static_cast<size_t>(bytes)).wait_and_throw();
        const double copy_ms = milliseconds(copy_start);
        q.memcpy(actual.data(), address, static_cast<size_t>(bytes)).wait_and_throw();
        if (actual != source) throw std::runtime_error("restored weight bytes differ");
        replay();
        std::printf("{\"payload_bytes\":%lld,\"physical_bytes\":%lld,\"round\":%d,"
                    "\"unmap_ms\":%.6f,\"map_ms\":%.6f,\"copy_ms\":%.6f,"
                    "\"same_address\":true,\"all_weight_bytes_identical\":true,\"captured_graph_replay_correct\":true}\n",
                    static_cast<long long>(bytes), static_cast<long long>(physical), round,
                    unmap_ms, map_ms, copy_ms);
        std::fflush(stdout);
    }
    q.wait_and_throw();
}

int main() try {
    auto& q = dpct::get_in_order_queue();
    std::printf("device=%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::fflush(stdout);
    check(1, (3ll << 20) + 20); // A partial segment with an unaligned-to-page payload.
    check(512, strata::kernels::cpu::BLOB); // The draft expert geometry.
    check(1, 106299ll * 2100); // The measured draft head subset and native row size.
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "cache lease probe: %s\n", error.what());
    return 1;
}
