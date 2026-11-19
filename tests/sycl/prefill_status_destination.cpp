// Main-authored. Repository LICENSE applies.
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/sycl/compat/cuda_runtime.h"
#include "strata/sycl/launch.hpp"
#include "strata/sycl/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
using strata::sycl_backend::checked_cleanup;
using strata::sycl_backend::queue_for;
using Clock = std::chrono::steady_clock;
constexpr uint32_t Guard = 0xfe17c3a9u;
constexpr uint32_t Pending = ~uint32_t{0};
constexpr size_t WorkItems = 262144;
constexpr unsigned Rounds = 192;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Stream {
    cudaStream_t handle{};

    Stream() {
        require(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking) == cudaSuccess,
                "could not create the compatibility stream");
    }

    void drain() const {
        require(cudaStreamSynchronize(handle) == cudaSuccess, "compatibility stream failed to drain");
    }

    ~Stream() {
        checked_cleanup([&] { drain(); }, [&] {
            require(cudaStreamDestroy(handle) == cudaSuccess, "could not retire the compatibility stream");
        }, "prefill status fixture retirement");
    }
};

struct Times {
    double enqueue_ms = 0;
    double drain_ms = 0;
};

void produce(sycl::queue& queue, uint32_t* scratch, uint32_t* status, uint32_t value) {
    queue.parallel_for(sycl::range<1>(WorkItems), [=](sycl::id<1> item) {
        const size_t i = item[0];
        uint32_t x = static_cast<uint32_t>(i) ^ value;
        for (unsigned step = 0; step < Rounds; ++step) {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x += step + 1;
        }
        scratch[i] = x;
    });
    queue.single_task([=] { *status = value ^ (scratch[37] & 0x0000ffffu); });
}

uint32_t expected_status(uint32_t value) {
    uint32_t x = 37u ^ value;
    for (unsigned step = 0; step < Rounds; ++step) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x += step + 1;
    }
    return value ^ (x & 0x0000ffffu);
}

Times transfer(Stream& stream, uint32_t* destination, const uint32_t* status, bool early_return) {
    destination[1] = Pending;
    const auto begin = Clock::now();
    // The ordinary and USM cases use the same production wrappers and order.
    // Any unconfirmed submission is drained before the destination can die.
    bool retired = false;
    struct PendingCopy {
        const Stream& stream;
        bool& retired;
        ~PendingCopy() {
            if (!retired)
                checked_cleanup([&] { stream.drain(); }, [] {}, "prefill status fixture cancelled copy");
        }
    } checked{stream, retired};
    require(cudaMemcpyAsync(destination + 1, status, sizeof(*status), cudaMemcpyDeviceToHost,
                           stream.handle) == cudaSuccess, "status copy failed");
    const auto submitted = Clock::now();
    if (early_return) return {};
    stream.drain();
    const auto ended = Clock::now();
    retired = true;
    return {elapsed_ms(begin, submitted), elapsed_ms(submitted, ended)};
}

void run(bool use_usm, bool timing) {
    auto runtime = strata::sycl_backend::runtime_for();
    require(runtime->device().get_backend() == sycl::backend::ext_oneapi_level_zero,
            "the qualified Level Zero device is required");
    std::array<uint32_t, 3> ordinary{Guard, Pending, Guard};
    Allocation scratch(runtime, (WorkItems + 1) * sizeof(uint32_t), MemoryKind::Device);
    Allocation host;
    const auto allocation_begin = Clock::now();
    if (use_usm) host = Allocation(runtime, sizeof(ordinary), MemoryKind::Host);
    const double allocation_ms = elapsed_ms(allocation_begin, Clock::now());
    Stream stream;
    auto& queue = queue_for(stream.handle);
    require(queue.get_context() == runtime->context() && queue.get_device() == runtime->device()
            && queue.is_in_order(), "unexpected compatibility stream domain or order");
    auto* destination = use_usm ? host.as<uint32_t>() : ordinary.data();
    if (use_usm) {
        strata::sycl_backend::validate_host_span(destination, sizeof(ordinary), queue);
        std::copy(ordinary.begin(), ordinary.end(), destination);
    }
    require(sycl::get_pointer_type(destination, queue.get_context()) ==
                (use_usm ? sycl::usm::alloc::host : sycl::usm::alloc::unknown),
            "actual destination placement differs");
    auto* status = scratch.as<uint32_t>() + WorkItems;
    auto check = [&](uint32_t expected) {
        require(destination[0] == Guard && destination[2] == Guard && destination[1] == expected,
                "status value or destination guard differs");
    };
    for (uint32_t value : {0u, 0x61530271u}) {
        queue.single_task([=] { *status = value; });
        transfer(stream, destination, status, false);
        check(value);
    }
    const uint32_t cancel_value = 0x61752100u;
    produce(queue, scratch.as<uint32_t>(), status, cancel_value);
    transfer(stream, destination, status, true);
    check(expected_status(cancel_value));
    std::array<double, 257> clock_checks{};
    for (double& interval : clock_checks) {
        const auto begin = Clock::now();
        const auto middle = Clock::now();
        const auto end = Clock::now();
        require(begin <= middle && middle <= end, "monotonic timing stamps invalid");
        interval = elapsed_ms(begin, end);
    }
    std::sort(clock_checks.begin(), clock_checks.end());
    std::cout << "placement=" << (use_usm ? "host-usm" : "ordinary")
              << " guarded_values=2 cancellation_drain=1 allocation_ms=" << allocation_ms
              << " clock_triplet_median_ms=" << clock_checks[clock_checks.size() / 2] << '\n';
    if (!timing) return;

    for (bool busy : {false, true}) {
        const unsigned count = busy ? 48 : 512;
        for (unsigned batch = 0; batch < 5; ++batch) {
            Times total;
            for (unsigned i = 0; i < count; ++i) {
                const uint32_t value = 0x40000000u | (batch << 16) | i;
                if (busy) {
                    produce(queue, scratch.as<uint32_t>(), status, value);
                } else {
                    queue.single_task([=] { *status = value; });
                    stream.drain();
                }
                const Times sample = transfer(stream, destination, status, false);
                total.enqueue_ms += sample.enqueue_ms;
                total.drain_ms += sample.drain_ms;
                check(busy ? expected_status(value) : value);
            }
            std::cout << "shape=" << (busy ? "queued" : "ready") << " batch=" << batch
                      << " copies=" << count << " enqueue_ms=" << total.enqueue_ms
                      << " drain_ms=" << total.drain_ms
                      << " total_ms=" << total.enqueue_ms + total.drain_ms << '\n';
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "choose ordinary or host-usm, and smoke or timing");
        const std::string_view placement(argv[1]), operation(argv[2]);
        require(placement == "ordinary" || placement == "host-usm", "unknown placement");
        require(operation == "smoke" || operation == "timing", "unknown fixture operation");
        std::cout << std::fixed << std::setprecision(9) << std::unitbuf;
        run(placement == "host-usm", operation == "timing");
        std::cout << "fixture_status=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
