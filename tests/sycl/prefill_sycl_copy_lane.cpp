// Main-authored dependency/lifetime fixture; repository LICENSE.
// No performance/model claim. Main must pin a fresh build and finite owner.
#include "strata/prefill/sycl_copy_lane.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>

namespace {
namespace sb = strata::sycl_backend;
using Lane = strata::prefill::detail::PrefillSyclCopyLane;
constexpr std::size_t bytes = 256;
void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void gate(const char* name) {
    std::cout << "{\"kind\":\"gate\",\"name\":\"" << name << "\"}" << std::endl;
    std::string line;
    need(bool(std::getline(std::cin, line)) && line == "continue", "Main gate refused");
}
template<class F> void refused(F&& f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    need(caught, "prefill SYCL lane accepted invalid transition");
}
struct Source {
    void* mapping = MAP_FAILED;
    Source() {
        mapping = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        need(mapping != MAP_FAILED, "ordinary source allocation");
        auto* p = static_cast<std::byte*>(mapping);
        std::memset(p, 0x31, bytes); std::memset(p + bytes, 0x73, bytes);
        if (mprotect(mapping, 4096, PROT_READ) != 0) {
            munmap(mapping, 4096); mapping = MAP_FAILED;
            throw std::runtime_error("ordinary source read-only protection");
        }
    }
    ~Source() { if (mapping != MAP_FAILED) munmap(mapping, 4096); }
    Source(const Source&) = delete;
    std::span<const std::byte> slice(std::size_t index) const {
        need(index < 2, "finite source index");
        return {static_cast<const std::byte*>(mapping) + index * bytes, bytes};
    }
};
struct Injected {};

int run(const std::string& mode, Lane::Placement placement, bool early) {
    gate("before-runtime");
    auto runtime = sb::runtime_for();
    auto& q = runtime->compute();
    Source source; // Outlives every lane and every accepted copy.
    sb::Allocation bank(runtime, bytes + 128, sb::MemoryKind::Device);
    sb::Allocation a(runtime, bytes, sb::MemoryKind::Device);
    sb::Allocation b(runtime, bytes, sb::MemoryKind::Device);
    sb::Allocation output(runtime, 2 * sizeof(std::uint32_t), sb::MemoryKind::Device);
    auto* destination = bank.as<std::byte>() + 64;
    gate("runtime-ready");
    need(sycl::get_pointer_type(source.mapping, runtime->context()) == sycl::usm::alloc::unknown,
         "ordinary source must not become a USM/imported allocation");
    if (mode != "metadata") {
        runtime->wait(q.memset(bank.data(), 0xa5, bank.size()));
        runtime->wait(q.memset(output.data(), 0, output.size()));
        if (mode == "throw-copy" || mode == "throw-reader") {
            bool caught = false;
            try {
                Lane lane(runtime, destination, bytes, placement, 2);
                auto ready = lane.prefill_copy(source.slice(0));
                if (mode == "throw-reader") {
                    lane.prefill_wait_ready(ready);
                    auto* values = a.as<std::byte>();
                    q.parallel_for<class StrataPrefillCopyExceptionalReader>(sycl::range<1>(bytes),
                        [=](sycl::id<1> i) { values[i] = destination[i]; });
                }
                throw Injected{};
            } catch (const Injected&) { caught = true; }
            need(caught, "injected exit not observed");
            // The lane's checked destructor must finish accepted copies/readers
            // while source, bank and consumer allocations still exist.
            std::array<std::byte, bytes> observed{};
            runtime->wait(q.memcpy(observed.data(), destination, bytes));
            need(std::memcmp(observed.data(), source.slice(0).data(), bytes) == 0,
                 "exceptional checked drain lost accepted copy");
        } else {
            need(mode == "ordered", "explicit finite mode required");
            Lane lane(runtime, destination, bytes, placement, 2);
            std::optional<Lane::Ready> previous;
            refused([&] { lane.prefill_copy(source.slice(0).first(bytes - 1)); });
            refused([&] { lane.prefill_copy({}); });
            for (std::size_t iteration = 0; iteration < 2; ++iteration) {
                auto ready = lane.prefill_copy(source.slice(iteration));
                if (previous) refused([&] { lane.prefill_wait_ready(*previous); });
                if (!previous) {
                    Lane foreign(runtime, destination, bytes, Lane::Placement::Compute, 2);
                    refused([&] { foreign.prefill_wait_ready(ready); });
                    refused([&] { foreign.prefill_record_readers(ready); });
                }
                refused([&] { lane.prefill_record_readers(ready); });
                previous = ready;
                refused([&] { lane.prefill_copy(source.slice(iteration)); });
                lane.prefill_wait_ready(ready);
                refused([&] { lane.prefill_wait_ready(ready); });
                auto* av = a.as<std::uint8_t>();
                auto* bv = b.as<std::uint8_t>();
                auto* result = output.as<std::uint32_t>();
                auto* quantized = reinterpret_cast<const std::uint8_t*>(destination);
                q.parallel_for<class StrataPrefillCopyFirstReader>(sycl::range<1>(bytes),
                    [=](sycl::id<1> i) { av[i] = quantized[i]; });
                q.parallel_for<class StrataPrefillCopySecondReader>(sycl::range<1>(bytes),
                    [=](sycl::id<1> i) { bv[i] = quantized[i]; });
                if (early) lane.prefill_record_readers(ready);
                q.single_task<class StrataPrefillCopySeparateWeightConsumers>([=] {
                    std::uint32_t value = 0;
                    for (std::size_t i = 0; i < bytes; ++i) value += av[i] + bv[i];
                    result[iteration] = value;
                });
                if (!early) lane.prefill_record_readers(ready);
                refused([&] { lane.prefill_record_readers(ready); });
            }
            refused([&] { lane.prefill_copy(source.slice(0)); }); // Capacity.
            lane.drain();
            std::array<std::uint32_t, 2> result{};
            runtime->wait(q.memcpy(result.data(), output.data(), output.size()));
            need(result[0] == 2 * bytes * 0x31 && result[1] == 2 * bytes * 0x73,
                 "two readers/weight consumers lost their dependency");
        }
        std::array<std::byte, bytes + 128> guarded{};
        runtime->wait(q.memcpy(guarded.data(), bank.data(), bank.size()));
        for (std::size_t i = 0; i < 64; ++i)
            need(guarded[i] == std::byte{0xa5} && guarded[bytes + 64 + i] == std::byte{0xa5},
                 "copy bank guard changed");
    }
    runtime->wait(); runtime->check(); gate("before-exit");
    std::cout << "{\"kind\":\"done\",\"passed\":true}" << std::endl;
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        need(argc == 4, "metadata|ordered|throw-copy|throw-reader compute|separate math|iq");
        const std::string mode = argv[1], placement = argv[2], retirement = argv[3];
        need(mode == "metadata" || mode == "ordered" || mode == "throw-copy" || mode == "throw-reader",
             "explicit finite fixture mode required");
        need(placement == "compute" || placement == "separate", "explicit queue placement required");
        need(retirement == "math" || retirement == "iq", "explicit retirement position required");
        return run(mode, placement == "compute" ? Lane::Placement::Compute : Lane::Placement::Separate,
                   retirement == "iq");
    } catch (const std::exception& e) {
        std::cerr << "prefill SYCL copy lane fixture failure: " << e.what() << std::endl;
        return 1;
    }
}
