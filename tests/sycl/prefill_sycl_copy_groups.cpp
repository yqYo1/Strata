// Main-authored grouped-copy lifetime fixture; repository LICENSE.
// Independent of model dispatch. Unbuilt until Main closes the current cohort.
#include "strata/prefill/sycl_copy_groups.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
using Groups = strata::prefill::detail::PrefillSyclCopyGroups;
constexpr std::size_t bytes = 256, source_count = 256, pad = 64;
void need(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void gate(const char* name) {
    std::cout << "{\"kind\":\"gate\",\"name\":\"" << name << "\"}" << std::endl;
    std::string line;
    need(bool(std::getline(std::cin, line)) && line == "continue", "Main gate refused");
}
template<class F> void refused(F&& f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    need(caught, "prefill copy groups accepted invalid transition");
}
struct Sources {
    void* mapping = MAP_FAILED;
    static std::uint8_t value(std::size_t i) { return std::uint8_t(1 + (17 * i) % 251); }
    Sources() {
        mapping = mmap(nullptr, bytes * source_count, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        need(mapping != MAP_FAILED, "ordinary source mapping");
        for (std::size_t i = 0; i < source_count; ++i)
            std::memset(static_cast<std::byte*>(mapping) + i * bytes, value(i), bytes);
        if (mprotect(mapping, bytes * source_count, PROT_READ) != 0) {
            munmap(mapping, bytes * source_count); mapping = MAP_FAILED;
            throw std::runtime_error("ordinary source read-only protection");
        }
    }
    ~Sources() { if (mapping != MAP_FAILED) munmap(mapping, bytes * source_count); }
    Sources(const Sources&) = delete;
    Sources& operator=(const Sources&) = delete;
    std::span<const std::byte> slice(std::size_t i) const {
        need(i < source_count, "finite source index");
        return {static_cast<const std::byte*>(mapping) + i * bytes, bytes};
    }
    std::vector<std::span<const std::byte>> group(std::size_t first, std::size_t count) const {
        std::vector<std::span<const std::byte>> result;
        for (std::size_t i = 0; i < count; ++i) result.push_back(slice(first + i));
        return result;
    }
};
struct Injected {};

int run(const std::string& mode, Groups::Placement placement, std::size_t width) {
    gate("before-runtime");
    auto runtime = sb::runtime_for();
    auto& q = runtime->compute();
    Sources sources;
    const std::size_t slot_bytes = 2 * width * bytes, experts = 3 * width + 1;
    sb::Allocation storage(runtime, slot_bytes + 2 * pad, sb::MemoryKind::Device);
    sb::Allocation a(runtime, bytes, sb::MemoryKind::Device), b(runtime, bytes, sb::MemoryKind::Device);
    sb::Allocation output(runtime, experts * sizeof(std::uint32_t), sb::MemoryKind::Device);
    auto* slots = storage.as<std::byte>() + pad;
    const Groups::Layout layout{bytes, width, 2};
    gate("runtime-ready");
    need(sycl::get_pointer_type(sources.mapping, runtime->context()) == sycl::usm::alloc::unknown,
         "source must remain ordinary RAM");
    if (mode == "metadata") {
        Groups groups(runtime, slots, layout, placement);
        groups.drain();
    } else {
        runtime->wait(q.memset(storage.data(), 0xa5, storage.size()));
        runtime->wait(q.memset(output.data(), 0, output.size()));
        const auto consume = [&](Groups& groups, Groups::Ready ready, std::size_t first,
                                 std::size_t count) {
            groups.prefill_wait_ready(ready);
            for (std::size_t i = 0; i < count; ++i) {
                const auto view = groups.prefill_expert(ready, i);
                const auto* quantized = reinterpret_cast<const std::uint8_t*>(view.data());
                auto* av = a.as<std::uint8_t>(); auto* bv = b.as<std::uint8_t>();
                auto* result = output.as<std::uint32_t>();
                q.parallel_for<class StrataPrefillGroupsFirstReader>(sycl::range<1>(bytes),
                    [=](sycl::id<1> j) { av[j] = quantized[j]; });
                q.parallel_for<class StrataPrefillGroupsSecondReader>(sycl::range<1>(bytes),
                    [=](sycl::id<1> j) { bv[j] = quantized[j]; });
                q.single_task<class StrataPrefillGroupsMath>([=] {
                    std::uint32_t sum = 0;
                    for (std::size_t j = 0; j < bytes; ++j) sum += av[j] + bv[j];
                    result[first + i] = sum;
                });
            }
        };
        if (mode == "throw-copy" || mode == "throw-reader") {
            bool caught = false;
            try {
                Groups groups(runtime, slots, layout, placement);
                auto ready = groups.prefill_copy_group(sources.group(0, width));
                if (mode == "throw-reader") consume(groups, ready, 0, width);
                throw Injected{};
            } catch (const Injected&) { caught = true; }
            need(caught, "injected exception not observed");
            std::vector<std::byte> observed(width * bytes);
            runtime->wait(q.memcpy(observed.data(), slots, observed.size()));
            for (std::size_t i = 0; i < width; ++i)
                need(std::memcmp(observed.data() + i * bytes, sources.slice(i).data(), bytes) == 0,
                     "exceptional drain lost an accepted copy");
            if (mode == "throw-reader") {
                std::vector<std::uint32_t> values(width);
                runtime->wait(q.memcpy(values.data(), output.data(), width * sizeof(std::uint32_t)));
                for (std::size_t i = 0; i < width; ++i)
                    need(values[i] == 2 * bytes * Sources::value(i), "exceptional drain lost readers");
            }
        } else {
            need(mode == "ordered", "explicit finite mode required");
            refused([&] { Groups bad(runtime, nullptr, layout, placement); });
            refused([&] { Groups bad(runtime, slots, {0, width, 2}, placement); });
            refused([&] { Groups bad(runtime, slots, {bytes, 0, 2}, placement); });
            refused([&] { Groups bad(runtime, slots, {bytes, width, 0}, placement); });
            refused([&] { Groups bad(runtime, slots, {bytes, 513, 1}, placement); });
            refused([&] { Groups bad(runtime, slots, {bytes, 512, 9}, placement); });
            refused([&] { Groups bad(runtime, slots, {std::numeric_limits<std::size_t>::max(), 2, 1}, placement); });
            refused([&] { Groups bad(runtime, slots, {slot_bytes + pad + 1, 1, 1}, placement); });
            refused([&] { Groups bad(runtime, sources.mapping, layout, placement); });
            refused([&] { Groups bad(runtime, slots, layout, static_cast<Groups::Placement>(2)); });
            refused([&] { Groups bad(runtime, slots, layout, placement, 0); });
            refused([&] { Groups bad(runtime, slots, layout, placement, 4097); });
            {
                Groups sealed(runtime, slots, layout, placement);
                sealed.drain(); sealed.drain();
                refused([&] { sealed.prefill_copy_group(sources.group(0, 1)); });
            }
            {
                Groups capacity(runtime, slots, layout, placement, 1);
                auto ready = capacity.prefill_copy_group(sources.group(0, 1));
                refused([&] { capacity.prefill_copy_group(sources.group(1, 1)); });
                capacity.drain();
                refused([&] { capacity.prefill_wait_ready(ready); });
                refused([&] { capacity.prefill_record_readers(ready); });
                refused([&] { capacity.prefill_expert(ready, 0); });
            }
            Groups groups(runtime, slots, layout, placement, 5);
            refused([&] { groups.prefill_copy_group({}); });
            refused([&] { groups.prefill_copy_group(sources.group(0, width + 1)); });
            auto malformed = sources.group(0, width);
            malformed.back() = sources.slice(0).first(bytes - 1);
            refused([&] { groups.prefill_copy_group(malformed); });
            std::array<std::span<const std::byte>, 1> aliased{{{slots, bytes}}};
            refused([&] { groups.prefill_copy_group(aliased); });
            auto first = groups.prefill_copy_group(sources.group(0, width));
            auto second = groups.prefill_copy_group(sources.group(width, width));
            refused([&] { groups.prefill_copy_group(sources.group(2 * width, width)); });
            refused([&] { groups.prefill_wait_ready(second); });
            refused([&] { groups.prefill_expert(first, 0); });
            refused([&] { groups.prefill_record_readers(first); });
            {
                Groups foreign(runtime, slots, layout, placement);
                refused([&] { foreign.prefill_wait_ready(first); });
            }
            consume(groups, first, 0, width);
            refused([&] { groups.prefill_wait_ready(first); });
            refused([&] { groups.prefill_expert(first, width); });
            groups.prefill_record_readers(first);
            refused([&] { groups.prefill_record_readers(first); });
            refused([&] { groups.prefill_expert(first, 0); });
            auto third = groups.prefill_copy_group(sources.group(2 * width, width));
            refused([&] { groups.prefill_wait_ready(first); }); // Bank recycled.
            consume(groups, second, width, width);
            groups.prefill_record_readers(second);
            auto partial = groups.prefill_copy_group(sources.group(3 * width, 1));
            consume(groups, third, 2 * width, width);
            groups.prefill_record_readers(third);
            consume(groups, partial, 3 * width, 1);
            refused([&] { groups.prefill_expert(partial, 1); });
            groups.prefill_record_readers(partial);
            groups.drain(); groups.drain();
            refused([&] { groups.prefill_copy_group(sources.group(0, 1)); }); // Spare capacity.
            std::vector<std::uint32_t> observed(experts);
            runtime->wait(q.memcpy(observed.data(), output.data(), output.size()));
            for (std::size_t i = 0; i < experts; ++i)
                need(observed[i] == 2 * bytes * Sources::value(i), "group readers/math lost dependency");
        }
        std::vector<std::byte> guarded(storage.size());
        runtime->wait(q.memcpy(guarded.data(), storage.data(), storage.size()));
        for (std::size_t i = 0; i < pad; ++i)
            need(guarded[i] == std::byte{0xa5} && guarded[pad + slot_bytes + i] == std::byte{0xa5},
                 "group copy storage guard changed");
    }
    runtime->wait(); runtime->check(); gate("before-exit");
    std::cout << "{\"kind\":\"done\",\"passed\":true,\"group_width\":" << width << "}" << std::endl;
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        need(argc == 4, "metadata|ordered|throw-copy|throw-reader compute|separate 1|2|64");
        const std::string mode = argv[1], placement = argv[2], width = argv[3];
        need(mode == "metadata" || mode == "ordered" || mode == "throw-copy" || mode == "throw-reader",
             "explicit finite fixture mode required");
        need(placement == "compute" || placement == "separate", "explicit queue placement required");
        need(width == "1" || width == "2" || width == "64", "explicit finite group width required");
        return run(mode, placement == "compute" ? Groups::Placement::Compute : Groups::Placement::Separate,
                   std::stoull(width));
    } catch (const std::exception& error) {
        std::cerr << "prefill grouped-copy fixture failure: " << error.what() << std::endl;
        return 1;
    }
}
