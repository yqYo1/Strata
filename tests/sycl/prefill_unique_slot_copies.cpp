// Main-authored unique-slot lifetime fixture; repository LICENSE.
// Unbuilt until its exact recipe and finite validation plan are committed.
#include "strata/prefill/sycl_unique_slot_copies.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <vector>

namespace {
namespace sb = strata::sycl_backend;
using Copies = strata::prefill::detail::PrefillSyclUniqueSlotCopies;
constexpr std::size_t bytes = 256, pad = 64, source_count = 512;
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
    need(caught, "prefill unique slots accepted an invalid transition");
}
struct Sources {
    void* mapping = MAP_FAILED;
    static std::uint8_t value(std::size_t slot, std::size_t byte) {
        // Both bytes of the slot index are represented: all512sources differ.
        return std::uint8_t((byte & 1) ? slot >> 8 : slot) ^ std::uint8_t(13 + 17 * byte);
    }
    Sources() {
        mapping = mmap(nullptr, source_count * bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        need(mapping != MAP_FAILED, "ordinary source mapping");
        for (std::size_t i = 0; i < source_count * bytes; ++i)
            static_cast<std::uint8_t*>(mapping)[i] = value(i / bytes, i % bytes);
        if (mprotect(mapping, source_count * bytes, PROT_READ) != 0) {
            munmap(mapping, source_count * bytes); mapping = MAP_FAILED;
            throw std::runtime_error("source read-only protection");
        }
    }
    ~Sources() { if (mapping != MAP_FAILED) munmap(mapping, source_count * bytes); }
    Sources(const Sources&) = delete;
    Sources& operator=(const Sources&) = delete;
    std::span<const std::byte> slice(std::size_t slot) const {
        need(slot < source_count, "source slot bounds");
        return {static_cast<const std::byte*>(mapping) + slot * bytes, bytes};
    }
};
void prefill_host_refusals() {
    for (const auto count : {std::size_t(0), std::size_t(513)})
        refused([&] { Copies owner({}, nullptr, bytes, count, Copies::Placement::Compute); });
}
struct Injected {};

int run(const std::string& mode, Copies::Placement placement, std::size_t width) {
    const std::size_t count = mode == "full512" ? 512 : mode == "partial129" ? 129 : 128;
    gate("before-runtime");
    auto runtime = sb::runtime_for(); auto& q = runtime->compute();
    Sources sources;
    sb::Allocation storage(runtime, count * bytes + 2 * pad, sb::MemoryKind::Device);
    sb::Allocation output(runtime, count * bytes * sizeof(std::uint32_t), sb::MemoryKind::Device);
    auto* slots = storage.as<std::byte>() + pad;
    auto* result = output.as<std::uint32_t>();
    gate("runtime-ready");
    need(sycl::get_pointer_type(sources.mapping, runtime->context()) == sycl::usm::alloc::unknown,
         "source must remain ordinary RAM");
    if (mode == "metadata") {
        Copies owner(runtime, slots, bytes, count, placement);
        need(owner.prefill_submitted_slots() == 0, "empty owner");
        owner.drain();
    } else {
        runtime->wait(q.memset(storage.data(), 0xa5, storage.size()));
        runtime->wait(q.memset(output.data(), 0, output.size()));
        if (mode == "throw") {
            bool caught = false;
            try {
                Copies owner(runtime, slots, bytes, count, placement);
                owner.prefill_copy_next_slot(0, sources.slice(0));
                owner.prefill_copy_next_slot(1, sources.slice(1));
                throw Injected{};
            } catch (const Injected&) { caught = true; }
            need(caught, "host exception not observed");
            std::array<std::byte, bytes * 2> observed;
            runtime->wait(q.memcpy(observed.data(), slots, observed.size()));
            for (std::size_t i = 0; i < observed.size(); ++i)
                need(observed[i] == std::byte(Sources::value(i / bytes, i % bytes)), "exception copy retirement");
        } else {
            Copies owner(runtime, slots, bytes, count, placement);
            refused([&] { owner.prefill_copy_next_slot(1, sources.slice(0)); });
            std::vector<Copies::Ready> tickets; tickets.reserve(count);
            tickets.push_back(owner.prefill_copy_next_slot(0, sources.slice(0)));
            refused([&] { owner.prefill_expert_bytes(tickets[0]); });
            refused([&] { owner.prefill_copy_next_slot(1, sources.slice(1).first(bytes - 1)); });
            refused([&] { owner.prefill_copy_next_slot(1, {slots + bytes, bytes}); });
            need(owner.prefill_submitted_slots() == 1, "refused copy advanced publication");
            if (mode == "terminal") {
                owner.drain();
                refused([&] { owner.prefill_copy_next_slot(1, sources.slice(1)); });
                refused([&] { owner.prefill_wait_ready(tickets[0]); });
                refused([&] { owner.prefill_expert_bytes(tickets[0]); });
                owner.drain();
            } else {
                std::optional<Copies::Ready> stale;
                {
                    sb::Allocation other_storage(runtime, bytes, sb::MemoryKind::Device);
                    Copies other(runtime, other_storage.data(), bytes, 1, placement);
                    stale = other.prefill_copy_next_slot(0, sources.slice(0));
                    refused([&] { owner.prefill_wait_ready(*stale); });
                    other.drain();
                }
                refused([&] { owner.prefill_wait_ready(*stale); });
                for (std::size_t i = 1; i < count; ++i)
                    tickets.push_back(owner.prefill_copy_next_slot(i, sources.slice(i)));
                refused([&] { owner.prefill_copy_next_slot(count, sources.slice(0)); });
                refused([&] { owner.prefill_wait_ready(std::span<const Copies::Ready>{}); });
                const std::array<Copies::Ready, 2> reversed{tickets[1], tickets[0]};
                refused([&] { owner.prefill_wait_ready(reversed); });
                for (std::size_t first = 0; first < count; first += width) {
                    const auto n = std::min(width, count - first);
                    const auto ready = std::span<const Copies::Ready>(tickets).subspan(first, n);
                    owner.prefill_wait_ready(ready);
                    refused([&] { owner.prefill_wait_ready(ready); });
                    for (std::size_t i = first; i < first + n; ++i) {
                        const auto span = owner.prefill_expert_bytes(tickets[i]);
                        need(span.data() == slots + i * bytes && span.size() == bytes, "wrong slot view");
                        const auto* input = reinterpret_cast<const std::uint8_t*>(span.data());
                        q.parallel_for<class StrataPrefillUniqueSlotConsumer>(sycl::range<1>(bytes),
                            [=](sycl::id<1> j) {
                                result[i * bytes + j] = std::uint32_t(input[j]) + std::uint32_t(i * 65536 + 17);
                            });
                    }
                }
                owner.drain();
                std::vector<std::uint32_t> observed(count * bytes);
                runtime->wait(q.memcpy(observed.data(), result, output.size()));
                for (std::size_t i = 0; i < observed.size(); ++i)
                    need(observed[i] == std::uint32_t(Sources::value(i / bytes, i % bytes)) +
                         std::uint32_t((i / bytes) * 65536 + 17), "GPU consumer saw wrong source");
                refused([&] { owner.prefill_expert_bytes(tickets[0]); });
                need(owner.prefill_submitted_slots() == count, "exact accepted copy count");
            }
        }
        std::array<std::byte, pad> guard;
        for (const auto* at : {storage.as<std::byte>(), slots + count * bytes}) {
            runtime->wait(q.memcpy(guard.data(), at, guard.size()));
            for (const auto value : guard) need(value == std::byte(0xa5), "destination guard overwritten");
        }
    }
    gate("before-exit");
    std::cout << "{\"kind\":\"done\",\"passed\":true,\"mode\":\"" << mode
              << "\",\"slots\":" << count << ",\"ready_width\":" << width << "}" << std::endl;
    return 0;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc == 2 && std::string(argv[1]) == "host") {
        gate("host-ready");
        prefill_host_refusals();
        gate("host-exit");
        std::cout << "{\"kind\":\"host\",\"passed\":true,\"capacity_refusals\":2}" << std::endl;
        return 0;
    }
    need(argc == 4, "mode placement ready_width");
    const std::string mode = argv[1], placement = argv[2], width = argv[3];
    need(mode == "metadata" || mode == "ordered" || mode == "partial129" ||
         mode == "full512" || mode == "throw" || mode == "terminal", "finite fixture mode");
    need(placement == "compute" || placement == "separate", "explicit placement");
    need(width == "1" || width == "64", "finite readiness width");
    return run(mode, placement == "separate" ? Copies::Placement::Separate : Copies::Placement::Compute,
               width == "64" ? 64 : 1);
} catch (const std::exception& error) {
    std::cerr << "prefill unique-slot fixture failure: " << error.what() << std::endl;
    return 1;
}
