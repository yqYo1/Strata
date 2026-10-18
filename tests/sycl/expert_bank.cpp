// Authored for Strata; repository LICENSE applies. Exercises the production
// upload helper and ExpertCache, not model arithmetic or session rollback.
#include "strata/prefill/expert_bank_upload.hpp"
#include "strata/prefill/draft_uploads.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

namespace pd = strata::prefill::detail;
namespace sb = strata::sycl_backend;
constexpr uint32_t Guard = 0xa5a5a5a5;
constexpr size_t G = 64;
void need(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
uint32_t value(int expert, size_t word, unsigned generation) {
    uint32_t x = uint32_t(word) * 747796405u ^ uint32_t(expert + 1) * 2891336453u ^ generation * 277803737u;
    x ^= x >> 16; x *= 2246822519u;
    return x ^ (x >> 13);
}
struct Stop {};

void run(int capacity, size_t stride, int experts, unsigned layers, bool unwind, size_t source_stride = 0) {
    if (!source_stride) source_stride = stride;
    auto runtime = sb::runtime_for();
    auto& q = runtime->compute();
    std::vector<uint32_t> host(size_t(512) * source_stride / 4 + 2 * G, Guard);
    auto* slab = reinterpret_cast<uint8_t*>(host.data() + G);
    const size_t slab_bytes = size_t(512) * source_stride, words = source_stride / 4 - 1;
    strata::core::ExpertCache cache;
    std::string error;
    if (!cache.open_uninitialized(capacity, 1, 512, int64_t(stride), error))
        throw std::runtime_error(error);
    sb::Allocation status_owner(runtime, 4, sb::MemoryKind::Device);
    auto* status = status_owner.as<uint32_t>();
    pd::ExpertBankSlots slots(capacity);
    uint32_t observed = ~uint32_t(0);
    unsigned drains = 0, cancellations = 0;
    uint64_t uploads = 0, hits = 0, consumers = 0;
    for (unsigned generation = 1; generation <= layers; ++generation) {
        // Previous generation's queue was checked before changing the slab/map.
        slots.reset();
        for (int e = 0; e < 512; ++e)
            for (size_t i = 0; i < source_stride / 4; ++i)
                host[G + size_t(e) * source_stride / 4 + i] = value(e, i, generation);
        bool unwound = false;
        try {
            pd::DraftUploadGuard lease([&] { runtime->wait(); ++drains; return true; },
                [] { sb::cleanup_fail_stop("expert-bank fixture retirement"); });
            lease.arm();
            q.memset(cache.device_slot(0), 0xa5, size_t(cache.bytes()));
            q.memset(status, 0, 4);
            for (int round = 0; round < 2; ++round) for (int e = 0; e < experts; ++e) {
                const auto ticket = pd::stage_expert_bank(slots, cache, slab, slab_bytes,
                    stride, words * 4, e, &q, source_stride);
                uploads += ticket.upload; hits += !ticket.upload;
                need(cache.slot_of(0, e) < 0, "private scratch/permanent map leaked to general cache");
                const auto* blob = reinterpret_cast<const uint32_t*>(cache.device_slot(ticket.slot));
                // Two queued readers before the next scratch overwrite model
                // multiple consumers without adding a wait after each expert.
                for (int job = 0; job < 2; ++job) {
                    const size_t global = (words + 255) / 256 * 256;
                    q.parallel_for(sycl::nd_range<1>(global, 256), [=](sycl::nd_item<1> item) {
                        const size_t i = item.get_global_linear_id();
                        const bool bad = i < words && blob[i] != value(e, i, generation);
                        const bool any = sycl::any_of_group(item.get_group(), bad);
                        if (item.get_local_linear_id() == 0 && any)
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                sycl::memory_scope::device, sycl::access::address_space::global_space>(*status).fetch_or(1);
                    });
                    ++consumers;
                }
                if (unwind && e == 4) {
                    q.memcpy(&observed, status, 4);
                    throw Stop{}; // slab/cache/status destinations outlive checked lease
                }
            }
            const auto* base = reinterpret_cast<const uint32_t*>(cache.device_slot(0));
            q.parallel_for(sycl::range<1>(size_t(capacity)), [=](sycl::id<1> item) {
                if (base[(size_t(item[0]) + 1) * (stride / 4) - 1] != Guard)
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                        sycl::memory_scope::device, sycl::access::address_space::global_space>(*status).fetch_or(2);
            });
            q.memcpy(&observed, status, 4);
            runtime->wait(); ++drains; lease.retire();
        } catch (const Stop&) { unwound = true; ++cancellations; }
        need(unwound == unwind && observed == 0, "queued reader, guard or unwind result failed");
        for (size_t i = 0; i < G; ++i)
            need(host[i] == Guard && host[host.size() - 1 - i] == Guard, "host guard changed");
        for (int e = 0; e < 512; ++e)
            for (size_t i = 0; i < source_stride / 4; ++i)
                need(host[G + size_t(e) * source_stride / 4 + i] == value(e, i, generation), "immutable slab changed");
    }
    if (!unwind) {
        const uint64_t permanent = uint64_t(std::min(capacity - 1, experts));
        need(uploads == layers * (2 * uint64_t(experts) - permanent) && hits == layers * permanent,
            "permanent/scratch admission totals");
    }
    need(drains == layers && cancellations == (unwind ? layers : 0), "checked retirement count");
    unsigned rejects = 0;
    auto reject = [&](auto action) {
        bool caught = false;
        try { action(); } catch (const std::invalid_argument&) { caught = true; }
        need(caught, "invalid admission accepted"); ++rejects;
    };
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes - 1, stride, words * 4, 0, &q, source_stride); });
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, stride + 1, 0, &q, source_stride); });
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 512, &q, source_stride); });
    reject([&] { pd::stage_expert_bank(slots, cache, nullptr, slab_bytes, stride, words * 4, 0, &q, source_stride); });
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 0, nullptr, source_stride); });
    pd::ExpertBankSlots wrong(capacity == 1 ? 2 : 1);
    reject([&] { pd::stage_expert_bank(wrong, cache, slab, slab_bytes, stride, words * 4, 0, &q, source_stride); });
    sycl::queue foreign{sycl::context{runtime->device()}, runtime->device(), sycl::property::queue::in_order{}};
    reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 0, &foreign, source_stride); });
    if (source_stride != stride) {
        reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 0, &q, source_stride - 8); });
        reject([&] { pd::stage_expert_bank(slots, cache, slab, slab_bytes, stride, words * 4, 0, &q, stride); });
    }
    runtime->wait();
    std::cout << "expert_bank_case slots=" << capacity << " stride=" << stride << " experts=" << experts
        << " source_stride=" << source_stride
        << " layers=" << layers << " uploads=" << uploads << " hits=" << hits << " consumers=" << consumers
        << " unwind=" << cancellations << " admission_rejections=" << rejects
        << " status=" << observed << " immutable_slab=PASS guards=PASS private_map=PASS checked_drain=PASS\n";
}

int main(int argc, char** argv) try {
    need(argc == 2 && (std::string(argv[1]) == "--smoke" || std::string(argv[1]) == "--validate"),
        "use --smoke|--validate");
    if (std::string(argv[1]) == "--smoke") {
        run(4, 256, 6, 2, false);
        run(1, 256, 6, 1, false);
        run(4, 256, 6, 1, true);
        run(4, 512, 6, 2, false, 256);
        run(1, 512, 6, 1, false, 256);
        run(4, 512, 6, 1, true, 256);
    } else {
        run(383, 256, 512, 1, false);
        run(384, 256, 512, 2, false);
        run(512, 256, 512, 1, false);
        run(384, 2662400, 512, 1, false);
        run(384, 512, 512, 2, false, 256);
        run(384, 2662400, 512, 1, false, 1382400);
    }
    std::cout << "expert_bank_fixture PASS model=UNRUN physical262144_RESTORE=UNQUALIFIED\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "expert_bank_fixture failed: " << error.what() << '\n';
    return 1;
}
