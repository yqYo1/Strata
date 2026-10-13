// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/phase_cache_owner.hpp"
#include "strata/core/head_upload_owner.hpp"
#include "strata/core/phase_serve_cycle.hpp"
#include "strata/prefill/workspace_layout.hpp"
#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <map>
#include <new>
#include <stdexcept>
#include <vector>
#ifdef __linux__
#include <cerrno>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
using Owner = strata::core::PhaseCacheOwner;
using Backend = strata::core::PhaseCacheBackend;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void gated_region_layout_fixture() {
    namespace layout = strata::prefill::workspace;
    check(!layout::exact_one(nullptr) && !layout::exact_one("0") && !layout::exact_one("01") &&
          !layout::exact_one("true") && layout::exact_one("1"), "gated alias exact opt-in");
    // G<B, G=B, G>B: all dimensions and cursor operations come from the production helper.
    for (uint64_t block : {uint64_t{1024}, uint64_t{512}, uint64_t{256}}) {
        for (bool alias : {false, true}) {
            const auto plan = layout::region_layout(block, 256, 256, 64, alias);
            const auto baseline = layout::region_layout(block, 256, 256, 64, false);
            check(plan.gate_extent == 512 && plan.block_max == block, "gate/block extent");
            check(plan.region_extent == (alias && block < 512 ? 512 : block), "region max");
            check(std::strcmp(plan.region_winner, alias && block < 512 ? "gated" : "gdn") == 0,
                  "gate winner mislabeled as block");
            uint64_t counted = 0, offset = 0;
            const uint64_t prefix = layout::take_bytes(7, sizeof(float));
            const uint64_t tail = layout::take_bytes(17, sizeof(uint16_t));
            check(layout::advance(prefix, counted, false, 0, offset), "count prefix");
            if (!alias) check(layout::advance(plan.gate_extent, counted, false, 0, offset), "count gate");
            check(layout::advance(plan.allocated_region, counted, false, 0, offset), "count region");
            check(layout::advance(tail, counted, false, 0, offset), "count tail");
            check(plan.saved_requested == (alias ? baseline.gate_extent + baseline.allocated_region -
                  plan.allocated_region : 0), "saved requested padding");
            alignas(256) std::array<uint8_t, 4096> storage{};
            // Fake owned requests and exact-capacity borrowed carve consume the same production cursor.
            for (bool bounded : {false, true}) {
                uint64_t used = 0, gate_offset = 0, region_offset = 0, tail_offset = 0;
                std::vector<uint64_t> owned_requests;
                auto take = [&](uint64_t bytes, uint64_t& start) {
                    const bool success = layout::advance(bytes, used, bounded, counted, start);
                    if (success && !bounded) owned_requests.push_back(bytes);
                    return success;
                };
                check(take(prefix, offset), "view prefix");
                if (!alias) check(take(plan.gate_extent, gate_offset), "view gate");
                check(take(plan.allocated_region, region_offset), "view region");
                if (alias) gate_offset = region_offset;
                check(take(tail, tail_offset) && used == counted, "count/view agreement");
                check(region_offset % 256 == 0 && tail_offset % 256 == 0, "view alignment");
                auto* region = storage.data() + region_offset;
                auto* gate = alias ? reinterpret_cast<uint8_t*>(layout::gate_view(region, plan)) :
                                    storage.data() + gate_offset;
                check((gate == region) == alias && region != storage.data() &&
                      region + plan.region_extent <= storage.data() + tail_offset, "pointer/range layout");
                uint64_t gate_used = 0, gate_start = 0;
                check(layout::advance(plan.gate_extent, gate_used, true,
                      alias ? plan.region_extent : plan.gate_extent, gate_start), "bounded gate view");
                check(gate_start == 0 && gate_used == plan.gate_extent, "gate view offset");
                check(bounded ? owned_requests.empty() : owned_requests.size() == (alias ? 3u : 4u),
                      "borrowed view acquired an owned take");
            }
            uint64_t short_used = counted - tail, failed_offset = 123;
            check(!layout::advance(tail, short_used, true, counted - 1, failed_offset) &&
                  short_used == counted - tail && failed_offset == 123, "small cap mutated cursor");
        }
    }
    // Shrink/regrow layouts independently rebuild extents; stock 12K retains Alloc's 256-byte padding.
    const auto large = layout::region_layout(2048, 1024, 512, 128, true);
    const auto small = layout::region_layout(1024, 512, 256, 64, true);
    check(small.region_extent < large.region_extent &&
          layout::region_layout(2048, 1024, 512, 128, true).region_extent == large.region_extent,
          "relayout shrink/regrow extent");
    const auto stock = layout::region_layout(1024, 1634831104, 1024,
        layout::product(12288, 10240), true);
    check(stock.saved_requested == 503316736, "stock 12K gate requested savings");
    bool small_view_rejected = false;
    alignas(256) std::array<uint8_t, 1024> short_storage{};
    try { layout::gate_view(short_storage.data(), layout::region_layout(256, 256, 256, 64, false)); }
    catch (const std::runtime_error&) { small_view_rejected = true; }
    check(small_view_rejected, "gate view exceeds block capacity without enlargement");
    int overflow_checks = 0;
    try { layout::product(std::numeric_limits<size_t>::max(), 2); }
    catch (const std::overflow_error&) { ++overflow_checks; }
    try { layout::take_bytes(std::numeric_limits<uint64_t>::max(), 4); }
    catch (const std::overflow_error&) { ++overflow_checks; }
    try { layout::region_layout(std::numeric_limits<uint64_t>::max(), 1, 1, 64, true); }
    catch (const std::overflow_error&) { ++overflow_checks; }
    uint64_t used = std::numeric_limits<uint64_t>::max(), offset = 0;
    try { layout::advance(1, used, false, 0, offset); }
    catch (const std::overflow_error&) { ++overflow_checks; }
    check(overflow_checks == 4 && used == std::numeric_limits<uint64_t>::max(), "layout overflow guards");
}
void head_upload_known_fixture() {
    using Head = strata::core::HeadUploadOwner;
    for (int allocation_fault : {0, 1, 2, 3, 4}) {
        Head owner;
        auto source = std::make_shared<const Head::Payload>(16, uint8_t{0x37});
        int allocations = 0, frees = 0, uploads = 0, conversions = 0;
        auto allocate = [&](void** pointer, size_t bytes) {
            ++allocations;
            if (allocation_fault == allocations) return false;
            if (allocation_fault == allocations + 2) throw std::bad_alloc();
            *pointer = new uint8_t[bytes]; return true;
        };
        auto free = [&](void* pointer) { ++frees; delete[] static_cast<uint8_t*>(pointer); };
        auto upload = [&](void* dst, const void* src, size_t bytes) {
            ++uploads; std::memcpy(dst, src, bytes); return true;
        };
        auto convert = [&](void*) { ++conversions; return true; };
        bool success = false;
        try { success = owner.stage(source, source->data(), 16, 8, allocate, free, upload, convert); }
        catch (const std::bad_alloc&) { check(allocation_fault >= 3, "unexpected head allocation throw"); }
        if (allocation_fault) {
            check(!success && !owner.uncertain() && !owner.weights() && !owner.scratch() && !owner.source(),
                  "head no-submission allocation failure retained/poisoned owner");
            check(uploads == 0 && conversions == 0 && frees == (allocation_fault == 2 || allocation_fault == 4),
                  "head allocation failure unsafe cleanup/submission");
            const int original_fault = allocation_fault;
            allocation_fault = 0; // same owner can retry a known no-submission failure
            success = owner.stage(source, source->data(), 16, 8, allocate, free, upload, convert);
            allocation_fault = original_fault;
        }
        check(success && !owner.uncertain() && owner.weights() && owner.scratch() && !owner.source(),
              "head successful staged ownership incomplete");
        const int calls = allocations;
        check(!owner.stage(source, source->data(), 16, 8, allocate, free, upload, convert) && allocations == calls,
              "head live owner accepted duplicate upload");
        check(owner.clear(free) && !owner.weights() && !owner.scratch(), "head known-complete cleanup failed");
    }
    // Same metadata/size but changed file-like bytes cannot replace a leased head.
    std::vector<uint8_t> file_bytes(16);
    for (size_t i = 0; i < file_bytes.size(); ++i) file_bytes[i] = static_cast<uint8_t>(i * 19 + 7);
    const auto expected = file_bytes;
    auto original = Head::snapshot(file_bytes.data(), file_bytes.size());
    file_bytes.assign(16, 0x91);
    Head owner;
    auto allocate = [](void** pointer, size_t bytes) { *pointer = new uint8_t[bytes]; return true; };
    auto free = [](void* pointer) { delete[] static_cast<uint8_t*>(pointer); };
    auto upload = [](void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); return true; };
    for (int cycle = 0; cycle < 2; ++cycle) {
        check(owner.stage(original, original->data(), 16, 8, allocate, free, upload,
                          [](void*) { return true; }), "immutable head initial/reload upload failed");
        check(std::memcmp(owner.weights(), expected.data(), expected.size()) == 0 && *original == expected,
              "head reload used changed same-metadata source");
        check(owner.clear(free), "immutable head release failed");
    }
}
#ifdef __linux__
void head_upload_destructor_fixture() {
    const pid_t child = fork();
    check(child >= 0, "fork head destructor fixture failed");
    if (child == 0) {
        std::set_terminate([] { _exit(77); });
        {
            strata::core::HeadUploadOwner owner;
            auto source = std::make_shared<const strata::core::HeadUploadOwner::Payload>(16, uint8_t{0x37});
            owner.stage(source, source->data(), 16, 8,
                [](void** pointer, size_t bytes) { *pointer = new uint8_t[bytes]; return true; },
                [](void*) { _exit(78); }, // freeing an unknown allocation is a failure
                [](void*, const void*, size_t) { return false; },
                [](void*) { return true; });
        }
        _exit(79); // an unknown owner must never complete destruction
    }
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    check(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 77,
          "head unknown destructor failed to terminate before cleanup");
}

[[noreturn]] void head_upload_unknown_fixture(int fault) {
    using Head = strata::core::HeadUploadOwner;
    // This fixture exits its child without destructors, matching retained owners.
    auto* owner = new Head;
    auto source = std::make_shared<const Head::Payload>(16, uint8_t{0x37});
    std::weak_ptr<const Head::Payload> retained = source;
    const void* source_address = source->data();
    int allocations = 0, frees = 0, uploads = 0, conversions = 0;
    auto allocate = [&](void** pointer, size_t bytes) {
        ++allocations; *pointer = new uint8_t[bytes]; return true;
    };
    auto free = [&](void* pointer) { ++frees; delete[] static_cast<uint8_t*>(pointer); };
    auto upload = [&](void*, const void*, size_t) {
        ++uploads;
        if (fault == 0) return false;
        if (fault == 1) throw std::runtime_error("injected upload submission/wait");
        return true;
    };
    auto convert = [&](void*) {
        ++conversions;
        if (fault == 2) return false;
        if (fault == 3) throw std::runtime_error("injected conversion submission");
        throw std::runtime_error("injected conversion completion");
    };
    bool success = false, threw = false;
    try { success = owner->stage(source, source_address, 16, 8, allocate, free, upload, convert); }
    catch (const std::runtime_error&) { threw = true; }
    check(threw == (fault == 1 || fault >= 3), "head unknown-completion fault exception mismatch");
    source.reset();
    check(!success && owner->uncertain() && owner->weights() && owner->scratch() &&
          owner->bytes() == 16 && owner->scratch_bytes() == 8 && owner->source() == source_address &&
          !retained.expired() && frees == 0, "head unknown completion lost allocations/source");
    const int calls = allocations;
    check(!owner->clear(free) && frees == 0, "head unknown completion freed pending DMA");
    auto retry_source = retained.lock();
    check(!owner->stage(retry_source, source_address, 16, 8, allocate, free, upload, convert) &&
          allocations == calls && uploads == 1 && conversions == (fault >= 2),
          "head unknown completion retried callbacks");
    _exit(0); // all assertions passed; OS tears down the intentionally retained owner
}
[[noreturn]] void head_release_unknown_fixture(int fail_at, bool after_release) {
    using Head = strata::core::HeadUploadOwner;
    auto* owner = new Head;
    auto source = std::make_shared<const Head::Payload>(16, uint8_t{0x37});
    auto allocate = [](void** pointer, size_t bytes) { *pointer = new uint8_t[bytes]; return true; };
    auto known_free = [](void* pointer) { delete[] static_cast<uint8_t*>(pointer); };
    check(owner->stage(source, source->data(), 16, 8, allocate, known_free,
                      [](void*, const void*, size_t) { return true; }, [](void*) { return true; }),
          "head release setup failed");
    void* const weights = owner->weights();
    void* const scratch = owner->scratch();
    int frees = 0;
    auto release = [&](void* pointer) {
        ++frees;
        if (frees == fail_at) {
            if (after_release) known_free(pointer);
            throw std::runtime_error("injected uncertain release");
        }
        known_free(pointer);
    };
    bool threw = false;
    try { owner->clear(release); }
    catch (const std::runtime_error&) { threw = true; }
    check(threw && owner->uncertain() && frees == fail_at && owner->weights() == weights &&
          owner->scratch() == (fail_at == 1 ? scratch : nullptr),
          "head release failure lost uncertain pointer or kept confirmed freed pointer");
    check(!owner->clear(release) && frees == fail_at, "head uncertain release retried free");
    check(!owner->stage(source, source->data(), 16, 8, allocate, release,
                       [](void*, const void*, size_t) { return true; }, [](void*) { return true; }),
          "head uncertain release accepted retry");
    _exit(0); // all assertions passed; no uncertain owner destructor may run
}
#endif

class Mock final : public Backend {
public:
    enum class Fault { None, Caps, Reserve, Physical, Attach, Detach, Write, Read, Release, Drain, Corrupt };
    Fault fault = Fault::None;
    int nth = 1, calls = 0, destroyed = 0, unsafe_destroy = 0, released = 0;
    int detach_attempts = 0, release_attempts = 0;
    uint64_t free_bytes = uint64_t{8} << 30;
    uint64_t recommended = uint64_t{2} << 20, device_minimum = 65536, context_minimum = 65536;
    bool misaligned_reserve = false;
    std::vector<uint8_t> storage;
    std::map<void*, bool> handles;
    uint8_t* pending_read = nullptr;
    bool force_detach_failure = false;
    // Two synthetic post-error states; neither claims actual provider semantics.
    bool failed_attach_unmapped = false, failed_detach_unmapped = false;
    bool hit(Fault kind) { return fault == kind && ++calls == nth; }
    void inject(Fault kind, int index = 1) { fault = kind; nth = index; calls = 0; }
    bool caps(Caps& out, std::string& error) noexcept override {
        if (hit(Fault::Caps)) { error = "injected caps"; return false; }
        out = {true, true, recommended, uint64_t{64} << 20, free_bytes, device_minimum, context_minimum};
        return true;
    }
    void* reserve(uint64_t bytes, std::string& error) noexcept override {
        if (hit(Fault::Reserve)) { error = "injected reserve"; return nullptr; }
        const uint64_t align = std::max(context_minimum, device_minimum);
        storage.resize(static_cast<size_t>(bytes + align), 0xA5);
        const auto raw = reinterpret_cast<uintptr_t>(storage.data());
        return reinterpret_cast<void*>(raw + (align - raw % align) % align + (misaligned_reserve ? 1 : 0));
    }
    bool release(void*, uint64_t, std::string& error) noexcept override {
        ++release_attempts;
        if (hit(Fault::Release)) { error = "injected release"; return false; }
        if (!handles.empty()) { error = "live physical handles at release"; return false; }
        ++released;
        return true;
    }
    void* physical(uint64_t, std::string& error) noexcept override {
        if (hit(Fault::Physical)) { error = "injected physical"; return nullptr; }
        try {
            auto handle = std::make_unique<int>(7);
            handles.emplace(handle.get(), false);
            return handle.release();
        } catch (const std::bad_alloc&) { error = "mock physical allocation refused"; return nullptr; }
    }
    bool attach(void* handle, void* address, uint64_t, std::string& error) noexcept override {
        addresses[address] = handle;
        handles[handle] = true; // model an error after the backend may have attached
        if (hit(Fault::Attach)) {
            if (failed_attach_unmapped) { handles[handle] = false; addresses.erase(address); }
            error = "injected uncertain attach"; return false;
        }
        return true;
    }
    bool detach(void* address, uint64_t, std::string& error) noexcept override {
        ++detach_attempts;
        if (force_detach_failure || hit(Fault::Detach)) {
            if (failed_detach_unmapped) {
                auto selected = addresses.find(address);
                if (selected != addresses.end()) handles[selected->second] = false;
            }
            error = "injected detach"; return false;
        }
        auto selected = addresses.find(address);
        if (selected == addresses.end()) { error = "detach address has no owned mapping"; return false; }
        handles[selected->second] = false;
        return true;
    }
    std::map<void*, void*> addresses;
    void destroy_physical(void* handle) noexcept override {
        if (handles[handle]) ++unsafe_destroy;
        handles.erase(handle);
        for (auto it = addresses.begin(); it != addresses.end();) {
            if (it->second == handle) it = addresses.erase(it); else ++it;
        }
        delete static_cast<int*>(handle);
        ++destroyed;
    }
    bool drain(std::string& error) noexcept override {
        if (hit(Fault::Drain)) { error = "injected drain"; return false; }
        if (pending_read) { pending_read[0] = 0x55; pending_read = nullptr; }
        return true;
    }
    bool write(void* out, const void* in, uint64_t bytes, std::string& error) noexcept override {
        std::memcpy(out, in, static_cast<size_t>(bytes));
        if (hit(Fault::Write)) { error = "injected write"; return false; }
        return true;
    }
    bool read(void* out, const void* in, uint64_t bytes, std::string& error) noexcept override {
        if (hit(Fault::Read)) {
            pending_read = static_cast<uint8_t*>(out); // actual owner scratch must survive failed D2H
            error = "injected pending read";
            return false;
        }
        std::memcpy(out, in, static_cast<size_t>(bytes));
        if (hit(Fault::Corrupt)) static_cast<uint8_t*>(out)[0] ^= 1;
        return true;
    }
};
const Owner::Quiesce quiet = [](std::string&) { return true; }; // CPU backend only, no claimed GPU proof
const Owner::Publish publish = [](const int32_t*, size_t, std::string&) { return true; };
void finish(Owner& owner, const std::shared_ptr<Mock>& backend, std::string& error) {
    backend->inject(Mock::Fault::None); // only known-state failures allow fixture cleanup
    backend->force_detach_failure = false;
    check(owner.close(quiet, error), "mock checked close failed");
    check(backend->handles.empty() && backend->unsafe_destroy == 0 && !owner.open(), "owner leaked/deleted mapped handle");
}
#ifdef __linux__
void retained_unknown(Owner& owner, const std::shared_ptr<Mock>& backend, std::string& error) {
    const auto snapshots = owner.mapping_snapshot();
    check(std::any_of(snapshots.begin(), snapshots.end(), [](const Owner::Mapping& map) { return map.uncertain; }),
          "failed map/unmap did not retain unknown state");
    const auto handles = backend->handles;
    const uint64_t bytes = owner.retained_physical_bytes();
    void* const base = owner.base();
    const int detaches = backend->detach_attempts, destroys = backend->destroyed, releases = backend->release_attempts;
    // Removing synthetic faults must not turn production uncertainty into proof.
    backend->inject(Mock::Fault::None);
    backend->force_detach_failure = false;
    int proofs = 0, publications = 0;
    const Owner::Quiesce proof = [&](std::string&) { ++proofs; return true; };
    const Owner::Publish publication = [&](const int32_t*, size_t, std::string&) { ++publications; return true; };
    for (int repeat = 0; repeat < 3; ++repeat) {
        check(!owner.close(proof, error) && owner.open() && owner.poisoned(), "unknown close recovered/promoted PASS");
        check(!owner.map_base(proof, error) && !owner.map_extras(proof, error) &&
              !owner.unmap_extras(proof, error) && !owner.publish(publication, error), "poisoned owner accepted request");
        check(!owner.readers_ready() && !owner.device_slot(0) && owner.resident_slot(0, 0) == -1,
              "unknown owner published readers");
        check(owner.base() == base && owner.retained_physical_bytes() == bytes && backend->handles == handles,
              "unknown owner lost native ownership");
        check(backend->detach_attempts == detaches && backend->destroyed == destroys &&
              backend->release_attempts == releases && backend->unsafe_destroy == 0,
              "unknown close/request retried unmap/destroy/free");
    }
    check(proofs == 0 && publications == 0, "unknown owner called request callbacks");
}
void isolated_unknown(const std::function<void()>& fixture) {
    const pid_t child = fork();
    check(child >= 0, "fork unknown-state fixture failed");
    if (child == 0) {
        try {
            fixture();
            // Unknown native ownership intentionally survives. No owner or backend
            // destructors run; OS process teardown is the only cleanup here.
            _exit(0);
        } catch (const std::exception& exception) {
            std::fprintf(stderr, "FAIL unknown-state child: %s\n", exception.what());
            _exit(1);
        }
    }
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    check(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0, "unknown-state subprocess failed");
}
#endif
}
int main() {
    try {
        gated_region_layout_fixture();
        head_upload_known_fixture();
#ifdef __linux__
        for (int fail_at : {1, 2})
            for (bool after_release : {false, true})
                isolated_unknown([=] { head_release_unknown_fixture(fail_at, after_release); });
        for (int fault = 0; fault < 5; ++fault)
            isolated_unknown([fault] { head_upload_unknown_fixture(fault); });
        head_upload_destructor_fixture();
#else
        std::puts("SKIP head unknown-completion subprocess fixtures: Linux fork required");
#endif
        std::string error;
        std::vector<uint8_t> bytes(1024);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i * 19 + 7);
        {
            strata::core::PhaseFreshState fresh;
            fresh.capacities_zero = fresh.healthy_baseline = fresh.session_geometry = true;
            check(strata::core::phase_fresh_admission(fresh), "stock omitted ckpt bit cannot admit actual fresh state");
            for (bool strata::core::PhaseFreshState::* member : {
                     &strata::core::PhaseFreshState::live, &strata::core::PhaseFreshState::checkpoint,
                     &strata::core::PhaseFreshState::parked, &strata::core::PhaseFreshState::reuse,
                     &strata::core::PhaseFreshState::slot, &strata::core::PhaseFreshState::incoming,
                     &strata::core::PhaseFreshState::from_live, &strata::core::PhaseFreshState::split,
                     &strata::core::PhaseFreshState::failed}) {
                auto dirty = fresh;
                dirty.*member = true;
                check(!strata::core::phase_fresh_admission(dirty), "actual reusable/fault state admitted");
            }
            fresh.resume = 1;
            check(!strata::core::phase_fresh_admission(fresh), "resume admitted");
            fresh.resume = 0; fresh.turn_at = 0;
            check(!strata::core::phase_fresh_admission(fresh), "checkpoint split admitted");
            fresh.turn_at = -1; fresh.capacities_zero = false;
            check(!strata::core::phase_fresh_admission(fresh), "nonzero cache capacities admitted");
        }
        {
            auto backend = std::make_shared<Mock>();
            backend->context_minimum = uint64_t{4} << 20;
            Owner owner(backend);
            check(owner.reserve({1024}, 8, uint64_t{16} << 20, error), "distinct context granularity refused");
            for (const auto& map : owner.mapping_snapshot())
                check(map.offset % backend->context_minimum == 0 && map.bytes % backend->context_minimum == 0,
                      "context/device extent geometry not reconciled");
            finish(owner, backend, error);
        }
        for (int geometry = 0; geometry < 4; ++geometry) {
            auto backend = std::make_shared<Mock>();
            if (geometry == 0) backend->device_minimum = 3; // incompatible recommended/minimum
            if (geometry == 1) backend->context_minimum = 0; // missing query
            if (geometry == 2) backend->context_minimum = UINT64_MAX; // checked LCM overflow
            if (geometry == 3) backend->misaligned_reserve = true;
            Owner owner(backend);
            check(!owner.reserve({1024}, 8, uint64_t{16} << 20, error) && owner.poisoned(),
                  "invalid/overflow/misaligned geometry admitted");
            check(backend->handles.empty(), "invalid geometry allocated physical memory");
            finish(owner, backend, error);
        }
        {
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve({1024, 1024}, 8, uint64_t{16} << 20, error), "reserve");
            const uint64_t* offsets = owner.capacity_offsets();
            const int32_t* table = owner.host_residency();
            check(owner.capacity() == 10 && owner.mapped_slots() == 0 && !owner.device_slot(0), "unmapped reserve exposed");
            check(owner.map_base(quiet, error) && owner.mapped_slots() == 6 && !owner.device_slot(6), "base membership");
            void* old = owner.device_slot(3);
            check(offsets[3] == 3072 && offsets[6] == (uint64_t{2} << 20) && owner.slot_stride(5) == 1024,
                  "old geometry moved");
            check(owner.fill_checked(0, 0, 0, bytes.data(), bytes.size(), error), "baseline fill");
            check(owner.resident_slot(0, 0) == -1, "resident published before checked upload");
            check(owner.publish(publish, error) && owner.resident_slot(0, 0) == 0, "baseline publication");
            for (int request = 0; request < 2; ++request) {
                check(owner.map_extras(quiet, error) && owner.mapped_slots() == 10, "extras map");
                check(owner.device_slot(3) == old && owner.capacity_offsets() == offsets && owner.host_residency() == table,
                      "reader pointer changed across phase");
                check(owner.fill_checked(0, 2, 6, bytes.data(), bytes.size(), error), "extra fill");
                check(owner.resident_slot(0, 2) == -1, "extra published before copy/check/upload");
                check(owner.publish(publish, error) && owner.resident_slot(0, 2) == 6, "extra publish");
                check(owner.invalidate_extras(publish, error), "extra invalidation");
                check(owner.resident_slot(0, 2) == -1 && owner.resident_slot(0, 0) == 0, "base residency lost");
                check(owner.unmap_extras(quiet, error) && owner.mapped_slots() == 6 && !owner.device_slot(6), "extra detach");
                check(std::memcmp(owner.device_slot(0), bytes.data(), bytes.size()) == 0, "baseline bytes changed");
            }
            finish(owner, backend, error);
        }
        for (const auto fault : {Mock::Fault::Caps, Mock::Fault::Reserve}) {
            auto backend = std::make_shared<Mock>();
            backend->inject(fault);
            Owner owner(backend);
            check(!owner.reserve({1024}, 8, uint64_t{16} << 20, error) && !owner.open() && backend->handles.empty(),
                  "reserve/caps fault leaked or promoted PASS");
        }
        {
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve({1024}, 8, uint64_t{16} << 20, error) && owner.map_base(quiet, error) &&
                  owner.fill_checked(0, 0, 0, bytes.data(), bytes.size(), error), "publish failure setup");
            const Owner::Publish rejected = [](const int32_t*, size_t, std::string& message) {
                message = "injected upload failure"; return false;
            };
            check(!owner.publish(rejected, error) && owner.host_residency()[0] == -1 && owner.poisoned(),
                  "failed upload published host residency");
            finish(owner, backend, error);
        }
        for (const auto fault : {Mock::Fault::Physical}) {
            for (int index = 1; index <= 2; ++index) {
                auto backend = std::make_shared<Mock>();
                Owner owner(backend);
                check(owner.reserve({1024, 1024}, 8, uint64_t{16} << 20, error) && owner.map_base(quiet, error), "rollback setup");
                backend->inject(fault, index);
                check(!owner.map_extras(quiet, error) && owner.poisoned(), "map fault promoted PASS");
                check(owner.retained_physical_bytes() == (uint64_t{2} << 20), "rollback failed to preserve base only");
                finish(owner, backend, error);
            }
        }
#ifdef __linux__
        for (const bool unmapped : {false, true}) {
            for (int index = 0; index <= 2; ++index) isolated_unknown([&] {
                auto backend = std::make_shared<Mock>();
                backend->failed_attach_unmapped = unmapped;
                auto* owner = new Owner(backend); // retained until child process teardown
                check(owner->reserve({1024, 1024}, 8, uint64_t{16} << 20, error), "unknown attach reserve");
                if (index) {
                    check(owner->map_base(quiet, error) &&
                          owner->fill_checked(0, 0, 0, bytes.data(), bytes.size(), error) &&
                          owner->publish(publish, error) && owner->readers_ready(), "unknown attach base setup");
                }
                backend->inject(Mock::Fault::Attach, index ? index : 1);
                check(!(index ? owner->map_extras(quiet, error) : owner->map_base(quiet, error)) && owner->poisoned(),
                      "attach fault promoted PASS");
                check(error == "injected uncertain attach", "original attach outcome lost");
                const uint64_t retained = static_cast<uint64_t>(index + 1) * (uint64_t{2} << 20);
                check(owner->retained_physical_bytes() == retained && backend->destroyed == 0 &&
                      backend->detach_attempts == 0 && backend->release_attempts == 0,
                      "unknown attach attempted rollback or lost known earlier maps");
                check(owner->mapping_snapshot()[index].uncertain, "failed extent not uncertain");
                retained_unknown(*owner, backend, error);
                std::weak_ptr<Mock> lifetime = backend;
                backend.reset();
                check(!lifetime.expired(), "unknown owner lost backend/context lifetime");
            });
        }
        for (const bool unmapped : {false, true}) isolated_unknown([&] {
            auto backend = std::make_shared<Mock>();
            backend->failed_detach_unmapped = unmapped;
            auto* retained = new Owner(backend);
            Owner& owner = *retained;
            check(owner.reserve({1024}, 8, uint64_t{16} << 20, error) && owner.map_base(quiet, error) &&
                  owner.map_extras(quiet, error), "detach setup");
            backend->inject(Mock::Fault::Detach);
            const int destroyed = backend->destroyed;
            check(!owner.unmap_extras(quiet, error) && owner.poisoned(), "detach failure promoted PASS");
            check(error == "injected detach", "original detach outcome lost");
            check(backend->destroyed == destroyed && owner.retained_physical_bytes() == (uint64_t{4} << 20),
                  "failed detach deleted physical handle");
            check(!owner.device_slot(0) && owner.mapping_snapshot()[1].uncertain,
                  "poisoned/uncertain owner remained readable or was reported complete");
            retained_unknown(owner, backend, error);
        });
        isolated_unknown([&] {
            auto backend = std::make_shared<Mock>();
            auto* retained = new Owner(backend);
            Owner& owner = *retained;
            check(owner.reserve({1024, 1024}, 8, uint64_t{16} << 20, error) && owner.map_base(quiet, error),
                  "rollback-detach failure setup");
            backend->inject(Mock::Fault::Physical, 2);
            backend->force_detach_failure = true;
            check(!owner.map_extras(quiet, error) && owner.retained_physical_bytes() == (uint64_t{4} << 20) &&
                  backend->destroyed == 0, "failed rollback deleted a possibly mapped physical handle");
            check(error == "injected detach" && owner.mapping_snapshot()[1].uncertain,
                  "rollback detach failure outcome/state lost");
            retained_unknown(owner, backend, error);
        });
        for (const bool unmapped : {false, true}) isolated_unknown([&] {
            auto backend = std::make_shared<Mock>();
            backend->failed_detach_unmapped = unmapped;
            auto* owner = new Owner(backend);
            check(owner->reserve({1024}, 8, uint64_t{16} << 20, error) && owner->map_base(quiet, error),
                  "close detach setup");
            backend->inject(Mock::Fault::Detach);
            check(!owner->close(quiet, error) && error == "injected detach", "close detach failure promoted/lost");
            check(backend->detach_attempts == 1 && backend->destroyed == 0 && backend->release_attempts == 0,
                  "failed close detached/freed beyond unknown range");
            retained_unknown(*owner, backend, error);
        });
#else
        std::puts("SKIP unknown-mapping subprocess fixtures: Linux fork required");
#endif
        for (const auto fault : {Mock::Fault::Write, Mock::Fault::Read, Mock::Fault::Corrupt}) {
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve({1024}, 8, uint64_t{16} << 20, error) && owner.map_base(quiet, error), "fill fault setup");
            backend->inject(fault);
            check(!owner.fill_checked(0, 0, 0, bytes.data(), bytes.size(), error) && owner.poisoned(), "bad fill promoted");
            check(owner.host_residency()[0] == -1, "failed bytes published");
            finish(owner, backend, error); // ASan observes pending read into retained owner scratch.
        }
        for (const auto fault : {Mock::Fault::Release, Mock::Fault::Drain}) {
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve({1024}, 8, uint64_t{16} << 20, error) && owner.map_base(quiet, error), "close fault setup");
            backend->inject(fault);
            check(!owner.close(quiet, error) && owner.open(), "failed close lost virtual owner");
            finish(owner, backend, error);
        }
        {
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve({1024}, 8, uint64_t{16} << 20, error), "budget setup");
            backend->free_bytes = Owner::guard_bytes + (uint64_t{2} << 20) - 1;
            check(!owner.map_base(quiet, error) && owner.retained_physical_bytes() == 0, "fresh guard ignored");
            finish(owner, backend, error);
        }
        {
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve({1024}, 8, uint64_t{16} << 20, error), "missing proof setup");
            check(!owner.map_base({}, error), "missing quiescence fabricated");
            finish(owner, backend, error);
        }
        {
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve({1024}, 8, uint64_t{16} << 20, error) && owner.map_base(quiet, error) &&
                  owner.map_extras(quiet, error) && owner.fill_checked(0, 0, 3, bytes.data(), bytes.size(), error) &&
                  owner.publish(publish, error), "published unmap setup");
            check(!owner.unmap_extras(quiet, error) && backend->destroyed == 0, "unmapped published resident");
            finish(owner, backend, error);
        }
        {
            using strata::core::phase_cache_opt_in;
#ifdef __linux__
            const bool frozen = strata::core::phase_cache_requested();
            check(setenv("STRATA_PHASE_DECODE_CACHE", frozen ? "0" : "1", 1) == 0 &&
                  strata::core::phase_cache_requested() == frozen, "phase opt-in changed after freezing");
#endif
            check(!phase_cache_opt_in(nullptr) && phase_cache_opt_in("1"), "exact phase opt-in");
            for (const char* value : {"0", "", "01", "1 ", " 1", "true", "+1", "2"})
                check(!phase_cache_opt_in(value), "non-exact flag enabled phase");
            for (const std::string command : {"SAVE x", "RESTORE x", "GENI 1 x 2", "BGEN 0 1 2", "VRAM", "BSTOP 0", ""})
                check(!strata::core::phase_request_kind(command), "incompatible command admitted");
            check(strata::core::phase_request_kind("QUIT") && strata::core::phase_request_kind("GEN 128 ckpt=0 1,2"),
                  "fresh GEN/QUIT command refused");
        }
        {
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve(std::vector<int64_t>(48, 1024), 512, uint64_t{128} << 20, error, 16),
                  "production 48-layer/16-extra geometry refused");
            check(owner.capacity() == 912 && owner.map_base(quiet, error) && owner.mapped_slots() == 144,
                  "production capacity/base membership wrong");
            const auto* offsets = owner.capacity_offsets();
            void* old = owner.device_slot(141);
            check(offsets[141] == 144384 && owner.slot_stride(143) == 1024, "old last-layer slots moved");
            check(owner.map_extras(quiet, error) && owner.mapped_slots() == 912 &&
                  owner.layer_slots(47).back() == 911 && owner.device_slot(141) == old,
                  "production append layout/old addresses wrong");
            finish(owner, backend, error);
        }
        {
            // Exact production orchestration, actual owner, synthetic workspace callback.
            // No enum here is offered as a GPU-completion or SessionState proof.
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve({1024}, 32, uint64_t{16} << 20, error, 16) && owner.map_base(quiet, error),
                  "sixteen-extra caller setup");
            for (uint32_t row = 0; row < 3; ++row)
                check(owner.fill_checked(0, row, row, bytes.data(), bytes.size(), error), "caller base fill");
            check(owner.publish(publish, error) && owner.readers_ready(), "caller baseline publish");
            const auto* offsets = owner.capacity_offsets();
            const auto* host_table = owner.host_residency();
            void* old = owner.device_slot(0);
            bool workspace = true;
            std::vector<int> calls;
            strata::core::PhaseServeCycle cycle;
            cycle.quiesce = [&](std::string& e) { calls.push_back(1); return backend->drain(e); };
            cycle.retire_readers = [&](std::string&) { calls.push_back(6); return true; };
            cycle.retire_workspace = [&](std::string&) { calls.push_back(2); workspace = false; return true; };
            cycle.grow_fill_publish = [&](std::string& e) {
                calls.push_back(3);
                if (workspace || !owner.map_extras(quiet, e)) return false;
                for (uint32_t row = 3; row < 19; ++row)
                    if (!owner.fill_checked(0, row, row, bytes.data(), bytes.size(), e)) return false;
                return owner.publish(publish, e);
            };
            cycle.invalidate_detach = [&](std::string& e) {
                calls.push_back(4);
                return owner.invalidate_extras(publish, e) && owner.unmap_extras(quiet, e);
            };
            cycle.reinit_workspace = [&](std::string&) {
                calls.push_back(5); workspace = true; return owner.mapped_slots() == 3;
            };
            for (int request = 0; request < 2; ++request) {
                check(cycle.begin_gen(error) && workspace && owner.mapped_slots() == 3, "fresh GEN reinit failed");
                check(cycle.successful_prompt(error) && !workspace && owner.mapped_slots() == 19 && owner.readers_ready(),
                      "actual caller retire/map/fill/publish failed");
                check(owner.device_slot(0) == old && owner.capacity_offsets() == offsets && owner.host_residency() == host_table,
                      "caller changed base/full table/host map pointers");
                check(cycle.complete_gen(true, error), "successful GEN did not complete");
            }
            check(cycle.requests() == 2 && calls == std::vector<int>({1, 2, 3, 1, 6, 4, 5, 1, 2, 3}),
                  "two-GEN actual caller operation order differs");
            finish(owner, backend, error);
        }
        {
            auto backend = std::make_shared<Mock>();
            Owner owner(backend);
            check(owner.reserve({1024}, 32, uint64_t{16} << 20, error, 16) && owner.map_base(quiet, error), "caller fault setup");
            strata::core::PhaseServeCycle cycle;
            cycle.quiesce = [&](std::string& e) { return backend->drain(e); };
            cycle.retire_workspace = [](std::string&) { return true; };
            cycle.grow_fill_publish = [&](std::string& e) { return owner.map_extras(quiet, e); };
            backend->inject(Mock::Fault::Physical);
            check(cycle.begin_gen(error) && !cycle.successful_prompt(error) && !cycle.begin_gen(error),
                  "caller map failure silently retried/fell back");
            finish(owner, backend, error);
        }
        for (int failed_operation = 0; failed_operation < 3; ++failed_operation) {
            strata::core::PhaseServeCycle cycle;
            cycle.quiesce = cycle.retire_workspace = cycle.grow_fill_publish = [](std::string&) { return true; };
            check(cycle.begin_gen(error) && cycle.successful_prompt(error) && cycle.complete_gen(true, error),
                  "first GEN fault setup");
            int mutation = 0;
            cycle.retire_readers = [&](std::string&) { ++mutation; return failed_operation != 0; };
            cycle.invalidate_detach = [&](std::string&) { ++mutation; return failed_operation != 1; };
            cycle.reinit_workspace = [&](std::string&) { ++mutation; return failed_operation != 2; };
            check(!cycle.begin_gen(error) && mutation == failed_operation + 1 && cycle.failed(),
                  "second GEN did not stop at retirement fault");
            const int before = mutation;
            check(!cycle.begin_gen(error) && mutation == before, "second GEN retirement fault retried");
        }
        {
            strata::core::PhaseServeCycle cycle;
            cycle.quiesce = cycle.retire_workspace = cycle.grow_fill_publish = [](std::string&) { return true; };
            check(cycle.begin_gen(error) && cycle.successful_prompt(error) &&
                  !cycle.complete_gen(false, error) && !cycle.begin_gen(error),
                  "cancel/nonfinite decode allowed another GEN");
        }
        {
            strata::core::PhaseServeCycle cycle;
            int calls = 0;
            cycle.release_head = [&](std::string&) { ++calls; return true; };
            check(!cycle.begin_gen(error) && calls == 0 && cycle.failed(), "incomplete head lease admitted");
        }
        // Lifetime fixture: the head stays absent through prompt work and workspace reset.
        // Callback failures poison the cycle; no reload or later transition may infer completion.
        for (int fault = -1; fault < 7; ++fault) {
            strata::core::PhaseServeCycle cycle;
            bool head = true, workspace = false, drained = false, retired = false;
            int attempt = 0;
            auto pass = [&]() { return attempt++ != fault; };
            cycle.quiesce = [&](std::string&) { if (!pass()) return false; drained = true; return true; };
            cycle.retire_readers = [&](std::string&) { if (!pass() || !drained) return false; retired = true; return true; };
            cycle.release_head = [&](std::string&) {
                if (!pass() || !drained || !retired || !head || workspace) return false;
                head = false; drained = retired = false; return true;
            };
            cycle.reinit_workspace = [&](std::string&) {
                if (!pass() || head || workspace) return false;
                workspace = true; return true;
            };
            cycle.retire_workspace = [&](std::string&) {
                if (!pass() || head || !workspace || !drained) return false;
                workspace = false; return true;
            };
            cycle.reload_head = [&](std::string&) {
                if (!pass() || head || workspace || !drained) return false;
                head = true; return true;
            };
            cycle.grow_fill_publish = [&](std::string&) { return head && !workspace; };
            cycle.invalidate_detach = [](std::string&) { return true; };
            const bool begun = cycle.begin_gen(error);
            if (begun) check(!head && workspace, "head coexists with prompt workspace");
            const bool done = begun && cycle.successful_prompt(error);
            if (fault < 0) {
                check(done && head && !workspace && cycle.complete_gen(true, error), "head lease first GEN");
                check(cycle.begin_gen(error) && !head && workspace && cycle.successful_prompt(error) &&
                      head && !workspace && cycle.complete_gen(true, error), "head lease subsequent GEN");
            } else {
                check(!done && cycle.failed(), "head lease fault accepted");
                const int prior = attempt;
                check(!cycle.begin_gen(error) && attempt == prior, "head lease uncertain completion retried");
            }
        }
        {
            using strata::core::phase_prefill_capacity_admitted;
            check(phase_prefill_capacity_admitted(true, false, false, false, false, 8192), "default 8K denied");
            check(phase_prefill_capacity_admitted(true, true, true, true, false, 12288), "explicit 12K denied");
            check(!phase_prefill_capacity_admitted(false, true, true, true, false, 12288), "12K without phase admitted");
            check(!phase_prefill_capacity_admitted(true, false, true, true, false, 12288), "12K without opt-in admitted");
            check(!phase_prefill_capacity_admitted(true, true, true, true, false, 8192), "mixed 8K/12K capacity admitted");
            check(!phase_prefill_capacity_admitted(true, true, true, true, true, 12288), "auto 12K admitted");
            check(!phase_prefill_capacity_admitted(true, true, false, true, false, 12288), "12K without lease admitted");
            check(!phase_prefill_capacity_admitted(true, true, true, false, false, 12288), "12K without alias admitted");
            check(!phase_prefill_capacity_admitted(true, true, true, true, false,
                  std::numeric_limits<int64_t>::max()), "overflow-sized capacity admitted");
            uint64_t requested = std::numeric_limits<uint64_t>::max() - 3;
            check(!strata::core::phase_requested_add(4, requested) &&
                  requested == std::numeric_limits<uint64_t>::max() - 3, "peak overflow changed count");
            check(strata::core::phase_requested_add(3, requested) &&
                  requested == std::numeric_limits<uint64_t>::max(), "exact peak sum rejected");
        }
        std::puts("PASS actual phase owner CPU bookkeeping/fault contract");
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "FAIL phase owner CPU fixture: %s\n", exception.what());
        return 1;
    }
}
