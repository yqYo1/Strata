// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Manually owned, runtime-dependent fixture; no unguarded GPU CTest.
#include "strata/core/phase_cache_owner.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/runtime.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void progress(const char* text) { std::puts(text); std::fflush(stdout); }
int full_lease(const std::shared_ptr<strata::core::PhaseCacheBackend>& backend,
               const strata::core::PhaseCacheBackend::Caps& caps) {
    using Owner = strata::core::PhaseCacheOwner;
    constexpr uint64_t limit = uint64_t{16} << 20, unit = uint64_t{2} << 20;
    uint64_t gran = caps.granularity;
    if (!caps.device_minimum || !caps.context_minimum || gran % caps.device_minimum) return 77;
    for (const uint64_t alignment : {unit, caps.device_minimum, caps.context_minimum, uint64_t{256}}) {
        const uint64_t factor = alignment / std::gcd(gran, alignment);
        if (gran > std::numeric_limits<uint64_t>::max() / factor) return 77;
        gran *= factor;
    }
    if (gran > (limit - 65536) / 3 || gran > caps.maximum || caps.free < Owner::guard_bytes + gran * 3) {
        progress("SKIP real full-lease geometry exceeds finite small fixture admission");
        return 77;
    }
    Owner owner(backend);
    std::string error;
    std::vector<uint8_t> padding(static_cast<size_t>(gran), 0xA5), readback(static_cast<size_t>(gran));
    std::vector<uint8_t> expected(static_cast<size_t>(gran), 0xA5);
    std::unique_ptr<strata::core::PinnedArena> arena;
    void* d_res = nullptr;
    std::array<int32_t, 16> table{};
    const Owner::Quiesce quiet = [&](std::string& message) { return backend->drain(message); };
    int result = 1;
    try {
        progress("BEGIN tiny two-layer real full physical lease");
        arena = std::make_unique<strata::core::PinnedArena>(4096, std::vector<uint64_t>{0, 4096}, 1024, true);
        auto* source = arena->at(0);
        check(arena->owned_host_range(source, 2048), "full-lease immutable source range unavailable");
        for (size_t i = 0; i < 2048; ++i) source[i] = static_cast<uint8_t>(i * 23 + i / 1024 + 9);
        d_res = strata::gpu::alloc_device(sizeof(table));
        check(d_res && strata::gpu::is_host_usm(source), "full-lease fixture allocation failed");
        check(owner.reserve({1024, 1024}, 8, limit - 65536, error), error);
        check(owner.capacity() == 10 && owner.reserved_bytes() == gran * 3, "full-lease bounded geometry");
        const auto* offsets = owner.capacity_offsets();
        const auto* host_table = owner.host_residency();
        void* const virtual_base = owner.base();
        const Owner::Publish upload = [&](const int32_t* rows, size_t count, std::string& message) {
            if (count != table.size()) { message = "full-lease residency geometry mismatch"; return false; }
            if (!strata::gpu::copy(d_res, rows, sizeof(table))) { message = strata::gpu::last_error(); return false; }
            if (!strata::gpu::copy(table.data(), d_res, sizeof(table))) { message = strata::gpu::last_error(); return false; }
            if (!std::equal(table.begin(), table.end(), rows)) { message = "full-lease device table readback mismatch"; return false; }
            return true;
        };
        const auto refill_base = [&] {
            check(!owner.readers_ready() && owner.mapped_slots() == 6, "baseline remap prematurely readable");
            check(strata::gpu::copy(owner.base(), padding.data(), padding.size()), strata::gpu::last_error());
            std::fill(expected.begin(), expected.end(), 0xA5);
            for (uint32_t layer = 0; layer < 2; ++layer) {
                for (uint32_t expert = 0; expert < 3; ++expert) {
                    const uint32_t slot = layer * 3 + expert;
                    check(owner.fill_checked(layer, expert, slot, source + static_cast<size_t>(layer) * 1024, 1024, error), error);
                    std::memcpy(expected.data() + offsets[slot], source + static_cast<size_t>(layer) * 1024, 1024);
                }
            }
            check(!owner.readers_ready() && owner.publish(upload, error), error);
            check(owner.readers_ready(), "baseline checked publication did not enable readers");
            check(strata::gpu::copy(readback.data(), owner.base(), readback.size()), strata::gpu::last_error());
            check(readback == expected, "baseline payload/padding readback mismatch");
        };
        check(owner.map_base(quiet, error), error);
        refill_base();
        for (int request = 0; request < 2; ++request) {
            progress("FULL LEASE extras fill/publish/readback");
            check(owner.map_extras(quiet, error) && owner.mapped_slots() == 10 && owner.mapped_bytes() == gran * 3,
                  error);
            for (uint32_t layer = 0; layer < 2; ++layer) {
                const uint32_t slot = 6 + layer * 2;
                check(strata::gpu::copy(owner.device_slot(slot), padding.data(), padding.size()), strata::gpu::last_error());
                check(owner.fill_checked(layer, 3, slot, source + static_cast<size_t>(layer) * 1024, 1024, error), error);
            }
            check(owner.publish(upload, error), error);
            for (uint32_t layer = 0; layer < 2; ++layer) {
                check(owner.resident_slot(layer, 3) == static_cast<int32_t>(6 + layer * 2), "extra layer residency mismatch");
                check(strata::gpu::copy(readback.data(), owner.device_slot(6 + layer * 2), readback.size()), strata::gpu::last_error());
                check(std::memcmp(readback.data(), source + static_cast<size_t>(layer) * 1024, 1024) == 0 &&
                      std::all_of(readback.begin() + 1024, readback.end(), [](uint8_t byte) { return byte == 0xA5; }),
                      "extra layer payload/padding mismatch");
            }
            progress("FULL LEASE checked empty table and all physical extents detached");
            check(owner.begin_prefill(quiet, upload, error), error);
            check(owner.prefill_leased() && !owner.readers_ready() && owner.mapped_slots() == 0 &&
                  owner.mapped_bytes() == 0 && owner.retained_physical_bytes() == 0,
                  "full lease retained physical backing/readers");
            check(owner.base() == virtual_base && owner.capacity_offsets() == offsets && owner.host_residency() == host_table,
                  "full lease changed retained virtual/table pointers");
            check(std::all_of(table.begin(), table.end(), [](int32_t slot) { return slot == -1; }) &&
                  std::all_of(host_table, host_table + table.size(), [](int32_t slot) { return slot == -1; }),
                  "full lease failed to invalidate the entire published table");
            for (uint32_t slot = 0; slot < owner.capacity(); ++slot)
                check(!owner.device_slot(slot), "unmapped full lease exposed a device slot");
            // Never dereference the retained virtual address while unmapped.
            progress("FULL LEASE baseline remap/refill/checked publication");
            check(owner.finish_prefill(quiet, error) && !owner.prefill_leased() && !owner.readers_ready(), error);
            check(owner.base() == virtual_base && owner.capacity_offsets() == offsets && owner.host_residency() == host_table,
                  "baseline restore changed retained pointers");
            refill_base();
        }
        result = 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "FAIL full physical lease fixture: %s\n", exception.what());
    }
    if (!owner.close(quiet, error) || !backend->drain(error)) {
        std::fprintf(stderr, "FAIL full-lease checked close; retaining all owners: %s\n", error.c_str());
        std::fflush(stderr);
        std::_Exit(1);
    }
    arena.reset();
    strata::gpu::free(d_res);
    if (result == 0) progress("PASS tiny actual full physical lease closed: two layers, two cycles (no model inference)");
    return result;
}
}
int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "--small-vmm") != 0 && std::strcmp(argv[1], "--small-full-lease") != 0)) return 2;
    using Owner = strata::core::PhaseCacheOwner;
    std::shared_ptr<strata::core::PhaseCacheBackend> backend;
    try { backend = strata::core::make_phase_cache_gpu_backend(0); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "SKIP runtime unavailable: %s\n", error.what());
        return 77;
    }
    std::string error;
    strata::core::PhaseCacheBackend::Caps caps;
    if (!backend->caps(caps, error) || !caps.vmm || !caps.free_known || !caps.granularity) {
        std::fprintf(stderr, "SKIP actual VMM/free capability unavailable: %s\n", error.c_str());
        return 77;
    }
    if (std::strcmp(argv[1], "--small-full-lease") == 0) return full_lease(backend, caps);
    constexpr uint64_t limit = uint64_t{16} << 20, unit = uint64_t{2} << 20;
    uint64_t gran = caps.granularity;
    if (!caps.device_minimum || !caps.context_minimum || gran % caps.device_minimum) return 77;
    for (const uint64_t alignment : {unit, caps.device_minimum, caps.context_minimum, uint64_t{256}}) {
        const uint64_t factor = alignment / std::gcd(gran, alignment);
        if (gran > std::numeric_limits<uint64_t>::max() / factor) return 77;
        gran *= factor;
    }
    if (gran > (limit - 65536) / 2 || gran > caps.maximum || caps.free < Owner::guard_bytes + gran * 2) {
        progress("SKIP real granularity/maximum/free exceeds finite small fixture admission");
        return 77;
    }
    Owner owner(backend);
    std::vector<uint8_t> padding(static_cast<size_t>(gran), 0xA5);
    std::vector<uint8_t> readback(static_cast<size_t>(gran));
    // All host/device buffers survive catch and failed completion until checked close.
    std::unique_ptr<strata::core::PinnedArena> source_arena;
    void* source = nullptr;
    void* d_res = nullptr;
    std::array<int32_t, 8> map_readback{};
    const Owner::Quiesce quiet = [&](std::string& message) { return backend->drain(message); };
    int result = 1;
    try {
        progress("BEGIN tiny one-layer real VMM owner");
        source_arena = std::make_unique<strata::core::PinnedArena>(4096, std::vector<uint64_t>{0, 4096}, 1024, true);
        source = source_arena->at(0);
        check(source_arena->owned_host_range(source, 1024), "genuine owned Host-USM range rejected");
        check(!source_arena->owned_host_range(static_cast<uint8_t*>(source) + 5120, 1) &&
              !source_arena->owned_host_range(static_cast<uint8_t*>(source) + 4095, 1026),
              "allocation-end/cross-allocation range admitted");
        d_res = strata::gpu::alloc_device(map_readback.size() * sizeof(int32_t));
        check(source && d_res && strata::gpu::is_host_usm(source), "fixture USM allocation failed");
        for (size_t i = 0; i < 1024; ++i) static_cast<uint8_t*>(source)[i] = static_cast<uint8_t>(i * 23 + 9);
        check(owner.reserve({1024}, 8, limit - 65536, error), error);
        check(owner.capacity() == 5 && owner.reserved_bytes() <= limit && !owner.device_slot(0), "reserved membership");
        const auto* offsets = owner.capacity_offsets();
        const auto* host_table = owner.host_residency();
        check(owner.map_base(quiet, error) && owner.mapped_slots() == 3, error);
        check(strata::gpu::copy(owner.base(), padding.data(), padding.size()), strata::gpu::last_error());
        void* old = owner.device_slot(0);
        check(owner.fill_checked(0, 0, 0, source, 1024, error), error);
        const Owner::Publish upload = [&](const int32_t* table, size_t count, std::string& message) {
            if (count != map_readback.size()) { message = "residency geometry mismatch"; return false; }
            if (!strata::gpu::copy(d_res, table, count * sizeof(int32_t))) {
                message = strata::gpu::last_error(); return false;
            }
            return true;
        };
        check(owner.publish(upload, error), error);
        for (int request = 0; request < 2; ++request) {
            progress("PHASE map/fill/publish/verify extras");
            check(owner.map_extras(quiet, error), error);
            check(owner.mapped_slots() == 5 && owner.mapped_bytes() <= limit &&
                  owner.retained_physical_bytes() <= limit, "physical budget/membership");
            check(owner.device_slot(0) == old && owner.capacity_offsets() == offsets && owner.host_residency() == host_table,
                  "full reader pointers moved");
            check(strata::gpu::copy(owner.device_slot(3), padding.data(), padding.size()), strata::gpu::last_error());
            check(owner.fill_checked(0, 1, 3, source, 1024, error), error);
            check(owner.resident_slot(0, 1) == -1 && owner.publish(upload, error), "premature/failed residency publication");
            check(strata::gpu::copy(map_readback.data(), d_res, sizeof(map_readback)), strata::gpu::last_error());
            check(std::equal(map_readback.begin(), map_readback.end(), owner.host_residency()), "host/device residency mismatch");
            check(strata::gpu::copy(readback.data(), owner.device_slot(3), readback.size()), strata::gpu::last_error());
            check(std::memcmp(readback.data(), source, 1024) == 0 &&
                  std::all_of(readback.begin() + 1024, readback.end(), [](uint8_t byte) { return byte == 0xA5; }),
                  "extra payload/padding canary changed");
            check(owner.invalidate_extras(upload, error) && owner.unmap_extras(quiet, error), error);
            check(owner.mapped_slots() == 3 && !owner.device_slot(3), "extra invalidation/detach incomplete");
            check(strata::gpu::copy(readback.data(), owner.device_slot(0), readback.size()), strata::gpu::last_error());
            check(std::memcmp(readback.data(), source, 1024) == 0 &&
                  std::all_of(readback.begin() + 1024, readback.end(), [](uint8_t byte) { return byte == 0xA5; }),
                  "retained base payload/padding changed");
        }
        result = 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "FAIL VMM owner fixture: %s\n", exception.what());
    }
    // No graphs/verifier/other readers in this fixture. The callback explicitly
    // retires fixture access and drains the bound Runtime before releasing owners.
    if (!owner.close(quiet, error) || !backend->drain(error)) {
        std::fprintf(stderr, "FAIL checked close; retaining owners/buffers until process exit: %s\n", error.c_str());
        std::fflush(stderr);
        std::_Exit(1);
    }
    source_arena.reset(); // only after actual backend/owner drains; pending source stays alive on failure
    strata::gpu::free(d_res);
    if (result == 0) progress("PASS tiny actual VMM owner closed (no model inference)");
    return result;
}
