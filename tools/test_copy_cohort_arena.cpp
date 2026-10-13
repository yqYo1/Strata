// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Actual allocation-owner fixture. Runtime/Host-USM required; no model or GPU arithmetic.
#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/pinned.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace core = strata::core;
namespace cpu = strata::kernels::cpu;
namespace gpu = strata::gpu;
namespace fs = std::filesystem;

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message); // Active in NDEBUG builds.
}
void progress(const char* phase) {
    std::fprintf(stderr, "copy cohort arena: phase=%s\n", phase);
    std::fflush(stderr);
}
const uint8_t* at_address(const uint8_t* pointer, size_t offset) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    require(offset <= UINTPTR_MAX - address, "fixture address overflow");
    return reinterpret_cast<const uint8_t*>(address + offset);
}
struct Pack {
    fs::path path;
    void create() {
        const std::string pattern = (fs::temp_directory_path() / "strata-copy-arena-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        char* directory = ::mkdtemp(name.data()); // Unique, private 0700 directory; Linux fixture.
        require(directory != nullptr, "could not create private fixture directory");
        path = directory;
    }
    void write(const cpu::ExpertLayout& layout) const {
        require(layout.total == 8294400 && !layout.native, "unexpected canonical fixture geometry");
        std::ofstream file(path / "experts.bin", std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(file), "could not create experts.bin");
        const std::array<char, 65536> zeros{};
        uint64_t remaining = layout.total;
        while (remaining) {
            const size_t bytes = remaining < zeros.size() ? static_cast<size_t>(remaining) : zeros.size();
            file.write(zeros.data(), static_cast<std::streamsize>(bytes));
            require(static_cast<bool>(file), "could not write canonical payload");
            remaining -= bytes;
        }
        for (int64_t layer = 0; layer < 2; ++layer) {
            for (int64_t expert = 0; expert < 3; ++expert) {
                file.seekp(static_cast<std::streamoff>(layout.blob_offset(layer, expert)));
                file.put(static_cast<char>(1 + layer * 3 + expert));
                require(static_cast<bool>(file), "could not write expert marker");
            }
        }
        file.close();
        require(!file.fail() && fs::file_size(path / "experts.bin") == layout.total, "payload close/size failure");
    }
    void retire() const {
        require(fs::remove(path / "experts.bin"), "could not remove owned payload");
        require(fs::remove(path), "could not remove owned private directory");
    }
};
void rejected_source(const core::ArenaExpertSource& source, int64_t layer, int64_t expert, size_t count) {
    const uint8_t sentinel = 0x5a;
    core::ExpertSource::CopyRange range{&sentinel, &sentinel, size_t(0x5a5a)};
    require(!source.copy_range(layer, expert, count, range), "invalid source range accepted");
    require(range.first == &sentinel && range.allocation == &sentinel && range.extent == 0x5a5a,
            "source rejection changed output sentinel");
}
void source_ranges(core::ArenaExpertSource& source, const cpu::ExpertLayout& layout) {
    for (int64_t layer = 0; layer < 2; ++layer) {
        const size_t width = static_cast<size_t>(layout.blob_bytes(layer));
        const size_t extent = width * 3 + static_cast<size_t>(layout.max_blob);
        core::ExpertSource::CopyRange first;
        require(source.copy_range(layer, 0, 2, first), "legal two-expert range rejected");
        require(first.first == source.blob(layer, 0) && first.allocation == first.first && first.extent == extent,
                "actual allocation base/extent mismatch");
        require(gpu::is_host_usm(first.first) && gpu::is_host_usm(first.allocation), "real Host-USM not classified");
        core::ExpertSource::CopyRange all, tail;
        require(source.copy_range(layer, 0, 3, all), "complete layer range rejected");
        require(source.copy_range(layer, 1, 2, tail), "legal expert tail rejected");
        require(all.first == first.first && all.allocation == first.allocation && all.extent == extent &&
                width * 3 <= extent && tail.allocation == first.allocation &&
                tail.extent == extent && tail.first == at_address(first.first, width), "tail owner mismatch");
        for (int64_t expert = 0; expert < 3; ++expert) {
            const uint8_t* actual = source.blob(layer, expert);
            require(actual == at_address(first.first, static_cast<size_t>(expert) * width), "source row adjacency");
            require(actual[0] == static_cast<uint8_t>(1 + layer * 3 + expert), "actual arena load marker mismatch");
        }
        rejected_source(source, layer, 2, 2); // Padding is not permission to cross the logical layer.
        rejected_source(source, layer, 0, 4);
    }
    for (size_t count : {size_t(0), size_t(1), size_t(17), std::numeric_limits<size_t>::max()})
        rejected_source(source, 0, 0, count);
    rejected_source(source, -1, 0, 2);
    rejected_source(source, 2, 0, 2);
    rejected_source(source, 0, -1, 2);
    rejected_source(source, 0, 3, 2);
}
void rejected_piece(const core::PinnedArena& arena, const uint8_t* pointer, size_t bytes) {
    const uint8_t sentinel = 0x33;
    const uint8_t* allocation = &sentinel;
    size_t extent = 0x3333;
    require(!arena.host_usm_range(pointer, bytes, allocation, extent), "invalid allocation span accepted");
    require(allocation == &sentinel && extent == 0x3333, "piece rejection changed output sentinel");
}
void piece_ranges(const core::PinnedArena& split, const core::PinnedArena& single) {
    const uint8_t* first = split.at(0);
    const uint8_t* second = split.at(64);
    const uint8_t* allocation = nullptr;
    size_t extent = 0;
    require(gpu::is_host_usm(first) && gpu::is_host_usm(second), "small owner not real Host-USM");
    require(split.host_usm_range(at_address(first, 60), 20, allocation, extent) && allocation == first && extent == 80,
            "exact split-allocation end rejected");
    require(split.host_usm_range(at_address(second, 79), 1, allocation, extent) && allocation == second && extent == 80,
            "second allocation end rejected");
    rejected_piece(split, at_address(first, 60), 21);
    rejected_piece(split, at_address(first, 60), 68); // A logical layer transition is not one allocation.
    rejected_piece(split, nullptr, 1);
    rejected_piece(split, first, 0);
    const uint8_t* base = single.at(0);
    require(single.host_usm_range(base, 80, allocation, extent) && allocation == base && extent == 80,
            "entire single allocation rejected");
    rejected_piece(single, at_address(base, 80), 1); // Single piece avoids an adjacent-allocation address coincidence.
    rejected_piece(single, at_address(base, 79), 2);
}
} // namespace

int main(int argc, char** argv) {
    const bool registration = argc == 2 && std::string(argv[1]) == "--registered";
    if (argc != 1 && !registration) return 2;
    core::Runtime* runtime = nullptr;
    Pack pack;
    // Owners survive catch so a failed drain never unwinds past live runtime state.
    std::unique_ptr<core::ArenaExpertSource> source;
    std::unique_ptr<core::PinnedArena> split, single, mapped;
    try {
        progress("environment");
        require(::setenv("STRATA_ARENA_HOST_USM", "1", 1) == 0 && ::setenv("STRATA_ARENA_MMAP", "0", 1) == 0,
                "could not freeze fixture arena mode before first open");
        if (core::Runtime::count() == 0) {
            std::fprintf(stderr, "copy cohort arena: SKIP no usable runtime device\n");
            std::fflush(stderr);
            return 77;
        }
        progress("runtime");
        runtime = &core::Runtime::get();
        if (!runtime->device().has(sycl::aspect::usm_host_allocations)) {
            runtime->finish();
            std::fprintf(stderr, "copy cohort arena: SKIP selected runtime lacks Host-USM\n");
            std::fflush(stderr);
            return 77;
        }
        if (runtime->device().get_info<sycl::info::device::max_mem_alloc_size>() < 5529600) {
            runtime->finish();
            std::fprintf(stderr, "copy cohort arena: SKIP reported allocation limit below fixture requirement\n");
            std::fflush(stderr);
            return 77;
        }
        progress("pack");
        pack.create();
        std::fprintf(stderr, "copy cohort arena: fixture_pack=%.512s payload_bytes=8294400\n", pack.path.string().c_str());
        std::fflush(stderr);
        std::string error;
        const bool layout_ok = cpu::expert_layout_load(pack.path.string(), 2, 3, error);
        require(layout_ok, "canonical layout: " + error);
        const auto& layout = cpu::expert_layout();
        pack.write(layout);
        source = std::make_unique<core::ArenaExpertSource>();
        rejected_source(*source, 0, 0, 2);
        progress("arena_open");
        const bool open_ok = source->open(pack.path.string(), 2, 3, 1, error);
        require(open_ok, "real arena open: " + error);
        source_ranges(*source, layout);
        progress("arena_close_reopen");
        runtime->finish();
        source->close();
        rejected_source(*source, 0, 0, 2);
        error.clear();
        const bool reopen_ok = source->open(pack.path.string(), 2, 3, 1, error);
        require(reopen_ok, "real arena reopen: " + error);
        source_ranges(*source, layout);
        progress("piece_boundaries");
        split = std::make_unique<core::PinnedArena>(128, std::vector<uint64_t>{0, 64, 128}, 16, true);
        single = std::make_unique<core::PinnedArena>(64, std::vector<uint64_t>{0, 64}, 16, true);
        piece_ranges(*split, *single);
        if (registration) {
            progress("registered_mapping");
            mapped = std::make_unique<core::PinnedArena>(4096, std::vector<uint64_t>{0, 2048, 4096}, 0, false);
            require(mapped->valid() && mapped->layer_base.empty(), "registered negative control is not a mapping");
            rejected_piece(*mapped, mapped->at(0), 32);
        }
        progress("drain_close");
        runtime->finish();
        mapped.reset(); single.reset(); split.reset();
        source->close();
        rejected_source(*source, 0, 0, 2);
        source.reset();
        pack.retire();
        std::fprintf(stderr, "copy cohort arena: PASS scope=actual_arena_owner_and_piece_ranges registered=%s\n",
                     registration ? "checked" : "not_requested");
        std::fflush(stderr);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "copy cohort arena: FAILED %.512s\n", error.what());
        std::fflush(stderr);
        if (runtime) {
            try { runtime->finish(); }
            catch (...) {
                std::fprintf(stderr, "copy cohort arena: drain failed; retaining live owners, fail-stop\n");
                std::fflush(stderr);
                std::_Exit(1);
            }
        }
        mapped.reset(); single.reset(); split.reset(); source.reset();
        if (!pack.path.empty()) std::fprintf(stderr, "copy cohort arena: retained failed pack %.512s\n",
                                             pack.path.string().c_str());
        std::fflush(stderr);
        return 1;
    }
}
