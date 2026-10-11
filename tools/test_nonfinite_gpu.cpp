// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// No-model device parity of the actual non-finite detector, its mapped flags and captured replays.
#include "strata/core/gpu.hpp"
#include "strata/core/nonfinite.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace gpu = strata::gpu;
namespace kernels = strata::kernels;
constexpr int kRows = kernels::kVerifyMaxT;
constexpr int kBlocks = kernels::kNonfiniteBlocks;
constexpr int kMaxVocab = 4099;
constexpr uint32_t kSentinel = 0xa55a5aa5u;
static_assert(kRows == 8, "extend the explicit row cohorts when the production bound changes");

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
void checked(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string(what) + ": " + gpu::last_error());
}
void drain() noexcept {
    if (!gpu::device_sync()) {
        std::fprintf(stderr, "nonfinite_gpu_test FAIL: completion unconfirmed: %s; storage not released\n",
                     gpu::last_error());
        std::fflush(nullptr);
        std::_Exit(1);
    }
}

// The same owner/context/device and host-USM alias contract as production mapped(). Staging is a
// member so exceptional queue submission cannot outlive an already-destroyed host copy source.
struct Fixture {
    gpu::Stream stream = nullptr;
    float* logits = nullptr;
    uint32_t* flags = nullptr;
    uint32_t* mapped = nullptr;
    std::vector<uint32_t> staging;
    std::array<gpu::Graph*, kRows + 1> graphs{};
    bool capturing = false;
    int checks = 0;

    ~Fixture() {
        if (capturing) gpu::abandon_capture(stream);
        drain();
        for (auto* graph : graphs) if (graph) gpu::graph_destroy(graph);
        gpu::free(logits);
        gpu::free(flags);
        if (stream) gpu::stream_destroy(stream);
    }

    void init() {
        auto& rt = strata::core::Runtime::get();
        stream = gpu::stream_create();
        require(stream != nullptr, "owned queue allocation");
        auto& q = rt.stream(stream);
        require(q.is_in_order() && q.get_device() == rt.device() && q.get_context() == rt.context(),
                "explicit owning in-order device/context queue");
        checked(gpu::alloc_device(&logits, (size_t) kRows * kMaxVocab * sizeof(float)), "logit allocation");
        checked(gpu::alloc_host(&flags, (size_t) (kRows * kBlocks + 2) * sizeof(uint32_t)), "host-USM flags");
        void* alias = nullptr;
        checked(gpu::device_pointer(&alias, flags), "mapped host-USM alias");
        require(alias == flags && strata::core::Runtime::owner(flags) == &rt &&
                strata::core::Runtime::owner(logits) == &rt && strata::core::Runtime::owner_of_stream(stream) == &rt,
                "allocation and queue ownership");
        mapped = static_cast<uint32_t*>(alias) + 1;
        // One guard word on either side catches writes outside the complete maximum-row storage.
        std::fill(flags, flags + (size_t) kRows * kBlocks + 2, kSentinel);
    }

    void upload(const std::vector<uint32_t>& words) {
        // Every prior use is completed before either host staging or flags can be rewritten.
        checked(gpu::stream_sync(stream), "before staging reuse");
        staging = words;
        checked(gpu::copy_async(logits, staging.data(), staging.size() * sizeof(uint32_t), stream), "input copy");
    }

    void record(int rows, int vocab) {
        checked(gpu::stream_sync(stream), "before graph capture");
        checked(gpu::begin_capture(stream), "begin actual graph capture");
        capturing = true;
        kernels::logits_nonfinite_rows(logits, rows, vocab, mapped, stream);
        checked(gpu::end_capture(stream, &graphs[(size_t) rows]), "finalize actual graph");
        capturing = false;
        require(graphs[(size_t) rows] && gpu::graph_nodes(graphs[(size_t) rows]) > 0, "nonempty detector graph");
    }

    void clear_graphs() {
        checked(gpu::stream_sync(stream), "before graph release");
        for (auto*& graph : graphs) {
            if (graph) gpu::graph_destroy(graph);
            graph = nullptr;
        }
    }

    void run(const std::vector<uint32_t>& words, int rows, int vocab, bool graph, bool reseed) {
        require(words.size() == (size_t) kRows * (size_t) vocab, "exact fixture extent");
        checked(gpu::stream_sync(stream), "before flag seed");
        if (reseed) std::fill(flags, flags + (size_t) kRows * kBlocks + 2, kSentinel);
        const std::vector<uint32_t> previous(mapped, mapped + (size_t) kRows * kBlocks);
        upload(words);
        if (graph) checked(gpu::graph_launch(graphs[(size_t) rows], stream), "graph replay");
        else kernels::logits_nonfinite_rows(logits, rows, vocab, mapped, stream);
        // ONLY this successful queue wait transfers flag ownership to the host. No concurrent poll.
        checked(gpu::stream_sync(stream), "detector completion");
        require(flags[0] == kSentinel && flags[kRows * kBlocks + 1] == kSentinel, "flag storage canaries");
        uint32_t mask = 0;
        const int slice = (vocab + kBlocks - 1) / kBlocks;
        for (int row = 0; row < kRows; ++row) {
            for (int block = 0; block < kBlocks; ++block) {
                const size_t at = (size_t) row * kBlocks + (size_t) block;
                if (row >= rows) {
                    require(mapped[at] == previous[at], "inactive tail flag was not written");
                    continue;
                }
                uint32_t expected = 0;
                // CPU observer of explicitly chosen input bits, never a replacement device kernel.
                for (int col = block * slice; col < std::min(vocab, (block + 1) * slice); ++col)
                    if ((words[(size_t) row * (size_t) vocab + (size_t) col] & 0x7f800000u) == 0x7f800000u)
                        expected = 1;
                if (expected) mask |= uint32_t{1} << row;
                if (mapped[at] != expected) {
                    std::fprintf(stderr, "flag mismatch graph=%d rows=%d vocab=%d row=%d block=%d got=%u expected=%u\n",
                                 graph ? 1 : 0, rows, vocab, row, block, mapped[at], expected);
                    throw std::runtime_error("exact per-block flags");
                }
            }
        }
        require(strata::core::nonfinite_mask(mapped, rows, kBlocks) == mask, "actual production active-row mask");
        ++checks;
        std::printf("PASS case=%d graph=%d rows=%d vocab=%d mask=%u\n", checks, graph ? 1 : 0, rows, vocab, mask);
        std::fflush(stdout);
    }
};

std::vector<uint32_t> finite(int vocab) {
    constexpr uint32_t bits[] = {0u, 0x80000000u, 0x7f7fffffu, 0xff7fffffu,
                                 1u, 0x80000001u, 0x007fffffu, 0x807fffffu, 0x3f800000u};
    std::vector<uint32_t> out((size_t) kRows * (size_t) vocab);
    for (size_t i = 0; i < out.size(); ++i) out[i] = bits[i % (sizeof(bits) / sizeof(bits[0]))];
    return out;
}

void cases(Fixture& f, int vocab, bool graph) {
    constexpr uint32_t bad[] = {0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc54321u};
    for (int rows : {1, 2, kRows}) {
        if (graph) f.record(rows, vocab);
        const auto clean = finite(vocab);
        f.run(clean, rows, vocab, graph, true);   // includes empty slices for vocab17
        for (uint32_t pattern : bad)
            for (int col : {0, vocab / 2, vocab - 1}) {
                auto words = clean;
                // Every active row gets a known bad value; each block still needs its own exact flag.
                for (int row = 0; row < rows; ++row)
                    words[(size_t) row * (size_t) vocab + (size_t) col] = pattern;
                f.run(words, rows, vocab, graph, true);
                f.run(clean, rows, vocab, graph, false);   // all active words MUST overwrite prior poison
            }
        auto mixed = clean;
        mixed[(size_t) (rows - 1) * (size_t) vocab + (size_t) (vocab - 1)] = bad[2];
        f.run(mixed, rows, vocab, graph, true);   // only last active row is bad
    }
    auto all_bad = finite(vocab);
    for (int row = 0; row < kRows; ++row) all_bad[(size_t) row * (size_t) vocab] = bad[0];
    f.run(all_bad, kRows, vocab, graph, true);
    f.run(finite(vocab), 2, vocab, graph, false);  // stale tail still bad; actual helper ignores it
    f.run(finite(vocab), 1, vocab, graph, false);
    require(strata::core::nonfinite_mask(f.mapped, kRows, kBlocks) != 0, "shrunk window leaves explicit poisoned tail");
}

void invalid_arguments() {
    float dummy = 0.0f;
    uint32_t flag = 0;
    auto rejects = [&](const float* input, int rows, int vocab, uint32_t* flags) {
        bool refused = false;
        try { kernels::logits_nonfinite_rows(input, rows, vocab, flags, nullptr); }
        catch (const strata::core::DeviceError& e) {
            refused = std::string(e.what()) == "logits_nonfinite_rows: invalid arguments";
            if (!refused) { drain(); throw; }
        }
        catch (...) { drain(); throw; }
        // A regression accepting an invalid pointer must drain while the stack fixture is alive.
        if (!refused) drain();
        require(refused, "invalid arguments rejected before queue lookup/enqueue");
    };
    rejects(nullptr, 1, 17, &flag);
    rejects(&dummy, 1, 17, nullptr);
    rejects(&dummy, 0, 17, &flag);
    rejects(&dummy, -1, 17, &flag);
    rejects(&dummy, kRows + 1, 17, &flag);
    rejects(&dummy, 1, 0, &flag);
    rejects(&dummy, 1, -1, &flag);
    std::puts("PASS invalid-arguments (before Runtime initialization)");
}
}  // namespace

int main() {
    try {
        invalid_arguments();
        Fixture fixture;
        fixture.init();
        for (int vocab : {17, kMaxVocab}) cases(fixture, vocab, false);
        const auto& device = strata::core::Runtime::get().device();
        const bool graph = device.has(sycl::aspect::ext_oneapi_graph) || device.has(sycl::aspect::ext_oneapi_limited_graph);
        if (graph) {
            for (int vocab : {17, kMaxVocab}) {
                cases(fixture, vocab, true);
                fixture.clear_graphs();
            }
        } else std::puts("SKIP graph-replay: device reports neither full nor limited command graph support");
        checked(gpu::stream_sync(fixture.stream), "final owned queue completion");
        std::printf("nonfinite_gpu_test PASS: %d actual detector cases; graph_supported=%d\n", fixture.checks, graph ? 1 : 0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nonfinite_gpu_test FAIL: %s\n", e.what());
        return 1;
    }
}
