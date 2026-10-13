// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/gpu.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace gpu = strata::gpu;
int main() {
    constexpr size_t width = 37, sp = 48, dp = 64, height = 4;
    gpu::Stream copy = nullptr, reader = nullptr;
    unsigned char *src = nullptr, *dst = nullptr, *snapshot = nullptr, *result = nullptr;
    gpu::Event* used[height]{};
    gpu::Event* done[height]{};
    const auto require = [](bool ok) { if (!ok) throw std::runtime_error(gpu::last_error()); };
    try {
        copy = gpu::stream_create(); reader = gpu::stream_create();
        src = static_cast<unsigned char*>(gpu::alloc_host(sp * height));
        dst = static_cast<unsigned char*>(gpu::alloc_device(dp * height));
        snapshot = static_cast<unsigned char*>(gpu::alloc_host(dp * height));
        result = static_cast<unsigned char*>(gpu::alloc_host(dp * height));
        if (!copy || !reader || !src || !dst || !snapshot || !result) throw std::runtime_error("allocation");
        std::memset(result, 0x5a, dp * height);
        require(gpu::copy(dst, result, dp * height));
        for (size_t row = 0; row < height; ++row) {
            require(gpu::event_create(&used[row])); require(gpu::event_create(&done[row]));
            if (!used[row] || !done[row]) throw std::runtime_error("event allocation");
            for (size_t col = 0; col < sp; ++col) src[row * sp + col] = static_cast<unsigned char>(row * 53 + col);
            // Actual previous reader on another queue, each protected separately.
            require(gpu::copy_async_after(snapshot + row * dp, dst + row * dp, dp, reader, nullptr, used[row]));
        }
        const gpu::Event* dependencies[height] = {used[0], used[1], used[2], used[3]};
        require(gpu::copy_cohort_async(dst, dp, src, sp, width, height, copy, dependencies, done));
        require(gpu::stream_sync(copy)); require(gpu::stream_sync(reader));
        for (size_t row = 0; row < height; ++row) require(gpu::event_sync(done[row]));
        require(gpu::copy(result, dst, dp * height));
        for (size_t row = 0; row < height; ++row)
            for (size_t col = 0; col < dp; ++col) {
                if (snapshot[row * dp + col] != 0x5a || result[row * dp + col] !=
                    (col < width ? src[row * sp + col] : 0x5a)) throw std::runtime_error("copy/padding/reader mismatch");
            }
        // One-row fallback is the unchanged actual 1D command, never a native2D retry.
        src[0] ^= 1;
        require(gpu::copy_async_after(dst, src, width, copy, used[0], done[0]));
        require(gpu::stream_sync(copy)); require(gpu::copy(result, dst, dp * height));
        if (std::memcmp(src, result, width)) throw std::runtime_error("one-row fallback mismatch");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "copy cohort fixture FAILED: %s\n", error.what());
        // Fail-stop on unsuccessful drains; never free pending GPU or copy-source state.
        if ((copy && !gpu::stream_sync(copy)) || (reader && !gpu::stream_sync(reader))) std::_Exit(1);
        return 1; // Main owns failed-process retirement; retain allocations until process exit.
    }
    for (size_t row = 0; row < height; ++row) { gpu::event_destroy(used[row]); gpu::event_destroy(done[row]); }
    gpu::free(src); gpu::free(dst); gpu::free(snapshot); gpu::free(result);
    gpu::stream_destroy(copy); gpu::stream_destroy(reader);
    std::puts("copy cohort fixture PASS");
}
