// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>

// Manually supervised GPU fixture: real runtime queues, cross-queue pinned copies, production grouped GEMM.
int main(int argc, char**) {
    if (argc != 1) return 2;
    using namespace strata;
    auto& runtime = core::Runtime::get();
    auto* stream = runtime.create_stream();
    const bool requested = core::service_events_requested();
    const auto properties = [requested](const sycl::queue& queue) {
        return queue.has_property<sycl::property::queue::in_order>() &&
               queue.has_property<sycl::property::queue::enable_profiling>() == requested;
    };
    if (!properties(runtime.compute()) || !properties(runtime.transfer()) || !properties(*stream)) return 1;
    constexpr int G = 2, N = 1280, K = 2560;
    constexpr size_t XC = K, WC = (size_t) G * N * K, YC = N + 16;
    const size_t limit = runtime.device().get_info<sycl::info::device::max_mem_alloc_size>();
    if (!runtime.device().has(sycl::aspect::usm_host_allocations) ||
        !runtime.device().has(sycl::aspect::usm_device_allocations) || WC * 2 > limit) return 77;
    uint16_t *hx = nullptr, *hw = nullptr, *dx = nullptr, *dw = nullptr;
    float *hy = nullptr, *dy = nullptr;
    int32_t *hb = nullptr, *db = nullptr;
    bool done = false;
    try {
        hx = sycl::aligned_alloc_host<uint16_t>(64, XC, runtime.compute());
        hw = sycl::aligned_alloc_host<uint16_t>(64, WC, runtime.compute());
        hy = sycl::aligned_alloc_host<float>(64, YC, runtime.compute());
        hb = sycl::aligned_alloc_host<int32_t>(64, G + 1, runtime.compute());
        dx = sycl::aligned_alloc_device<uint16_t>(64, XC, runtime.compute());
        dw = sycl::aligned_alloc_device<uint16_t>(64, WC, runtime.compute());
        dy = sycl::aligned_alloc_device<float>(64, YC, runtime.compute());
        db = sycl::aligned_alloc_device<int32_t>(64, G + 1, runtime.compute());
        if (!hx || !hw || !hy || !hb || !dx || !dw || !dy || !db) throw std::runtime_error("fixture allocation");
        std::fill(hx, hx + XC, uint16_t{0x3000});  // FP16 0.125, exact dot product = 320.
        std::fill(hw, hw + WC, uint16_t{0x3c00});
        std::fill(hy, hy + YC, -17.0f);
        hb[0] = 0; hb[1] = 1; hb[2] = 1;  // Last group is empty; Y has sixteen untouched canaries.
        core::ServiceScope scope(runtime.service_events());
        bool overlap_rejected = false;
        if (requested) {
            try { runtime.service_events().begin(); }
            catch (const std::runtime_error&) { overlap_rejected = true; }
            if (!overlap_rejected) throw std::runtime_error("overlapping scope admitted");
        }
        const auto ex = runtime.copy_async(dx, hx, XC * 2);
        const auto ew = runtime.copy_async(dw, hw, WC * 2);
        const auto ey = runtime.copy_async(dy, hy, YC * sizeof(float));
        const auto eb = runtime.copy_async(db, hb, (G + 1) * sizeof(int32_t));
        stream->ext_oneapi_submit_barrier({ex, ew, ey, eb});
        kernels::xmx_gemm_grouped(dx, dw, (int64_t) N * K, dy, db, G, 1, N, K, stream);
        if (!gpu::copy_async(hy, dy, YC * sizeof(float), stream) ||
            !gpu::copy_async(hy, dy, 0, stream)) throw std::runtime_error(gpu::last_error());
        runtime.finish();
        scope.complete();
        for (size_t i = 0; i < YC; ++i)
            if (hy[i] != (i < N ? 320.0f : -17.0f)) throw std::runtime_error("fixture output/canary mismatch");
        { core::ServiceScope reset(runtime.service_events()); reset.complete(); }
        done = true;
        std::printf("SERVICE_FIXTURE status=PASS profiling=%d queues=3 copies=5 grouped=1 outputs=%d canaries=16 "
                    "overlap_rejected=%d reset_scope=1 zero_copy_excluded=1\n",
                    (int) requested, N, (int) overlap_rejected);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "SERVICE_FIXTURE status=FAILED reason=%.256s\n", error.what());
        try { runtime.finish(); }
        catch (...) { std::_Exit(1); }  // Operands remain alive when completion cannot be established.
    }
    for (void* pointer : {static_cast<void*>(hx), static_cast<void*>(hw), static_cast<void*>(hy),
                         static_cast<void*>(hb), static_cast<void*>(dx), static_cast<void*>(dw),
                         static_cast<void*>(dy), static_cast<void*>(db)})
        if (pointer) runtime.free(pointer);
    runtime.destroy_stream(stream);
    return done ? 0 : 1;
}
