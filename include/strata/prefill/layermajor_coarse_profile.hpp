#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace strata::prefill::detail {
// Diagnostic only. Fixed storage, no per-expert marks. All methods are noexcept:
// observation failure must never bypass the wrapper's checked owner retirement.
struct LayerMajorCoarseProfile {
    using Clock = std::chrono::steady_clock;
    enum Metric { source, upload_submit, upload_drain, activation, input_submit_host, input_submit_device,
                  store_wall_host, store_wall_device, store_drain_host, store_drain_device,
                  chunk_drain, wrapper_drain, metric_count };
    enum Interval { input, qsa, gdn, moe, upload, store, interval_count };
    struct Row {
        double wall[metric_count]{};
        double queue[interval_count]{};
        uint64_t input_bytes[2]{}, store_bytes[2]{};
        double input_queue[2]{}, store_queue[2]{};
        uint64_t upload_bytes = 0;
        uint64_t samples[interval_count]{};
    } rows[48]{};
    struct Pair { cudaEvent_t a = nullptr, b = nullptr; bool begun = false, ended = false, device = false; int layer = -1; } pairs[interval_count];
    bool usable = true, complete = false, emitted = false;
    unsigned errors = 0;
    int first_status = 0;
    int layer = -1;
    double wrapper_ms = -1; // unavailable on an incomplete wrapper
    static double elapsed(Clock::time_point t) noexcept {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    }
    void fail() noexcept { usable = false; ++errors; }
    bool check(cudaError_t e) noexcept { if (e == cudaSuccess) return true; if (!first_status) first_status = int(e); fail(); return false; }
    void open() noexcept {
        for (auto& p : pairs) {
            if (!check(cudaEventCreate(&p.a))) return;
            if (!check(cudaEventCreate(&p.b))) return;
        }
    }
    // Called only after the existing outer checked lease retires, or by the
    // destructor AFTER that lease and Restore have run. compat EventDestroy is
    // delete-only (no wait/submission), audited at the pinned source revision.
    void close() noexcept {
        for (auto& p : pairs) {
            if (p.a) { check(cudaEventDestroy(p.a)); p.a = nullptr; }
            if (p.b) { check(cudaEventDestroy(p.b)); p.b = nullptr; }
        }
    }
    ~LayerMajorCoarseProfile() noexcept { close(); if (!emitted) emit(); }
    void begin(Interval k, cudaStream_t stream, bool device = false) noexcept {
        if (!usable) return;
        auto& p = pairs[k];
        if (layer < 0 || layer >= 48 || p.begun) { fail(); return; }
        p.layer = layer; p.device = device;
        if (check(cudaEventRecord(p.a, stream))) p.begun = true;
    }
    void end(Interval k, cudaStream_t stream) noexcept {
        if (!usable) return;
        auto& p = pairs[k];
        if (!p.begun || p.ended || p.layer != layer) { fail(); return; }
        if (check(cudaEventRecord(p.b, stream))) p.ended = true;
    }
    // The caller MUST have just successfully completed an existing checked
    // stream drain covering both marks. Never call from a submit-only seam.
    // ElapsedTime internally waits for b in this compatibility implementation;
    // here b is already retired, not a new synchronization dependency.
    void retired(Interval k) noexcept {
        if (!usable) return;
        auto& p = pairs[k];
        if (!p.begun) return;
        if (!p.ended || !check(cudaEventQuery(p.a)) || !check(cudaEventQuery(p.b))) { fail(); return; }
        float ms = 0;
        if (!check(cudaEventElapsedTime(&ms, p.a, p.b))) return;
        if (!std::isfinite(ms) || !(ms >= 0)) { fail(); return; }
        rows[p.layer].queue[k] += ms;
        if (k == input) rows[p.layer].input_queue[int(p.device)] += ms;
        if (k == store) rows[p.layer].store_queue[int(p.device)] += ms;
        ++rows[p.layer].samples[k];
        p.begun = p.ended = false; p.layer = -1;
    }
    void chunk_retired() noexcept { retired(input); retired(qsa); retired(gdn); retired(moe); }
    void add(Metric k, Clock::time_point t) noexcept { if (layer >= 0 && layer < 48) rows[layer].wall[k] += elapsed(t); }
    // 50 bounded lines including aggregate; scientific formatting bounds every
    // numeric field even for unusually long jobs. Output is below 64 KiB.
    void emit() noexcept {
        emitted = true;
        std::fprintf(stderr, "strata lm profile: v=2 complete=%d errors=%u first_status=%d wrapper_ms=%.6g denominator=started_through_checked_lease_retirement teardown_unmeasured=1 queue_intervals=NONEXCLUSIVE_including_host_submission_gaps bytes=REQUESTED_not_wire wall_order=source,upload_submit,upload_drain,activation,input_submit_host,input_submit_device,store_wall_host,store_wall_device,store_drain_host,store_drain_device,chunk_drain,wrapper_drain queue_order=input,QSA,GDN,MoE,upload,store\n", int(complete && usable), errors, first_status, wrapper_ms);
        Row total;
        auto line = [](int l, const Row& r) noexcept {
            std::fprintf(stderr, "strata lm profile: layer=%d wall_ms=", l);
            for (int k = 0; k < metric_count; ++k) std::fprintf(stderr, "%s%.6g", k ? "," : "", r.wall[k]);
            std::fputs(" queue_ms=", stderr);
            for (int k = 0; k < interval_count; ++k) std::fprintf(stderr, "%s%.6g", k ? "," : "", r.queue[k]);
            std::fputs(" samples=", stderr);
            for (int k = 0; k < interval_count; ++k) std::fprintf(stderr, "%s%llu", k ? "," : "", (unsigned long long)r.samples[k]);
            std::fprintf(stderr, " upload_bytes=%llu input_bytes_host_device=%llu,%llu store_bytes_host_device=%llu,%llu input_queue_host_device_ms=%.6g,%.6g store_queue_host_device_ms=%.6g,%.6g\n",
                (unsigned long long)r.upload_bytes,
                (unsigned long long)r.input_bytes[0],(unsigned long long)r.input_bytes[1],
                (unsigned long long)r.store_bytes[0],(unsigned long long)r.store_bytes[1],
                r.input_queue[0],r.input_queue[1],r.store_queue[0],r.store_queue[1]);
        };
        for (int l = 0; l < 48; ++l) {
            const auto& r = rows[l];
            line(l, r);
            for (int k = 0; k < metric_count; ++k) total.wall[k] += r.wall[k];
            for (int k = 0; k < interval_count; ++k) { total.queue[k] += r.queue[k]; total.samples[k] += r.samples[k]; }
            for (int k = 0; k < 2; ++k) {
                total.input_bytes[k] += r.input_bytes[k]; total.store_bytes[k] += r.store_bytes[k];
                total.input_queue[k] += r.input_queue[k]; total.store_queue[k] += r.store_queue[k];
            }
            total.upload_bytes += r.upload_bytes;
        }
        line(-1, total); // aggregate, not an exclusive sum across phase categories
    }
};
struct LayerMajorCoarseBorrow {
    LayerMajorCoarseProfile*& slot;
    LayerMajorCoarseProfile* prior;
    LayerMajorCoarseBorrow(LayerMajorCoarseProfile*& s, LayerMajorCoarseProfile* p) noexcept : slot(s), prior(s) { slot = p; }
    ~LayerMajorCoarseBorrow() noexcept { slot = prior; }
};
} // namespace strata::prefill::detail
