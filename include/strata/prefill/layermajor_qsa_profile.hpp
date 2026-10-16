#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace strata::prefill::detail {
// Own-fork diagnostic. A pair per section is reused only AFTER the existing
// checked chunk drain. No event waits or stream synchronization are added.
struct LayerMajorQsaProfile {
    using Clock = std::chrono::steady_clock;
    enum Section { projections, kv_stage, kv_append_query, indexer_append,
                   selection, attention, output_projection, section_count };
    struct Row {
        double queue[section_count]{}, host[section_count]{};
        uint64_t samples[section_count]{};
        uint64_t expected = 0, expected_queries = 0;
        uint64_t matrix_chunks = 0, fallback_chunks = 0;
        uint64_t matrix_queries = 0, fallback_queries = 0, staged_chunks = 0;
        int64_t cap = 0;
    } rows[48]{};
    struct Pair {
        cudaEvent_t a = nullptr, b = nullptr;
        bool begun = false, ended = false;
        int layer = -1;
        Clock::time_point host_start{};
    } pairs[section_count];
    bool usable = true, complete = false, emitted = false;
    unsigned errors = 0;
    int first_status = 0;
    int layer = -1;
    double wrapper_ms = -1;
    void fail() noexcept { usable = false; complete = false; ++errors; }
    bool check(cudaError_t e) noexcept {
        if (e == cudaSuccess) return true;
        if (!first_status) first_status = int(e);
        fail(); return false;
    }
    void open() noexcept {
        for (auto& p : pairs) {
            if (!check(cudaEventCreate(&p.a))) return;
            if (!check(cudaEventCreate(&p.b))) return;
        }
    }
    // Declaration/borrow order in run_layer_major keeps these owners alive
    // through both checked lease retirement and Restore, including exceptions.
    // The pinned SYCL compat EventDestroy is delete-only.
    void close() noexcept {
        for (auto& p : pairs) {
            if (p.a) { check(cudaEventDestroy(p.a)); p.a = nullptr; }
            if (p.b) { check(cudaEventDestroy(p.b)); p.b = nullptr; }
        }
    }
    ~LayerMajorQsaProfile() noexcept { close(); if (!emitted) emit(); }
    void begin(Section k, cudaStream_t stream) noexcept {
        if (!usable) return;
        auto& p = pairs[k];
        if (layer < 0 || layer >= 48 || !rows[layer].expected || p.begun ||
            rows[layer].samples[k] >= rows[layer].expected) { fail(); return; }
        p.layer = layer;
        p.host_start = Clock::now();
        if (check(cudaEventRecord(p.a, stream))) p.begun = true;
    }
    void end(Section k, cudaStream_t stream) noexcept {
        if (!usable) return;
        auto& p = pairs[k];
        if (!p.begun || p.ended || p.layer != layer) { fail(); return; }
        if (!check(cudaEventRecord(p.b, stream))) return;
        p.ended = true;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - p.host_start).count();
        if (!std::isfinite(ms) || ms < 0 || !std::isfinite(rows[p.layer].host[k] + ms)) { fail(); return; }
        rows[p.layer].host[k] += ms;
    }
    // MUST be called only after run_impl's existing checked callback drain.
    // Compat ElapsedTime waits internally: both events have already completed
    // here. Failure leaves the pair unreusable; owner retirement still runs.
    void chunk_retired() noexcept {
        if (!usable) return;
        for (int k = 0; k < section_count; ++k) {
            auto& p = pairs[k];
            if (!p.begun) continue;
            if (!p.ended || !check(cudaEventQuery(p.a)) || !check(cudaEventQuery(p.b))) { fail(); return; }
            float ms = 0;
            if (!check(cudaEventElapsedTime(&ms, p.a, p.b))) return;
            if (!std::isfinite(ms) || ms < 0 || !std::isfinite(rows[p.layer].queue[k] + ms)) { fail(); return; }
            rows[p.layer].queue[k] += ms;
            ++rows[p.layer].samples[k];
            p.begun = p.ended = false; p.layer = -1;
        }
    }
    // Record the ACTUAL dispatch result, independently of the existing optional
    // dispatch logger. Exactly once per attention layer/chunk; no new events.
    void dispatch(bool matrix, bool staged, int64_t queries, int64_t cap) noexcept {
        if (!usable) return;
        if (layer < 0 || layer >= 48) { fail(); return; }
        auto& row = rows[layer];
        const uint64_t chunks = row.matrix_chunks + row.fallback_chunks;
        const uint64_t prior_queries = row.matrix_queries + row.fallback_queries;
        if (!row.expected || chunks >= row.expected || queries <= 0 || queries > 8192 ||
            cap <= 0 || cap != row.cap || prior_queries > row.expected_queries ||
            uint64_t(queries) > row.expected_queries - prior_queries) { fail(); return; }
        if (matrix) { ++row.matrix_chunks; row.matrix_queries += uint64_t(queries); }
        else { ++row.fallback_chunks; row.fallback_queries += uint64_t(queries); }
        if (staged) ++row.staged_chunks;
    }
    void finish(double denominator) noexcept {
        if (!std::isfinite(denominator) || denominator < 0) { fail(); return; }
        wrapper_ms = denominator;
        for (const auto& p : pairs) if (p.begun || p.ended) { fail(); return; }
        int64_t shared_cap = 0;
        for (const auto& row : rows) {
            for (int k = 0; k < section_count; ++k)
                if (row.samples[k] != row.expected) { fail(); return; }
            if (row.matrix_chunks + row.fallback_chunks != row.expected ||
                row.matrix_queries + row.fallback_queries != row.expected_queries ||
                row.staged_chunks > row.expected) { fail(); return; }
            if (row.expected) {
                if (row.cap <= 0 || !row.expected_queries || (shared_cap && shared_cap != row.cap)) { fail(); return; }
                shared_cap = row.cap;
            } else if (row.cap || row.expected_queries || row.matrix_chunks || row.fallback_chunks ||
                       row.matrix_queries || row.fallback_queries || row.staged_chunks) { fail(); return; }
        }
        if (!shared_cap) { fail(); return; }
        complete = usable;
    }
    // Fixed 50 lines. Seven scientific-format values in each numeric vector,
    // bounded uint64 counts, 48 rows + aggregate: comfortably below 64 KiB.
    void emit() noexcept {
        emitted = true;
        std::fprintf(stderr, "strata lm qsa profile: v=1 complete=%d errors=%u first_status=%d wrapper_ms=%.6g denominator=started_through_checked_lease_retirement teardown_unmeasured=1 granularity=QSA_LAYER_CHUNK sections=projections,kv_stage,kv_append_query,indexer_append,selection,attention,output_projection queue_intervals=NONEXCLUSIVE_including_host_submission_gaps host_wall=NONEXCLUSIVE_submission_and_helper_waits_not_CPU_or_GPU_service markers_included=1 kv_stage_noop_sample=1 dispatch=ACTUAL_matrix_used_result queries=SUM_T_not_request_tokens\n", int(complete && usable), errors, first_status, wrapper_ms);
        Row total;
        auto line = [](int l, const Row& row) noexcept {
            std::fprintf(stderr, "strata lm qsa profile: layer=%d expected_chunks=%llu queue_ms=", l, (unsigned long long)row.expected);
            for (int k = 0; k < section_count; ++k) std::fprintf(stderr, "%s%.6g", k ? "," : "", row.queue[k]);
            std::fputs(" host_wall_ms=", stderr);
            for (int k = 0; k < section_count; ++k) std::fprintf(stderr, "%s%.6g", k ? "," : "", row.host[k]);
            std::fputs(" samples=", stderr);
            for (int k = 0; k < section_count; ++k) std::fprintf(stderr, "%s%llu", k ? "," : "", (unsigned long long)row.samples[k]);
            std::fprintf(stderr, " cap=%lld expected_queries=%llu dispatch_chunks=%llu query_total=%llu matrix_chunks=%llu fallback_chunks=%llu matrix_queries=%llu fallback_queries=%llu staged_chunks=%llu\n",
                (long long)row.cap, (unsigned long long)row.expected_queries,
                (unsigned long long)(row.matrix_chunks + row.fallback_chunks),
                (unsigned long long)(row.matrix_queries + row.fallback_queries),
                (unsigned long long)row.matrix_chunks, (unsigned long long)row.fallback_chunks,
                (unsigned long long)row.matrix_queries, (unsigned long long)row.fallback_queries,
                (unsigned long long)row.staged_chunks);
        };
        for (int l = 0; l < 48; ++l) {
            const auto& row = rows[l]; line(l, row);
            total.expected += row.expected;
            total.expected_queries += row.expected_queries;
            total.matrix_chunks += row.matrix_chunks; total.fallback_chunks += row.fallback_chunks;
            total.matrix_queries += row.matrix_queries; total.fallback_queries += row.fallback_queries;
            total.staged_chunks += row.staged_chunks;
            if (row.expected && !total.cap) total.cap = row.cap;
            for (int k = 0; k < section_count; ++k) {
                total.queue[k] += row.queue[k]; total.host[k] += row.host[k];
                total.samples[k] += row.samples[k];
            }
        }
        line(-1, total);
    }
};
struct LayerMajorQsaBorrow {
    LayerMajorQsaProfile*& slot;
    LayerMajorQsaProfile* prior;
    LayerMajorQsaBorrow(LayerMajorQsaProfile*& s, LayerMajorQsaProfile* p) noexcept : slot(s), prior(s) { slot = p; }
    ~LayerMajorQsaBorrow() noexcept { slot = prior; }
};
} // namespace strata::prefill::detail
