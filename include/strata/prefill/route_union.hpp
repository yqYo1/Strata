// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstdint>
#include <cstdio>
#include <limits>

namespace strata::prefill {
// Request-owned host receipt; no per-token routes are retained.
struct RouteUnion {
    struct Layer {
        uint64_t bits[8] = {};
        uint64_t rows = 0, chunks = 0, visits = 0, unique = 0, blob = 0;
    } layers[64];
    uint64_t rows = 0, chunks = 0, topk = 0;
    int64_t count = 0;
    bool invalid = false, eligible = false, completed = false;
    static bool product(uint64_t a, uint64_t b, uint64_t& out) {
        if (b && a > std::numeric_limits<uint64_t>::max() / b) return false;
        out = a * b; return true;
    }
    static bool add(uint64_t& a, uint64_t b) {
        if (b > std::numeric_limits<uint64_t>::max() - a) return false;
        a += b; return true;
    }
    void reset(int64_t n_layers, int64_t n_expert, uint64_t n_rows, uint64_t n_chunks, uint64_t k) {
        *this = RouteUnion{};
        count = n_layers; rows = n_rows; chunks = n_chunks; topk = k;
        uint64_t expected = 0;
        eligible = n_layers > 0 && n_layers <= 64 && n_expert == 512 && rows > 0 && chunks > 0 && k == 10;
        invalid = eligible && !product(rows, topk, expected);
    }
    void observe(int64_t layer, uint64_t offset, uint64_t tokens, const int32_t* ids,
                 uint64_t id_count, uint64_t blob_bytes) {
        if (!eligible || invalid) return;
        if (layer < 0 || layer >= count) { invalid = true; return; }
        auto& c = layers[layer];
        uint64_t expected = 0;
        if (!ids || !tokens || !blob_bytes || offset != c.rows || offset > rows || tokens > rows - offset ||
            c.chunks >= chunks || !product(tokens, topk, expected) || id_count != expected ||
            (c.chunks && c.blob != blob_bytes)) { invalid = true; return; }
        if ((layer && (layers[layer - 1].rows != rows || layers[layer - 1].chunks != chunks)) ||
            (layer + 1 < count && layers[layer + 1].chunks)) { invalid = true; return; }
        uint64_t local[8] = {};
        for (uint64_t i = 0; i < id_count; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= 512) { invalid = true; return; }
            local[e / 64] |= uint64_t(1) << (e % 64);
        }
        uint64_t active = 0, fresh = 0;
        for (int e = 0; e < 512; ++e) {
            const uint64_t mask = uint64_t(1) << (e % 64);
            if (local[e / 64] & mask) {
                ++active;
                if (!(c.bits[e / 64] & mask)) ++fresh;
            }
        }
        if (!add(c.visits, active) || !add(c.unique, fresh) || !add(c.rows, tokens) || !add(c.chunks, 1)) {
            invalid = true; return;
        }
        c.blob = blob_bytes;
        for (int i = 0; i < 8; ++i) c.bits[i] |= local[i];
    }
    bool finish(bool success) {
        completed = false;
        if (!eligible || invalid || !success) return false;
        uint64_t union_total = 0, full_total = 0;
        for (int64_t l = 0; l < count; ++l) {
            const auto& c = layers[l];
            uint64_t selected = 0, full = 0;
            if (c.rows != rows || c.chunks != chunks || !c.blob ||
                !product(c.unique, c.blob, selected) || !product(512, c.blob, full) ||
                !add(union_total, selected) || !add(full_total, full)) { invalid = true; return false; }
        }
        completed = true; return true;
    }
    // Caller invokes only after the existing checked request drain.
    void emit(bool success) {
        finish(success);
        uint64_t selected_total = 0, full_total = 0;
        if (eligible) for (int64_t l = 0; l < count; ++l) {
            const auto& c = layers[l];
            uint64_t selected = 0, full = 0;
            const bool extent = product(c.unique, c.blob, selected) && product(512, c.blob, full);
            if (completed) { add(selected_total, selected); add(full_total, full); }
            std::fprintf(stderr, "strata prefill route union v1: layer=%lld rows=%llu expected_rows=%llu chunks=%llu expected_chunks=%llu active_visits=%llu union_experts=%llu blob_bytes=%llu union_logical_bytes=%llu full_logical_bytes=%llu extent_valid=%d status=%s\n",
                (long long) l, (unsigned long long) c.rows, (unsigned long long) rows,
                (unsigned long long) c.chunks, (unsigned long long) chunks,
                (unsigned long long) c.visits, (unsigned long long) c.unique, (unsigned long long) c.blob,
                (unsigned long long) selected, (unsigned long long) full, (int) extent,
                completed ? "complete" : invalid ? "invalid" : "partial");
        }
        std::fprintf(stderr, "strata prefill route union total v1: layers=%lld rows=%llu chunks=%llu topk=%llu union_logical_bytes=%llu full_logical_bytes=%llu status=%s scope=fresh_request logical_extents_not_bus_traffic=1\n",
            (long long) count, (unsigned long long) rows, (unsigned long long) chunks,
            (unsigned long long) topk, (unsigned long long) selected_total, (unsigned long long) full_total,
            !eligible ? "declined_geometry_or_flow" : completed ? "complete" : invalid ? "invalid" : "partial");
    }
};
static_assert(sizeof(RouteUnion) < 8192, "route union receipt must remain bounded");
} // namespace strata::prefill
