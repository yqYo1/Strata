// Diagnostic host ledger only. No queue/event ownership or recovery contract.
#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <limits>

namespace strata::prefill::detail {
struct Iq4nlCallerCensus {
    static constexpr size_t layers = 48, experts = 512;
    struct Expert {
        uint64_t decisions = 0, rows = 0, n = 0;
        uint64_t selected[2]{}, returned[2]{}, compressed = 0, output = 0;
        int gu_type = -1, down_type = -1;
        int64_t embd = 0, ff = 0;
    };
    struct Layer {
        uint64_t chunks = 0, positions = 0, routed = 0, expected_rows = 0;
    };
    std::array<Expert, layers * experts> expert{};
    std::array<Layer, layers> layer{};
    uint64_t batched = 0;
    int64_t model_layers = 0, model_experts = 0;
    bool active = false, overflow = false, invalid = false, unsupported = false;
    bool layer_major = false, private_enabled = false;
    void add(uint64_t& dst, uint64_t value) noexcept {
        if (value > std::numeric_limits<uint64_t>::max() - dst) { overflow = true; return; }
        dst += value;
    }
    uint64_t mul(uint64_t a, uint64_t b) noexcept {
        if (b && a > std::numeric_limits<uint64_t>::max() / b) { overflow = true; return 0; }
        return a * b;
    }
    void begin(int64_t n, int64_t nl, int64_t ne, bool major, bool priv, bool unsupported_scope) noexcept {
        // Fixed storage was allocated in init; reset occurs before run_impl's timer.
        expert = {}; layer = {};
        active = true; overflow = false; unsupported = unsupported_scope;
        invalid = n <= 0 || nl <= 0 || nl > int64_t(layers) || ne <= 0 || ne > int64_t(experts);
        batched = n > 0 ? uint64_t(n) : 0; model_layers = nl; model_experts = ne;
        layer_major = major; private_enabled = priv;
    }
    bool bounds(int64_t l, int64_t e = 0) noexcept {
        if (!active) return false;
        if (l < 0 || l >= model_layers || l >= int64_t(layers) || e < 0 || e >= model_experts || e >= int64_t(experts)) {
            invalid = true; return false;
        }
        return true;
    }
    void chunk(int64_t l, int64_t c0, int64_t t) noexcept {
        if (!bounds(l)) return;
        auto& a = layer[size_t(l)];
        if (c0 < 0 || t <= 0 || uint64_t(c0) != a.positions || uint64_t(t) > batched - (a.positions <= batched ? a.positions : batched)) {
            invalid = true; return;
        }
        add(a.chunks, 1); add(a.positions, uint64_t(t));
    }
    void routed(int64_t l, int64_t t, int64_t k, uint64_t rows) noexcept {
        if (!bounds(l) || t < 0 || k < 0) { invalid = true; return; }
        add(layer[size_t(l)].routed, rows);
        add(layer[size_t(l)].expected_rows, mul(uint64_t(t), uint64_t(k)));
    }
    void decision(int64_t l, int64_t e, int64_t rows) noexcept {
        if (!bounds(l, e) || rows <= 0) { invalid = true; return; }
        auto& a = expert[size_t(l) * experts + size_t(e)];
        add(a.decisions, 1); add(a.rows, uint64_t(rows));
    }
    void selected(int64_t l, int64_t e, int gt, int dt, int64_t embd, int64_t ff, bool priv) noexcept {
        if (!bounds(l, e)) return;
        auto& a = expert[size_t(l) * experts + size_t(e)];
        if (embd <= 0 || ff <= 0) { invalid = true; return; }
        const auto n = mul(uint64_t(embd), uint64_t(ff));
        if (a.selected[0] || a.selected[1]) {
            if (a.gu_type != gt || a.down_type != dt || a.embd != embd || a.ff != ff || a.n != n) invalid = true;
        }
        a.gu_type = gt; a.down_type = dt; a.embd = embd; a.ff = ff; a.n = n;
        if (dt == 20 && n % 32) invalid = true;
        if (priv && dt != 20) invalid = true;
        add(a.selected[priv ? 1 : 0], 1);
    }
    void returned(int64_t l, int64_t e, bool priv) noexcept {
        if (!bounds(l, e)) return;
        auto& a = expert[size_t(l) * experts + size_t(e)];
        add(a.returned[priv ? 1 : 0], 1);
        if (a.down_type == 20) {
            add(a.compressed, mul(a.n / 32, 18));
            add(a.output, mul(a.n, 2));
        }
    }
    void report(bool normal_return, bool pipeline_drained) noexcept {
        if (!active) return;
        bool incomplete = !normal_return || !pipeline_drained;
        for (int64_t l = 0; l < model_layers && l < int64_t(layers); ++l) {
            auto& c = layer[size_t(l)];
            if (c.positions != batched || c.routed != c.expected_rows) invalid = true;
            uint64_t decisions = 0, rows = 0, sel[2]{}, ret[2]{}, type20[2]{}, bytes = 0, out = 0;
            const Expert* shape = nullptr;
            for (size_t e = 0; e < experts; ++e) {
                const auto& a = expert[size_t(l) * experts + e];
                if (a.selected[0] || a.selected[1]) {
                    if (!shape) shape = &a;
                    else if (shape->gu_type != a.gu_type || shape->down_type != a.down_type ||
                             shape->embd != a.embd || shape->ff != a.ff || shape->n != a.n) invalid = true;
                }
                add(decisions, a.decisions); add(rows, a.rows);
                for (int w = 0; w < 2; ++w) {
                    add(sel[w], a.selected[w]); add(ret[w], a.returned[w]);
                    if (a.down_type == 20) add(type20[w], a.returned[w]);
                    if (a.selected[w] != a.returned[w]) incomplete = true;
                }
                uint64_t selected_total = a.selected[0];
                add(selected_total, a.selected[1]);
                if (a.decisions != selected_total) invalid = true;
                add(bytes, a.compressed); add(out, a.output);
            }
            if (rows != c.routed) invalid = true;
            if (shape)
                std::fprintf(stderr, "strata IQ4NL census shape layer=%lld gu_type=%d down_type=%d n_embd=%lld n_ff=%lld n=%llu arithmetic=native-IQ-to-FP16-then-oneMKL\n",
                    (long long)l, shape->gu_type, shape->down_type, (long long)shape->embd,
                    (long long)shape->ff, (unsigned long long)shape->n);
            std::fprintf(stderr, "strata IQ4NL census layer=%lld chunks=%llu positions=%llu routed_rows=%llu expected_rows=%llu decisions=%llu generic_selected=%llu generic_returned=%llu private_selected=%llu private_returned=%llu type20_generic_returned=%llu type20_private_returned=%llu type20_compressed_read_bytes=%llu type20_fp16_output_bytes=%llu\n",
                (long long)l, (unsigned long long)c.chunks, (unsigned long long)c.positions,
                (unsigned long long)c.routed, (unsigned long long)c.expected_rows,
                (unsigned long long)decisions, (unsigned long long)sel[0], (unsigned long long)ret[0],
                (unsigned long long)sel[1], (unsigned long long)ret[1], (unsigned long long)type20[0],
                (unsigned long long)type20[1], (unsigned long long)bytes, (unsigned long long)out);
        }
        std::fprintf(stderr, "strata IQ4NL census scope=native-direct-down build=1 ledger_bytes=%zu expert_capacity=%zu layer_capacity=%zu n_batched=%llu at_least_32768=%d layer_major=%d private_optin=%d normal_return=%d pipeline_drained=%d overflow=%d invalid=%d unsupported=%d incomplete=%d async_success=UNIMPLEMENTED gate=UNQUALIFIED decode_gate=UNOBSERVED full_context_gate=UNOBSERVED bytes_scope=returned_type20_calls staging_bytes=UNOBSERVED wrapper_return_is_gpu_completion=0\n",
            sizeof(*this), layers * experts, layers, (unsigned long long)batched, batched >= 32768,
            layer_major, private_enabled, normal_return, pipeline_drained, overflow, invalid,
            unsupported, incomplete);
        active = false;
    }
};
} // namespace strata::prefill::detail
