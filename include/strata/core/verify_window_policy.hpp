#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::core {

// Prompt reads through the decode weight layout still have an independent
// execution policy. Sampling is a separate choice and does not identify phase.
enum class VerifyInput : uint8_t { prompt, decode };

constexpr bool valid_verify_input(VerifyInput input) {
    return input == VerifyInput::prompt || input == VerifyInput::decode;
}

// e44575158:q6_dense_c4_component_v2: these four non-head shapes won in
// all three fresh processes against the production columns entry. Keep the
// losing512-row shape and the separately faster sole-packed head unchanged.
inline constexpr std::array<std::array<int, 2>, 4> verify_q6_dense_dpas_shapes{{
    {6144, 2560}, {2560, 6144}, {2560, 10240}, {2560, 12288}
}};

constexpr std::size_t verify_q6_dense_dpas_index(int n_in, int n_out) {
    for (std::size_t i = 0; i < verify_q6_dense_dpas_shapes.size(); ++i)
        if (verify_q6_dense_dpas_shapes[i] == std::array<int, 2>{n_in, n_out}) return i;
    return verify_q6_dense_dpas_shapes.size();
}

constexpr bool verify_q6_dpas_dense(bool enabled, VerifyInput input, bool exact,
                                   int type, int n_in, int n_out, int columns) {
    return enabled && input == VerifyInput::decode && exact && type == 14 && columns == 4 &&
        verify_q6_dense_dpas_index(n_in, n_out) < verify_q6_dense_dpas_shapes.size();
}

// 0ccc4ed45:q6_canonical_component_v4: only the full target head/C4 won.
// The enabled owner supplies unchanged canonical Q6 rows. Never select the
// slower 2560x640 matrix. This is eligibility for a model trial only.
constexpr bool verify_q6_dpas_head(bool dpas_ready, VerifyInput input, bool exact,
                                  int type, int n_in, int n_out, int columns) {
    return dpas_ready && input==VerifyInput::decode && exact && type==14 &&
           n_in==2560 && n_out==248320 && columns==4;
}

// Only these shapes passed all three independent process medians in
// 733500802:q6_byte_load_component_v2. This is eligibility for a model comparison,
// not model adoption. Every other type, phase and shape keeps its reference.
constexpr bool verify_q6_columns(bool enabled, VerifyInput input, bool exact,
                                 int type, int n_in, int n_out, int columns) {
    if (!enabled || input != VerifyInput::decode || !exact || type != 14 ||
        (columns != 1 && columns != 2 && columns != 3 && columns != 4 && columns != 8)) return false;
    if (n_in == 6144 && n_out == 2560) return true;
    if (n_in != 2560) return false;
    if (n_out == 10240 || n_out == 12288 || n_out == 248320) return true;
    if (n_out == 6144) return true;
    if (n_out == 640) return columns != 2;
    return n_out == 512 && columns == 8;
}

// 2197e5063:q6_centering_component_v2:18 shapes passed every independent
// process median. Both arms use the same column geometry. Model gain remains
// unqualified; retain the scalar column entry outside this opt-in decode set.
constexpr bool verify_q6_centering(bool enabled, VerifyInput input, bool exact,
                                  int type, int n_in, int n_out, int columns) {
    if (!verify_q6_columns(enabled, input, exact, type, n_in, n_out, columns)) return false;
    if (n_in == 6144 && n_out == 2560) return columns == 1 || columns == 3 || columns == 4;
    if (n_in != 2560) return false;
    if (n_out == 10240) return columns != 4;
    if (n_out == 6144) return columns == 3;
    return n_out == 12288 || n_out == 248320;
}

// 3e6b836a4:q6_native_half_component_v3. Intersect the27 independently
// winning cells with the existing30-cell columns selector above:26 remain.
// In particular512x4 stays on its incumbent,despite its component gain.
// This is an opt-in comparison policy,not model adoption.
constexpr bool verify_q6_native_half(bool enabled, VerifyInput input, bool exact,
                                     int type, int n_in, int n_out, int columns) {
    return verify_q6_columns(enabled, input, exact, type, n_in, n_out, columns) &&
           n_out != 640;
}

enum class VerifyQ6Route : uint8_t { reference, columns, centered, native_half };

// Handles borrow session/weight operands. The caller must complete both owner
// queues before clearing. A failing destroy must throw or fail-stop; already
// destroyed handles are nulled so an interrupted host-only cleanup cannot
// destroy them twice. No runtime calls or process environment changes here.
template<class Handle>
class VerifyWindowGraphCache {
public:
    VerifyWindowGraphCache() = default;
    VerifyWindowGraphCache(const VerifyWindowGraphCache&) = delete;
    VerifyWindowGraphCache& operator=(const VerifyWindowGraphCache&) = delete;
    struct Window {
        std::vector<Handle> layer, resident, pcie;
        uint64_t q6_reference_calls = 0, q6_columns_calls = 0, q6_centered_calls = 0,
                 q6_native_half_calls = 0;
        // Packed target-head route is counted separately from canonical-row
        // MMVQ routes, preserving the existing Q6_ROUTE/SHAPE wire schema.
        uint64_t q6_dpas_head_calls = 0;
        uint64_t q6_packed_head_calls = 0;
        // Separate from the four-arm legacy Q6_SHAPE schema. These are static
        // capture counts, never a per-replay global/atomic counter.
        std::array<uint64_t, verify_q6_dense_dpas_shapes.size()> q6_dpas_dense_calls{};
        struct Q6Shape {
            int n_in, n_out, columns;
            std::array<uint64_t, 4> calls{};
        };
        // Optional host-only audit. No device allocation or execution event;
        // owner reads these only at a serialized request boundary.
        static constexpr std::size_t max_q6_shapes = 64;
        std::vector<Q6Shape> q6_shapes;
        uint64_t q6_capture_id = 0, q6_cache_hits = 0,
                 q6_replay_started = 0, q6_replay_completed = 0;
        bool q6_routes_complete = true;
        bool ready = false; // A nonempty partial capture must never be replayed.

        void record_q6(int n_in, int n_out, int columns, VerifyQ6Route route) {
            const std::size_t arm = static_cast<std::size_t>(route);
            if (arm >= 4) { q6_routes_complete = false; return; }
            for (auto& shape : q6_shapes)
                if (shape.n_in == n_in && shape.n_out == n_out && shape.columns == columns) {
                    ++shape.calls[arm];
                    return;
                }
            if (q6_shapes.size() == max_q6_shapes) { q6_routes_complete = false; return; }
            q6_shapes.push_back({n_in, n_out, columns, {}});
            ++q6_shapes.back().calls[arm];
        }

        template<class Destroy>
        void clear(Destroy&& destroy) {
            for (auto* family : {&layer, &resident, &pcie}) {
                for (auto& handle : *family) if (handle) {
                    destroy(handle);
                    handle = {};
                }
                family->clear();
            }
            q6_reference_calls = q6_columns_calls = q6_centered_calls = q6_native_half_calls = 0;
            q6_dpas_head_calls = 0;
            q6_packed_head_calls = 0;
            q6_dpas_dense_calls.fill(0);
            q6_shapes.clear();
            q6_capture_id = q6_cache_hits = q6_replay_started = q6_replay_completed = 0;
            q6_routes_complete = true;
            ready = false;
        }
    };

    Window& at(VerifyInput input, int tokens) {
        if (!valid_verify_input(input) || tokens < 1 || tokens > 8)
            throw std::out_of_range("verify graph phase/token key");
        return windows_[input == VerifyInput::decode ? 1 : 0][static_cast<std::size_t>(tokens)];
    }

    const Window& at(VerifyInput input, int tokens) const {
        if (!valid_verify_input(input) || tokens < 1 || tokens > 8)
            throw std::out_of_range("verify graph phase/token key");
        return windows_[input == VerifyInput::decode ? 1 : 0][static_cast<std::size_t>(tokens)];
    }

    template<class Destroy>
    void clear(Destroy&& destroy) {
        for (auto& phase : windows_) for (auto& window : phase) window.clear(destroy);
    }

private:
    std::array<std::array<Window, 9>, 2> windows_{};
};

} // namespace strata::core
