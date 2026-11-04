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
        uint64_t q6_reference_calls = 0, q6_columns_calls = 0;
        bool ready = false; // A nonempty partial capture must never be replayed.

        template<class Destroy>
        void clear(Destroy&& destroy) {
            for (auto* family : {&layer, &resident, &pcie}) {
                for (auto& handle : *family) if (handle) {
                    destroy(handle);
                    handle = {};
                }
                family->clear();
            }
            q6_reference_calls = q6_columns_calls = 0;
            ready = false;
        }
    };

    Window& at(VerifyInput input, int tokens) {
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
