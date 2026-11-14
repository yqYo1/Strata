// Main-authored prefill selection; repository LICENSE.
#pragma once
#include <cstdint>
#include <string_view>

namespace strata::prefill {
enum class Strategy { ttft, throughput };

constexpr bool valid_strategy(Strategy strategy) noexcept {
    return strategy == Strategy::ttft || strategy == Strategy::throughput;
}
constexpr const char* strategy_name(Strategy strategy) noexcept {
    return strategy == Strategy::ttft ? "ttft" : strategy == Strategy::throughput ? "throughput" : "invalid";
}
constexpr bool parse_strategy(std::string_view value, Strategy& strategy) noexcept {
    if (value == "ttft") { strategy = Strategy::ttft; return true; }
    if (value == "throughput") { strategy = Strategy::throughput; return true; }
    return false;
}
// This is route eligibility, not a tuned automatic strategy threshold. A later
// request-length resolver chooses the enum before entering prompt orchestration.
constexpr bool uses_layer_major(Strategy strategy, int64_t rows, int64_t chunk) noexcept {
    return strategy == Strategy::throughput && chunk > 0 && rows > chunk;
}
} // namespace strata::prefill
