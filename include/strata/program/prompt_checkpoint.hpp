#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace strata::program {

// Niko's opt-in trailing-effort turn stays outside the reusable checkpoint.
// This selects a state boundary; it never removes tokens from the request.
inline int64_t checkpoint_turn(std::span<const int64_t> ids, int64_t resume,
                               int64_t turn_token, int64_t tail_role_token) {
    if (ids.size() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        resume < 0 || static_cast<uint64_t>(resume) > ids.size())
        throw std::invalid_argument("invalid prompt checkpoint resume position");
    int64_t at = -1;
    if (turn_token >= 0)
        for (int64_t i = static_cast<int64_t>(ids.size()) - 1; i > resume; --i)
            if (ids[static_cast<std::size_t>(i)] == turn_token) { at = i; break; }
    if (at > 0 && tail_role_token >= 0)
        for (int64_t i = at - 1; i > resume; --i)
            if (ids[static_cast<std::size_t>(i)] == turn_token) {
                if (ids[static_cast<std::size_t>(i + 1)] == tail_role_token) at = i;
                break;
            }
    return at;
}

} // namespace strata::program
