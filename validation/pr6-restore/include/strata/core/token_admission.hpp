#pragma once
// New project code, repository LICENSE; no external implementation reference.
#include <cstdint>
#include <limits>

namespace strata::core::detail {
inline bool embedding_token(int64_t token, int64_t vocabulary) noexcept {
    return vocabulary > 0 && vocabulary <= std::numeric_limits<int32_t>::max() &&
           token >= 0 && token < vocabulary && token <= std::numeric_limits<int32_t>::max();
}
template<class Token>
inline bool embedding_tokens(const Token* ids, int64_t n, int64_t vocabulary) noexcept {
    if (n < 0 || n > 262144 || (n && !ids)) return false;
    for (int64_t i = 0; i < n; ++i) if (!embedding_token(ids[i], vocabulary)) return false;
    return vocabulary > 0 && vocabulary <= std::numeric_limits<int32_t>::max();
}
} // namespace strata::core::detail
