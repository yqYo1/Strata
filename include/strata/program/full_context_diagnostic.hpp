#pragma once

#include <cstddef>
#include <cstdint>

// Main-authored diagnostic policy; the repository LICENSE applies.
// Only the separately configured diagnostic executable includes this header.
namespace strata::program::full_context_diagnostic {
constexpr int64_t kContext = 262144;

inline bool request(int64_t context, std::size_t prompt_tokens, int64_t new_tokens) noexcept {
    return context == kContext && prompt_tokens == std::size_t(kContext) && new_tokens == 0;
}

inline bool last_window(bool first, int64_t prompt_tokens, int64_t position, int width,
                        int32_t token, int64_t final_prompt_token, std::size_t produced) noexcept {
    return first && prompt_tokens == kContext && position == kContext - 1 && width == 1 &&
           int64_t(token) == final_prompt_token && produced == 0;
}
} // namespace strata::program::full_context_diagnostic
