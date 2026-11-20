#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Main-authored diagnostic policy; the repository LICENSE applies.
// Only the separately configured diagnostic executable includes this header.
namespace strata::program::full_context_diagnostic {
constexpr int64_t kContext = 262144;

// Token-only stdin command for the diagnostic server. A null byte would
// otherwise let the C-string token parser silently ignore part of the request.
inline const char* server_prompt_ids(const std::string& line) noexcept {
    constexpr char prefix[] = "CONSUME_FULL ";
    constexpr auto length = sizeof(prefix) - 1;
    if (line.compare(0, length, prefix) != 0 || line.find('\0') != std::string::npos)
        return nullptr;
    return line.c_str() + length;
}

inline bool request(int64_t context, std::size_t prompt_tokens, int64_t new_tokens) noexcept {
    return context == kContext && prompt_tokens == std::size_t(kContext) && new_tokens == 0;
}

// One row belongs to the final verifier. The cache must leave enough rows
// before it for a batched prefill loan, even before any turn-boundary split.
inline bool can_borrow_after_prefix(int64_t prefix, int64_t short_read, bool windows_enabled) noexcept {
    if (prefix < 0 || prefix >= kContext - 1) return false;
    return !windows_enabled || kContext - 1 - prefix > short_read;
}

inline bool last_window(bool first, int64_t prompt_tokens, int64_t position, int width,
                        int32_t token, int64_t final_prompt_token, std::size_t produced) noexcept {
    return first && prompt_tokens == kContext && position == kContext - 1 && width == 1 &&
           int64_t(token) == final_prompt_token && produced == 0;
}
} // namespace strata::program::full_context_diagnostic
