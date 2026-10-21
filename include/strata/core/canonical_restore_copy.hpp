#pragma once
// Original Strata implementation; repository LICENSE applies.
#include <algorithm>
#include <cstddef>

namespace strata::core::detail {
inline constexpr size_t canonical_restore_chunk_bytes = 64ULL << 20;

// Schedule the original immutable snapshot on its existing queue. The caller
// owns the final checked drain and publication. A refused submission stops the
// schedule immediately; it never publishes or releases either owner here.
template<class Submit>
bool submit_canonical_restore(size_t bytes, bool chunked, Submit&& submit,
                              size_t& submitted_copies) {
    submitted_copies = 0;
    if (!bytes) return false;
    const size_t limit = chunked ? canonical_restore_chunk_bytes : bytes;
    for (size_t at = 0; at < bytes;) {
        const size_t count = std::min(limit, bytes - at);
        if (!submit(at, count)) return false;
        ++submitted_copies;
        at += count; // count <= bytes-at, including the SIZE_MAX boundary
    }
    return true;
}
} // namespace strata::core::detail
