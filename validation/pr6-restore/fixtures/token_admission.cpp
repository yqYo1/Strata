// New project fixture, repository LICENSE. Main alone builds/runs it.
#include "strata/core/token_admission.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <vector>

int main() {
    using strata::core::detail::embedding_token;
    using strata::core::detail::embedding_tokens;
    constexpr int64_t vocab = 248320;
    assert(embedding_token(0, vocab));
    assert(embedding_token(vocab - 1, vocab));
    assert(!embedding_token(vocab, vocab));
    assert(!embedding_token(-1, vocab));
    assert(!embedding_token(INT32_MAX, vocab));
    assert(!embedding_token(int64_t(INT32_MAX) + 1, vocab));
    assert(!embedding_token(INT64_MAX, vocab));
    assert(!embedding_token(0, 0));
    assert(!embedding_token(0, int64_t(INT32_MAX) + 1));
    const int64_t ids[] = {0, vocab - 1, vocab};
    assert(embedding_tokens(ids, 2, vocab));
    assert(!embedding_tokens(ids, 3, vocab));
    assert(!embedding_tokens<int32_t>(nullptr, 1, vocab));
    assert(!embedding_tokens(ids, -1, vocab));
    std::vector<int32_t> physical(262144, int32_t(vocab - 1));
    assert(embedding_tokens(physical.data(), 262144, vocab));
    physical.back() = int32_t(vocab);
    assert(!embedding_tokens(physical.data(), 262144, vocab));
    assert(!embedding_tokens(physical.data(), 262145, vocab));
}
