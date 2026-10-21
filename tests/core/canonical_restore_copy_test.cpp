// Original Strata host contract; repository LICENSE applies.
#include "strata/core/canonical_restore_copy.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace d = strata::core::detail;
using Span = std::pair<size_t, size_t>;
static void need(bool value) { if (!value) std::abort(); }
static void spans(size_t bytes, bool chunked, const std::vector<Span>& expected) {
    size_t copies = 99;
    std::vector<Span> actual;
    need(d::submit_canonical_restore(bytes, chunked,
        [&](size_t at, size_t n) {
            need(at <= bytes && n && n <= bytes - at);
            actual.emplace_back(at, n);
            return true;
        }, copies));
    need(actual == expected && copies == expected.size());
}
int main() {
    constexpr size_t C = 67108864;
    static_assert(d::canonical_restore_chunk_bytes == C);
    size_t copies = 99;
    bool called = false;
    for (bool chunked : {false, true}) {
        need(!d::submit_canonical_restore(0, chunked,
            [&](size_t, size_t) { called = true; return true; }, copies));
        need(!called && copies == 0);
    }
    for (bool chunked : {false, true}) spans(1, chunked, {{0, 1}});
    for (bool chunked : {false, true}) spans(C, chunked, {{0, C}});
    spans(C + 1, false, {{0, C + 1}});
    spans(C + 1, true, {{0, C}, {C, 1}});
    spans(C * 2, true, {{0, C}, {C, C}});
    spans(C * 2 + 17, true, {{0, C}, {C, C}, {C * 2, 17}});
    // Independent literal extent/count for the real canonical snapshot.
    constexpr size_t model = 1538035200;
    spans(model, false, {{0, model}});
    size_t cursor = 0;
    need(d::submit_canonical_restore(model, true,
        [&](size_t at, size_t n) {
            need(at == cursor && n == (at == 1476395008 ? 61640192 : C));
            cursor += n;
            return true;
        }, copies));
    need(cursor == model && copies == 23);
    const size_t largest = std::numeric_limits<size_t>::max();
    spans(largest, false, {{0, largest}});
    need(!d::submit_canonical_restore(largest, true,
        [&](size_t at, size_t n) { need(at == 0 && n == C); return false; }, copies));
    need(copies == 0);
    // Refusal after two successful submissions must not call the tail again.
    size_t calls = 0;
    need(!d::submit_canonical_restore(C * 3 + 1, true,
        [&](size_t at, size_t n) {
            need(at == calls * C && n == C);
            return ++calls < 3;
        }, copies));
    need(calls == 3 && copies == 2);
    std::puts("canonical restore copy host: cases=8 passed=1");
}
