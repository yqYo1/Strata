// New project file, repository LICENSE. Boundary cases from PR6's unguarded
// public entry and residual allocation contract; no external implementation.
#include "strata/prefill/range_guard.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <limits>
using namespace strata::prefill::detail;
int main() {
    // A continuation ending exactly at the physical limit, plus one invalid row.
    assert(pr6_position_span(131072, 131072, 262144));
    assert(!pr6_position_span(131073, 131072, 262144));
    assert(pr6_position_span(262144, 0, 262146)); // Reserved cells are not input.
    assert(!pr6_position_span(262145, 0, 262146));
    assert(pr6_position_span(32768, 0, 65538)); // Original receipt conditions.
    assert(pr6_position_span(0, 65538, 65538));
    assert(!pr6_position_span(1, 65538, 65538));
    assert(!pr6_position_span(0, 65539, 65538));
    assert(!pr6_position_span(INT64_MAX, 1, INT64_MAX));
    assert(!pr6_position_span(2, INT64_MAX, INT64_MAX));
    assert(!pr6_position_span(-1, 0, 65538));
    assert(!pr6_position_span(1, -1, 65538));
    assert(!pr6_position_span(1, 0, -1));
    assert(pr6_product(32768, 10240 * sizeof(float)) == size_t(1342177280));
    assert(pr6_product(0, std::numeric_limits<size_t>::max()) == 0);
    bool rejected = false;
    try { (void)pr6_product(std::numeric_limits<size_t>::max(), 2); }
    catch (const std::overflow_error&) { rejected = true; }
    assert(rejected);
}
