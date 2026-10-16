#include "strata/prefill/layermajor_qsa_profile.hpp"
#include <cassert>
#include <memory>
#include <stdexcept>
using P = strata::prefill::detail::LayerMajorQsaProfile;
using B = strata::prefill::detail::LayerMajorQsaBorrow;
static void chunk(P& p, int64_t queries = 8192, bool matrix = true, bool staged = false) {
    for (int k = 0; k < P::section_count; ++k) {
        p.begin(P::Section(k), nullptr);
        if (k == P::attention) p.dispatch(matrix, staged, queries, p.rows[p.layer].cap);
        p.end(P::Section(k), nullptr);
    }
    fake_existing_checked_drain(); // caller's already-existing checked drain
    p.chunk_retired();
}
static void geometry(int chunks) {
    P p;
    const uint64_t total_queries = chunks == 10 ? 78452 : 262144;
    for (int l = 0; l < 48; ++l) if (l % 4 == 3) {
        p.rows[l].expected = chunks;
        p.rows[l].expected_queries = total_queries;
        p.rows[l].cap = 2051;
    }
    const auto records = fake_records, queries = fake_queries, elapsed = fake_elapsed;
    p.open();
    for (int l = 0; l < 48; ++l) {
        p.layer = l;
        if (l % 4 == 3) for (int c = 0; c < chunks; ++c)
            chunk(p, c + 1 == chunks ? int64_t(total_queries - uint64_t(chunks - 1) * 8192) : 8192,
                  c % 2 == 0, c % 3 == 0);
        else p.chunk_retired(); // no marks in GDN chunks
    }
    p.finish(235039);
    assert(p.complete && p.usable);
    assert(fake_records - records == uint64_t(12 * chunks * 14));
    assert(fake_queries - queries == uint64_t(12 * chunks * 14));
    assert(fake_elapsed - elapsed == uint64_t(12 * chunks * 7));
    for (int l = 0; l < 48; ++l) for (int k = 0; k < 7; ++k)
        assert(p.rows[l].samples[k] == (l % 4 == 3 ? uint64_t(chunks) : 0));
    for (int l = 0; l < 48; ++l) {
        const auto& row = p.rows[l];
        if (l % 4 == 3) {
            assert(row.matrix_chunks == uint64_t((chunks + 1) / 2));
            assert(row.fallback_chunks == uint64_t(chunks / 2));
            assert(row.matrix_queries + row.fallback_queries == total_queries);
            assert(row.matrix_queries == uint64_t((chunks + 1) / 2) * 8192);
            assert(row.staged_chunks == uint64_t((chunks + 2) / 3));
            assert(row.cap == 2051);
        } else assert(!row.expected_queries && !row.cap && !row.matrix_chunks && !row.fallback_chunks &&
                      !row.matrix_queries && !row.fallback_queries && !row.staged_chunks);
    }
    p.close();
}
static void missing_geometry() {
    P p; p.rows[3].expected = 2; p.rows[3].expected_queries = 16384; p.rows[3].cap = 2051; p.layer = 3; p.open(); chunk(p);
    p.finish(1); assert(!p.complete && !p.usable);
}
// A local lease/Restore model asserts actual observer destruction is after
// retirement and restoration on both ordinary and exception exits. Main must
// also review the real run_layer_major declaration order/integration seam.
struct Restore { ~Restore() { fake_restored = true; } };
struct Lease { ~Lease() { fake_existing_checked_drain(); } };
static void lifetime(bool throws) {
    P* slot = nullptr;
    fake_require_restore = true; fake_restored = false;
    try {
        std::unique_ptr<P> p(new P);
        B borrow(slot, p.get());
        Restore restore;
        Lease lease;
        p->rows[3].expected = 1; p->layer = 3; p->open();
        p->begin(P::projections, nullptr); // pending marker, no end
        if (throws) throw std::runtime_error("fixture unwind");
    } catch (const std::runtime_error&) {}
    assert(slot == nullptr && fake_restored);
    fake_require_restore = false;
}
static void failure_arms() {
    for (int at : {0, 7, 13}) {
        P p; fake_create_fail_after = at; p.open(); assert(!p.usable); fake_create_fail_after = -1;
    }
    { P p; p.rows[3].expected = 1; p.layer = 3; p.open();
      p.begin(P::selection, nullptr); p.begin(P::selection, nullptr);
      assert(!p.usable); fake_existing_checked_drain(); }
    { P p; p.rows[3].expected = 1; p.layer = 3; p.open();
      fake_record_failure = true; p.begin(P::selection, nullptr); fake_record_failure = false;
      assert(!p.usable); fake_existing_checked_drain(); }
    { P p; p.rows[3].expected = 1; p.layer = 3; p.open();
      p.begin(P::selection, nullptr); fake_record_failure = true; p.end(P::selection, nullptr);
      fake_record_failure = false; assert(!p.usable); fake_existing_checked_drain(); }
    for (int arm = 0; arm < 2; ++arm) {
        P p; p.rows[3].expected = 1; p.layer = 3; p.open();
        p.begin(P::selection, nullptr); p.end(P::selection, nullptr); fake_existing_checked_drain();
        fake_query_failure = arm == 0; fake_elapsed_failure = arm == 1;
        p.chunk_retired(); assert(!p.usable && p.rows[3].samples[P::selection] == 0);
        fake_query_failure = fake_elapsed_failure = false;
    }
}
static void extended_failure_arms() {
    for (int elapsed_value : {1, 2}) {
        P p; p.rows[3].expected = 1; p.layer = 3; p.open();
        p.begin(P::attention, nullptr); p.end(P::attention, nullptr);
        fake_existing_checked_drain(); fake_elapsed_value = elapsed_value;
        p.chunk_retired(); assert(!p.usable && !p.complete && !p.rows[3].samples[P::attention]);
        fake_elapsed_value = 0;
    }
    { // finish must reject a live pair without querying/waiting on it.
        P p; p.rows[3].expected = 1; p.layer = 3; p.open();
        p.begin(P::attention, nullptr);
        const auto queries = fake_queries, elapsed = fake_elapsed;
        p.finish(1); assert(!p.complete && !p.usable);
        assert(fake_queries == queries && fake_elapsed == elapsed);
        fake_existing_checked_drain();
    }
    for (int arm = 0; arm < 5; ++arm) {
        P p; p.rows[3].expected = 1; p.rows[3].expected_queries = 8192; p.rows[3].cap = 2051;
        p.layer = 3; p.open(); chunk(p);
        if (arm == 0) ++p.rows[3].fallback_chunks;
        if (arm == 1) --p.rows[3].matrix_queries;
        if (arm == 2) p.rows[3].staged_chunks = 2;
        if (arm == 3) p.rows[0].cap = 2051; // all GDN fields must be zero
        if (arm == 4) {
            p.rows[7].expected = 1; p.rows[7].expected_queries = 8192; p.rows[7].cap = 2052;
            p.layer = 7; chunk(p); // individually consistent, cross-layer cap differs
        }
        p.finish(1); assert(!p.complete && !p.usable);
    }
    { P p; p.rows[3].expected = 1; p.rows[3].expected_queries = 8192; p.rows[3].cap = 2051;
      p.layer = 3; p.open(); p.dispatch(true, false, 8192, 2052); assert(!p.usable); }
    { P p; p.rows[3].expected = 1; p.rows[3].expected_queries = 8192; p.rows[3].cap = 2051;
      p.layer = 3; p.open(); chunk(p); p.finish(1); assert(p.complete);
      fake_destroy_failure = true; p.close(); fake_destroy_failure = false;
      assert(!p.complete && !p.usable && p.errors == 14); }
}
int main() {
    geometry(10); geometry(32); missing_geometry(); lifetime(false); lifetime(true); failure_arms(); extended_failure_arms();
    for (auto p : fake_owners) assert(p == nullptr);
}
