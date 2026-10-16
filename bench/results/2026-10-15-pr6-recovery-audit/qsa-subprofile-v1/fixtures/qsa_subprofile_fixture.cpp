#include "strata/prefill/layermajor_qsa_profile.hpp"
#include <cassert>
#include <memory>
#include <stdexcept>
using P = strata::prefill::detail::LayerMajorQsaProfile;
using B = strata::prefill::detail::LayerMajorQsaBorrow;
static void chunk(P& p) {
    for (int k = 0; k < P::section_count; ++k) {
        p.begin(P::Section(k), nullptr); p.end(P::Section(k), nullptr);
    }
    fake_existing_checked_drain(); // caller's already-existing checked drain
    p.chunk_retired();
}
static void geometry(int chunks) {
    P p;
    for (int l = 0; l < 48; ++l) p.rows[l].expected = l % 4 == 3 ? chunks : 0;
    const auto records = fake_records, queries = fake_queries, elapsed = fake_elapsed;
    p.open();
    for (int l = 0; l < 48; ++l) {
        p.layer = l;
        if (l % 4 == 3) for (int c = 0; c < chunks; ++c) chunk(p);
        else p.chunk_retired(); // no marks in GDN chunks
    }
    p.finish(235039);
    assert(p.complete && p.usable);
    assert(fake_records - records == uint64_t(12 * chunks * 14));
    assert(fake_queries - queries == uint64_t(12 * chunks * 14));
    assert(fake_elapsed - elapsed == uint64_t(12 * chunks * 7));
    for (int l = 0; l < 48; ++l) for (int k = 0; k < 7; ++k)
        assert(p.rows[l].samples[k] == (l % 4 == 3 ? uint64_t(chunks) : 0));
    p.close();
}
static void missing_geometry() {
    P p; p.rows[3].expected = 2; p.layer = 3; p.open(); chunk(p);
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
    { P p; fake_create_fail_after = 3; p.open(); assert(!p.usable); fake_create_fail_after = -1; }
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
int main() {
    geometry(10); geometry(32); missing_geometry(); lifetime(false); lifetime(true); failure_arms();
    for (auto p : fake_owners) assert(p == nullptr);
}
