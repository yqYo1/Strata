// Independent project fixture under repository LICENSE. Main alone builds/runs.
#include "strata/core/validation_control.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <stdexcept>

using namespace strata::core::detail;

static void progress_cases() {
    int errors = 0;
    auto poll = [&](bool complete, bool rang) {
        return ring_progress([&] { return complete; }, [&] { ++errors; }, [&] { return rang; });
    };
    assert(poll(false, false) == RingProgress::pending);
    assert(poll(true, false) == RingProgress::complete_without_ring);
    assert(poll(true, true) == RingProgress::reached);
    assert(poll(false, true) == RingProgress::reached); // later queue work may still be pending
    assert(errors == 4);
    bool ring_read = false;
    assert(ring_progress([]() -> bool { throw std::runtime_error("query"); },
                         [] {}, [&] { ring_read = true; return true; }) == RingProgress::query_error);
    assert(!ring_read);
    assert(ring_progress([] { return false; },
                         [] { throw std::runtime_error("async consumer"); },
                         [] { return true; }) == RingProgress::query_error);
    // A long logical sequence of successful NOT_READY queries must remain
    // pending. No GPU, provider or wall-clock performance claim is involved.
    for (int tick = 0; tick < 10001; ++tick) {
        const auto state = poll(tick == 10000, tick == 10000);
        assert(state == (tick == 10000 ? RingProgress::reached : RingProgress::pending));
    }
}

static void span_cases() {
    int token = 0;
    float residual = 0;
    const auto full = {int64_t(262144), int64_t(262144)};
    assert(prefill_admission(&residual, &token, 0, 262144, full));
    assert(prefill_admission(&residual, &token, 262143, 1, full));
    assert(!prefill_admission(&residual, &token, 262144, 1, full));
    assert(!prefill_admission(&residual, &token, 0, 262145, full));
    assert(prefill_admission(nullptr, nullptr, 262144, 0, full));
    assert(!prefill_admission(nullptr, nullptr, 262145, 0, full));
    assert(!prefill_admission(nullptr, &token, 0, 1, full));
    assert(!prefill_admission(&residual, nullptr, 0, 1, full));
    assert(!prefill_admission(&residual, &token, -1, 1, full));
    assert(!prefill_admission(&residual, &token, INT64_MIN, 1, full));
    assert(!prefill_admission(&residual, &token, INT64_MAX, 1, full));
    assert(!prefill_admission(&residual, &token, 0, INT64_MAX, full));
    assert(!prefill_admission(&residual, &token, 0, INT64_MIN, full));
    assert(!prefill_admission(&residual, &token, INT64_MAX - 1, 8, full));
    assert(!prefill_admission(&residual, &token, 0, -1, full));
    assert(!position_span(0, 1, {0, 262144}));
    assert(!position_span(0, 1, {-1, 262144}));
    assert(position_span(99, 1, {100, 200}));
    assert(!position_span(100, 1, {100, 200}));
    assert(!position_span(100, 1, {200, 100}));
    assert(!position_span(262144, 1, {INT64_MAX, INT64_MAX}));
    assert(!position_span(0, 1, full, INT64_MAX));
    assert(!position_span(0, 1, full, -1));
    // Catch-up at 262140..262141; chain from accepted row 1 writes
    // 262141..262142, leaving verifying position 262143 in context.
    assert(draft_admission(&residual, &token, &token, 262140, 2, 1, 4, 2, full));
    assert(draft_admission(&residual, &token, &token, 262141, 2, 1, 4, 1, full));
    assert(!draft_admission(&residual, &token, &token, 262142, 2, 1, 4, 1, full));
    // A valid catch-up span can still lack room for the draft chain/lookahead.
    assert(!draft_admission(&residual, &token, &token, 262141, 2, 1, 4, 3, full));
    assert(!draft_admission(&residual, &token, &token, INT64_MAX, 1, 0, 4, 1, full));
    assert(!draft_admission(&residual, &token, &token, INT64_MIN, 1, 0, 4, 1, full));
    assert(!draft_admission(&residual, &token, &token, -1, 1, 0, 4, 1, full));
    assert(!draft_admission(&residual, &token, &token, 0, 0, 0, 4, 1, full));
    assert(!draft_admission(&residual, &token, &token, 0, 1, -1, 4, 1, full));
    assert(!draft_admission(&residual, &token, &token, 0, 1, 1, 4, 1, full));
    assert(!draft_admission(&residual, &token, &token, 0, 1, 0, INT32_MAX, 1, full));
    assert(!draft_admission(&residual, &token, nullptr, 0, 1, 0, 4, 1, full));
    assert(!draft_admission(nullptr, &token, &token, 0, 1, 0, 4, 1, full));
    assert(!draft_admission(&residual, nullptr, &token, 0, 1, 0, 4, 1, full));
    assert(!draft_admission(&residual, &token, &token, 98, 2, 1, 4, 1, {100, 200}));
}

static void load_cases() {
    // The actual NativeEmbed member gate, not a model-shaped stand-in. Failure
    // at any point after begin consumes the attempt. Payload parsing, allocation
    // and DMA need separate Main qualification.
    OneAttemptLoad initial_failure;
    assert(initial_failure.begin());
    assert(!initial_failure.begin());
    assert(!initial_failure.begin());
    OneAttemptLoad success;
    assert(success.begin());
    assert(!success.begin()); // repeated supported reload never reaches mutation
    assert(!success.begin());
    OneAttemptLoad fresh;
    assert(fresh.begin()); // a fresh owner is the only retry path
}

int main() {
    static_assert(!concurrent_doorbell_admitted());
    progress_cases();
    span_cases();
    load_cases();
}
