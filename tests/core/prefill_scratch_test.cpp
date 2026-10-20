// Original Strata host contracts; repository LICENSE applies.
#include "strata/prefill/scratch_allocation.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
namespace pd = strata::prefill::detail;
constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
void need(bool ok) { if (!ok) throw std::runtime_error("scratch contract failed"); }
template<class F> void rejects(F&& f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    need(rejected);
}
void ready(pd::ScratchLayoutState& state) {
    pd::ScratchLayoutState::Update update(state, true);
    need(update.started() && !state.ready()); update.commit();
    need(state.ready() && state.generation() == 1);
}
void test(const std::string& name) {
    if (name == "guard_alignment") {
        for (uint64_t n : {0ull, 1ull, 63ull, 64ull, 255ull, 256ull, 257ull, 8192ull}) {
            const auto r = pd::reserve_scratch(0, n, 4);
            need(r.bytes == ((n * 4 + 511) & ~uint64_t(255)) && r.next == r.bytes);
        }
    } else if (name == "shape_bounds") {
        constexpr int64_t limit = std::numeric_limits<int32_t>::max();
        need(pd::scratch_rows(8192, 512, 262144, 10) == 8192);
        need(pd::scratch_rows(limit / 10, limit, limit, 10) == uint64_t(limit / 10));
        for (int64_t n : {int64_t(-1), int64_t(0), limit / 10 + 1})
            rejects([&] { (void)pd::scratch_rows(n, 512, 262144, 10); });
        rejects([&] { (void)pd::scratch_rows(8192, limit + 1, 262144, 10); });
        rejects([&] { (void)pd::scratch_rows(8192, 512, limit + 1, 10); });
        rejects([] { (void)pd::scratch_rows(8192, 512, 262144, 0); });
    } else if (name == "gemm_64mib") {
        need(pd::reserve_scratch(0, 32ull << 20, 2).bytes == (64ull << 20) + 256);
    } else if (name == "exact_capacity") {
        const auto a = pd::reserve_scratch(0, 8192 * 10240, 4);
        const auto b = pd::reserve_scratch(a.next, 8192 * 2560, 4);
        need(pd::reserve_scratch(a.next, 8192 * 2560, 4, b.next).next == b.next);
        rejects([&] { (void)pd::reserve_scratch(a.next, 8192 * 2560, 4, b.next - 1); });
    } else if (name == "payload_overflow") {
        rejects([] { (void)pd::reserve_scratch(0, maximum / 2 + 1, 2); });
        rejects([] { (void)pd::reserve_scratch(0, 0, 0); });
    } else if (name == "guard_overflow") {
        rejects([] { (void)pd::reserve_scratch(0, maximum - 510, 1); });
        need(pd::reserve_scratch(0, maximum - 511, 1).next == maximum - 255);
    } else if (name == "used_overflow") {
        rejects([] { (void)pd::reserve_scratch(maximum - 255, 0, 1); });
    } else if (name == "capacity_bounds") {
        rejects([] { (void)pd::reserve_scratch(513, 0, 1, 512); });
        rejects([] { (void)pd::reserve_scratch(0, 0, 1, 255); });
        need(pd::reserve_scratch(0, 0, 1, 256).next == 256);
    } else if (name == "slack_overflow") {
        rejects([] { (void)pd::scratch_with_slack(maximum - 7, 8); });
        need(pd::scratch_with_slack(maximum - 8, 8) == maximum);
    } else if (name == "initial_layout") {
        pd::ScratchLayoutState state; need(!state.ready()); ready(state);
        pd::ScratchLayoutState::Update duplicate(state, true); need(!duplicate.started() && state.ready());
    } else if (name == "relayout_generation") {
        pd::ScratchLayoutState state; ready(state);
        pd::ScratchLayoutState::Update update(state, false);
        need(update.started() && !state.ready()); update.commit();
        need(state.ready() && state.generation() == 2);
    } else if (name == "failed_init") {
        pd::ScratchLayoutState state;
        { pd::ScratchLayoutState::Update update(state, true); need(update.started()); }
        need(!state.ready() && state.generation() == 0);
        pd::ScratchLayoutState::Update retry(state, true); need(!retry.started());
    } else if (name == "failed_relayout") {
        // Inject host exceptions at retirement, bind, early/late layout takes
        // and final validation. This exercises the real state guard, not GPU
        // queue failures or Prefill::carve/Gemm fault injection.
        for (int fail_at = 0; fail_at != 9; ++fail_at) {
            pd::ScratchLayoutState state; ready(state);
            try {
                pd::ScratchLayoutState::Update update(state, false); need(update.started());
                for (int phase = 0; phase != 9; ++phase)
                    if (phase == fail_at) throw std::runtime_error("injected host phase failure");
                update.commit();
            } catch (const std::runtime_error&) {}
            need(!state.ready() && state.generation() == 1);
            pd::ScratchLayoutState::Update retry(state, false); need(!retry.started());
        }
    } else if (name == "nested_update") {
        pd::ScratchLayoutState state; ready(state);
        pd::ScratchLayoutState::Update outer(state, false); need(outer.started());
        { pd::ScratchLayoutState::Update inner(state, false); need(!inner.started()); }
        need(!state.ready()); outer.commit(); need(state.ready() && state.generation() == 2);
    } else if (name == "failed_take_and_reset") {
        pd::ScratchLayoutState old; ready(old);
        rejects([&] {
            pd::ScratchLayoutState::Update update(old, false); need(update.started());
            const auto a = pd::reserve_scratch(0, 32ull << 20, 2);
            (void)pd::reserve_scratch(a.next, 8192 * 10240, 4, a.next);
            update.commit();
        });
        need(!old.ready() && old.generation() == 1);
        pd::ScratchLayoutState fresh; ready(fresh); need(!old.ready());
    } else if (name == "qsa_global_ordinal") {
        need(pd::scratch_qsa_primary(48, 4, true, 0, 12) == 0);
        need(pd::scratch_qsa_primary(48, 4, true, 11, 1) == 11);
        rejects([] { (void)pd::scratch_qsa_primary(48, 4, false, 0, 12); });
        rejects([] { (void)pd::scratch_qsa_primary(48, 4, true, 12, 1); });
        rejects([] { (void)pd::scratch_qsa_primary(48, 4, true, 11, 2); });
        rejects([] { (void)pd::scratch_qsa_primary(48, 0, true, 0, 12); });
    } else if (name == "qsa_resident_plan") {
        const auto p = pd::scratch_qsa_plan(262144,262144,65536,65536,0,2,256,4,4,true,false,false);
        need(p.max_blocks == 65538 && p.rows == 524288 && p.values == 134217728 &&
            p.scales == 2097152 && p.q4_bytes == 150994944);
    } else if (name == "qsa_stream_ring_formats") {
        for (int mode : {1,2}) for (bool q4 : {false,true}) {
            const auto p = pd::scratch_qsa_plan(262144,262144,65536,512,mode,2,256,4,4,!q4,q4,false);
            need(p.rows == 524288);
        }
        need(pd::scratch_qsa_plan(1024,1024,256,256,0,2,256,4,4,false,false,true).values == 524288);
        rejects([] { (void)pd::scratch_qsa_plan(1024,1024,256,128,1,2,256,4,4,false,false,true); });
    } else if (name == "qsa_bad_metadata") {
        rejects([] { (void)pd::scratch_qsa_plan(1024,1023,256,256,0,2,256,4,4,true,false,false); });
        rejects([] { (void)pd::scratch_qsa_plan(1024,1024,255,255,0,2,256,4,4,true,false,false); });
        rejects([] { (void)pd::scratch_qsa_plan(1024,1024,256,128,0,2,256,4,4,true,false,false); });
        rejects([] { (void)pd::scratch_qsa_plan(1024,1024,256,256,1,2,256,4,4,true,false,false); });
        rejects([] { (void)pd::scratch_qsa_plan(1024,1024,256,257,0,2,256,4,4,true,false,false); });
        rejects([] { (void)pd::scratch_qsa_plan(1024,1024,256,256,3,2,256,4,4,true,false,false); });
        rejects([] { (void)pd::scratch_qsa_plan(1024,1024,256,256,0,2,252,4,4,true,false,false); });
        rejects([] { (void)pd::scratch_qsa_plan(1024,1024,256,256,0,2,192,4,4,false,true,false); });
        rejects([] { (void)pd::scratch_qsa_plan(1024,1024,256,256,0,2,256,4,4,true,true,false); });
    } else if (name == "qsa_product_overflow") {
        rejects([] { (void)pd::scratch_product(maximum,2); });
        rejects([] {
            constexpr auto m = std::numeric_limits<int32_t>::max();
            (void)pd::scratch_qsa_plan(m,m,m,m,0,m,m-3,1,4,false,false,false);
        });
    } else if (name == "qsa_block_overflow") {
        rejects([] {
            constexpr auto m = std::numeric_limits<int32_t>::max();
            (void)pd::scratch_qsa_plan(m,m,m,m,0,2,256,1,1,true,false,false);
        });
    } else if (name == "qsa_tail_rounding") {
        const auto p = pd::scratch_qsa_plan(1025,1025,257,257,0,2,256,4,4,true,false,false);
        need(p.rows == 2056 && p.max_blocks == 258);
        rejects([] { (void)pd::scratch_qsa_plan(0,0,0,0,0,2,256,4,4,true,false,false); });
    } else throw std::runtime_error("unknown case");
}
}
int main() try {
    constexpr std::array names = {"guard_alignment", "shape_bounds", "gemm_64mib", "exact_capacity", "payload_overflow",
        "guard_overflow", "used_overflow", "capacity_bounds", "slack_overflow", "initial_layout",
        "relayout_generation", "failed_init", "failed_relayout", "nested_update", "failed_take_and_reset",
        "qsa_global_ordinal", "qsa_resident_plan", "qsa_stream_ring_formats", "qsa_bad_metadata",
        "qsa_product_overflow", "qsa_block_overflow", "qsa_tail_rounding"};
    for (const auto* name : names) {
        test(name); std::cout << "prefill_scratch case=" << name << " status=PASS\n";
    }
    std::cout << "prefill_scratch_fixture cases=22 status=PASS\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
