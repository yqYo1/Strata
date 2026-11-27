// Original Strata host contracts; repository LICENSE applies.
#include "strata/prefill/scratch_allocation.hpp"
#include "strata/core/cache_scratch_lease.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
void loan_test(const std::string& name) {
    using State = strata::core::detail::CacheScratchState;
    using Phase = State::Phase;
    int parent = 0;
    auto state = std::make_shared<State>(&parent, reinterpret_cast<void*>(0x1000), 0x4000);
    auto to_used = [&] {
        const auto g=state->prepare(256,8192);
        need(state->mask(g,[]{return true;})); need(state->activate(g,[]{return true;}));
        need(state->retire(g,[]{return true;})); return g;
    };
    if (name=="loan_cycle") {
        std::vector<int> order;
        const auto g=state->prepare(256,8192); need(g==1 && !state->readable());
        rejects([&]{(void)state->read();}); rejects([&]{state->close();});
        need(state->mask(g,[&]{order.push_back(1);return true;}));
        need(state->activate(g,[&]{order.push_back(2);return true;}));
        need(state->retire(g,[&]{order.push_back(3);return true;}));
        need(state->phase()==Phase::used && !state->readable());
        state->begin_refill(g); need(!state->readable());
        need(state->publish_refilled(g,[&]{order.push_back(4);return true;}));
        need(state->readable() && order==std::vector<int>({1,2,3,4})); state->close();
        need(state->phase()==Phase::closed); rejects([&]{(void)state->read();});
    } else if(name=="loan_bounds") {
        for(auto [at,bytes]:std::array<std::pair<uint64_t,uint64_t>,5>{{{0,0},{1,256},{0x4000,1},{256,0x4000},{maximum,1}}})
            rejects([&]{(void)state->prepare(at,bytes);});
        need(state->readable());
        rejects([&]{(void)std::make_shared<State>(&parent,reinterpret_cast<void*>(maximum-255),256);});
        rejects([&]{(void)std::make_shared<State>(nullptr,reinterpret_cast<void*>(0x1000),256);});
        rejects([&]{(void)std::make_shared<State>(&parent,nullptr,256);});
        need(state->prepare(0,0x4000)==1);
    } else if(name=="loan_read_admission") {
        { auto reader=state->read(); auto moved=std::move(reader);
          rejects([&]{(void)state->prepare(256,8192);}); rejects([&]{state->close();}); }
        const auto g=state->prepare(256,8192); rejects([&]{(void)state->read();});
        need(state->mask(g,[]{return true;})); rejects([&]{(void)state->read();});
    } else if(name=="loan_stale_generation") {
        const auto g=to_used(); state->begin_refill(g); need(state->publish_refilled(g,[]{return true;}));
        const auto next=state->prepare(512,8192); need(next==g+1);
        int calls=0; rejects([&]{state->mask(g,[&]{++calls;return true;});});
        rejects([&]{state->mask(0,[&]{++calls;return true;});});
        need(calls==0 && state->phase()==Phase::prepared);
    } else if(name=="loan_wrong_order") {
        const auto g=state->prepare(256,8192); int calls=0;
        rejects([&]{state->activate(g,[&]{++calls;return true;});});
        rejects([&]{state->begin_refill(g);});
        rejects([&]{state->publish_refilled(g,[&]{++calls;return true;});});
        need(state->mask(g,[]{return true;})); need(state->activate(g,[]{return true;}));
        rejects([&]{state->begin_refill(g);}); need(calls==0 && state->phase()==Phase::exclusive);
    } else if(name=="loan_callback_false" || name=="loan_callback_throw") {
        for(int fail=0;fail<4;++fail) {
            auto s=std::make_shared<State>(&parent,reinterpret_cast<void*>(0x1000),0x4000);
            const auto g=s->prepare(256,8192); int stage=0;
            auto callback=[&] {
                const bool bad=stage++==fail;
                if(bad && name=="loan_callback_throw") throw std::runtime_error("injected phase error");
                return !bad;
            };
            auto run=[&] {
                if(!s->mask(g,callback) || !s->activate(g,callback) || !s->retire(g,callback))return false;
                s->begin_refill(g); return s->publish_refilled(g,callback);
            };
            if(name=="loan_callback_throw") rejects(run); else need(!run());
            need(s->phase()==Phase::poisoned && !s->readable());
            rejects([&]{(void)s->read();}); rejects([&]{s->close();});
        }
    } else if(name=="loan_nested_transition") {
        const auto g=state->prepare(256,8192); int nested=0;
        need(state->mask(g,[&] {
            rejects([&]{state->mask(g,[&]{++nested;return true;});}); return true;
        }));
        need(nested==0 && state->phase()==Phase::masked);
    } else if(name=="loan_poison_during_callback") {
        const auto g=state->prepare(256,8192);
        rejects([&]{state->mask(g,[&]{state->poison();return true;});});
        need(state->phase()==Phase::poisoned && !state->readable());
    } else if(name=="loan_reuse_masked_range") {
        const auto g=to_used(); need(state->activate(g,[]{return true;}));
        need(state->retire(g,[]{return true;})); need(!state->readable());
        state->begin_refill(g); need(state->publish_refilled(g,[]{return true;})); need(state->readable());
    } else if(name=="loan_writer_admission") {
        { auto writer=state->write(reinterpret_cast<void*>(0x1100),256);
          auto moved=std::move(writer);
          rejects([&]{(void)state->prepare(256,8192);}); rejects([&]{state->close();});
          rejects([&]{(void)state->write(reinterpret_cast<void*>(0x1200),256);}); }
        need(state->prepare(256,8192)==1);
        rejects([&]{(void)state->write(reinterpret_cast<void*>(0x1100),256);});
    } else if(name=="loan_writer_refill_interval") {
        const auto g=to_used(); state->begin_refill(g);
        for(auto [at,bytes]:std::array<std::pair<uintptr_t,uint64_t>,5>{{
            {0x1000,256},{0x3100,1},{0x1100,8193},{0x1100,0},{maximum-255,512}}})
            rejects([&]{(void)state->write(reinterpret_cast<void*>(at),bytes);});
        { auto writer=state->write(reinterpret_cast<void*>(0x1100),8192); }
        need(state->publish_refilled(g,[]{return true;}));
    } else if(name=="loan_writer_publication") {
        const auto g=to_used(); state->begin_refill(g); int calls=0;
        { auto writer=state->write(reinterpret_cast<void*>(0x1100),256);
          rejects([&]{state->publish_refilled(g,[&]{++calls;return true;});}); }
        need(calls==0 && state->phase()==Phase::refilling);
        need(state->publish_refilled(g,[&] {
            ++calls;
            rejects([&]{(void)state->write(reinterpret_cast<void*>(0x1100),256);});
            rejects([&]{(void)state->read();}); rejects([&]{state->close();}); return true;
        }));
        need(calls==1 && state->readable());
    } else if(name=="loan_writer_exception") {
        rejects([&]{auto writer=state->write(reinterpret_cast<void*>(0x1100),256);
                    throw std::runtime_error("injected fill failure");});
        const auto g=state->prepare(256,8192); need(g==1);
        need(state->mask(g,[]{return true;})); need(state->activate(g,[]{return true;}));
    } else if(name=="loan_writer_each_phase") {
        const auto no_write=[&]{rejects([&]{(void)state->write(reinterpret_cast<void*>(0x1100),256);});};
        const auto g=state->prepare(256,8192); no_write();
        need(state->mask(g,[&]{no_write();return true;})); no_write();
        need(state->activate(g,[&]{no_write();return true;})); no_write();
        need(state->retire(g,[&]{no_write();return true;})); no_write();
        state->poison(); no_write();
        auto closed=std::make_shared<State>(&parent,reinterpret_cast<void*>(0x1000),0x4000);
        closed->close(); rejects([&]{(void)closed->write(reinterpret_cast<void*>(0x1100),256);});
        need(!strata::core::CacheScratchLease{}.pending());
    } else throw std::runtime_error("unknown loan case");
}
void test(const std::string& name) {
    if (name.rfind("loan_",0)==0) { loan_test(name); return; }
    if (name == "borrowed_used_extent") {
        const auto pointer=[](uintptr_t n){return reinterpret_cast<void*>(n);};
        need(pd::scratch_span_inside(pointer(0x1000),0x2000,pointer(0x1000),0x2000));
        need(pd::scratch_span_inside(pointer(0x1000),0x2000,pointer(0x2fff),1));
        need(!pd::scratch_span_inside(pointer(0x1000),0x2000,pointer(0x3000),1));
        need(!pd::scratch_span_inside(pointer(0x1000),0x2000,pointer(0x0fff),1));
        need(!pd::scratch_span_inside(pointer(maximum-255),256,pointer(maximum-255),1));
        need(!pd::scratch_span_inside(pointer(0x1000),0x2000,pointer(0x1100),0));
    } else if (name == "borrowed_view_disjointness") {
        const auto a=reinterpret_cast<void*>(0x1000), b=reinterpret_cast<void*>(0x1100);
        need(pd::scratch_spans_disjoint(a,256,b,256) && pd::scratch_spans_disjoint(b,256,a,256));
        need(!pd::scratch_spans_disjoint(a,257,b,256) && !pd::scratch_spans_disjoint(b,256,a,257));
        need(!pd::scratch_spans_disjoint(a,256,a,256));
        need(!pd::scratch_spans_disjoint(a,0,b,256) && !pd::scratch_spans_disjoint(nullptr,1,b,256));
        need(!pd::scratch_spans_disjoint(reinterpret_cast<void*>(maximum-255),256,a,256));
    } else if (name == "guard_alignment") {
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
        "qsa_product_overflow", "qsa_block_overflow", "qsa_tail_rounding",
        "borrowed_used_extent","borrowed_view_disjointness",
        "loan_cycle","loan_bounds","loan_read_admission","loan_stale_generation","loan_wrong_order",
        "loan_callback_false","loan_callback_throw","loan_nested_transition","loan_poison_during_callback",
        "loan_reuse_masked_range","loan_writer_admission","loan_writer_refill_interval",
        "loan_writer_publication","loan_writer_exception","loan_writer_each_phase"};
    for (const auto* name : names) {
        test(name); std::cout << "prefill_scratch case=" << name << " status=PASS\n";
    }
    std::cout << "prefill_scratch_fixture cases=39 status=PASS\n";
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
