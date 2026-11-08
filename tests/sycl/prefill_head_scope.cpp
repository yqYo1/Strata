// Authored for Strata; repository LICENSE (MIT) applies.
// Host-only tests of the policy used by actual standalone and joint boundaries.
// These do not qualify device readers, graph retirement or restored tensors.
#include "strata/core/prefill_head_scope.hpp"
#include "strata/core/cache_scratch_lease.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {
using strata::core::detail::PrefillHeadState;
using strata::core::detail::begin_prefill_head_release;
using strata::core::detail::finish_prepared_prefill_head_release;
using strata::core::detail::with_prefill_head_release;
using strata::core::detail::CacheScratchState;
using Phase = PrefillHeadState::Phase;
enum Event { prepare, cancel, other_readers, head_readers, detach, prompt, other_restore, restore, resume };
int checks = 0, cases = 0;
void check(bool ok) noexcept {
    ++checks;
    if(!ok) { std::fprintf(stderr,"head scope check %d failed\n",checks); std::abort(); }
}
struct Backend {
    PrefillHeadState state;
    std::array<Event,16> events{};
    size_t size=0;
    bool accept=true,throw_prepare=false,joint=false,other_retired=false,other_restored=false;
    void event(Event e) noexcept { check(size<events.size());events[size++]=e; }
    bool prepare(std::string&) {
        event(::prepare);
        check(state.readable() && !other_retired);
        if(throw_prepare)throw std::runtime_error("RAM prepare");
        if(!accept)return false;
        check(state.transition(Phase::ready,Phase::prepared));return true;
    }
    void require_prepared() const noexcept {check(state.phase()==Phase::prepared);}
    void cancel_prepared() noexcept {
        require_prepared();event(cancel);check(state.cancel_prepared());
    }
    void retire_readers() noexcept {
        check(!joint || other_retired);
        event(head_readers);check(state.transition(Phase::prepared,Phase::retired));
    }
    void detach() noexcept { event(::detach);check(state.transition(Phase::retired,Phase::detached)); }
    void restore() noexcept {
        check(!joint || other_restored);
        event(::restore);check(state.transition(Phase::detached,Phase::restored));
    }
    void resume() noexcept { event(::resume);check(state.transition(Phase::restored,Phase::ready)); }
    void retire_other() noexcept {
        check(joint && state.phase()==Phase::prepared && !other_retired);
        event(other_readers);other_retired=true;
    }
    void restore_other() noexcept {
        check(joint && state.phase()==Phase::detached && other_retired);
        event(other_restore);other_restored=true;
    }
    template<size_t N> void expect(const std::array<Event,N>& wanted) const noexcept {
        check(size==N);
        for(size_t i=0;i<N;++i)check(events[i]==wanted[i]);
    }
};
// Exercise the real host lease state, without minting a forged GPU lease or
// pretending that host callbacks prove device completion/publication.
struct Cache {
    alignas(256) std::array<unsigned char,512> bytes{};
    std::shared_ptr<CacheScratchState> state;
    uint64_t generation;
    Cache() : state(std::make_shared<CacheScratchState>(this,bytes.data(),bytes.size())),
              generation(state->prepare(0,bytes.size())) {
        check(state->mask(generation,[]() noexcept {return true;}));
    }
    void finish_used() {
        check(state->phase()==CacheScratchState::Phase::used);
        state->begin_refill(generation);
        check(state->publish_refilled(generation,[]() noexcept {return true;}));
        check(state->readable());state->close();
        check(state->phase()==CacheScratchState::Phase::closed);
    }
    void prove_unpoisoned() {
        check(state->phase()==CacheScratchState::Phase::masked);
        check(state->activate(generation,[]() noexcept {return true;}));
        check(state->retire(generation,[]() noexcept {return true;}));
        finish_used();
    }
};

// Instrument the production weight-participant composition, independently of
// the existing fake verifier/cache boundary. No device completion is inferred.
struct WeightEvents {
    std::array<int,32> values{};size_t size=0;
    void add(int value) noexcept {check(size<values.size());values[size++]=value;}
    template<size_t N> void expect(const std::array<int,N>& wanted) const noexcept {
        check(size==N);for(size_t i=0;i<N;++i)check(values[i]==wanted[i]);
    }
};
struct TargetWeights {
    WeightEvents& log;
    bool prepared=false,detached=false;int prepare_result=1;
    const uint64_t identity=0x12345678;
    bool prepare_prefill_weights(std::string& err) {
        check(!prepared && !detached);log.add(2);
        if(prepare_result==2)throw std::runtime_error("target RAM preparation");
        if(!prepare_result){err="target preparation refused";return false;}
        prepared=true;return true;
    }
    void cancel_prefill_weights() noexcept {check(prepared && !detached);log.add(10);prepared=false;}
    void detach_prefill_weights() noexcept {check(prepared && !detached);log.add(4);detached=true;}
    void restore_prefill_weights() noexcept {check(prepared && detached);log.add(6);detached=false;}
    void resume_prefill_weights() noexcept {check(prepared && !detached);log.add(8);prepared=false;}
};
struct DraftWeights {
    WeightEvents& log;TargetWeights& target;
    bool accept=true,prepared=false,retired=false,detached=false;
    bool prepare_prefill_head(const TargetWeights& h,std::string& err) {
        check(&h==&target && !prepared && !retired && !detached && !h.prepared);log.add(1);
        if(!accept){err="draft preparation refused";return false;}
        prepared=true;return true;
    }
    void cancel_prefill_head() noexcept {check(prepared && !retired && !detached);log.add(11);prepared=false;}
    void retire_prefill_head_readers() noexcept {
        check(prepared && !retired && !target.detached);log.add(3);retired=true;
    }
    void detach_prefill_head() noexcept {
        check(prepared && retired && !detached && target.detached);log.add(5);detached=true;
    }
    void restore_prefill_head() noexcept {
        check(prepared && retired && detached && !target.detached);log.add(7);detached=false;
    }
    void resume_prefill_head() noexcept {
        check(prepared && retired && !detached && !target.prepared);log.add(9);prepared=false;retired=false;
    }
};
struct WeightBoundary {
    strata::core::detail::PrefillHeadWeights<TargetWeights,DraftWeights> weights;
    PrefillHeadState state;
    bool prepare(std::string& err) {
        check(state.readable());if(!weights.prepare(err))return false;
        check(state.transition(Phase::ready,Phase::prepared));return true;
    }
    void require_prepared() const noexcept {check(state.phase()==Phase::prepared);}
    void retire_readers() noexcept {
        weights.retire_draft_readers();check(state.transition(Phase::prepared,Phase::retired));
    }
    void detach() noexcept {weights.detach();check(state.transition(Phase::retired,Phase::detached));}
    void restore() noexcept {weights.restore();check(state.transition(Phase::detached,Phase::restored));}
    void resume() noexcept {weights.resume();check(state.transition(Phase::restored,Phase::ready));}
};
}

int main() {
    std::string err="unchanged";
    {
        ++cases;
        using D=strata::core::detail::PrefillHeadDrafter;
        int allowed=0,rows=0;
        for(bool single:{false,true})for(int spec=1;spec<=8;++spec)
            for(bool mtp:{false,true})for(bool df2:{false,true})for(bool live:{false,true}) {
                ++rows;
                const auto selected=strata::core::detail::prefill_head_drafter(single,spec,mtp,df2,live);
                const bool eligible=single && spec==4 && !live && mtp!=df2;
                check((selected!=D::unsupported)==eligible);
                if(eligible){++allowed;check(selected==(mtp?D::mtp4:D::absent_dflash2));}
            }
        check(rows==128 && allowed==2);
    }
    for(bool with_mtp:{false,true}) {
        for(int outcome:{0,1,2}) {
            ++cases;
            WeightEvents log;TargetWeights head{log};DraftWeights draft{log,head};
            WeightBoundary boundary{{head,with_mtp?&draft:nullptr},{}};
            bool caught=false,result=false;
            try {
                result=with_prefill_head_release(boundary,[&]() {
                    check(boundary.state.phase()==Phase::detached && head.detached);
                    check(with_mtp?draft.detached:!draft.prepared);
                    if(outcome==2)throw std::runtime_error("prompt failure after detach");
                    return outcome==1;
                },err);
            }catch(const std::runtime_error&){caught=true;}
            check(result==(outcome==1) && caught==(outcome==2) && boundary.state.readable());
            check(!head.prepared && !head.detached && head.identity==0x12345678);
            check(!draft.prepared && !draft.retired && !draft.detached && err=="unchanged");
            if(with_mtp)log.expect(std::array<int,9>{1,2,3,4,5,6,7,8,9});
            else log.expect(std::array<int,4>{2,4,6,8});
            // A restored owner must admit another request rather than only
            // surviving destruction after one prompt.
            log.size=0;
            check(with_prefill_head_release(boundary,[](){return true;},err));
            check(boundary.state.readable() && !head.prepared && !draft.prepared);
        }
        for(int failure:{0,2}) {
            ++cases;
            WeightEvents log;TargetWeights head{log};head.prepare_result=failure;
            DraftWeights draft{log,head};WeightBoundary boundary{{head,with_mtp?&draft:nullptr},{}};
            std::string failure_err;bool caught=false,called=false;
            try {check(!with_prefill_head_release(boundary,[&](){called=true;return true;},failure_err));}
            catch(const std::runtime_error&){caught=true;}
            check(caught==(failure==2) && !called && boundary.state.readable());
            check(!head.prepared && !head.detached && !draft.prepared && !draft.retired);
            if(with_mtp)log.expect(std::array<int,3>{1,2,11});else log.expect(std::array<int,1>{2});
        }
        {
            ++cases;
            WeightEvents log;TargetWeights head{log};DraftWeights draft{log,head};
            strata::core::detail::PrefillHeadWeights<TargetWeights,DraftWeights> weights{head,with_mtp?&draft:nullptr};
            check(weights.prepare(err));weights.cancel_prepared();
            check(!head.prepared && !head.detached && !draft.prepared && !draft.retired);
            if(with_mtp)log.expect(std::array<int,4>{1,2,10,11});else log.expect(std::array<int,2>{2,10});
        }
    }
    {
        ++cases;
        WeightEvents log;TargetWeights head{log};DraftWeights draft{log,head};draft.accept=false;
        strata::core::detail::PrefillHeadWeights<TargetWeights,DraftWeights> weights{head,&draft};
        std::string refusal;check(!weights.prepare(refusal));
        check(refusal=="draft preparation refused" && !head.prepared && !draft.prepared);
        log.expect(std::array<int,1>{1});
    }
    for(bool throws : {false,true}) {
        ++cases;
        Backend b;b.joint=true;b.accept=false;b.throw_prepare=throws;
        bool caught=false;
        try { check(!begin_prefill_head_release(b,[&]() noexcept {b.retire_other();},err)); }
        catch(const std::runtime_error&) {caught=true;}
        check(caught==throws && b.state.readable() && !b.other_retired);
        b.expect(std::array<Event,1>{prepare});
    }
    {
        ++cases;
        Backend b;b.joint=true;
        check(begin_prefill_head_release(b,[&]() noexcept {b.retire_other();},err));
        check(b.state.phase()==Phase::detached && b.other_retired);
        b.event(prompt);b.restore_other();b.restore();b.resume();
        check(b.state.readable());
        b.expect(std::array<Event,8>{prepare,other_readers,head_readers,detach,prompt,other_restore,restore,resume});
    }
    for(int outcome : {0,1,2}) {
        ++cases;
        Backend b;bool caught=false,result=false;
        try {
            result=with_prefill_head_release(b,[&]() {
                b.event(prompt);check(b.state.phase()==Phase::detached);
                if(outcome==2)throw std::runtime_error("prompt cancelled");
                return outcome==1;
            },err);
        }catch(const std::runtime_error&) {caught=true;}
        check(caught==(outcome==2) && result==(outcome==1) && b.state.readable());
        check(err=="unchanged");
        b.expect(std::array<Event,6>{prepare,head_readers,detach,prompt,restore,resume});
    }
    {
        ++cases;
        PrefillHeadState s;
        check(!s.cancel_prepared());
        check(!s.transition(Phase::ready,Phase::detached));
        check(s.transition(Phase::ready,Phase::prepared));
        check(!s.transition(Phase::ready,Phase::prepared));
        check(!s.transition(Phase::prepared,Phase::ready));
        check(s.cancel_prepared() && s.readable());
        check(!s.cancel_prepared());
        check(s.transition(Phase::ready,Phase::prepared));
        check(s.transition(Phase::prepared,Phase::retired));
        check(!s.cancel_prepared());
        check(!s.transition(Phase::retired,Phase::ready));
        check(s.transition(Phase::retired,Phase::detached));
        check(!s.cancel_prepared());
        check(s.transition(Phase::detached,Phase::restored));
        check(!s.cancel_prepared());
        check(!s.transition(Phase::restored,Phase::retired));
        check(s.transition(Phase::restored,Phase::ready) && s.readable());
    }
    // Reproduce why fallible prepare INSIDE activate is forbidden. Do not
    // weaken the parent's poison policy to make this failure disappear.
    for(bool throws : {false,true}) {
        ++cases;
        Cache c;Backend b;b.joint=true;b.accept=false;b.throw_prepare=throws;
        bool caught=false;
        try {
            check(!c.state->activate(c.generation,[&]() {
                return begin_prefill_head_release(b,[&]() noexcept {b.retire_other();},err);
            }));
        }catch(const std::runtime_error&) {caught=true;}
        check(caught==throws && b.state.readable() && !b.other_retired);
        check(c.state->phase()==CacheScratchState::Phase::poisoned);
        bool refused=false;
        try {c.state->close();}catch(const std::logic_error&) {refused=true;}
        check(refused);b.expect(std::array<Event,1>{prepare});
    }
    // In the corrected order preparation failure never enters activation.
    for(bool throws : {false,true}) {
        ++cases;
        Cache c;Backend b;b.joint=true;b.accept=false;b.throw_prepare=throws;
        bool caught=false;
        try {check(!b.prepare(err));}catch(const std::runtime_error&) {caught=true;}
        check(caught==throws && b.state.readable() && !b.other_retired);
        b.expect(std::array<Event,1>{prepare});c.prove_unpoisoned();
    }
    {
        ++cases;
        Cache c;Backend b;b.joint=true;
        check(b.prepare(err));bool called=false,caught=false;
        try {
            c.state->activate(c.generation+1,[&]() noexcept {
                called=true;finish_prepared_prefill_head_release(b,[&]() noexcept {b.retire_other();});
                return true;
            });
        }catch(const std::logic_error&) {caught=true;}
        check(caught && !called && !b.other_retired);
        b.cancel_prepared();check(b.state.readable());
        b.expect(std::array<Event,2>{prepare,cancel});c.prove_unpoisoned();
    }
    for(int outcome : {0,1,2}) {
        ++cases;
        Cache c;Backend b;b.joint=true;check(b.prepare(err));
        check(c.state->activate(c.generation,[&]() noexcept {
            finish_prepared_prefill_head_release(b,[&]() noexcept {b.retire_other();});return true;
        }));
        bool caught=false,result=false;
        try {
            b.event(prompt);check(b.state.phase()==Phase::detached);
            if(outcome==2)throw std::runtime_error("joint prompt cancelled");
            result=outcome==1;
        }catch(const std::runtime_error&) {caught=true;}
        check(caught==(outcome==2) && result==(outcome==1));
        check(c.state->retire(c.generation,[]() noexcept {return true;}));
        b.restore_other();b.restore();b.resume();
        check(b.state.readable());c.finish_used();
        b.expect(std::array<Event,8>{prepare,other_readers,head_readers,detach,prompt,other_restore,restore,resume});
    }
    std::printf("{\"kind\":\"prefill_head_scope\",\"cases\":%d,\"checks\":%d,\"status\":\"PASS\"}\n",cases,checks);
}
