// Authored for Strata; repository LICENSE (MIT) applies.
// Host-only tests of the policy used by actual standalone and joint boundaries.
// These do not qualify device readers, graph retirement or restored tensors.
#include "strata/core/prefill_head_scope.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {
using strata::core::detail::PrefillHeadState;
using strata::core::detail::begin_prefill_head_release;
using strata::core::detail::with_prefill_head_release;
using Phase = PrefillHeadState::Phase;
enum Event { prepare, other_readers, head_readers, detach, prompt, other_restore, restore, resume };
int checks = 0;
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
}

int main() {
    std::string err="unchanged";
    for(bool throws : {false,true}) {
        Backend b;b.joint=true;b.accept=false;b.throw_prepare=throws;
        bool caught=false;
        try { check(!begin_prefill_head_release(b,[&]() noexcept {b.retire_other();},err)); }
        catch(const std::runtime_error&) {caught=true;}
        check(caught==throws && b.state.readable() && !b.other_retired);
        b.expect(std::array<Event,1>{prepare});
    }
    {
        Backend b;b.joint=true;
        check(begin_prefill_head_release(b,[&]() noexcept {b.retire_other();},err));
        check(b.state.phase()==Phase::detached && b.other_retired);
        b.event(prompt);b.restore_other();b.restore();b.resume();
        check(b.state.readable());
        b.expect(std::array<Event,8>{prepare,other_readers,head_readers,detach,prompt,other_restore,restore,resume});
    }
    for(int outcome : {0,1,2}) {
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
        PrefillHeadState s;
        check(!s.transition(Phase::ready,Phase::detached));
        check(s.transition(Phase::ready,Phase::prepared));
        check(!s.transition(Phase::ready,Phase::prepared));
        check(!s.transition(Phase::prepared,Phase::ready));
        check(s.transition(Phase::prepared,Phase::retired));
        check(!s.transition(Phase::retired,Phase::ready));
        check(s.transition(Phase::retired,Phase::detached));
        check(s.transition(Phase::detached,Phase::restored));
        check(!s.transition(Phase::restored,Phase::retired));
        check(s.transition(Phase::restored,Phase::ready) && s.readable());
    }
    std::printf("{\"kind\":\"prefill_head_scope\",\"cases\":7,\"checks\":%d,\"status\":\"PASS\"}\n",checks);
}
