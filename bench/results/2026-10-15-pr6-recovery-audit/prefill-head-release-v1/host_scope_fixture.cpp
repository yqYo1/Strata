// SOURCE ONLY. Main compiles/runs with a host C++17 compiler after native retirement.
// Includes the byte-identical proposed production helper, without CUDA/SYCL.
#include "prefill_head_scope.hpp"
#include <stdexcept>
#include <exception>
#include <vector>
#include <string>
#include <cstdio>
using strata::core::detail::PrefillHeadState;
using strata::core::detail::with_prefill_head_release;
using strata::core::detail::valid_prefill_head_ids;
static void need(bool ok){if(!ok)throw std::runtime_error("head scope contract failed");}
struct Fake {
    using Phase=PrefillHeadState::Phase;
    PrefillHeadState state;
    std::vector<int> calls;
    std::vector<int32_t> ids{9,0,4,4},saved=ids;
    bool reject=false,target=true,draft=true,verify=true;
    void move(Phase from,Phase to) noexcept {if(!state.transition(from,to))std::terminate();}
    bool prepare(std::string& err){
        if(!state.readable() || reject || !valid_prefill_head_ids(ids.data(),ids.size(),10)){err="preflight rejected";return false;}
        calls.push_back(1);move(Phase::ready,Phase::prepared);return true;
    }
    void retire_readers() noexcept {calls.push_back(2);verify=false;move(Phase::prepared,Phase::retired);}
    void detach() noexcept {calls.push_back(3);if(verify)std::terminate();target=false;draft=false;move(Phase::retired,Phase::detached);}
    void restore() noexcept {
        calls.push_back(5);if(ids!=saved || verify)std::terminate();
        target=true; // original packed target before duplicate gather
        if(!target)std::terminate();draft=true;move(Phase::detached,Phase::restored);
    }
    void resume() noexcept {calls.push_back(6);if(!target || !draft)std::terminate();verify=true;move(Phase::restored,Phase::ready);}
};
int main(){
    std::string err;
    for(bool ok:{true,false}){
        Fake f;
        const bool result=with_prefill_head_release(f,[&](){
            f.calls.push_back(4);need(!f.state.readable() && !f.target && !f.draft && !f.verify);
            return ok; // false stands for normal cancellation/error return
        },err);
        need(result==ok && f.state.readable() && f.target && f.draft && f.verify);
        need(f.calls==std::vector<int>({1,2,3,4,5,6}) && f.ids==f.saved);
        // A second independent warm-serve request reuses the same stable map.
        need(with_prefill_head_release(f,[](){return true;},err));
    }
    Fake failed;failed.reject=true;
    need(!with_prefill_head_release(failed,[]()->bool{throw std::runtime_error("must not run");},err));
    need(failed.calls.empty() && failed.state.readable());
    Fake thrown;bool caught=false;
    try{with_prefill_head_release(thrown,[&]()->bool{thrown.calls.push_back(4);throw std::runtime_error("original prompt error");},err);}
    catch(const std::runtime_error& e){caught=std::string(e.what())=="original prompt error";}
    need(caught && thrown.state.readable() && thrown.calls==std::vector<int>({1,2,3,4,5,6}));
    Fake reentry;
    need(with_prefill_head_release(reentry,[&](){
        std::string nested;
        need(!with_prefill_head_release(reentry,[](){return true;},nested));
        return true;
    },err));
    need(reentry.calls==std::vector<int>({1,2,3,5,6}));
    PrefillHeadState s;using P=PrefillHeadState::Phase;
    need(!s.transition(P::detached,P::restored)); // restore before detach
    need(s.transition(P::ready,P::prepared));
    need(!s.transition(P::ready,P::prepared)); // duplicate open
    need(!s.transition(P::prepared,P::ready)); // skipping retirement/detach/restore rejected
    // A fresh state verifies the production edge sequence explicitly.
    PrefillHeadState stages;
    for(auto edge:{std::pair<P,P>{P::ready,P::prepared},{P::prepared,P::retired},{P::retired,P::detached},{P::detached,P::restored},{P::restored,P::ready}}){
        need(stages.transition(edge.first,edge.second));
        need(!stages.transition(edge.first,edge.second));
    }
    need(valid_prefill_head_ids(nullptr,0,10));
    need(!valid_prefill_head_ids(nullptr,1,10));
    need(!valid_prefill_head_ids(nullptr,0,0));
    for(auto ids:std::vector<std::vector<int32_t>>{{-1},{10},std::vector<int32_t>(11,0)})
        need(!valid_prefill_head_ids(ids.data(),ids.size(),10));
    const int32_t good[]={9,0,4,4};need(valid_prefill_head_ids(good,4,10));
    std::puts("host head-boundary contract: PASS");
}
