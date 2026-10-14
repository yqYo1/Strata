// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// One owner and one close receipt for the adapted archived route/service collectors.
#pragma once
#include "strata/core/decode_route_ledger.hpp"
#include "strata/core/progress.hpp"
#include <memory>
#include <cstdlib>
namespace strata::core {
// call() borrows job/format owners from the caller. Fail-stop here before
// unwinding that caller if native success (done + parked) was not observed.
// Do not inspect worker slots or release observer/job owners in this branch.
template<class NativeCall>
void decode_native_owner_call(DecodeRouteLedger& owner,NativeCall&& call) {
    try { call(); }
    catch (...) {
        std::fprintf(stderr,"strata decode joined close request=%llu coupled=%d complete=0 reason=native_pool_exception fail_stop=1 attempted=%llu verified=%llu W=%llu R=%llu D=%llu A=%llu C=%llu E=%llu\n",
            (unsigned long long)owner.sequence,owner.coupled,(unsigned long long)owner.attempted_windows,
            (unsigned long long)owner.verified_windows,(unsigned long long)owner.W,(unsigned long long)owner.R,
            (unsigned long long)owner.D,(unsigned long long)owner.A,(unsigned long long)owner.C,(unsigned long long)owner.E);
        strata::core::release_gpu_waits(stderr); std::fflush(stderr); std::abort();
    }
}
struct ExpertDispatch;
class DecodeServiceBudgetScope {
    ExpertDispatch& dispatch_;
    std::unique_ptr<DecodeRouteLedger> state_;
    DecodeLedgerSources sources_;
    bool requested_=false,finished_=false;
    uint64_t sequence_=0;
    bool coupled_=false;
    const char* refusal_="ineligible";
    DecodeLedgerSnapshot snapshot();
public:
    DecodeServiceBudgetScope(ExpertDispatch&,int64_t layers,int64_t k,bool eligible,
                            int64_t prompt_n,int64_t max_new,int64_t context,bool coupled,
                            const DecodeLedgerSources&);
    ~DecodeServiceBudgetScope();
    void begin_window(int T) { if (state_) state_->begin_window(T); }
    void verified(int T) { if (state_) state_->verified(T); }
    void commit_attempt() { if (state_) state_->commit_attempt(); }
    void committed(int a,int keep) { if (state_) state_->committed(a,keep); }
    void emitted(int64_t n,bool terminal) { if (state_) state_->emitted(n,terminal); }
    void context_limit() { if (state_) state_->context_limit=true; }
    void final_wait_begin() {
        if (state_) {
            state_->add(state_->final_wait_attempts,1);
            state_->final_wait_active=true;
            state_->final_wait_start=strata::kernels::cpu::service_wall_now();
        }
    }
    void wait_commit_finished(bool success) {
        if (state_) {
            const uint64_t now=strata::kernels::cpu::service_wall_now();
            if (!state_->final_wait_active || now<state_->final_wait_start) state_->invalid=true;
            else state_->add(state_->final_wait_ns,now-state_->final_wait_start);
            state_->final_wait_active=false; state_->add(state_->final_wait_returns,1); state_->wait_commit_ok=success;
        }
    }
    void finish(const char* status,int64_t emitted,int64_t offered,int64_t accepted,double decode_ms);
    DecodeServiceBudgetScope(const DecodeServiceBudgetScope&)=delete;
    DecodeServiceBudgetScope& operator=(const DecodeServiceBudgetScope&)=delete;
};
}
