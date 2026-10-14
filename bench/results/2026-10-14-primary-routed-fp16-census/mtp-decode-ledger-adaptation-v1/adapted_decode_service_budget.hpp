// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// One owner and one close receipt for the adapted archived route/service collectors.
#pragma once
#include "strata/core/decode_route_ledger.hpp"
#include <memory>
namespace strata::core {
struct ExpertDispatch;
class DecodeServiceBudgetScope {
    ExpertDispatch& dispatch_;
    std::unique_ptr<DecodeRouteLedger> state_;
    DecodeLedgerSources sources_;
    bool requested_=false,finished_=false;
    uint64_t sequence_=0,prompt_hash_=0;
    const char* refusal_="ineligible";
    DecodeLedgerSnapshot snapshot();
public:
    DecodeServiceBudgetScope(ExpertDispatch&,int64_t layers,int64_t k,bool eligible,
                            const int64_t* prompt,int64_t prompt_n,int64_t max_new,int64_t context,
                            const DecodeLedgerSources&);
    ~DecodeServiceBudgetScope();
    void begin_window(int T) { if (state_) state_->begin_window(T); }
    void verified(int T) { if (state_) state_->verified(T); }
    void committed(int a,int keep) { if (state_) state_->committed(a,keep); }
    void emitted(int64_t n,bool terminal) { if (state_) state_->emitted(n,terminal); }
    void context_limit() { if (state_) state_->context_limit=true; }
    void wait_commit_succeeded() { if (state_) state_->wait_commit_ok=true; }
    void finish(const char* status,int64_t emitted,int64_t offered,int64_t accepted,double decode_ms);
    DecodeServiceBudgetScope(const DecodeServiceBudgetScope&)=delete;
    DecodeServiceBudgetScope& operator=(const DecodeServiceBudgetScope&)=delete;
};
}
