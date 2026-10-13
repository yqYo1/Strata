// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstdint>
#include <memory>
namespace strata::core {
struct ExpertDispatch;
class DecodeServiceBudget;
class DecodeServiceBudgetScope {
    ExpertDispatch& dispatch_;
    std::shared_ptr<DecodeServiceBudget> state_;
    bool requested_=false,finished_=false;
public:
    DecodeServiceBudgetScope(ExpertDispatch& d,int64_t layers,bool eligible);
    ~DecodeServiceBudgetScope();
    void finish(const char* status,int64_t emitted,double decode_ms=0);
    DecodeServiceBudgetScope(const DecodeServiceBudgetScope&)=delete;
    DecodeServiceBudgetScope& operator=(const DecodeServiceBudgetScope&)=delete;
};
}
