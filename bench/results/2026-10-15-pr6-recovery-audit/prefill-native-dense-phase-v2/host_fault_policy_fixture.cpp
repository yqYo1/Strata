// Unexecuted actual-policy host fixture; Main owns compilation/execution.
#include "strata/core/native_dense_bank_policy.hpp"
#include <cstdlib>
using namespace strata::core::detail;
int main() {
    int owner;
    constexpr int success=0, oom=2, unknown=999;
    auto check=[&](int status, const void* p, BankAllocation expected) {
        if(classify_bank_allocation(status,success,oom,p)!=expected) std::abort();
    };
    check(success,&owner,BankAllocation::success);
    check(success,nullptr,BankAllocation::uncertain);
    check(oom,nullptr,BankAllocation::retryable_oom);
    check(oom,&owner,BankAllocation::uncertain);
    check(unknown,nullptr,BankAllocation::uncertain);
    check(unknown,&owner,BankAllocation::uncertain);
}
