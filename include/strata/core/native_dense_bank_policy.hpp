#pragma once
namespace strata::core::detail {
enum class BankAllocation { success, retryable_oom, uncertain };
// Only confirmed OOM with no returned owner permits an ordinary return.
constexpr BankAllocation classify_bank_allocation(int status, int success, int oom,
                                                   const void* owner) noexcept {
    if (status == success) return owner ? BankAllocation::success : BankAllocation::uncertain;
    if (status == oom && !owner) return BankAllocation::retryable_oom;
    return BankAllocation::uncertain;
}
}
