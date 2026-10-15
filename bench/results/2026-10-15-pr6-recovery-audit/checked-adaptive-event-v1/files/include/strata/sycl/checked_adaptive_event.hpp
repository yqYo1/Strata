#pragma once
#include "strata/sycl/checked_cleanup.hpp"
#include <exception>

namespace strata::sycl_backend {
// Called on the adapt issuer, immediately after DMA enqueues and before it can
// return to pending publication. Failure must not unwind DMA/commit owners.
template<class Record,class Describe>
void checked_adaptive_event_record(Record&& record,Describe&& describe,
                                   const char* where) noexcept {
    try {
        const int code=std::forward<Record>(record)();
        if(code==0)return;
        // Read/print TLS-backed detail immediately: no intervening runtime API,
        // allocation, queue drain or deferred cudaGetLastError lookup.
        const char* detail=std::forward<Describe>(describe)(code);
        std::fprintf(stderr,"Strata SYCL adaptive event: %s status=%d: %s\n",
                     where,code,detail?detail:"unknown API error");
        std::fflush(stderr);
    } catch(const std::exception& error) {
        std::fprintf(stderr,"Strata SYCL adaptive event: %s exception: %s\n",where,error.what());
        std::fflush(stderr);
    } catch(...) {
        std::fprintf(stderr,"Strata SYCL adaptive event: %s unknown exception\n",where);
        std::fflush(stderr);
    }
    cleanup_fail_stop("adaptive event recording failed; pending residency must not be published");
}
} // namespace strata::sycl_backend
