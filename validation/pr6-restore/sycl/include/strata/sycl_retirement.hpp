#pragma once
// New project code under repository LICENSE; SYCL API contracts only.
#include <sycl/sycl.hpp>
#include "strata/prefill/retirement.hpp"

namespace strata {
inline void retire_event(sycl::event event) noexcept {
    try {
        prefill::detail::RetirementDeadline deadline;
        prefill::detail::retirement_until([&] {
            return event.get_info<sycl::info::event::command_execution_status>() ==
                   sycl::info::event_command_status::complete;
        }, "event did not reach completion");
        // Invoke the selected queue's async handler before accepting success.
        event.wait_and_throw();
    } catch (...) { prefill::detail::retirement_fail_stop("event completion/error unavailable"); }
}
inline void retire_queue(sycl::queue& queue) noexcept {
    try {
        prefill::detail::RetirementDeadline deadline;
        // The extension is already used by the historical engine. This barrier
        // closes all work previously accepted by this queue, including partial submits.
        retire_event(queue.ext_oneapi_submit_barrier());
        queue.throw_asynchronous();
    } catch (...) { prefill::detail::retirement_fail_stop("queue retirement unavailable"); }
}
} // namespace strata
