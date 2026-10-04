#include "strata/sycl_upstream/handoff.hpp"
#include <stdexcept>
#include <algorithm>

namespace strata::sycl_upstream {
namespace {
sycl::event unchanged_tail(sycl::queue& q) {
    return q.ext_oneapi_get_last_event().value_or(sycl::event{});
}
inline void system_fence() {
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
}
void validate(HandoffWaitLimit limit) {
    if (limit.max_polls && !limit.timed_out)
        throw std::invalid_argument("bounded handoff wait requires a timeout status");
}
template<bool Bounded, class Ready>
inline void poll(Ready ready, HandoffWaitLimit limit) {
    uint32_t polls = 0;
    while (!ready()) {
        // B570 probes timed out with volatile loads and only pre/post fences.
        // A fence in the loop is required by the measured implementation. Do
        // not substitute a plain load, a cached local value, or a host wait.
        system_fence();
        if constexpr (Bounded) {
            if (++polls == limit.max_polls) {
                *limit.timed_out = 1;
                if (limit.observed_polls) *limit.observed_polls = polls;
                return;
            }
        }
    }
    system_fence();
    if constexpr (Bounded) if (limit.observed_polls) *limit.observed_polls = polls;
}
template<bool Bounded>
sycl::event wait_eq(sycl::queue& q, const uint32_t* flag, const uint32_t* seq, HandoffWaitLimit limit) {
    return q.single_task([=] {
        const uint32_t want = *static_cast<const volatile uint32_t*>(seq);
        poll<Bounded>([=] { return *static_cast<const volatile uint32_t*>(flag) == want; }, limit);
    });
}
template<bool Bounded>
sycl::event wait_ge(sycl::queue& q, const uint32_t* flag, uint32_t value,
                    const uint32_t* skip, HandoffWaitLimit limit) {
    return q.single_task([=] {
        if (skip && *static_cast<const volatile uint32_t*>(skip) == value) return;
        poll<Bounded>([=] { return *static_cast<const volatile uint32_t*>(flag) >= value; }, limit);
    });
}
}
sycl::event doorbell_ring(sycl::queue& q, uint32_t* seq) {
    if (!seq) return unchanged_tail(q);
    return q.single_task([=] {
        system_fence();
        volatile auto* p = seq;
        *p = *p + 1u;
        system_fence();
    });
}
sycl::event doorbell_publish(sycl::queue& q, const float* x, const int32_t* ids,
    const float* weights, int64_t n, int64_t k, float* x_out, int32_t* ids_out,
    float* weights_out, uint32_t* seq) {
    if (n < 0 || k < 0 || k > 1024) throw std::invalid_argument("invalid doorbell payload size");
    return q.parallel_for(sycl::nd_range<1>{1024, 1024}, [=](sycl::nd_item<1> item) {
        const int i = item.get_local_linear_id();
        for (int64_t j = i; j < n; j += 1024) x_out[j] = x[j];
        if (i < k) { ids_out[i] = ids[i]; weights_out[i] = weights[i]; }
        system_fence();
        item.barrier(sycl::access::fence_space::global_and_local);
        if (i == 0) {
            system_fence();
            volatile auto* p = seq;
            *p = *p + 1u; // Read on every replay, never capture a host literal.
            system_fence();
        }
    });
}
sycl::event doorbell_wait(sycl::queue& q, const uint32_t* flag, const uint32_t* seq, HandoffWaitLimit limit) {
    if (!flag || !seq) return unchanged_tail(q);
    validate(limit);
    return limit.max_polls ? wait_eq<true>(q, flag, seq, limit) : wait_eq<false>(q, flag, seq, limit);
}
sycl::event wait_flag_ge(sycl::queue& q, const uint32_t* flag, uint32_t value, HandoffWaitLimit limit) {
    return wait_flag_ge_or(q, flag, value, nullptr, limit);
}
sycl::event wait_flag_ge_or(sycl::queue& q, const uint32_t* flag, uint32_t value,
                          const uint32_t* skip, HandoffWaitLimit limit) {
    validate(limit);
    return limit.max_polls ? wait_ge<true>(q, flag, value, skip, limit) : wait_ge<false>(q, flag, value, skip, limit);
}
sycl::event copy_from_mapped(sycl::queue& q, float* dst, const float* src, int64_t n) {
    if (n <= 0) return unchanged_tail(q);
    if (n % 4 || (reinterpret_cast<uintptr_t>(src) & 15) || (reinterpret_cast<uintptr_t>(dst) & 15))
        throw std::invalid_argument("mapped float copy requires a multiple of four and 16-byte alignment");
    const size_t threads = std::min<size_t>((size_t(n) / 4 + 255) / 256, 64) * 256;
    return q.parallel_for(sycl::nd_range<1>{threads, 256}, [=](sycl::nd_item<1> item) {
        for (int64_t i = item.get_global_linear_id(); i < n / 4; i += item.get_global_range(0)) {
            sycl::float4 value;
            value.load(0, src + i * 4);
            value.store(0, dst + i * 4);
        }
    });
}
sycl::event copy_i32_from_mapped(sycl::queue& q, int32_t* dst, const int32_t* src, int64_t n) {
    if (n <= 0) return unchanged_tail(q);
    return q.parallel_for(sycl::nd_range<1>{128, 128}, [=](sycl::nd_item<1> item) {
        for (int64_t j = item.get_local_linear_id(); j < n; j += 128)
            dst[j] = static_cast<const volatile int32_t*>(src)[j];
    });
}
} // namespace strata::sycl_upstream
