#pragma once
#include <cassert>
#include <cstdint>
#include <vector>
#include <limits>
// Standalone host shim only. No device/runtime dependency. The absence of wait
// and synchronize APIs makes an added observer drain a compile-time failure.
struct FakeEvent { bool submitted = false, complete = false; uint64_t stamp = 0; };
using cudaEvent_t = FakeEvent*;
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorNotReady = 1, cudaErrorUnknown = 2;
inline std::vector<FakeEvent*> fake_owners;
inline uint64_t fake_clock = 0, fake_records = 0, fake_queries = 0, fake_elapsed = 0;
inline int fake_create_fail_after = -1;
inline bool fake_record_failure = false, fake_query_failure = false, fake_elapsed_failure = false;
inline bool fake_destroy_failure = false;
inline int fake_elapsed_value = 0; // 1 NaN, 2 negative
inline bool fake_require_restore = false, fake_restored = false;
inline cudaError_t cudaEventCreate(cudaEvent_t* out) {
    if (fake_create_fail_after == 0) return cudaErrorUnknown;
    if (fake_create_fail_after > 0) --fake_create_fail_after;
    *out = new FakeEvent; fake_owners.push_back(*out); return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t p, cudaStream_t) {
    if (fake_record_failure) return cudaErrorUnknown;
    assert(p); p->submitted = true; p->complete = false;
    p->stamp = ++fake_clock; ++fake_records; return cudaSuccess;
}
inline void fake_existing_checked_drain() {
    for (auto p : fake_owners) if (p) p->complete = true;
}
inline cudaError_t cudaEventQuery(cudaEvent_t p) {
    ++fake_queries;
    if (fake_query_failure) return cudaErrorUnknown;
    return p->complete ? cudaSuccess : cudaErrorNotReady;
}
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    ++fake_elapsed;
    assert(a->complete && b->complete); // prohibit pending ElapsedTime
    if (fake_elapsed_failure) return cudaErrorUnknown;
    *ms = fake_elapsed_value == 1 ? std::numeric_limits<float>::quiet_NaN() :
          fake_elapsed_value == 2 ? -1.f : float(b->stamp - a->stamp);
    return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t p) {
    assert(!p->submitted || p->complete); // owner retained until checked drain
    assert(!fake_require_restore || fake_restored);
    for (auto& q : fake_owners) if (q == p) { q = nullptr; break; }
    delete p; return fake_destroy_failure ? cudaErrorUnknown : cudaSuccess;
}
