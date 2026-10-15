#pragma once
#include <cstddef>
// Host-only API shim for observer ownership/status tests, never a device oracle.
struct MockEvent;
using cudaStream_t = void*;
using cudaEvent_t = MockEvent*;
using cudaError_t = int;
constexpr cudaError_t cudaSuccess=0, cudaErrorUnknown=999, cudaErrorNotReady=600;
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*,cudaEvent_t,cudaEvent_t);
