#pragma once
// Host API adapter for the shared Strata engine. Kernels use SYCL directly.
// This header is on the include path only in STRATA_ENABLE_SYCL builds.
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace strata::sycl_backend::compat {
struct Event;
struct Graph;
struct GraphExec;
struct Node;
} // namespace strata::sycl_backend::compat
using cudaStream_t = sycl::queue *;
using cudaEvent_t = strata::sycl_backend::compat::Event *;
using cudaGraph_t = strata::sycl_backend::compat::Graph *;
using cudaGraphExec_t = strata::sycl_backend::compat::GraphExec *;
using cudaGraphNode_t = strata::sycl_backend::compat::Node *;
enum cudaError_t {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorNotReady = 34,
  cudaErrorNotSupported = 801,
  cudaErrorUnknown = 999
};
enum cudaMemcpyKind {
  cudaMemcpyHostToHost,
  cudaMemcpyHostToDevice,
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyDefault
};
enum cudaDeviceAttr {
  cudaDevAttrClockRate,
  cudaDevAttrMultiProcessorCount,
  cudaDevAttrMaxSharedMemoryPerBlockOptin
};
enum cudaStreamCaptureMode { cudaStreamCaptureModeThreadLocal };
enum cudaGraphNodeType {
  cudaGraphNodeTypeKernel,
  cudaGraphNodeTypeMemcpy,
  cudaGraphNodeTypeMemset,
  cudaGraphNodeTypeEmpty,
  cudaGraphNodeTypeHost
};
struct cudaKernelNodeParams {
  void *func = nullptr;
};
struct cudaDeviceProp {
  char name[256]{};
  size_t totalGlobalMem = 0;
  int major = 0, minor = 0, multiProcessorCount = 0;
};
inline constexpr unsigned cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2;
inline constexpr unsigned cudaHostAllocDefault = 0, cudaHostAllocPortable = 1,
                          cudaHostAllocMapped = 2;
inline constexpr unsigned cudaHostRegisterPortable = 1,
                          cudaHostRegisterMapped = 2;
inline constexpr unsigned cudaDeviceScheduleSpin = 1, cudaDeviceMapHost = 8;
#define CUDART_VERSION 0
cudaError_t cudaGetDeviceCount(int *);
cudaError_t cudaGetDevice(int *);
cudaError_t cudaSetDevice(int);
cudaError_t cudaInitDevice(int, unsigned, unsigned);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
cudaError_t cudaDeviceGetAttribute(int *, cudaDeviceAttr, int);
cudaError_t cudaMemGetInfo(size_t *, size_t *);
cudaError_t cudaMalloc(void **, size_t);
cudaError_t cudaHostAlloc(void **, size_t, unsigned);
cudaError_t cudaMallocHost(void **, size_t);
cudaError_t cudaFree(void *);
cudaError_t cudaFreeHost(void *);
cudaError_t cudaHostRegister(void *, size_t, unsigned);
cudaError_t cudaHostUnregister(void *);
cudaError_t cudaHostGetDevicePointer(void **, void *, unsigned);
cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind,
                            cudaStream_t = nullptr);
cudaError_t cudaMemcpyPeerAsync(void *, int, const void *, int, size_t,
                                cudaStream_t = nullptr);
cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t,
                              size_t, cudaMemcpyKind, cudaStream_t = nullptr);
cudaError_t cudaMemset(void *, int, size_t);
cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t = nullptr);
cudaError_t cudaStreamCreate(cudaStream_t *);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaEventCreate(cudaEvent_t *);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
cudaError_t cudaLaunchHostFunc(cudaStream_t, void (*)(void *), void *);
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t *);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t,
                                 unsigned long long);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t,
                                 cudaGraphNode_t *, char *, size_t);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
namespace strata::sycl_backend::compat {
// Explicit native recording for graphs that never query or update nodes.
// Generic CUDA-compatible capture keeps the SYCL node model.
cudaError_t stream_begin_capture_native(cudaStream_t);
// Bind a disabled-timing event to the graph submission's completion. Covers
// every graph node without submitting a separate queue barrier. The stream
// must be outside capture; reusing the event replaces its previous recording.
cudaError_t graph_launch_with_completion(cudaGraphExec_t, cudaStream_t,
                                        cudaEvent_t);
}
cudaError_t cudaGraphUpload(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphGetNodes(cudaGraph_t, cudaGraphNode_t *, size_t *);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t, cudaGraphNodeType *);
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t,
                                         cudaKernelNodeParams *);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
const char *cudaGetErrorString(cudaError_t);

// CUDA's C++ headers accept typed allocation outputs at host call sites.
template <class T> cudaError_t cudaMalloc(T **out, size_t bytes) {
  return cudaMalloc(reinterpret_cast<void **>(out), bytes);
}
template <class T> cudaError_t cudaMallocHost(T **out, size_t bytes) {
  return cudaMallocHost(reinterpret_cast<void **>(out), bytes);
}
template <class T>
cudaError_t cudaHostAlloc(T **out, size_t bytes, unsigned flags) {
  return cudaHostAlloc(reinterpret_cast<void **>(out), bytes, flags);
}
template <class T>
cudaError_t cudaHostGetDevicePointer(T **out, void *ptr, unsigned flags) {
  return cudaHostGetDevicePointer(reinterpret_cast<void **>(out), ptr, flags);
}
