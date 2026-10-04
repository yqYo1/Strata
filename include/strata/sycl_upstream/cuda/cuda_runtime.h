#pragma once
// Source frontend for the audited SYCL runtime subset. This header is selected
// only by targets explicitly adding this directory; it is not a CUDA binary ABI.
#include <cstddef>
struct strata_cuda_stream;
struct strata_cuda_event;
struct strata_cuda_graph;
struct strata_cuda_graph_exec;
struct strata_cuda_graph_node;
using cudaStream_t = strata_cuda_stream*;
using cudaEvent_t = strata_cuda_event*;
using cudaGraph_t = strata_cuda_graph*;
using cudaGraphExec_t = strata_cuda_graph_exec*;
using cudaGraphNode_t = strata_cuda_graph_node*;
using cudaHostFn_t = void (*)(void*);
enum cudaError_t {
    cudaSuccess=0, cudaErrorInvalidValue=1, cudaErrorMemoryAllocation=2,
    cudaErrorInitializationError=3, cudaErrorInvalidDevice=101,
    cudaErrorInvalidResourceHandle=400, cudaErrorNotReady=600,
    cudaErrorPeerAccessAlreadyEnabled=704, cudaErrorPeerAccessNotEnabled=705,
    cudaErrorNotSupported=801, cudaErrorStreamCaptureUnsupported=900,
    cudaErrorStreamCaptureInvalidated=901, cudaErrorUnknown=999
};
enum cudaMemcpyKind {
    cudaMemcpyHostToHost=0, cudaMemcpyHostToDevice=1,
    cudaMemcpyDeviceToHost=2, cudaMemcpyDeviceToDevice=3, cudaMemcpyDefault=4
};
enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal=0, cudaStreamCaptureModeThreadLocal=1,
    cudaStreamCaptureModeRelaxed=2
};
enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone=0, cudaStreamCaptureStatusActive=1,
    cudaStreamCaptureStatusInvalidated=2
};
inline constexpr unsigned cudaStreamDefault=0, cudaStreamNonBlocking=1;
inline constexpr unsigned cudaDeviceScheduleSpin=1, cudaDeviceMapHost=8;
inline constexpr unsigned cudaEventDefault=0, cudaEventBlockingSync=1,
    cudaEventDisableTiming=2, cudaEventInterprocess=4;
inline constexpr unsigned cudaHostAllocDefault=0, cudaHostAllocPortable=1,
    cudaHostAllocMapped=2, cudaHostAllocWriteCombined=4;
inline constexpr unsigned cudaHostRegisterDefault=0, cudaHostRegisterPortable=1,
    cudaHostRegisterMapped=2, cudaHostRegisterIoMemory=4, cudaHostRegisterReadOnly=8;
const char* cudaGetErrorString(cudaError_t) noexcept;
const char* cudaGetErrorName(cudaError_t) noexcept;
cudaError_t cudaGetLastError() noexcept;
cudaError_t cudaPeekAtLastError() noexcept;
cudaError_t cudaGetDeviceCount(int*) noexcept;
cudaError_t cudaGetDevice(int*) noexcept;
cudaError_t cudaSetDevice(int) noexcept;
cudaError_t cudaInitDevice(int, unsigned, unsigned) noexcept;
cudaError_t cudaDeviceCanAccessPeer(int*, int, int) noexcept;
cudaError_t cudaDeviceEnablePeerAccess(int, unsigned = 0) noexcept;
cudaError_t cudaDeviceDisablePeerAccess(int) noexcept;
cudaError_t cudaMemGetInfo(size_t*, size_t*) noexcept;
cudaError_t cudaMalloc(void**, size_t) noexcept;
cudaError_t cudaMallocHost(void**, size_t) noexcept;
cudaError_t cudaHostAlloc(void**, size_t, unsigned) noexcept;
template<class T> cudaError_t cudaMalloc(T** p, size_t bytes) noexcept { return cudaMalloc(reinterpret_cast<void**>(p),bytes); }
template<class T> cudaError_t cudaMallocHost(T** p, size_t bytes) noexcept { return cudaMallocHost(reinterpret_cast<void**>(p),bytes); }
cudaError_t cudaFree(void*) noexcept;
cudaError_t cudaFreeHost(void*) noexcept;
cudaError_t cudaHostRegister(void*, size_t, unsigned) noexcept;
cudaError_t cudaHostUnregister(void*) noexcept;
cudaError_t cudaHostGetDevicePointer(void**, void*, unsigned) noexcept;
cudaError_t cudaStreamCreate(cudaStream_t*) noexcept;
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned) noexcept;
cudaError_t cudaStreamDestroy(cudaStream_t) noexcept;
cudaError_t cudaStreamSynchronize(cudaStream_t) noexcept;
cudaError_t cudaStreamQuery(cudaStream_t) noexcept;
cudaError_t cudaDeviceSynchronize() noexcept;
cudaError_t cudaEventCreate(cudaEvent_t*) noexcept;
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned) noexcept;
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t) noexcept;
cudaError_t cudaEventDestroy(cudaEvent_t) noexcept;
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr) noexcept;
cudaError_t cudaEventQuery(cudaEvent_t) noexcept;
cudaError_t cudaEventSynchronize(cudaEvent_t) noexcept;
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0) noexcept;
cudaError_t cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t, void*) noexcept;
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) noexcept;
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t = nullptr) noexcept;
cudaError_t cudaMemcpyPeerAsync(void*, int, const void*, int, size_t, cudaStream_t = nullptr) noexcept;
cudaError_t cudaMemcpy2DAsync(void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t = nullptr) noexcept;
cudaError_t cudaMemset(void*, int, size_t) noexcept;
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t = nullptr) noexcept;
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) noexcept;
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) noexcept;
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus*) noexcept;
cudaError_t cudaGraphGetNodes(cudaGraph_t, cudaGraphNode_t*, size_t*) noexcept;
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, cudaGraphNode_t*, char*, size_t) noexcept;
cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, unsigned long long = 0) noexcept;
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) noexcept;
cudaError_t cudaGraphDestroy(cudaGraph_t) noexcept;
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) noexcept;
