// Original Gemm::bf16/f16 call oneMKL on the opaque stream's actual queue.
#include "strata/sycl_upstream/cuda/cublas_v2.h"
#include "strata/sycl_upstream/cuda_backend.hpp"
#include <oneapi/mkl.hpp>
#include <algorithm>
#include <limits>
#include <map>
#include <mutex>

namespace {
namespace backend = strata::sycl_upstream::cuda;
static_assert(sizeof(sycl::half)==2 && sizeof(oneapi::mkl::bfloat16)==2);
struct State {
    sycl::device device;
    cudaStream_t stream = nullptr;
    std::mutex mutex;
    explicit State(sycl::device device): device(std::move(device)) {}
};
struct Registry {
    std::mutex mutex;
    uintptr_t next = 1;
    std::map<cublasHandle_t, std::shared_ptr<State>> handles;
};
Registry& registry() { static Registry value; return value; }
std::shared_ptr<State> get(cublasHandle_t handle) {
    auto& r = registry();std::lock_guard lock(r.mutex);
    const auto it = r.handles.find(handle);
    return it == r.handles.end() ? nullptr : it->second;
}
cublasStatus_t status(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return CUBLAS_STATUS_SUCCESS;
        case cudaErrorMemoryAllocation: return CUBLAS_STATUS_ALLOC_FAILED;
        case cudaErrorInvalidResourceHandle: return CUBLAS_STATUS_NOT_INITIALIZED;
        case cudaErrorNotSupported: return CUBLAS_STATUS_NOT_SUPPORTED;
        case cudaErrorInvalidDevice: case cudaErrorInvalidValue: return CUBLAS_STATUS_INVALID_VALUE;
        default: return CUBLAS_STATUS_EXECUTION_FAILED;
    }
}
template<class F> cublasStatus_t call(F&& body) noexcept {
    try { return body(); }
    catch (const std::bad_alloc&) { return CUBLAS_STATUS_ALLOC_FAILED; }
    catch (...) { return CUBLAS_STATUS_EXECUTION_FAILED; }
}
size_t matrix_bytes(int rows, int cols, int ld, size_t element) {
    const size_t count = cols ? size_t(cols-1)*size_t(ld)+size_t(rows) : 0;
    if (count > std::numeric_limits<size_t>::max()/element) throw std::bad_alloc();
    return count*element;
}
}
cublasStatus_t cublasCreate(cublasHandle_t* output) noexcept { return call([&] {
    if (!output) return CUBLAS_STATUS_INVALID_VALUE;
    *output=nullptr;sycl::device device;
    const auto error=backend::stream_device(nullptr,&device);if(error!=cudaSuccess)return status(error);
    auto value=std::make_shared<State>(device);auto& r=registry();std::lock_guard lock(r.mutex);
    if(r.next==std::numeric_limits<uintptr_t>::max())return CUBLAS_STATUS_ALLOC_FAILED;
    const auto handle=reinterpret_cast<cublasHandle_t>(r.next++);
    r.handles.emplace(handle,std::move(value));*output=handle;return CUBLAS_STATUS_SUCCESS;
}); }
cublasStatus_t cublasDestroy(cublasHandle_t handle) noexcept { return call([&] {
    auto& r=registry();std::lock_guard lock(r.mutex);
    return r.handles.erase(handle) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_INITIALIZED;
}); }
cublasStatus_t cublasSetStream(cublasHandle_t handle,cudaStream_t stream) noexcept { return call([&] {
    auto value=get(handle);if(!value)return CUBLAS_STATUS_NOT_INITIALIZED;
    sycl::device device;const auto error=backend::stream_device(stream,&device);if(error!=cudaSuccess)return status(error);
    std::lock_guard lock(value->mutex);if(device!=value->device)return CUBLAS_STATUS_INVALID_VALUE;
    value->stream=stream;return CUBLAS_STATUS_SUCCESS;
}); }
cublasStatus_t cublasSetMathMode(cublasHandle_t handle,cublasMath_t mode) noexcept {
    if(!get(handle))return CUBLAS_STATUS_NOT_INITIALIZED;
    // Every product explicitly requests standard FP32 accumulation.
    return mode==CUBLAS_DEFAULT_MATH ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_SUPPORTED;
}
cublasStatus_t cublasSetWorkspace(cublasHandle_t handle,void*,size_t) noexcept {
    if(!get(handle))return CUBLAS_STATUS_NOT_INITIALIZED;
    // oneMKL BLAS exposes no caller workspace API. The original optional setup
    // call reports this failure and continues. Never promise a fixed allocation
    // budget or pretend to bind a buffer which the library does not use.
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
cublasStatus_t cublasGemmEx(cublasHandle_t handle,cublasOperation_t ta,cublasOperation_t tb,
    int m,int n,int k,const void* alpha,const void* a,cudaDataType_t at,int lda,
    const void* b,cudaDataType_t bt,int ldb,const void* beta,void* c,cudaDataType_t ct,int ldc,
    cublasComputeType_t compute,cublasGemmAlgo_t algo) noexcept { return call([&] {
    auto value=get(handle);if(!value)return CUBLAS_STATUS_NOT_INITIALIZED;
    if((ta!=CUBLAS_OP_N && ta!=CUBLAS_OP_T && ta!=CUBLAS_OP_C) ||
       (tb!=CUBLAS_OP_N && tb!=CUBLAS_OP_T && tb!=CUBLAS_OP_C) || m<0 || n<0 || k<0 ||
       lda<std::max(1,ta==CUBLAS_OP_N?m:k) || ldb<std::max(1,tb==CUBLAS_OP_N?k:n) || ldc<std::max(1,m))
        return CUBLAS_STATUS_INVALID_VALUE;
    if(at!=bt || (at!=CUDA_R_16F && at!=CUDA_R_16BF) || ct!=CUDA_R_32F ||
       compute!=CUBLAS_COMPUTE_32F || algo!=CUBLAS_GEMM_DEFAULT)return CUBLAS_STATUS_NOT_SUPPORTED;
    if(!m || !n)return CUBLAS_STATUS_SUCCESS;
    if(!alpha || !beta || !c || (k && (!a || !b)))return CUBLAS_STATUS_INVALID_VALUE;
    const float av=*static_cast<const float*>(alpha),bv=*static_cast<const float*>(beta);
    std::lock_guard lock(value->mutex);
    sycl::device device;auto error=backend::stream_device(value->stream,&device);
    if(error!=cudaSuccess)return status(error);
    if(device!=value->device)return CUBLAS_STATUS_INVALID_VALUE;
    for(const auto& buffer : {std::pair<const void*,size_t>{a,k?matrix_bytes(ta==CUBLAS_OP_N?m:k,ta==CUBLAS_OP_N?k:m,lda,2):0},
                              {b,k?matrix_bytes(tb==CUBLAS_OP_N?k:n,tb==CUBLAS_OP_N?n:k,ldb,2):0},
                              {c,matrix_bytes(m,n,ldc,4)}}) {
        error=backend::validate_device_buffer(value->stream,buffer.first,buffer.second);
        if(error!=cudaSuccess)return status(error);
    }
    const auto transa=ta==CUBLAS_OP_N?oneapi::mkl::transpose::nontrans:oneapi::mkl::transpose::trans;
    const auto transb=tb==CUBLAS_OP_N?oneapi::mkl::transpose::nontrans:oneapi::mkl::transpose::trans;
    return status(backend::submit(value->stream,[=](sycl::queue& q) {
        if(!k)return q.parallel_for(sycl::range<2>(size_t(n),size_t(m)),[=](sycl::id<2> i){
            auto* out=static_cast<float*>(c);const size_t at=i[0]*size_t(ldc)+i[1];out[at]=bv==0 ? 0 : bv*out[at];
        });
        constexpr auto mode=oneapi::mkl::blas::compute_mode::standard;
        if(at==CUDA_R_16F)return oneapi::mkl::blas::column_major::gemm(q,transa,transb,m,n,k,av,
            static_cast<const sycl::half*>(a),lda,static_cast<const sycl::half*>(b),ldb,bv,static_cast<float*>(c),ldc,mode);
        return oneapi::mkl::blas::column_major::gemm(q,transa,transb,m,n,k,av,
            static_cast<const oneapi::mkl::bfloat16*>(a),lda,static_cast<const oneapi::mkl::bfloat16*>(b),ldb,bv,static_cast<float*>(c),ldc,mode);
    }));
}); }
