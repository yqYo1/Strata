#pragma once
// Source frontend for Strata's GEMM calls, not the cuBLAS binary ABI.
#include "cuda_runtime.h"
struct strata_cublas_handle;
using cublasHandle_t = strata_cublas_handle*;
enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_NOT_INITIALIZED=1,
    CUBLAS_STATUS_ALLOC_FAILED=3, CUBLAS_STATUS_INVALID_VALUE=7,
    CUBLAS_STATUS_ARCH_MISMATCH=8, CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_NOT_SUPPORTED=15
};
enum cublasOperation_t { CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_OP_C=2 };
enum cudaDataType_t { CUDA_R_32F=0, CUDA_R_16F=2, CUDA_R_16BF=14 };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F=68 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT=-1 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH=0 };
cublasStatus_t cublasCreate(cublasHandle_t*) noexcept;
cublasStatus_t cublasDestroy(cublasHandle_t) noexcept;
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) noexcept;
cublasStatus_t cublasSetWorkspace(cublasHandle_t, void*, size_t) noexcept;
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) noexcept;
cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const void*, const void*, cudaDataType_t, int,
    const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int,
    cublasComputeType_t, cublasGemmAlgo_t) noexcept;
