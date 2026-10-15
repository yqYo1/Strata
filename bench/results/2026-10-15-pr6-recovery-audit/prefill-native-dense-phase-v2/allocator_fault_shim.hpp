#pragma once
// FORCE-INCLUDE ONLY when compiling the proposed native_dense.cpp for the
// fault fixture. Do not use for any other object or production build.
#include <cuda_runtime.h>
cudaError_t strata_dense_fixture_malloc(void**, size_t);
#define cudaMalloc strata_dense_fixture_malloc
