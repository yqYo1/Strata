#include "strata/prefill/gemm.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/sycl/launch.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <oneapi/mkl/blas.hpp>

namespace strata::prefill {
namespace {
using namespace sycl_backend;
void shape(int64_t T, int64_t N, int64_t K, int64_t ld, float beta) {
  if (T <= 0 || N <= 0 || K <= 0 || ld < N || !std::isfinite(beta))
    throw std::invalid_argument("invalid SYCL prefill GEMM shape");
  if (checked_count(T, ld) > SIZE_MAX / 4 ||
      checked_count(N, K) > SIZE_MAX / 2 || checked_count(T, K) > SIZE_MAX / 2)
    throw std::invalid_argument("SYCL prefill GEMM byte overflow");
}
template <class Input>
void multiply(const uint16_t *x, const uint16_t *w, float *y, int64_t T,
              int64_t N, int64_t K, int64_t ld, float beta, void *stream) {
  shape(T, N, K, ld, beta);
  validate_spans({{y, size_t(T - 1) * ld * 4 + size_t(N) * 4}},
                 {{x, size_t(T) * K * 2}, {w, size_t(N) * K * 2}});
  auto event = oneapi::mkl::blas::row_major::gemm(
      queue_for(stream), oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::trans, T, N, K, 1.f,
      reinterpret_cast<const Input *>(x), K, reinterpret_cast<const Input *>(w),
      K, beta, y, ld);
  finish(stream, event);
}
void scratch_valid(uint16_t *scratch, int64_t count) {
  if (count < 0 || uint64_t(count) > SIZE_MAX / 2 ||
      (count && (!scratch || uintptr_t(scratch) % 4)))
    throw std::invalid_argument("invalid SYCL GEMM dequantization scratch");
}
} // namespace
Gemm::~Gemm() {
  if (!external_ && scratch_)
    (void)cudaFree(scratch_);
}
bool Gemm::init_external(void *stream, uint16_t *scratch, int64_t count,
                         void *workspace, size_t, std::string &err) {
  try {
    if (handle_)
      throw std::invalid_argument("SYCL GEMM is already initialized");
    scratch_valid(scratch, count);
    auto &q = queue_for(stream);
    stream_ = stream;
    scratch_ = scratch;
    scratch_elems_ = count;
    workspace_ = workspace;
    external_ = true;
    handle_ = &q;
    return true;
  } catch (const std::exception &e) {
    err = e.what();
    return false;
  }
}
bool Gemm::init(void *stream, int64_t count, std::string &err) {
  if (handle_ || count < 0 || uint64_t(count) > SIZE_MAX / 2) {
    err = "invalid SYCL GEMM initialization";
    return false;
  }
  uint16_t *scratch = nullptr;
  if (count) {
    auto status = cudaMalloc(&scratch, size_t(count) * 2);
    if (status != cudaSuccess) {
      err = cudaGetErrorString(status);
      return false;
    }
  }
  if (!init_external(stream, scratch, count, nullptr, 0, err)) {
    if (scratch)
      (void)cudaFree(scratch);
    return false;
  }
  external_ = false;
  return true;
}
void Gemm::rebind(uint16_t *scratch, int64_t count, void *workspace, size_t) {
  if (!handle_ || !external_)
    throw std::invalid_argument("SYCL GEMM rebind needs caller-owned buffers");
  scratch_valid(scratch, count);
  scratch_ = scratch;
  scratch_elems_ = count;
  workspace_ = workspace;
}
void Gemm::bf16(const uint16_t *x, const uint16_t *w, float *y, int64_t T,
                int64_t N, int64_t K, int64_t ld, float beta) {
  if (!handle_)
    throw std::invalid_argument("uninitialized SYCL GEMM");
  multiply<oneapi::mkl::bfloat16>(x, w, y, T, N, K, ld > 0 ? ld : N, beta,
                                  stream_);
}
void Gemm::f16(const uint16_t *x, const uint16_t *w, float *y, int64_t T,
               int64_t N, int64_t K, int64_t ld, float beta) {
  if (!handle_)
    throw std::invalid_argument("uninitialized SYCL GEMM");
  multiply<sycl::half>(x, w, y, T, N, K, ld > 0 ? ld : N, beta, stream_);
}
void Gemm::native(const uint16_t *x, int type, const void *w, float *y,
                  int64_t T, int64_t N, int64_t K, int64_t ld, float beta) {
  if (!handle_)
    throw std::invalid_argument("uninitialized SYCL GEMM");
  if (ld <= 0)
    ld = N;
  shape(T, N, K, ld, beta);
  if (!strata::kernels::iq_row_bytes(type, K) || K % 256 || scratch_elems_ < K)
    throw std::invalid_argument(
        "unsupported SYCL native GEMM format or insufficient scratch");
  validate_spans({{y, size_t(T - 1) * ld * 4 + size_t(N) * 4},
                  {scratch_, size_t(scratch_elems_) * 2}},
                 {{x, size_t(T) * K * 2},
                  {w, size_t(N) * strata::kernels::iq_row_bytes(type, K)}});
  const int64_t rows = scratch_elems_ / K;
  for (int64_t r = 0; r < N; r += rows) {
    const int64_t count = std::min(rows, N - r);
    strata::kernels::iq_dequant_f16(
        type,
        static_cast<const uint8_t *>(w) +
            size_t(r) * strata::kernels::iq_row_bytes(type, K),
        count * K, scratch_, stream_);
    f16(x, scratch_, y + r, T, count, K, ld, beta);
  }
}
} // namespace strata::prefill
