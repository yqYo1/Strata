// include/strata/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are BF16 on the device - either already (the pack's BF16 tensors) or dequantized from
// their native GGUF blocks into a reusable scratch (`dequant_bf16`) right before the product - and activations are
// rounded to BF16, which is also what llama.cpp's batched CUDA path does.  Tensor-core GEMM through cuBLAS.
#pragma once

#include <cstddef>
#include <cstdint>
#if defined(STRATA_ENABLE_SYCL)
#include <memory>
#endif
#include <string>

namespace strata::prefill {
#if defined(STRATA_ENABLE_SYCL)
struct Native16BackendState;
#endif

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// `scratch_elems`: BF16 elements of the dequantization scratch (the largest weight dequantized at once).
    bool init(void* stream, int64_t scratch_elems, std::string& err);
    /// The same with caller-owned device buffers (the prompt path borrowing expert-cache slots).
    bool init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                       std::string& err);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major).  `beta` = 1 adds.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f);

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

#if defined(STRATA_ENABLE_SYCL)
    /// SYCL native matrix units: X[T,K] . W[N,K]^T -> 16-bit Y[T,N].
    /// Inputs/weights share the named format; output_bf16 chooses Y's format.
    /// Accumulator and beta epilogue are FP32; one final RNE output rounding.
    /// T > 0; N,K > 0 and multiples of 16; ldy=0 means N, otherwise ldy>=N.
    /// All spans must be 2-byte aligned registered Strata device allocations in
    /// this stream's context/device. Y must not overlap X or W. beta is finite.
    /// No implicit scratch, fallback, clamp, or allocation. Unsupported devices
    /// and geometry throw before submission. Caller retains buffers until the
    /// in-order stream completes; null stream follows synchronous finish().
    /// Optional backend=vllm retains these tensor/storage meanings, but requires
    /// beta=0, N%64=K%32=0, 64-byte bases, ldy%8=0 and a live full Y[T,ldy]
    /// rectangle. See docs/SYCL_VLLM_XE2_NATIVE16.md for all limits. No fallback.
    void f16_out16(const uint16_t* X, const uint16_t* W, uint16_t* Y,
                   int64_t T, int64_t N, int64_t K, bool output_bf16 = false,
                   int64_t ldy = 0, float beta = 0.0f);
    void bf16_out16(const uint16_t* X, const uint16_t* W, uint16_t* Y,
                    int64_t T, int64_t N, int64_t K, bool output_bf16 = false,
                    int64_t ldy = 0, float beta = 0.0f);
#endif

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f);

    /// Caller-owned buffers only: the scratch and workspace moved (the prompt path laid its buffers out again).
    void rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
#if defined(STRATA_ENABLE_SYCL)
    // Bit 0: queried; bits 1/2: native half/BF16 with FP32 C and D;
    // bit 3: FP16 scalar output. Calls on a single Gemm require serialized
    // host ownership, as do init/rebind.
    uint8_t out16_support_ = 0;
    // Populated only for vllm. Retains the original Runtime; borrows stream_.
    // shared_ptr supports the incomplete state type at this public ABI boundary.
    std::shared_ptr<Native16BackendState> out16_backend_;
#endif
    void* handle_ = nullptr;
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    void* workspace_ = nullptr;
    bool external_ = false;
    [[maybe_unused]] void* hipblaslt_state_ = nullptr;
};



}  // namespace strata::prefill
