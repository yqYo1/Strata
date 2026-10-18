// Native BF16 tensor boundaries, with FP32 local arithmetic and reductions.
// See docs/SYCL_NATIVE16_PREFILL.md for design sources and remaining bridges.
#pragma once
#include <cstdint>
#include <cstddef>

namespace strata::prefill::detail { class MoeWindowRoutes; }

namespace strata::prefill::native16 {
// Startup configuration only; active execution is restricted to long layer-major.
bool requested();
void restore(const uint16_t* x, float* y, int64_t n, void* stream);
void pack(const float* x, uint16_t* y, int64_t n, void* stream);
void broadcast(const float* embedding, uint16_t* residual, int64_t rows, void* stream);
void norm(const uint16_t* residual, const float* weight, float eps, float* scales,
          uint16_t* normalized, int64_t rows, void* stream);
void silu(const uint16_t* lo, uint16_t* out, int64_t rows, void* stream);
void mix(const uint16_t* residual, const float* scales, const float* weight,
         const uint16_t* gated, uint16_t* mixed_bf16, uint16_t* mixed_fp16,
         int64_t rows, void* stream);
void write(uint16_t* residual, const uint16_t* output, const float* inject,
           int64_t rows, void* stream);
void swiglu(const uint16_t* gate_up, uint16_t* hidden_fp16, int64_t rows, void* stream);
void swiglu_pair(const uint16_t* gate, const uint16_t* up, uint16_t* hidden_fp16,
                 int64_t rows, void* stream);
// Routing weights/scalar gate and the sum over ten experts stay FP32 locally.
void combine(const uint16_t* down, const int32_t* slot, const float* weight,
             const uint16_t* shared, const float* shared_gate, uint16_t* output,
             int64_t rows, int64_t down_rows, void* stream);
// Private immutable GPU copies must match the validated host plan. Its unique
// token/expert contract plus an in-order queue permits FP32 updates without
// atomics. The owner retains the plan, uploads and buffers through checked drain.
void moe_job_weights(const detail::MoeWindowRoutes& plan, const float* weights,
                     const int32_t* original_slots, float* job_weights,
                     uint32_t* status, void* stream);
// Two logical FP16 banks, with checked per-device-row routing. Invalid maps
// write zero and set sticky status bit4 before any operand load. No early bank
// reuse: all routed and shared consumers must retire before either owner moves.
void moe_gather_mixed(const detail::MoeWindowRoutes& plan, size_t job_index,
                      const uint16_t* primary, const uint16_t* tail, int64_t primary_rows,
                      const int32_t* source, uint16_t* gathered, uint32_t* status, void* stream);
void moe_accumulate(const detail::MoeWindowRoutes& plan, size_t job_index,
                    const uint16_t* down, const int32_t* source,
                    const int32_t* original_slots, const float* job_weights,
                    float* accumulator, uint32_t* status, void* stream);
void moe_finish(const float* accumulator, const uint16_t* shared,
                const float* shared_gate, uint16_t* output, int64_t rows, void* stream);
} // namespace strata::prefill::native16
