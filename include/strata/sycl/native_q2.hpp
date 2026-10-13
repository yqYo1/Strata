#pragma once
#include <cstdint>
namespace strata::kernels {
// Experimental native Q2_0 expert path with FP32 activation scales, reciprocal
// quantization and the eight-lane AVX2 accumulator/reduction shape. The scalar
// correction is explicitly fused. CPU compiler and expf differences can still
// change last bits; this is not a cross-platform bit-identity promise.
// Fixed 2560/640 geometry. One pointer per hit; dst indexes cap output rows.
// count is device-side (0..cap). Input uses private 34-byte activation blocks:
// an int16 code-sum header and 32 int8 codes, plus separate FP32 scales. They
// share Q8_0's stride but are not Q8_0 weights. They are produced by
// native_q2_quantize_cpu_order. Scratch: native_expert_scratch_bytes(cap, 640).
// Matches the AVX2 activation quantizer: rounded maximum/127, reciprocal
// division, then half-away rounding. Do not replace the first division with
// multiplication by a pre-rounded reciprocal.
// Finite inputs with a representable reciprocal follow that contract. A block
// with non-finite input or scale/inverse is marked by a NaN scale and zero codes
// and sum, avoiding invalid float-to-int conversion. Callers must reject that
// marker; it is not a valid quantized activation or CPU parity promise.
void native_q2_quantize_cpu_order(const float *input, uint8_t *blocks,
                                  float *scales, int64_t n, void *stream);
void native_q2_expert_cpu_order(const unsigned long long *pointers,
                                const int32_t *destinations,
                                const int32_t *count, int cap, const uint8_t *x,
                                const float *scales, void *scratch,
                                float *output, void *stream);
} // namespace strata::kernels
