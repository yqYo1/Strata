#pragma once
#include <cstdint>
// Private SYCL prefill representation experiment; public/decode APIs remain log-gate.
namespace strata::prefill {
enum class GdnGateEncoding { LogGate, DecayFactor };
bool gdn_prefill_gate_factor_enabled();
void gdn_gates_encoded(GdnGateEncoding encoding,const float* ab,const float* dt,const float* ssm_a,
                       float* gate,float* beta,int64_t T,void* stream);
void gdn_recurrence_encoded(GdnGateEncoding encoding,int variant,float* state,const float* h,
 const float* gate,const float* beta,const float* z,const float* gamma,float eps,float* y,
 uint16_t* y16,int64_t T,void* stream,int64_t ld16=0);
} // namespace strata::prefill
