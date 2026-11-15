// Main-authored adapter; repository LICENSE applies.
// Use the existing independent oracle/guard/full-component cases unchanged.
// Include declarations before remapping only the fixture's tested entrypoint.
#include "strata/kernels/qsa_prefill_workgroup64.hpp"
#define qsa_bf16_prefix_batch strata::kernels::prefill::qsa_bf16_prefix_workgroup64_batch
#include "qsa_bf16_prefix.cpp"
