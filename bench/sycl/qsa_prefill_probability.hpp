// Strata prefill component fixture. Repository LICENSE applies.
#pragma once
#include "strata/kernels/qsa_bf16_prefix.hpp"

namespace strata::kernels::prefill_probability_reuse {
decltype(strata::kernels::qsa_bf16_prefix_ordered_exp_batch) qsa_bf16_prefix_ordered_exp_batch;
decltype(strata::kernels::qsa_bf16_prefix_state_probe_batch) qsa_bf16_prefix_state_probe_batch;
}
