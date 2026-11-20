#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before constructing/capturing sessions. Existing captures keep their
// selected kernels; changing this flag does not rewrite an existing graph.
void native_moe_combine_set_enabled(bool enabled);
bool native_moe_combine_enabled();

// Pinned CUDA weighted-reduction contract for one token, k in [1, 15].
// The backend fuses k=2..15; k=1 is its ordinary multiply/contiguous/add path.
// parts: k contiguous F32 rows of n_embd; weights: k F32 values. The first
// product rounds to F32, following products accumulate with FMA in expert order,
// and optional shared is added once afterward. Shared is not router weighted.
// Requires a nonnull ordered stream and disjoint output. No allocation or sync.
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream);
/// n_tok rows (parts [n,k,N], weights [n,k], shared/output [n,N]) in one launch, each as the single call.
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream);
#ifdef STRATA_ENABLE_SYCL
// Decode-only 4-token verifier shape, 10 routed experts and 2560 columns.
// gpu_rows names the valid rows of gpu_parts. The other rows come from
// mapped_cpu_parts. Both sources use [token, expert, column] indexing.
// Keep the CPU publication and GPU projections alive until stream completion.
void decode_moe_combine_mapped_rows(const float* mapped_cpu_parts, const float* gpu_parts,
                                    const int32_t* gpu_rows, const int32_t* gpu_count,
                                    const float* weights, const float* shared, float* output, void* stream);
#endif
}
