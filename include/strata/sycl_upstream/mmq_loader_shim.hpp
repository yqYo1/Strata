#pragma once
#include <sycl/sycl.hpp>
#include <ggml.h>
#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include <ggml-common.h>

namespace strata::sycl_upstream::original_loaders {
// CUDA loader threads are logical lanes. No warp instructions are used in
// these loaders; SYCL subgroup geometry is independent of this indexing.
struct ThreadIdx { int x, y; };
struct alignas(8) int2 { int x, y; };
struct alignas(8) uint2 { unsigned x, y; };
inline int2 make_int2(int x, int y) { return {x, y}; }
inline int min(int a, int b) { return a < b ? a : b; }
inline float __half2float(sycl::half v) { return float(v); }
inline uint32_t __popc(uint32_t v) { return sycl::popcount(v); }
inline uint32_t __byte_perm(uint32_t a, uint32_t b, uint32_t selector) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t index = (selector >> (4 * i)) & 7u;
        result |= (((index < 4 ? a : b) >> (8 * (index % 4))) & 255u) << (8 * i);
    }
    return result;
}
inline uint32_t __vcmpne4(uint32_t a, uint32_t b) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i)
        result |= (((a >> (8 * i)) & 255u) != ((b >> (8 * i)) & 255u) ? 255u : 0u) << (8 * i);
    return result;
}
inline uint32_t __vsub4(uint32_t a, uint32_t b) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i)
        result |= (((a >> (8 * i)) - (b >> (8 * i))) & 255u) << (8 * i);
    return result;
}
constexpr int MMQ_ITER_K = 256;
constexpr int MMQ_TILE_NE_K = 32;
constexpr int kWeightRows = 16;
constexpr int kThreads = 128;
// Padding accommodates both eight 32-element and sixteen 16-element scales.
constexpr int kSramStride = 84;
constexpr int ggml_cuda_get_physical_warp_size() { return 32; }
constexpr int ggml_cuda_mmq_get_nthreads(ggml_type, int, bool) { return kThreads; }
constexpr int ggml_cuda_mmq_get_I(ggml_type, int, bool) { return kWeightRows; }
constexpr int ggml_cuda_mmq_get_sram_stride(ggml_type, int, bool) { return kSramStride; }
} // namespace strata::sycl_upstream::original_loaders
