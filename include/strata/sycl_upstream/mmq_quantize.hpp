#pragma once
#include <sycl/sycl.hpp>
#include <ggml.h>
#include <cstddef>
#include <cstdint>

namespace strata::sycl_upstream {
// GGML 3cf03257 ggml-cuda/mmq.cuh; see third_party/ggml/LICENSE.
// This is the CUDA MMQ layout, NOT four ordinary block_q8_1 records.
enum class MmqLayout { D4, DS4, D2S6 };
struct MmqBlock {
    union {
        float d4[4];
        sycl::half2 ds4[4];
        sycl::half d2s6[8];
    };
    int8_t qs[128];
};
static_assert(sizeof(MmqBlock) == 144);
static_assert(offsetof(MmqBlock, qs) == 16);
MmqLayout mmq_layout(ggml_type type);
size_t mmq_q8_bytes(int64_t rows, int64_t cols);
// The Strata quantize() entry's 2D specialization. Queue must be in-order;
// pointers are device accessible. IDs select physical input rows, as upstream.
// The returned event owns completion; no internal synchronization is introduced.
sycl::event mmq_quantize(sycl::queue& queue, const float* x, const int32_t* ids,
                        MmqBlock* out, ggml_type type, int64_t cols,
                        int64_t ld, int64_t rows);
} // namespace strata::sycl_upstream
