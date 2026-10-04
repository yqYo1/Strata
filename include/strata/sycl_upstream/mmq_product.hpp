#pragma once
#include "strata/sycl_upstream/mmq_quantize.hpp"

namespace strata::sycl_upstream {
// Routed Strata Product operation for the nine default MMQ formats.
// Bounds and unique in-range output IDs are caller-owned device data.
// Weights need the original 4096-byte tail
// guard; activations come from mmq_quantize, including its guard and padding.
struct MmqProduct {
    const void* weights;
    ggml_type type;
    int weight_rows, weight_cols;
    size_t expert_bytes;
    int experts;
    const MmqBlock* x;
    const int32_t* bounds; // [experts+1], offsets in the compact activation rows
    const int32_t* ids;    // compact activation row -> destination row
    int total_rows, max_rows;
    float* dst;
    int ld_dst;
};
bool mmq_product_supported(ggml_type type);
size_t mmq_matrix_bytes(ggml_type type, int rows, int cols);
sycl::event mmq_product(sycl::queue& queue, const MmqProduct& product);
} // namespace strata::sycl_upstream
