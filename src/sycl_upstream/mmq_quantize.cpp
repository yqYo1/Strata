// Port of pinned GGML ggml-cuda/quantize.cu:quantize_mmq_q8_1 and
// ggml-cuda/mmq.cuh:mmq_get_q8_1_ds_layout (MIT; third_party/ggml/LICENSE).
#include "strata/sycl_upstream/mmq_quantize.hpp"
#include <sycl/ext/intel/math.hpp>
#include <limits>
#include <stdexcept>

namespace strata::sycl_upstream {
MmqLayout mmq_layout(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0: case GGML_TYPE_Q2_0: case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0: case GGML_TYPE_MXFP4: case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q3_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL:
            return MmqLayout::D4;
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_IQ1_S:
            return MmqLayout::DS4;
        case GGML_TYPE_Q2_K: return MmqLayout::D2S6;
        default: throw std::invalid_argument("no upstream MMQ Q8 layout for type");
    }
}
namespace {
int64_t padded(int64_t cols) {
    if (cols <= 0 || cols % 4 || cols > std::numeric_limits<int64_t>::max() - 511)
        throw std::invalid_argument("MMQ columns must be a positive multiple of four");
    return (cols + 511) / 512 * 512;
}
template<MmqLayout layout>
sycl::event submit(sycl::queue& queue, const float* x, const int32_t* ids,
                   MmqBlock* out, int64_t cols, int64_t ld, int64_t rows) {
    const int64_t ne0 = padded(cols);
    constexpr int vals_per_scale = layout == MmqLayout::D2S6 ? 64 : 32;
    constexpr int vals_per_sum = layout == MmqLayout::D2S6 ? 16 : 32;
    // CUDA grid=(rows, padded_cols/512), block=(128): four floats per lane.
    return queue.parallel_for(sycl::nd_range<2>({size_t(rows), size_t(ne0 / 4)}, {1, 128}),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
            const int64_t row = it.get_global_id(0);
            const int64_t i0 = int64_t(it.get_global_id(1)) * 4;
            const int64_t source_row = ids ? ids[row] : row;
            float v[4] = {};
            if (i0 < cols)
                for (int j = 0; j < 4; ++j) v[j] = x[source_row * ld + i0 + j];
            float amax = sycl::fabs(v[0]);
            for (int j = 1; j < 4; ++j) amax = sycl::fmax(amax, sycl::fabs(v[j]));
            auto sg = it.get_sub_group();
            for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1)
                amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, offset));
            float sum = 0;
            if constexpr (layout != MmqLayout::D4) {
                sum = ((v[0] + v[1]) + v[2]) + v[3];
                for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1)
                    sum += sycl::permute_group_by_xor(sg, sum, offset);
            }
            // Preserve the source expression order. CUDA builds this with
            // -use_fast_math; cross-device reciprocal error remains to be measured.
            const float inv = sycl::ext::intel::math::fdiv_rn(127.f, amax);
            const float d = sycl::ext::intel::math::fdiv_rn(1.f, inv);
            const int64_t ib = (i0 / 128) * rows + row;
            const int iqs = int(i0 % 128);
            for (int j = 0; j < 4; ++j) {
                // CUDA converts roundf(0 * inf) to int8 in a zero block.
                // Make zero codes defined here; actual CUDA byte parity pending.
                // Keep the FP32 multiply rounded before roundf and integer conversion.
                // The fused expression crossed a half-way boundary on B570.
                out[ib].qs[iqs + j] = amax == 0.f ? 0 : int8_t(sycl::round(sycl::ext::intel::math::fmul_rn(v[j], inv)));
            }
            if constexpr (layout == MmqLayout::D2S6) {
                if (iqs % 16 == 0 && iqs < 96) {
                    out[ib].d2s6[2 + iqs / 16] = sycl::half(sum);
                    if (iqs % 64 == 0) out[ib].d2s6[iqs / 64] = sycl::half(d);
                }
            } else if (iqs % 32 == 0) {
                if constexpr (layout == MmqLayout::DS4)
                    out[ib].ds4[iqs / 32] = sycl::half2(sycl::half(d), sycl::half(sum));
                else out[ib].d4[iqs / 32] = d;
            }
        });
}
} // namespace
size_t mmq_q8_bytes(int64_t rows, int64_t cols) {
    const size_t blocks_per_row = size_t(padded(cols) / 128);
    constexpr size_t guard = 128;
    if (rows < 0 || size_t(rows) > (std::numeric_limits<size_t>::max() / sizeof(MmqBlock) - guard) / blocks_per_row)
        throw std::invalid_argument("MMQ allocation overflow");
    return (size_t(rows) * blocks_per_row + guard) * sizeof(MmqBlock);
}
sycl::event mmq_quantize(sycl::queue& queue, const float* x, const int32_t* ids,
                        MmqBlock* out, ggml_type type, int64_t cols, int64_t ld, int64_t rows) {
    if (rows <= 0) return {};
    if (!queue.has_property<sycl::property::queue::in_order>() || !x || !out || ld < cols || ld % 4)
        throw std::invalid_argument("invalid MMQ queue/pointers/row stride");
    (void) mmq_q8_bytes(rows, cols);
    switch (mmq_layout(type)) {
        case MmqLayout::D4: return submit<MmqLayout::D4>(queue, x, ids, out, cols, ld, rows);
        case MmqLayout::DS4: return submit<MmqLayout::DS4>(queue, x, ids, out, cols, ld, rows);
        case MmqLayout::D2S6: return submit<MmqLayout::D2S6>(queue, x, ids, out, cols, ld, rows);
    }
    throw std::invalid_argument("invalid MMQ layout");
}
} // namespace strata::sycl_upstream
