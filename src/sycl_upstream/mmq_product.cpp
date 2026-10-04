// GGML MMQ operation port. Original packed-weight tile loaders are extracted
// from pinned mmq-load-tiles.cuh; integer MMA is implemented by Intel XMX.
#include "strata/sycl_upstream/mmq_product.hpp"
#include "upstream_mmq_loaders.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/math.hpp>
#include <stdexcept>
#include <climits>

namespace strata::sycl_upstream {
namespace {
namespace original = original_loaders;
namespace mx = sycl::ext::oneapi::experimental::matrix;
constexpr int I = original::kWeightRows, J = 64, WG = original::kThreads;
constexpr int XS = original::kSramStride;
template<ggml_type> struct Format;
#define FORMAT(TYPE, name, QK, HALF_SCALE) \
    template<> struct Format<TYPE> { \
        static constexpr int qk = QK, bytes = sizeof(block_##name); \
        static constexpr bool half_scale = HALF_SCALE; \
        static void load(const char* w, int* tile, int offset, int last, int stride, original::ThreadIdx thread) { \
            original::ggml_cuda_mmq_load_tiles_##name<TYPE, J, true>(w, tile, offset, last, stride, thread); \
        } \
    };
FORMAT(GGML_TYPE_Q2_0, q2_0, QK2_0, false)
FORMAT(GGML_TYPE_IQ2_XXS, iq2_xxs, QK_K, false)
FORMAT(GGML_TYPE_IQ2_XS, iq2_xs, QK_K, true)
FORMAT(GGML_TYPE_IQ2_S, iq2_s, QK_K, true)
FORMAT(GGML_TYPE_IQ3_XXS, iq3_xxs, QK_K, false)
FORMAT(GGML_TYPE_IQ3_S, iq3_s, QK_K, false)
FORMAT(GGML_TYPE_IQ4_NL, iq4_nl, QK4_NL, false)
FORMAT(GGML_TYPE_IQ4_XS, iq4_xs, QK_K, false)
FORMAT(GGML_TYPE_Q8_0, q8_0, QK8_0, false)
#undef FORMAT

template<ggml_type Type>
sycl::event launch(sycl::queue& queue, MmqProduct p) {
    using F = Format<Type>;
    const int tile_rows = 1 + (p.weight_rows - 1) / I;
    const int tile_cols = 1 + (p.max_rows - 1) / J;
    const size_t groups = size_t(p.experts) * tile_rows * tile_cols;
    if (groups > INT_MAX || p.weight_cols > INT_MAX - 255)
        throw std::invalid_argument("MMQ launch dimensions overflow");
    return queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> wtile(I * XS, h);
        sycl::local_accessor<int8_t, 1> a_local(J * 32, h), b_local(32 * I, h);
        sycl::local_accessor<int, 1> part0(J * I, h), part1(J * I, h);
        h.parallel_for(sycl::nd_range<1>(groups * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid = int(it.get_local_linear_id());
                const int g = int(it.get_group_linear_id());
                const int tile_i = g % tile_rows, tile_j = (g / tile_rows) % tile_cols;
                const int expert = g / (tile_rows * tile_cols);
                const int begin = p.bounds[expert] + tile_j * J, end = p.bounds[expert + 1];
                if (begin >= end) return; // uniform across the workgroup
                const int wi = tile_i * I;
                const int last = sycl::min(I, p.weight_rows - wi) - 1;
                const auto* weights = reinterpret_cast<const char*>(p.weights) + expert * p.expert_bytes;
                const int row_stride = p.weight_cols / F::qk;
                const int weight_offset = wi * row_stride;
                auto* wt = wtile.template get_multi_ptr<sycl::access::decorated::no>().get();
                auto* ap = a_local.template get_multi_ptr<sycl::access::decorated::no>().get();
                auto* bp = b_local.template get_multi_ptr<sycl::access::decorated::no>().get();
                auto* c0 = part0.template get_multi_ptr<sycl::access::decorated::no>().get();
                auto* c1 = part1.template get_multi_ptr<sycl::access::decorated::no>().get();
                const int8_t* wc = reinterpret_cast<const int8_t*>(wt);
                const float* wd = reinterpret_cast<const float*>(wt) + 64;
                auto sg = it.get_sub_group();
                const int sg_index = tid / 16;
                using A = mx::joint_matrix<sycl::sub_group, int8_t, mx::use::a, 8, 32, mx::layout::row_major>;
                using B = mx::joint_matrix<sycl::sub_group, int8_t, mx::use::b, 32, I, mx::layout::ext_intel_packed>;
                using C = mx::joint_matrix<sycl::sub_group, int, mx::use::accumulator, 8, I>;
                float sums[J * I / WG] = {};
                for (int kb = 0; kb < p.weight_cols; kb += 256) {
                    F::load(weights, wt, weight_offset + kb / F::qk, last, row_stride, {tid % 32, tid / 32});
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int k = 0; k < 256; k += 32) {
                        // IQ2_XS/S perform two independent 16-element integer products,
                        // then combine their differently scaled sums (CUDA MMA path).
                        for (int half = 0; half < (F::half_scale ? 2 : 1); ++half) {
                            const int width = F::half_scale ? 16 : 32;
                            for (int idx = tid; idx < J * 32; idx += WG) {
                                const int row = begin + idx / 32, kk = idx % 32;
                                const int channel = kb + k + half * 16 + kk;
                                ap[idx] = row < end && kk < width ?
                                    p.x[size_t(channel / 128) * p.total_rows + row].qs[channel % 128] : 0;
                            }
                            for (int idx = tid; idx < 32 * I; idx += WG) {
                                const int kk = idx / I, row = idx % I;
                                bp[(kk / 4) * (I * 4) + row * 4 + kk % 4] = kk < width && row <= last ?
                                    wc[row * XS * 4 + k + half * 16 + kk] : 0;
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                            A a; B b; C c;
                            mx::joint_matrix_fill(sg, c, 0);
                            mx::joint_matrix_load(sg, a, a_local.template get_multi_ptr<sycl::access::decorated::no>() + sg_index * 8 * 32, 32);
                            mx::joint_matrix_load(sg, b, b_local.template get_multi_ptr<sycl::access::decorated::no>(), I * 4);
                            mx::joint_matrix_mad(sg, c, a, b, c);
                            mx::joint_matrix_store(sg, c, (half == 0 ? part0 : part1).template get_multi_ptr<sycl::access::decorated::no>() + sg_index * 8 * I, I, mx::layout::row_major);
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        for (int n = 0; n < J * I / WG; ++n) {
                            const int idx = tid + n * WG, row = begin + idx / I, wr = idx % I;
                            if (row >= end || wr > last) continue;
                            const float db = p.x[size_t((kb + k) / 128) * p.total_rows + row].d4[((kb + k) % 128) / 32];
                            float term;
                            if constexpr (F::half_scale) {
                                const float da0 = wd[wr * XS + k / 16], da1 = wd[wr * XS + k / 16 + 1];
                                term = sycl::fma(float(c0[idx]), da0, sycl::ext::intel::math::fmul_rn(float(c1[idx]), da1));
                            } else {
                                term = sycl::ext::intel::math::fmul_rn(float(c0[idx]), wd[wr * XS + k / 32]);
                            }
                            sums[n] = sycl::fma(term, db, sums[n]);
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                }
                for (int n = 0; n < J * I / WG; ++n) {
                    const int idx = tid + n * WG, row = begin + idx / I, wr = idx % I;
                    if (row < end && wr <= last)
                        p.dst[size_t(p.ids[row]) * p.ld_dst + wi + wr] = sums[n];
                }
            });
    });
}
struct Info { int qk, bytes; };
Info info(ggml_type type) {
#define CASE(TYPE) case TYPE: return {Format<TYPE>::qk, Format<TYPE>::bytes};
    switch (type) {
        CASE(GGML_TYPE_Q2_0) CASE(GGML_TYPE_IQ2_XXS) CASE(GGML_TYPE_IQ2_XS)
        CASE(GGML_TYPE_IQ2_S) CASE(GGML_TYPE_IQ3_XXS) CASE(GGML_TYPE_IQ3_S)
        CASE(GGML_TYPE_IQ4_NL) CASE(GGML_TYPE_IQ4_XS) CASE(GGML_TYPE_Q8_0)
        default: return {0, 0};
    }
#undef CASE
}
} // namespace
bool mmq_product_supported(ggml_type type) { return info(type).qk != 0; }
size_t mmq_matrix_bytes(ggml_type type, int rows, int cols) {
    auto f = info(type);
    if (!f.qk || rows <= 0 || cols <= 0 || cols % f.qk)
        throw std::invalid_argument("invalid MMQ matrix geometry/type");
    return size_t(rows) * (cols / f.qk) * f.bytes;
}
sycl::event mmq_product(sycl::queue& queue, const MmqProduct& p) {
    if (p.experts <= 0 || p.max_rows <= 0) return {};
    if (!queue.has_property<sycl::property::queue::in_order>() || !p.weights || !p.x || !p.bounds || !p.ids || !p.dst ||
        p.total_rows <= 0 || p.ld_dst < p.weight_rows || p.expert_bytes < mmq_matrix_bytes(p.type, p.weight_rows, p.weight_cols))
        throw std::invalid_argument("invalid MMQ product");
#define CASE(TYPE) case TYPE: return launch<TYPE>(queue, p);
    switch (p.type) {
        CASE(GGML_TYPE_Q2_0) CASE(GGML_TYPE_IQ2_XXS) CASE(GGML_TYPE_IQ2_XS)
        CASE(GGML_TYPE_IQ2_S) CASE(GGML_TYPE_IQ3_XXS) CASE(GGML_TYPE_IQ3_S)
        CASE(GGML_TYPE_IQ4_NL) CASE(GGML_TYPE_IQ4_XS) CASE(GGML_TYPE_Q8_0)
        default: throw std::invalid_argument("unsupported MMQ product type");
    }
#undef CASE
}
} // namespace strata::sycl_upstream
