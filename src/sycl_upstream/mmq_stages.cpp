// Processing/layout port of upstream src/prefill/moe_mmq.cu.
#include "strata/sycl_upstream/mmq_stages.hpp"
#include <stdexcept>

namespace strata::sycl_upstream {
namespace {
void ordered(const sycl::queue& q) {
    if (!q.is_in_order()) throw std::invalid_argument("MMQ stages require an in-order queue");
}
size_t rounded(size_t n) { return (n + 255) / 256 * 256; }
template<class T>
sycl::event copy(sycl::queue& q, const void* gate, const void* up, size_t na,
                 const void* down, size_t nc, void* gu, void* dn) {
    const auto* a = static_cast<const T*>(gate);
    const auto* b = static_cast<const T*>(up);
    const auto* c = static_cast<const T*>(down);
    auto* ab = static_cast<T*>(gu);
    auto* cd = static_cast<T*>(dn);
    return q.parallel_for(sycl::nd_range<1>(rounded(2 * na + nc), 256), [=](sycl::nd_item<1> it) {
        size_t i = it.get_global_linear_id();
        if (i < na) ab[i] = a[i];
        else if (i < 2 * na) ab[i] = b[i - na];
        else if (i < 2 * na + nc) cd[i - 2 * na] = c[i - 2 * na];
    });
}
}
sycl::event mmq_gather_native(sycl::queue& q, const void* gate, const void* up,
    size_t na, const void* down, size_t nc, void* gu, void* dn) {
    ordered(q);
    const uintptr_t alignment = uintptr_t(gate) | uintptr_t(up) | uintptr_t(down) |
                                uintptr_t(gu) | uintptr_t(dn) | na | nc;
    if (alignment % 16 == 0) return copy<sycl::uint4>(q, gate, up, na / 16, down, nc / 16, gu, dn);
    return copy<uint8_t>(q, gate, up, na, down, nc, gu, dn);
}
bool mmq_gather_native_group(sycl::queue& q, const prefill::mmq::GatherGroup& g,
    size_t up_off, size_t na, size_t down_off, size_t nc,
    void* gu, size_t gu_stride, void* dn, size_t dn_stride) {
    if (g.first < 0 || g.n <= g.first || g.n > prefill::mmq::kGatherGroupMax) return false;
    uintptr_t alignment = uintptr_t(gu) | uintptr_t(dn) | up_off | na | down_off | nc | gu_stride | dn_stride;
    for (int e = g.first; e < g.n; ++e) alignment |= uintptr_t(g.blob[e]);
    if (alignment % 16 != 0) return false;
    ordered(q);
    // Capture the pointer table by value, as CUDA's GroupArgs kernel argument.
    q.parallel_for(sycl::nd_range<2>({size_t(g.n - g.first), rounded((2 * na + nc) / 16)}, {1, 256}),
        [=](sycl::nd_item<2> it) {
            const int e = g.first + int(it.get_global_id(0));
            const size_t i = it.get_global_id(1), a = na / 16, c = nc / 16;
            const auto* src = reinterpret_cast<const sycl::uint4*>(g.blob[e]);
            auto* ab = reinterpret_cast<sycl::uint4*>(static_cast<uint8_t*>(gu) + e * gu_stride);
            auto* cd = reinterpret_cast<sycl::uint4*>(static_cast<uint8_t*>(dn) + e * dn_stride);
            if (i < a) ab[i] = src[i];
            else if (i < 2 * a) ab[i] = src[up_off / 16 + i - a];
            else if (i < 2 * a + c) cd[i - 2 * a] = src[down_off / 16 + i - 2 * a];
        });
    return true;
}
sycl::event mmq_gather_strata_q2(sycl::queue& q, const uint8_t* blob, void* gu, void* dn) {
    ordered(q);
    return q.parallel_for(sycl::nd_range<1>(rounded(1280 * 40 + 2560 * 10), 256), [=](sycl::nd_item<1> it) {
        const size_t i = it.get_global_linear_id();
        constexpr size_t ngu = 1280 * 40, nd = 2560 * 10;
        if (i >= ngu + nd) return;
        const bool gate_up = i < ngu;
        const size_t j = gate_up ? i : i - ngu;
        constexpr size_t d_codes = 1280 * 640, gu_sc = d_codes + 2560 * 160,
                         d_sc = gu_sc + 1280 * 40 * 2;
        const auto* codes = blob + (gate_up ? 0 : d_codes) + j * 16;
        const auto* scales = reinterpret_cast<const uint16_t*>(blob + (gate_up ? gu_sc : d_sc));
        auto* out = static_cast<uint16_t*>(gate_up ? gu : dn) + j * 9;
        const sycl::uint4 packed = *reinterpret_cast<const sycl::uint4*>(codes);
        const auto words = packed.as<sycl::vec<uint16_t, 8>>();
        out[0] = scales[j];
        for (int k = 0; k < 8; ++k) out[k + 1] = words[k];
    });
}
sycl::event mmq_swiglu(sycl::queue& q, const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved) {
    if (rows <= 0) return {};
    ordered(q);
    if (n_ff <= 0) throw std::invalid_argument("MMQ SwiGLU requires positive width");
    return q.parallel_for(sycl::nd_range<1>(rounded(rows * n_ff), 256), [=](sycl::nd_item<1> it) {
        const int64_t i = it.get_global_linear_id();
        if (i >= rows * n_ff) return;
        const int64_t r = i / n_ff, k = i % n_ff;
        const float* row = gu + r * 2 * n_ff;
        const float g = interleaved ? row[2 * k] : row[k];
        const float u = interleaved ? row[2 * k + 1] : row[n_ff + k];
        // CUDA uses __expf. Native exp is the target's corresponding approximate
        // operation; cross-vendor bit equality is not claimed.
        h[i] = g / (1.0f + sycl::native::exp(-g)) * u;
    });
}
sycl::event mmq_iota(sycl::queue& q, int32_t* dst, int64_t n) {
    if (n <= 0) return {};
    ordered(q);
    return q.parallel_for(sycl::nd_range<1>(rounded(n), 256), [=](sycl::nd_item<1> it) {
        const int64_t i = it.get_global_linear_id();
        if (i < n) dst[i] = int32_t(i);
    });
}
} // namespace strata::sycl_upstream
