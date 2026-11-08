// Main-authored interleaved Q6_K x Q8_1 C4 component; repository LICENSE.
// DPAS operand-layout reference: this project's native_mmq_xmx.cpp and
// ggml-org/llama.cpp 34ca79122b44d43eda7efb203b82ec91db94c55c esimd.hpp.
// The latter's two K16 halves preserve Q6's per-16 signed scales. This uses a
// separately owned uint16_t interleave, not GGML's in-place SOA/reorder path.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "q6_dpas_component.hpp"
#include "strata/sycl/esimd_half.hpp"
#include "strata/sycl/launch.hpp"
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <string>

namespace strata::kernels::component {
namespace {
namespace e = sycl::ext::intel::esimd;
using U = e::simd<uint32_t, 16>;
using F = e::simd<float, 16>;
using namespace sycl_backend;

// Offsets are bytes into a uint16_t interleave. Every halfword is naturally
// aligned, including original Q6 blocks whose 210-byte stride is not aligned4.
U word(const uint16_t* weights, U base, uint32_t byte) SYCL_ESIMD_FUNCTION {
    return e::convert<uint32_t>(e::gather<uint16_t, 16>(weights, base + byte * 16)) |
        (e::convert<uint32_t>(e::gather<uint16_t, 16>(weights, base + (byte + 2) * 16)) << 16);
}
struct Q6DpasC4 {
    const uint16_t* weights;
    const uint8_t* activation;
    float* output;
    int k, n;
    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const uint32_t first = uint32_t(item.get_global_linear_id()) * 16;
        if (first >= uint32_t(n)) return; // no collective, barrier or shared memory
        const U rows(first, 1);
        const e::simd_mask<16> valid = rows < uint32_t(n);
        const U lane_bytes(0, 2);
        e::simd<float, 64> sum = 0.f;
        const int blocks = k / 256;
        const size_t pitch = size_t(k / 32) * 36;
        for (int block = 0; block < blocks; ++block) {
            const U base = lane_bytes + uint32_t((size_t(first / 16) * blocks + block) * (16 * 210));
            const F super = esimd_half_decode<16>(e::convert<uint32_t>(
                e::gather<uint16_t, 16>(weights, base + 208u * 16)));
#pragma unroll 1
            for (int unit = 0; unit < 8; ++unit) {
                const int part = unit % 4;
                const uint32_t ql = uint32_t((unit / 4) * 64 + (part % 2) * 32);
                const uint32_t qh = uint32_t(128 + (unit / 4) * 32);
                e::simd<uint32_t, 128> w;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const U low = (word(weights, base, ql + uint32_t(4 * j)) >> (4 * (part / 2))) & 0x0f0f0f0fu;
                    const U high = ((word(weights, base, qh + uint32_t(4 * j)) >> (2 * part)) & 0x03030303u) << 4;
                    // Each byte is0..63. This exact packed subtraction cannot
                    // borrow across byte lanes because of the high guard bit.
                    w.template select<16, 1>(16 * j) = (((low | high) | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
                }
                const U sc = e::convert<uint32_t>(e::gather<uint16_t, 16>(weights, base + uint32_t(192 + 2 * unit) * 16));
                auto lo8 = e::convert<uint8_t>(sc & 255u);
                auto hi8 = e::convert<uint8_t>(sc >> 8);
                const F s0 = e::convert<float>(e::simd<int8_t, 16>(lo8.template bit_cast_view<int8_t>())) * super;
                const F s1 = e::convert<float>(e::simd<int8_t, 16>(hi8.template bit_cast_view<int8_t>())) * super;
                e::simd<int32_t, 32> a0 = 0, a1 = 0;
                F dx = 0.f;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const auto* p = activation + size_t(c) * pitch + size_t(block * 8 + unit) * 36;
                    const auto* words = reinterpret_cast<const int32_t*>(p);
                    a0.template select<4, 1>(c * 8) = e::gather<int32_t, 4>(words, e::simd<uint32_t, 4>(4, 4));
                    a1.template select<4, 1>(c * 8 + 4) = e::gather<int32_t, 4>(words, e::simd<uint32_t, 4>(20, 4));
                    dx[c] = esimd_half_decode<1>(e::convert<uint32_t>(e::gather<uint16_t, 1>(
                        reinterpret_cast<const uint16_t*>(p), e::simd<uint32_t, 1>(0))))[0];
                }
                const e::simd<int32_t, 64> zero = 0;
                auto d0 = e::xmx::dpas<8, 4, int32_t, int32_t, uint32_t, int32_t,
                    e::xmx::dpas_argument_type::s8, e::xmx::dpas_argument_type::s8>(zero, w, a0);
                auto d1 = e::xmx::dpas<8, 4, int32_t, int32_t, uint32_t, int32_t,
                    e::xmx::dpas_argument_type::s8, e::xmx::dpas_argument_type::s8>(zero, w, a1);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const F low = e::convert<float>(e::simd<int32_t, 16>(d0.template select<16, 1>(c * 16)));
                    const F high = e::convert<float>(e::simd<int32_t, 16>(d1.template select<16, 1>(c * 16)));
                    sum.template select<16, 1>(c * 16) += (low * s0 + high * s1) * F(dx[c]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < 4; ++c)
            e::scatter<float, 16>(output, (rows + uint32_t(c * n)) * 4u,
                F(sum.template select<16, 1>(c * 16)), valid);
    }
    constexpr auto get(sycl::ext::oneapi::experimental::properties_tag) const {
        return sycl::ext::oneapi::experimental::properties{
            sycl::ext::intel::experimental::grf_size<128>};
    }
};
} // namespace

void q6_dpas_c4(const uint16_t* packed, const void* q8, float* output,
                int k, int n, int columns, void* stream) {
    if (!q6_dpas_c4_shape(k, n, columns) || !stream)
        throw std::invalid_argument("Q6 DPAS needs explicit queue and eligible C4 shape");
    auto& q = queue_for(stream);
    const size_t wb = q6_dpas_packed_bytes(k, n), xb = size_t(k / 32) * 36 * 4, yb = size_t(n) * 4 * 4;
    validate_spans({{output, yb}}, {{packed, wb}, {q8, xb}});
    validate_device_span(packed, wb, q); validate_device_span(q8, xb, q); validate_device_span(output, yb, q);
    const auto device = q.get_device();
    if (device.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
        device.get_info<sycl::info::device::name>().find("B570") == std::string::npos || !device.has(sycl::aspect::fp16))
        throw std::invalid_argument("Q6 DPAS component is scoped to B570/fp16");
    const size_t items = (size_t(n) + 15) / 16;
    const auto event = q.parallel_for(sycl::nd_range<1>(((items + 3) / 4) * 4, 4),
        Q6DpasC4{packed, static_cast<const uint8_t*>(q8), output, k, n});
    finish(stream, event);
}
} // namespace strata::kernels::component
