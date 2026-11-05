// Main-authored expression discriminator for the decode Q6 packed-byte dot.
// The scalar reference is packed_dot from Strata native_mmvq.cpp at
// 9922f8a5aff9631c87cda6b31e9526f0e4e4e266, adapted there from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. Retain its notice:
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/dot_product.hpp>
#include <cstdint>
#include <stdexcept>

namespace strata::bench {
// This translation unit has no main, device discovery, allocation or execution
// driver. First use is offline LLVM/SPIR-V inspection, not a device benchmark.
// Its two kernel instantiations have identical geometry and unknown operands.
// They isolate the signed four-byte expression with accumulator zero, as used
// by Q6. They do not reproduce Q6 reconstruction, scales or column reductions.
template <bool Vector> class Q6DotExpression;

template <bool Vector>
int32_t q6_dot_expression(int32_t a, int32_t b) {
    if constexpr (Vector) {
        sycl::vec<int8_t, 4> av, bv;
        for (int lane = 0; lane < 4; ++lane) {
            av[lane] = int8_t(uint32_t(a) >> (8 * lane));
            bv[lane] = int8_t(uint32_t(b) >> (8 * lane));
        }
        return sycl::ext::oneapi::dot_acc(av, bv, 0);
    } else {
        int32_t acc = 0;
        for (int lane = 0; lane < 4; ++lane)
            acc += int(int8_t(uint32_t(a) >> (8 * lane))) *
                   int(int8_t(uint32_t(b) >> (8 * lane)));
        return acc;
    }
}

// Future execution requires disjoint, live same-context USM spans of count
// int32 elements. No production caller exists. All signed-byte dot results fit
// int32: four products are bounded by 65536 in magnitude; neither expression
// uses saturating accumulation or a packed-overload type-punning shortcut.
template <bool Vector>
void enqueue_q6_dot_expression(sycl::queue& queue, const int32_t* a,
                               const int32_t* b, int32_t* out, size_t count) {
    if (!count || count % 128)
        throw std::invalid_argument("Q6 dot discriminator needs whole 128-item groups");
    queue.parallel_for<Q6DotExpression<Vector>>(
        sycl::nd_range<1>(count, 128),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const size_t i = item.get_global_linear_id();
            out[i] = q6_dot_expression<Vector>(a[i], b[i]);
        });
}

template void enqueue_q6_dot_expression<false>(sycl::queue&, const int32_t*,
                                               const int32_t*, int32_t*, size_t);
template void enqueue_q6_dot_expression<true>(sycl::queue&, const int32_t*,
                                              const int32_t*, int32_t*, size_t);
} // namespace strata::bench
