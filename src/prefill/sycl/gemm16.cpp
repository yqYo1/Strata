// Authored for Strata; repository LICENSE (MIT) applies.
// Native 16-bit producer: FP32 matrix accumulation and FP32 beta epilogue,
// followed by one output rounding inside this kernel. No full FP32 tensor.
#include "strata/prefill/gemm.hpp"
#include "strata/sycl/cpu_wait_profile.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace strata::prefill {
namespace {
using namespace sycl_backend;
namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace mi = sycl::ext::intel::experimental::matrix;
using BFloat = sycl::ext::oneapi::bfloat16;
constexpr size_t BM = 16, BN = 32, BK = 32, WG = 64;
// Each subgroup owns one 8x16 FP32 accumulator; A is row major. The W row
// tile is exactly a column-major B[K,N] tile, so loads stay contiguous in K.
template <class Input> struct alignas(64) Tiles {
  Input a[BM * BK];
  Input b[BN * BK];
};
static_assert(sizeof(Tiles<sycl::half>) == 3072);
static_assert(sizeof(Tiles<BFloat>) == 3072);
static_assert(std::is_trivially_copyable_v<BFloat> && sizeof(BFloat) == 2);
static_assert(sycl::is_device_copyable_v<Tiles<BFloat>>);

size_t product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::invalid_argument("SYCL GEMM16 size overflow");
  return a * b;
}
struct Geometry {
  size_t t, n, k, ld, xbytes, wbytes, ybytes, ntiles, global;
};
Geometry geometry(int64_t t, int64_t n, int64_t k, int64_t ld, float beta) {
  if (ld == 0)
    ld = n;
  if (t <= 0 || n <= 0 || k <= 0 || ld < n || !std::isfinite(beta))
    throw std::invalid_argument("invalid SYCL GEMM16 dimensions/stride/beta");
  if (n % 16 || k % 16)
    throw std::invalid_argument("SYCL GEMM16 requires N and K multiples of 16");
  const size_t xs = checked_count(t, k), ws = checked_count(n, k);
  // Conservatively reject overflow of the complete strided rectangle, even
  // though validation touches only through the final live output element.
  (void)product(checked_count(t, ld), sizeof(uint16_t));
  Geometry g{size_t(t), size_t(n), size_t(k), size_t(ld),
             product(xs, 2), product(ws, 2),
             product(size_t(t - 1) * size_t(ld) + size_t(n), 2),
             size_t(n - 1) / BN + 1, 0};
  g.global = product(product(size_t(t - 1) / BM + 1, g.ntiles), WG);
  return g;
}
bool accepts(size_t fixed, size_t maximum, size_t wanted) {
  return fixed ? fixed == wanted : maximum >= wanted;
}
uint8_t matrix_support(const sycl::device &device) {
  // Native Intel GPU kernels only; no CPU or software GEMM fallback.
  if (!device.is_gpu() ||
      device.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
      device.get_info<sycl::info::device::max_work_group_size>() < WG ||
      device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] < WG ||
      device.get_info<sycl::info::device::local_mem_size>() < 3072)
    return 1;
  const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
  if (std::find(sizes.begin(), sizes.end(), 16) == sizes.end())
    return 1;
  uint8_t support = device.has(sycl::aspect::fp16) ? 9 : 1;
  try {
    for (const auto &c : device.get_info<sycl::ext::oneapi::experimental::info::
                                             device::matrix_combinations>()) {
      if (c.ctype != mx::matrix_type::fp32 || c.dtype != mx::matrix_type::fp32 ||
          !accepts(c.msize, c.max_msize, 8) ||
          !accepts(c.nsize, c.max_nsize, 16) ||
          !accepts(c.ksize, c.max_ksize, 16))
        continue;
      if (c.atype == mx::matrix_type::fp16 && c.btype == c.atype &&
          device.has(sycl::aspect::fp16))
        support |= 2;
      if (c.atype == mx::matrix_type::bf16 && c.btype == c.atype)
        support |= 4;
    }
  } catch (const sycl::exception &) {
    return 1;
  }
  return support;
}

template <class Input, class Output>
void submit(const uint16_t *x, const uint16_t *w, uint16_t *y,
            Geometry g, float beta, void *stream, sycl::queue &q) {
  cpu_wait_profile::Host submission(cpu_wait_profile::gemm_submit);
  auto event = q.submit([&](sycl::handler &h) {
    sycl::local_accessor<Tiles<Input>, 1> storage(1, h);
    h.parallel_for(sycl::nd_range<1>(g.global, WG),
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const size_t group = it.get_group_linear_id();
        const size_t row0 = (group / g.ntiles) * BM;
        const size_t col0 = (group % g.ntiles) * BN;
        const size_t tid = it.get_local_linear_id();
        const auto sg = it.get_sub_group();
        const size_t sub = sg.get_group_linear_id();
        const size_t ar = (sub / 2) * 8, bc = (sub % 2) * 16;
        auto a_ptr = sycl::address_space_cast<
            sycl::access::address_space::local_space,
            sycl::access::decorated::no>(&storage[0].a[0]);
        auto b_ptr = sycl::address_space_cast<
            sycl::access::address_space::local_space,
            sycl::access::decorated::no>(&storage[0].b[0]);
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> acc;
        mx::joint_matrix<sycl::sub_group, Input, mx::use::a, 8, 16,
                         mx::layout::row_major> a;
        mx::joint_matrix<sycl::sub_group, Input, mx::use::b, 16, 16,
                         mx::layout::col_major> b;
        mx::joint_matrix_fill(sg, acc, 0.f);
        for (size_t k0 = 0; k0 < g.k; k0 += BK) {
          for (size_t i = tid; i < BM * BK; i += WG) {
            const size_t r = row0 + i / BK, k = k0 + i % BK;
            const uint16_t bits = r < g.t && k < g.k ? x[r * g.k + k] : 0;
            a_ptr[i] = sycl::bit_cast<Input>(bits);
          }
          for (size_t i = tid; i < BN * BK; i += WG) {
            const size_t c = col0 + i / BK, k = k0 + i % BK;
            const uint16_t bits = c < g.n && k < g.k ? w[c * g.k + k] : 0;
            b_ptr[i] = sycl::bit_cast<Input>(bits);
          }
          it.barrier(sycl::access::fence_space::local_space);
          for (size_t ki = 0; ki < BK; ki += 16) {
            mx::joint_matrix_load(sg, a, a_ptr + ar * BK + ki, BK);
            mx::joint_matrix_load(sg, b, b_ptr + bc * BK + ki, BK);
            mx::joint_matrix_mad(sg, acc, a, b, acc);
          }
          // All four subgroups finish reading before any rewrites the tile.
          it.barrier(sycl::access::fence_space::local_space);
        }
        // Callback coordinates name unique logical elements of this subgroup's
        // accumulator. Predication is solely inside the element callback;
        // every subgroup member enters each collective above and below.
        mi::joint_matrix_apply(sg, acc, [=](float &value, size_t r, size_t c) {
          const size_t row = row0 + ar + r, col = col0 + bc + c;
          if (row < g.t && col < g.n) {
            const size_t offset = row * g.ld + col;
            if (beta != 0.f)
              value = sycl::fma(beta, float(sycl::bit_cast<Output>(y[offset])), value);
            y[offset] = sycl::bit_cast<uint16_t>(Output(value));
          }
        });
      });
  });
  submission.finish();
  finish(stream, event);
}
template <class Input>
void dispatch(const uint16_t *x, const uint16_t *w, uint16_t *y,
              int64_t t, int64_t n, int64_t k, bool output_bf16,
              int64_t ld, float beta, void *stream, uint8_t &support) {
  const auto g = geometry(t, n, k, ld, beta);
  validate_spans({{y, g.ybytes}}, {{x, g.xbytes}, {w, g.wbytes}}, 2);
  auto &q = queue_for(stream);
  validate_device_span(x, g.xbytes, q);
  validate_device_span(w, g.wbytes, q);
  validate_device_span(y, g.ybytes, q);
  if (!(support & 1))
    support = matrix_support(q.get_device());
  constexpr uint8_t required = std::is_same_v<Input, BFloat> ? 4 : 2;
  if (!(support & required))
    throw std::invalid_argument("SYCL GEMM16 needs native 8x16x16 input/FP32 matrix support");
  if (output_bf16)
    submit<Input, BFloat>(x, w, y, g, beta, stream, q);
  else {
    if (!(support & 8))
      throw std::invalid_argument("SYCL GEMM16 FP16 output requires fp16 support");
    submit<Input, sycl::half>(x, w, y, g, beta, stream, q);
  }
}
} // namespace
void Gemm::f16_out16(const uint16_t *x, const uint16_t *w, uint16_t *y,
                     int64_t t, int64_t n, int64_t k, bool output_bf16,
                     int64_t ld, float beta) {
  if (!handle_)
    throw std::invalid_argument("uninitialized SYCL GEMM16");
  dispatch<sycl::half>(x, w, y, t, n, k, output_bf16, ld, beta,
                       stream_, out16_support_);
}
void Gemm::bf16_out16(const uint16_t *x, const uint16_t *w, uint16_t *y,
                      int64_t t, int64_t n, int64_t k, bool output_bf16,
                      int64_t ld, float beta) {
  if (!handle_)
    throw std::invalid_argument("uninitialized SYCL GEMM16");
  dispatch<BFloat>(x, w, y, t, n, k, output_bf16, ld, beta,
                   stream_, out16_support_);
}
} // namespace strata::prefill
