// Authored for Strata; repository LICENSE (MIT) applies.
// Dense adapter/policies use public pinned SYCL-TLA APIs, not the unlicensed
// vLLM policy header or a replacement joint_matrix implementation.
#include "native16.hpp"
#include "xe_gemm_bsd.hpp"
#include "strata/sycl/launch.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace strata::experimental::vllm_xe2_native16 {
namespace {
using namespace cute;
using sycl_backend::validate_device_span;
using sycl_backend::validate_spans;
static_assert(sizeof(long) == 8, "Pinned TLA block2D payload ABI needs LP64");
static_assert(sizeof(cutlass::half_t) == 2 && sizeof(cutlass::bfloat16_t) == 2);

size_t mul(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::invalid_argument("Xe2 GEMM: size multiplication overflow");
  return a * b;
}

struct Geometry {
  int t, n, k, ld;
  size_t xb, wb, yb, groups_m, groups_n, threads;
};

Geometry geometry(const Problem &p, Tile tile) {
  if (p.t <= 0 || p.n <= 0 || p.k <= 0 || p.ldy < 0 ||
      p.beta != 0 || !std::isfinite(p.beta))
    throw std::invalid_argument("Xe2 GEMM: positive dimensions, beta=0 required");
  if ((p.input != Storage::fp16 && p.input != Storage::bf16) ||
      (p.output != Storage::fp16 && p.output != Storage::bf16))
    throw std::invalid_argument("Xe2 GEMM: invalid storage kind");
  const int64_t ld = p.ldy ? p.ldy : p.n;
  // V1 intentionally accepts full K tiles and N multiples of 64 only. M is
  // logical, unpadded, including 1/3/17. No assumption of K-tail safety.
  if (ld < p.n || p.n % 64 || p.k % 32 || ld % 8)
    throw std::invalid_argument("Xe2 GEMM: N%64=0, K%32=0, ldy>=N, ldy%8=0 required");
  constexpr int64_t surface_max = 0xFFFFFF; // strict pinned TLA assertion bound
  if (p.t > surface_max || p.n > surface_max || p.k > surface_max ||
      ld > surface_max || p.n * 2 > surface_max ||
      p.k * 2 > surface_max || ld * 2 > surface_max)
    throw std::invalid_argument("Xe2 GEMM: block2D width/pitch/height limit");
  const size_t xc = mul(size_t(p.t), size_t(p.k));
  const size_t wc = mul(size_t(p.n), size_t(p.k));
  const size_t yc = mul(size_t(p.t), size_t(ld));
  // CuTe dynamic shapes/strides/coordinate products use int in this adapter.
  // Reject the complete byte rectangles conservatively, including row padding.
  const size_t xb = mul(xc, 2), wb = mul(wc, 2), yb = mul(yc, 2);
  if (std::max({xb, wb, yb}) > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Xe2 GEMM: int coordinate extent limit");
  int bm, bn;
  size_t threads;
  switch (tile) {
  case Tile::compact_16x64: bm = 16; bn = 64; threads = 64; break;
  case Tile::wide_128x128: bm = 128; bn = 128; threads = 128; break;
  default: throw std::invalid_argument("Xe2 GEMM: invalid tile");
  }
  const size_t gm = size_t(p.t - 1) / size_t(bm) + 1;
  const size_t gn = size_t(p.n - 1) / size_t(bn) + 1;
  if (mul(mul(gm, gn), threads) > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Xe2 GEMM: int launch extent limit");
  return {int(p.t), int(p.n), int(p.k), int(ld), xb, wb, yb, gm, gn, threads};
}

bool check_device(sycl_backend::Runtime &runtime, const sycl::queue &q) {
  if (!q.has_property<sycl::property::queue::in_order>() ||
      q.get_context() != runtime.context() || q.get_device() != runtime.device())
    throw std::invalid_argument("Xe2 GEMM: caller Runtime in-order queue required");
  const auto d = q.get_device();
  namespace ex = sycl::ext::oneapi::experimental;
  if (!d.is_gpu() || d.get_info<sycl::info::device::vendor_id>() != 0x8086 ||
      q.get_backend() != sycl::backend::ext_oneapi_level_zero ||
      !d.has(sycl::aspect::usm_device_allocations))
    throw std::invalid_argument("Xe2 GEMM: Intel Level Zero device USM GPU required");
  // Architecture query, not a name substring or assumed device ordinal.
  // Unsupported query throws; there is no software or generic-device fallback.
  const auto arch = d.get_info<ex::info::device::architecture>();
  if (arch != ex::architecture::intel_gpu_bmg_g21 &&
      arch != ex::architecture::intel_gpu_bmg_g31)
    throw std::invalid_argument("Xe2 GEMM: BMG G21/G31 architecture required");
  const auto sgs = d.get_info<sycl::info::device::sub_group_sizes>();
  const auto max_items = d.get_info<sycl::info::device::max_work_item_sizes<3>>();
  if (std::find(sgs.begin(), sgs.end(), size_t(16)) == sgs.end() ||
      d.get_info<sycl::info::device::max_work_group_size>() < 128 ||
      max_items[2] < 128)
    throw std::invalid_argument("Xe2 GEMM: SG16 and 128 work-items required");
  runtime.check();
  return d.has(sycl::aspect::fp16);
}

// Independently selected geometry, expressed through the public TLA helper.
// No policy/header code copied from gemm_xe2_policy.hpp. These are qualification
// variants, not a measured shape selector or occupancy/performance claim.
template <Tile> struct ShapePolicy;
template <> struct ShapePolicy<Tile::compact_16x64> {
  using Shape = cute::Shape<Int<16>, Int<64>, Int<32>>;
  using Subgroups = Layout<cute::Shape<Int<1>, Int<4>, Int<1>>,
                          Stride<Int<4>, Int<1>, Int<0>>>;
  static constexpr int threads = 64;
};
template <> struct ShapePolicy<Tile::wide_128x128> {
  using Shape = cute::Shape<Int<128>, Int<128>, Int<32>>;
  using Subgroups = Layout<cute::Shape<Int<4>, Int<2>, Int<1>>,
                          Stride<Int<2>, Int<1>, Int<0>>>;
  static constexpr int threads = 128;
};
template <class Input, class Output, Tile Policy> class DenseKernel;

template <class Input, class Output, Tile Policy>
sycl::event enqueue(sycl::queue &q, const Problem &p, Geometry g) {
  using P = ShapePolicy<Policy>;
  using Atom = MMA_Atom<XE_DPAS_TT<8, float, Input>>;
  using MMA = typename TiledMMAHelper<Atom, Layout<typename P::Shape>,
                                     typename P::Subgroups>::TiledMMA;
  static_assert(size(typename P::Subgroups{}) * 16 == P::threads);
  static_assert(decltype(size(MMA{}))::value == P::threads);
  static_assert(P::threads % 16 == 0);
  static_assert(get<0>(MMA{}.tile_mnk()) == get<0>(typename P::Shape{}));
  static_assert(get<1>(MMA{}.tile_mnk()) == get<1>(typename P::Shape{}));
  static_assert(get<2>(MMA{}.tile_mnk()) == 32);
  static_assert(std::is_same_v<typename MMA::ValTypeC, float>);
  static_assert(std::is_same_v<typename MMA::ValTypeD, float>);
  const auto x = reinterpret_cast<const Input *>(p.x);
  const auto w = reinterpret_cast<const Input *>(p.w);
  auto y = reinterpret_cast<Output *>(p.y);
  namespace ex = sycl::ext::oneapi::experimental;
  namespace ix = sycl::ext::intel::experimental;
  const ex::properties props{ex::sub_group_size<16>, ix::grf_size<256>};
  const sycl::range<3> local(1, 1, P::threads);
  const sycl::range<3> global(g.groups_m, g.groups_n, P::threads);
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<DenseKernel<Input, Output, Policy>>(
        sycl::nd_range<3>(global, local), props, [=](sycl::nd_item<3> item) {
      // Pass full logical surfaces to TLA, never a padded-M allocation view.
      // All tensors are coordinate views; OOB tile coordinates are suppressed
      // by block2D surfaces, without divergent workgroup control flow.
      auto a = make_tensor(make_gmem_ptr(x),
          make_layout(make_shape(g.t, g.k), make_stride(g.k, _1{})));
      auto b = make_tensor(make_gmem_ptr(w),
          make_layout(make_shape(g.n, g.k), make_stride(g.k, _1{})));
      auto d = make_tensor(make_gmem_ptr(y),
          make_layout(make_shape(g.t, g.n), make_stride(g.ld, _1{})));
      auto coord = make_coord(int(item.get_group(0)), int(item.get_group(1)), _, 0);
      derived::xe_gemm_beta0<void, void, void>(a, b, d, coord, MMA{});
    });
  });
}

template <class I, class O>
sycl::event select_tile(sycl::queue &q, const Problem &p, Geometry g, Tile tile) {
  if (tile == Tile::compact_16x64)
    return enqueue<I, O, Tile::compact_16x64>(q, p, g);
  return enqueue<I, O, Tile::wide_128x128>(q, p, g);
}
} // namespace

Gemm::Gemm(sycl_backend::Runtime &runtime, sycl::queue &q)
    : runtime_(runtime), queue_(q), has_fp16_(check_device(runtime, q)) {}

sycl::event Gemm::beta0(const Problem &p, Tile tile) {
  const auto g = geometry(p, tile);
  if (!has_fp16_ && (p.input == Storage::fp16 || p.output == Storage::fp16))
    throw std::invalid_argument("Xe2 GEMM: native fp16 support required");
  validate_spans({{p.y, g.yb}}, {{p.x, g.xb}, {p.w, g.wb}}, 64);
  validate_device_span(p.x, g.xb, queue_);
  validate_device_span(p.w, g.wb, queue_);
  validate_device_span(p.y, g.yb, queue_);
  runtime_.check();
  if (p.input == Storage::fp16) {
    if (p.output == Storage::fp16)
      return select_tile<cutlass::half_t, cutlass::half_t>(queue_, p, g, tile);
    return select_tile<cutlass::half_t, cutlass::bfloat16_t>(queue_, p, g, tile);
  }
  if (p.output == Storage::fp16)
    return select_tile<cutlass::bfloat16_t, cutlass::half_t>(queue_, p, g, tile);
  return select_tile<cutlass::bfloat16_t, cutlass::bfloat16_t>(queue_, p, g, tile);
}
} // namespace strata::experimental::vllm_xe2_native16
