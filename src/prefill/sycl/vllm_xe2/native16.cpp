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
#include <vector>

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
  static constexpr int rows = 16;
};
template <> struct ShapePolicy<Tile::wide_128x128> {
  using Shape = cute::Shape<Int<128>, Int<128>, Int<32>>;
  using Subgroups = Layout<cute::Shape<Int<4>, Int<2>, Int<1>>,
                          Stride<Int<2>, Int<1>, Int<0>>>;
  static constexpr int threads = 128;
  static constexpr int rows = 128;
};
template <class Input, Tile Policy>
using TileMMA = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, Input>>,
    Layout<typename ShapePolicy<Policy>::Shape>,
    typename ShapePolicy<Policy>::Subgroups>::TiledMMA;
template <class Input, class Output, Tile Policy> class DenseKernel;
template <Tile Policy, bool Capture, ScatterLayout Mapping> class DownScatterKernel;
template <Tile Policy, bool Capture> class GuSwiGLUKernel;

template <bool Capture> struct GuSwiGLUEpilogue {
  GuSwiGLUProblem p;
  template <class Fragment, class Coordinates, class MMA>
  CUTE_DEVICE void operator()(const Fragment &fragment, const Coordinates &tile,
                              const MMA &mma, int lane) const {
    auto rounded = make_fragment_like<cutlass::bfloat16_t>(fragment.tensor());
    reorder(fragment.tensor(), rounded, fragment.tv_layout(), fragment.tv_layout());
    auto coordinates = mma.get_slice(lane).partition_C(tile);
    static_assert(decltype(size(rounded))::value == decltype(size(coordinates))::value);
    static_assert(decltype(size(rounded))::value % 2 == 0);
    const auto sg = sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_sub_group();
    CUTE_UNROLL
    for (int i = 0; i < size(rounded); i += 2) {
      // Both BF16s keep their individual rounding. Exchange BEFORE logical
      // masking, with identical loop/mask participation by every SG16 lane.
      const uint32_t own = uint32_t(sycl::bit_cast<uint16_t>(rounded(i))) |
          (uint32_t(sycl::bit_cast<uint16_t>(rounded(i + 1))) << 16);
      const uint32_t partner = sycl::permute_group_by_xor(sg, own, 1);
      const bool odd = (int(get<1>(coordinates(i))) & 1) != 0;
      // Adjacent lanes split the TWO fragment indices: even-column lane uses
      // i, odd-column lane uses i+1. Host inspection proves unique H writers.
      const int index = i + int(odd), shift = int(odd) * 16;
      const uint16_t gb = uint16_t((odd ? partner : own) >> shift);
      const uint16_t ub = uint16_t((odd ? own : partner) >> shift);
      const auto coordinate = coordinates(index);
      const int row = get<0>(coordinate), col = get<1>(coordinate);
      if constexpr (Capture) {
        CUTE_UNROLL
        for (int j = 0; j < 2; ++j) {
          const auto c = coordinates(i + j);
          if (get<0>(c) >= 0 && get<0>(c) < p.rows && get<1>(c) >= 0 && get<1>(c) < 1280)
            p.rounded_diagnostic[size_t(get<0>(c)) * 1280 + size_t(get<1>(c))] =
                sycl::bit_cast<uint16_t>(rounded(i + j));
        }
      }
      if (row < 0 || row >= p.rows || col < 0 || col >= 1280) continue;
      const float g = float(sycl::bit_cast<cutlass::bfloat16_t>(gb));
      const float u = float(sycl::bit_cast<cutlass::bfloat16_t>(ub));
      const float v = (g * (1.f / (1.f + sycl::exp(-g)))) * u;
      const float clipped = sycl::isnan(v) ? v : sycl::clamp(v, -65504.f, 65504.f);
      p.y[size_t(row) * 640 + size_t(col / 2)] = sycl::bit_cast<uint16_t>(sycl::half(clipped));
    }
  }
};

template <Tile Policy, bool Capture>
sycl::event enqueue_gu(sycl::queue &q, GuSwiGLUProblem p, Geometry g) {
  using P = ShapePolicy<Policy>;
  using MMA = TileMMA<cutlass::half_t, Policy>;
  namespace ex = sycl::ext::oneapi::experimental;
  namespace ix = sycl::ext::intel::experimental;
  const ex::properties props{ex::sub_group_size<16>, ix::grf_size<256>};
  const sycl::nd_range<3> range(sycl::range<3>(g.groups_m, g.groups_n, P::threads),
                              sycl::range<3>(1, 1, P::threads));
  return q.submit([&](sycl::handler &h) {
    h.parallel_for<GuSwiGLUKernel<Policy, Capture>>(range, props, [=](sycl::nd_item<3> item) {
      auto a = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::half_t *>(p.x)),
          make_layout(make_shape(g.t, g.k), make_stride(g.k, _1{})));
      auto b = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::half_t *>(p.w)),
          make_layout(make_shape(g.n, g.k), make_stride(g.k, _1{})));
      auto c = make_identity_tensor(make_shape(g.t, g.n));
      const auto coord = make_coord(int(item.get_group(0)), int(item.get_group(1)), _, 0);
      derived::xe_gemm_epilogue<void, void, void>(a, b, c, coord, MMA{}, GuSwiGLUEpilogue<Capture>{p});
    });
  });
}

template <bool Capture, ScatterLayout Mapping> struct DownEpilogue {
  DownScatterProblem p;
  int t, token_rows;
  size_t routes, offset;

  template <class Fragment, class Coordinates, class MMA>
  CUTE_DEVICE void operator()(const Fragment &fragment, const Coordinates &tile,
                              const MMA &mma, int lane) const {
    if constexpr (Mapping == ScatterLayout::output_copy) {
      // Public layout-only factory: no fabricated pointer or global BF16 slab.
      // partition_sg_fragment_S constructs BF16 values in the SAME per-thread
      // shape as partition_S(tile). Dense output uses this exact conversion.
      auto copy = make_block_2d_copy_D<cutlass::bfloat16_t>(mma, make_stride(2560, _1{}));
      auto thread = copy.get_slice(lane);
      auto rounded = thread.partition_sg_fragment_S(tile);
      auto coordinates = thread.partition_S(tile);
      static_assert(decltype(size(coordinates))::value ==
                    decltype(size(rounded.tensor()))::value);
      reorder(fragment, rounded);
      scatter(rounded.tensor(), coordinates);
    } else {
      auto rounded = make_fragment_like<cutlass::bfloat16_t>(fragment.tensor());
      reorder(fragment.tensor(), rounded, fragment.tv_layout(), fragment.tv_layout());
      auto coordinates = mma.get_slice(lane).partition_C(tile);
      static_assert(decltype(size(coordinates))::value == decltype(size(rounded))::value);
      scatter(rounded, coordinates);
    }
  }

  template <class Values, class Coordinates>
  CUTE_DEVICE void scatter(const Values &rounded, const Coordinates &coordinates) const {
    CUTE_UNROLL
    for (int i = 0; i < size(rounded); ++i) {
      const auto coordinate = coordinates(i);
      const int row = get<0>(coordinate), col = get<1>(coordinate);
      // All collectives have completed. Mask logical coordinates BEFORE any
      // map read/address computation, including the last partial M tile.
      if (row < 0 || row >= t || col < 0 || col >= 2560) continue;
      if constexpr (Capture)
        p.rounded_diagnostic[size_t(row) * 2560 + size_t(col)] =
            sycl::bit_cast<uint16_t>(rounded(i));
      const size_t routed = offset + size_t(row);
      const int32_t slot = p.original[routed], token = p.source[routed];
      uint32_t error = 0;
      if (slot < 0 || size_t(slot) >= routes) error = 1;
      else if (token < 0 || token >= token_rows || token != slot / 10) error = 2;
      if (error) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
            sycl::memory_scope::device, sycl::access::address_space::global_space>(
                *p.status).fetch_or(error);
        continue;
      }
      const size_t out = size_t(token) * 2560 + size_t(col);
      p.accumulator[out] = sycl::fma(p.job_weights[routed], float(rounded(i)), p.accumulator[out]);
    }
  }
};

template <bool Capture> struct CachedDownEpilogue {
  DownScatterProblem p;
  int t, base;
  sycl::local_accessor<int32_t, 1> tokens;
  sycl::local_accessor<float, 1> weights;

  template <class Fragment, class Coordinates, class MMA>
  CUTE_DEVICE void operator()(const Fragment &fragment, const Coordinates &tile,
                              const MMA &mma, int lane) const {
    auto rounded = make_fragment_like<cutlass::bfloat16_t>(fragment.tensor());
    reorder(fragment.tensor(), rounded, fragment.tv_layout(), fragment.tv_layout());
    auto coordinates = mma.get_slice(lane).partition_C(tile);
    static_assert(decltype(size(coordinates))::value == decltype(size(rounded))::value);
    CUTE_UNROLL
    for (int i = 0; i < size(rounded); ++i) {
      const auto coordinate = coordinates(i);
      const int row = get<0>(coordinate), col = get<1>(coordinate);
      if (row < 0 || row >= t || col < 0 || col >= 2560) continue;
      // Capture is observable even when the associated route is invalid.
      if constexpr (Capture)
        p.rounded_diagnostic[size_t(row) * 2560 + size_t(col)] =
            sycl::bit_cast<uint16_t>(rounded(i));
      const int local = row - base;
      if (local < 0 || size_t(local) >= tokens.size()) continue;
      const int32_t token = tokens[local];
      if (token < 0) continue;
      const size_t out = size_t(token) * 2560 + size_t(col);
      p.accumulator[out] = sycl::fma(weights[local], float(rounded(i)), p.accumulator[out]);
    }
  }
};

template <Tile Policy, bool Capture, ScatterLayout Mapping>
sycl::event enqueue_down(sycl::queue &q, DownScatterProblem p, Geometry g,
                         int token_rows, size_t routes, size_t offset) {
  using P = ShapePolicy<Policy>;
  using MMA = TileMMA<cutlass::half_t, Policy>;
  namespace ex = sycl::ext::oneapi::experimental;
  namespace ix = sycl::ext::intel::experimental;
  const ex::properties props{ex::sub_group_size<16>, ix::grf_size<256>};
  // The host-only plan pointer is never sent to or dereferenced by the kernel.
  p.plan = nullptr;
  const sycl::nd_range<3> range(sycl::range<3>(g.groups_m, g.groups_n, P::threads),
                              sycl::range<3>(1, 1, P::threads));
  return q.submit([&](sycl::handler &h) {
    if constexpr (Mapping == ScatterLayout::matrix_row_cache) {
      static_assert(P::rows <= P::threads);
      static_assert(int(get<0>(MMA{}.tile_mnk())) == P::rows);
      sycl::local_accessor<int32_t, 1> tokens(sycl::range<1>(P::rows), h);
      sycl::local_accessor<float, 1> weights(sycl::range<1>(P::rows), h);
      h.parallel_for<DownScatterKernel<Policy, Capture, Mapping>>(range, props, [=](sycl::nd_item<3> item) {
        const int base = int(item.get_group(0)) * P::rows;
        const int local = int(item.get_local_linear_id());
        // One producer initializes every local slot, including masked M tails.
        // No lane returns before the unconditional publication barrier.
        if (local < P::rows) {
          const int row = base + local;
          int32_t token = -1;
          float weight = 0.f;
          if (row < g.t) {
            const size_t routed = offset + size_t(row);
            const int32_t slot = p.original[routed], source = p.source[routed];
            uint32_t error = 0;
            if (slot < 0 || size_t(slot) >= routes) error = 1;
            else if (source < 0 || source >= token_rows || source != slot / 10) error = 2;
            if (error) {
              sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                  sycl::memory_scope::device, sycl::access::address_space::global_space>(
                      *p.status).fetch_or(error);
            } else {
              token = source;
              weight = p.job_weights[routed];
            }
          }
          tokens[local] = token;
          weights[local] = weight;
        }
        item.barrier(sycl::access::fence_space::local_space);
        auto a = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::half_t *>(p.x)),
            make_layout(make_shape(g.t, g.k), make_stride(g.k, _1{})));
        auto b = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::half_t *>(p.w)),
            make_layout(make_shape(g.n, g.k), make_stride(g.k, _1{})));
        auto c = make_identity_tensor(make_shape(g.t, g.n));
        const auto coord = make_coord(int(item.get_group(0)), int(item.get_group(1)), _, 0);
        derived::xe_gemm_epilogue<void, void, void>(a, b, c, coord, MMA{},
            CachedDownEpilogue<Capture>{p, g.t, base, tokens, weights});
      });
    } else {
      h.parallel_for<DownScatterKernel<Policy, Capture, Mapping>>(range, props, [=](sycl::nd_item<3> item) {
        auto a = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::half_t *>(p.x)),
            make_layout(make_shape(g.t, g.k), make_stride(g.k, _1{})));
        auto b = make_tensor(make_gmem_ptr(reinterpret_cast<const cutlass::half_t *>(p.w)),
            make_layout(make_shape(g.n, g.k), make_stride(g.k, _1{})));
        // A shape/coordinate view, without a fabricated uint16_t output pointer.
        auto c = make_identity_tensor(make_shape(g.t, g.n));
        const auto coord = make_coord(int(item.get_group(0)), int(item.get_group(1)), _, 0);
        derived::xe_gemm_epilogue<void, void, void>(a, b, c, coord, MMA{},
            DownEpilogue<Capture, Mapping>{p, g.t, token_rows, routes, offset});
      });
    }
  });
}

template <Tile Policy, ScatterLayout Mapping>
DownOwnership inspect_ownership(int rows) {
  using P = ShapePolicy<Policy>;
  using MMA = TileMMA<cutlass::half_t, Policy>;
  const MMA mma{};
  const auto shape = mma.tile_mnk();
  auto identity = make_identity_tensor(make_shape(rows, 2560));
  std::vector<uint8_t> visits(size_t(rows) * 2560, 0);
  DownOwnership result{visits.size(), 0, 0, 0};
  for (int m = 0; m < (rows + int(get<0>(shape)) - 1) / int(get<0>(shape)); ++m)
    for (int n = 0; n < (2560 + int(get<1>(shape)) - 1) / int(get<1>(shape)); ++n) {
      auto tile = local_tile(identity, shape, make_coord(m, n, 0), Step<_1, _1, X>{});
      if constexpr (Mapping == ScatterLayout::matrix_row_cache) {
        std::vector<uint8_t> writes(P::rows, 0);
        for (int lane = 0; lane < P::threads; ++lane)
          if (lane < P::rows) ++writes[size_t(lane)];
        result.metadata_slots += writes.size();
        for (auto count : writes) {
          result.metadata_missing += count == 0;
          result.metadata_multiple += count > 1;
        }
      }
      for (int lane = 0; lane < P::threads; ++lane) {
        const auto coordinates = [&] {
          if constexpr (Mapping == ScatterLayout::output_copy) {
            auto copy = make_block_2d_copy_D<cutlass::bfloat16_t>(mma, make_stride(2560, _1{}));
            return copy.get_slice(lane).partition_S(tile);
          } else return mma.get_slice(lane).partition_C(tile);
        }();
        for (int i = 0; i < size(coordinates); ++i) {
          const auto coordinate = coordinates(i);
          const int row = get<0>(coordinate), col = get<1>(coordinate);
          if constexpr (Mapping == ScatterLayout::matrix_row_cache)
            result.metadata_outside_tile += row < m * P::rows || row >= (m + 1) * P::rows;
          if (row < 0 || row >= rows || col < 0 || col >= 2560) { ++result.masked; continue; }
          auto &count = visits[size_t(row) * 2560 + size_t(col)];
          if (count < 2) ++count;
        }
      }
    }
  for (auto count : visits) {
    result.missing += count == 0;
    result.multiple += count > 1;
  }
  return result;
}

template <Tile Policy, GuLayout Mapping>
GuPairOwnership inspect_gu_pair(int rows) {
  using P = ShapePolicy<Policy>;
  using MMA = TileMMA<cutlass::half_t, Policy>;
  const MMA mma{};
  const auto shape = mma.tile_mnk();
  auto identity = make_identity_tensor(make_shape(rows, 1280));
  std::vector<uint8_t> gu(size_t(rows) * 1280, 0), hidden(size_t(rows) * 640, 0),
      packed(size_t(rows) * 640, 0);
  GuPairOwnership result;
  result.gu_values = gu.size(); result.hidden_values = hidden.size();
  for (int m = 0; m < (rows + int(get<0>(shape)) - 1) / int(get<0>(shape)); ++m)
    for (int n = 0; n < (1280 + int(get<1>(shape)) - 1) / int(get<1>(shape)); ++n) {
      auto tile = local_tile(identity, shape, make_coord(m, n, 0), Step<_1, _1, X>{});
      const auto partition = [&](int lane) {
        if constexpr (Mapping == GuLayout::output_copy) {
          auto copy = make_block_2d_copy_D<cutlass::bfloat16_t>(mma, make_stride(1280, _1{}));
          return copy.get_slice(lane).partition_S(tile);
        } else return mma.get_slice(lane).partition_C(tile);
      };
      for (int lane = 0; lane < P::threads; ++lane) {
        const auto own = partition(lane), partner = partition(lane ^ 1);
        for (int i = 0; i < size(own); ++i) {
          const auto c = own(i), p = partner(i);
          const int row = get<0>(c), col = get<1>(c);
          if (row < 0 || row >= rows || col < 0 || col >= 1280) { ++result.masked; continue; }
          auto &visited = gu[size_t(row) * 1280 + size_t(col)];
          if (visited < 2) ++visited;
          if (col % 2) continue;
          auto &written = hidden[size_t(row) * 640 + size_t(col / 2)];
          if (written < 2) ++written;
          ++result.checked_pairs;
          result.partner_mismatch += get<0>(p) != row || get<1>(p) != col + 1;
        }
        if (size(own) % 2) ++result.packed_orientation_mismatch;
        for (int i = 0; i + 1 < size(own); i += 2) {
          const int parity = int(get<1>(own(i))) & 1;
          result.packed_orientation_mismatch += (int(get<1>(own(i + 1))) & 1) != parity;
          const auto c = own(i + parity);
          const int row = get<0>(c), col = get<1>(c);
          if (row < 0 || row >= rows || col < 0 || col >= 1280) continue;
          auto &written = packed[size_t(row) * 640 + size_t(col / 2)];
          if (written < 2) ++written;
        }
      }
    }
  for (auto count : gu) { result.gu_missing += count == 0; result.gu_multiple += count > 1; }
  for (auto count : hidden) { result.hidden_missing += count == 0; result.hidden_multiple += count > 1; }
  for (auto count : packed) { result.packed_missing += count == 0; result.packed_multiple += count > 1; }
  return result;
}

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
    : runtime_(runtime), queue_(q), has_fp16_(check_device(runtime, q)),
      local_mem_bytes_(q.get_device().get_info<sycl::info::device::local_mem_size>()) {}

DownOwnership inspect_down_ownership(int64_t rows, Tile tile, ScatterLayout mapping) {
  if (rows < 1 || rows > 8192)
    throw std::invalid_argument("Down ownership: rows outside [1,8192]");
  const auto inspect = [&]<Tile Policy>() {
    if (mapping == ScatterLayout::matrix)
      return inspect_ownership<Policy, ScatterLayout::matrix>(int(rows));
    if (mapping == ScatterLayout::output_copy)
      return inspect_ownership<Policy, ScatterLayout::output_copy>(int(rows));
    if (mapping == ScatterLayout::matrix_row_cache)
      return inspect_ownership<Policy, ScatterLayout::matrix_row_cache>(int(rows));
    throw std::invalid_argument("Down ownership: invalid scatter layout");
  };
  if (tile == Tile::compact_16x64) return inspect.template operator()<Tile::compact_16x64>();
  if (tile == Tile::wide_128x128) return inspect.template operator()<Tile::wide_128x128>();
  throw std::invalid_argument("Down ownership: invalid tile");
}

GuPairOwnership inspect_gu_pair_ownership(int64_t rows, Tile tile, GuLayout mapping) {
  if (rows < 1 || rows > 8192)
    throw std::invalid_argument("GU ownership: rows outside [1,8192]");
  const auto inspect = [&]<Tile Policy>() {
    if (mapping == GuLayout::matrix)
      return inspect_gu_pair<Policy, GuLayout::matrix>(int(rows));
    if (mapping == GuLayout::output_copy)
      return inspect_gu_pair<Policy, GuLayout::output_copy>(int(rows));
    throw std::invalid_argument("GU ownership: invalid layout");
  };
  if (tile == Tile::compact_16x64) return inspect.template operator()<Tile::compact_16x64>();
  if (tile == Tile::wide_128x128) return inspect.template operator()<Tile::wide_128x128>();
  throw std::invalid_argument("GU ownership: invalid tile");
}

sycl::event Gemm::down_scatter(const DownScatterProblem &p, Tile tile, ScatterLayout mapping) {
  if (mapping != ScatterLayout::matrix && mapping != ScatterLayout::output_copy &&
      mapping != ScatterLayout::matrix_row_cache)
    throw std::invalid_argument("Down scatter: invalid layout");
  using Routes = strata::prefill::detail::MoeWindowRoutes;
  if (!has_fp16_ || !p.plan || p.plan->rows() < 1 || p.plan->rows() > Routes::max_accum_rows)
    throw std::invalid_argument("Down scatter: native FP16 and published accumulator plan required");
  const size_t routes = size_t(p.plan->rows()) * Routes::top_k;
  if (p.plan->source_rows().size() != routes || p.plan->original_slots().size() != routes)
    throw std::invalid_argument("Down scatter: inverse plan required");
  const auto &job = p.plan->jobs().at(p.job_index);
  if (job.rows < 1 || job.rows > 8192 || job.global_row < 0 ||
      size_t(job.rows) > routes ||
      size_t(job.global_row) > routes - size_t(job.rows))
    throw std::invalid_argument("Down scatter: invalid job interval");
  Problem shape;
  shape.t = job.rows; shape.n = 2560; shape.k = 640;
  const auto g = geometry(shape, tile);
  if (mapping == ScatterLayout::matrix_row_cache) {
    const size_t local_bytes = size_t(tile == Tile::compact_16x64 ? 16 : 128) *
        (sizeof(int32_t) + sizeof(float));
    if (local_mem_bytes_ < local_bytes)
      throw std::invalid_argument("Down scatter: insufficient local memory for row cache");
  }
  const size_t ab = mul(mul(size_t(p.plan->rows()), 2560), 4), mb = mul(routes, 4);
  const std::initializer_list<sycl_backend::Span> inputs{
      {p.x, g.xb}, {p.w, g.wb}, {p.source, mb}, {p.original, mb}, {p.job_weights, mb}};
  validate_spans({{p.accumulator, ab}, {p.status, 4}}, inputs);
  if (p.rounded_diagnostic)
    validate_spans({{p.accumulator, ab}, {p.status, 4}, {p.rounded_diagnostic, g.yb}}, inputs);
  for (auto span : inputs) validate_device_span(span.pointer, span.bytes, queue_);
  validate_device_span(p.accumulator, ab, queue_);
  validate_device_span(p.status, 4, queue_);
  if (p.rounded_diagnostic) validate_device_span(p.rounded_diagnostic, g.yb, queue_);
  // Only X/W participate in block2D IO; retain the adapter's 64B base contract.
  if (reinterpret_cast<uintptr_t>(p.x) % 64 || reinterpret_cast<uintptr_t>(p.w) % 64)
    throw std::invalid_argument("Down scatter: block2D inputs require 64B alignment");
  if ((reinterpret_cast<uintptr_t>(p.source) | reinterpret_cast<uintptr_t>(p.original) |
       reinterpret_cast<uintptr_t>(p.job_weights) | reinterpret_cast<uintptr_t>(p.accumulator) |
       reinterpret_cast<uintptr_t>(p.status)) % 4 ||
      (p.rounded_diagnostic && reinterpret_cast<uintptr_t>(p.rounded_diagnostic) % 2))
    throw std::invalid_argument("Down scatter: typed map/output alignment required");
  runtime_.check();
  const auto submit = [&]<Tile Policy, ScatterLayout Mapping>() {
    if (p.rounded_diagnostic)
      return enqueue_down<Policy, true, Mapping>(queue_, p, g, p.plan->rows(), routes, size_t(job.global_row));
    return enqueue_down<Policy, false, Mapping>(queue_, p, g, p.plan->rows(), routes, size_t(job.global_row));
  };
  const auto select = [&]<ScatterLayout Mapping>() {
    if (tile == Tile::compact_16x64)
      return submit.template operator()<Tile::compact_16x64, Mapping>();
    return submit.template operator()<Tile::wide_128x128, Mapping>();
  };
  if (mapping == ScatterLayout::output_copy)
    return select.template operator()<ScatterLayout::output_copy>();
  if (mapping == ScatterLayout::matrix_row_cache)
    return select.template operator()<ScatterLayout::matrix_row_cache>();
  return select.template operator()<ScatterLayout::matrix>();
}

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

sycl::event Gemm::gu_swiglu(const GuSwiGLUProblem &p, Tile tile) {
  if (!has_fp16_ || p.rows < 1 || p.rows > 8192)
    throw std::invalid_argument("GU SwiGLU: native FP16 and rows in [1,8192] required");
  Problem shape;
  shape.t = p.rows; shape.n = 1280; shape.k = 2560;
  const auto g = geometry(shape, tile);
  const size_t hb = mul(mul(size_t(p.rows), 640), 2);
  validate_spans({{p.y, hb}}, {{p.x, g.xb}, {p.w, g.wb}}, 2);
  if (p.rounded_diagnostic)
    validate_spans({{p.y, hb}, {p.rounded_diagnostic, g.yb}}, {{p.x, g.xb}, {p.w, g.wb}}, 2);
  validate_device_span(p.x, g.xb, queue_); validate_device_span(p.w, g.wb, queue_);
  validate_device_span(p.y, hb, queue_);
  if (p.rounded_diagnostic) validate_device_span(p.rounded_diagnostic, g.yb, queue_);
  if (reinterpret_cast<uintptr_t>(p.x) % 64 || reinterpret_cast<uintptr_t>(p.w) % 64)
    throw std::invalid_argument("GU SwiGLU: block2D inputs require 64B alignment");
  runtime_.check();
  const auto submit = [&]<Tile Policy>() {
    if (p.rounded_diagnostic) return enqueue_gu<Policy, true>(queue_, p, g);
    return enqueue_gu<Policy, false>(queue_, p, g);
  };
  if (tile == Tile::compact_16x64) return submit.template operator()<Tile::compact_16x64>();
  return submit.template operator()<Tile::wide_128x128>();
}
} // namespace strata::experimental::vllm_xe2_native16
