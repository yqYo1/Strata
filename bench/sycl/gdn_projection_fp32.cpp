// Main-authored; repository LICENSE (MIT). CPU-only TLA ownership proof.
// This enumerates the actual public D-copy layouts, not a handwritten store map.
// It creates no SYCL runtime, queue, allocation or device submission.
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cutlass/numeric_types.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

using namespace cute;
namespace {
void need(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
using SG = Layout<Shape<Int<4>, Int<2>, Int<1>>, Stride<Int<2>, Int<1>, Int<0>>>;
using MMA = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, cutlass::half_t>>,
    Layout<Shape<Int<128>, Int<128>, Int<32>>>, SG>::TiledMMA;
static_assert(decltype(size(MMA{}))::value == 128);
static_assert(std::is_same_v<typename MMA::ValTypeC, float> &&
              std::is_same_v<typename MMA::ValTypeD, float>);

void ownership() {
  alignas(64) float dummy[16]{};
  auto tensor = [&](int t, int n, int ld) {
    // The only address inspected is the first element. No matrix is accessed.
    return make_tensor(make_gmem_ptr(dummy),
        make_layout(make_shape(t, n), make_stride(ld, _1{})));
  };
  MMA mma;
  auto d = tensor(128, 10240, 10256);
  auto copy = get_block_2d_copy_D<void>(mma, d);
  using Op = typename decltype(copy)::CopyOp;
  static_assert(Op::CopyBits == 32 && sizeof(typename decltype(copy)::ValType) == 4);
  auto identity = make_identity_tensor(shape(d));
  auto tile = local_tile(identity, mma.tile_mnk(), make_coord(0, 0, 0), Step<_1, _1, X>{});
  std::array<int, 128 * 128> source{}, destination{}, accumulated{};
  std::array<int, 129> clipped_cells{};
  auto index = [&](const auto &c) {
    const int row = get<0>(c), col = get<1>(c);
    need(row >= 0 && row < 128 && col >= 0 && col < 128, "copy coordinate escapes its tile");
    return row * 128 + col;
  };
  for (int sg = 0; sg < 8; ++sg) {
    std::array<int, 128 * 128> subgroup_source{}, subgroup_destination{}, subgroup_accumulated{};
    for (int lane = 0; lane < 16; ++lane) {
      const int thread = sg * 16 + lane;
      auto s = copy.get_slice(thread).partition_S(tile);
      auto a = mma.get_slice(thread).partition_C(tile);
      for (int i = 0; i < size(s); ++i) { ++source[index(s(i))]; ++subgroup_source[index(s(i))]; }
      for (int i = 0; i < size(a); ++i) { ++accumulated[index(a(i))]; ++subgroup_accumulated[index(a(i))]; }
    }
    // The destination layout is replicated across the16 lanes. One block2D
    // message is collective for the subgroup: enumerate its descriptor once.
    auto target = copy.get_slice(sg * 16).partition_D(tile);
    for (int i = 0; i < size(target); ++i) {
      ++destination[index(target(i))]; ++subgroup_destination[index(target(i))];
    }
    need(subgroup_source == subgroup_destination && subgroup_source == subgroup_accumulated,
         "subgroup accumulator/payload/destination cells disagree");
  }
  for (int i = 0; i < 128 * 128; ++i)
    need(source[i] == 1 && destination[i] == 1 && accumulated[i] == 1,
         "missing or multiple tile cell owners");
  for (int tail = 1; tail <= 128; ++tail) {
    for (int row = 0; row < tail; ++row)
      for (int col = 0; col < 128; ++col) clipped_cells[tail] += destination[row * 128 + col];
    need(clipped_cells[tail] == tail * 128, "tail descriptor ownership");
  }
  uint64_t shapes = 0;
  for (int t = 1; t <= 8192; ++t) for (int n : {6144, 10240}) {
    // Whole-tile translations are disjoint Cartesian intervals. The final
    // M tile uses the descriptor height; N has no tail for either producer.
    const int full = t / 128, tail = t % 128;
    need(uint64_t(full * 16384 + clipped_cells[tail]) * (n / 128) == uint64_t(t) * n,
         "translated tiles do not cover every logical output exactly once");
    const int maximum_ld = int(std::min<int64_t>(0xFFFFFF / 4,
        std::numeric_limits<int>::max() / (int64_t(t) * 4))) & ~3;
    for (int ld : {n, n + 4, n + 16, maximum_ld}) {
      need(ld >= n && ld % 4 == 0 && int64_t(t) * ld * 4 <= std::numeric_limits<int>::max(),
           "invalid fixture pitch");
      auto typed = get_block_2d_copy_D<void>(mma, tensor(t, n, ld));
      const auto &descriptor = cute::detail::as_xe2d_base(typed);
      need(descriptor.width == uint32_t(n * 4) && descriptor.height == uint32_t(t) &&
           descriptor.pitch == uint32_t(ld * 4) && descriptor.base_ptr == uintptr_t(dummy),
           "FP32 descriptor width/height/pitch/base mismatch");
      const uint64_t span = (uint64_t(t - 1) * ld + n) * 4;
      const uint64_t last = (uint64_t(t - 1) * ld + (n - 1)) * 4 + 3;
      need(last < span && (t == 1 || uint64_t(ld) * 4 >= uint64_t(n) * 4),
           "valid-cell byte interval escapes or overlaps another row");
      ++shapes;
    }
  }
  std::cout << "{\"ownership\":\"passed\",\"tile_cells\":16384,\"subgroups\":8,"
            << "\"subgroup_size\":16,\"copy_bits\":" << Op::CopyBits
            << ",\"atom_height\":" << Op::AtomHeight << ",\"atom_width\":" << Op::AtomWidth
            << ",\"checked_shapes\":" << shapes
            << ",\"T_min\":1,\"T_max\":8192,\"GPU_submissions\":0,"
            << "\"hardware_tail_masking_executed\":false,\"physical262144_qualified\":false}\n";
}
}
int main() {
  try { ownership(); return 0; }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
