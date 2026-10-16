// Authored for Strata; repository LICENSE (MIT) applies.
// Compile-only contract probe of the exact production template/types; no GPU run.
#include "../../../src/kernels/sycl/qsa_bf16_prefix.cpp"
#include <cstddef>
namespace {
using BFloat = sycl::ext::oneapi::bfloat16;
static_assert(sizeof(BFloat) == 2);
template<int Count> constexpr bool matrix_tile_contract() {
  using Tile = strata::kernels::MatrixTile<Count>;
  static_assert(std::is_standard_layout_v<Tile>);
  static_assert(std::is_trivially_copyable_v<Tile>);
  static_assert(alignof(Tile) >= 4);
  static_assert(offsetof(Tile, values) == 0);
  static_assert(offsetof(Tile, values) % 4 == 0);
  static_assert(sizeof(Tile) == Count * sizeof(BFloat));
  return true;
}
static_assert(matrix_tile_contract<16 * 256>());
static_assert(matrix_tile_contract<32 * 256>());
static_assert(matrix_tile_contract<16 * 32>());
} // namespace
