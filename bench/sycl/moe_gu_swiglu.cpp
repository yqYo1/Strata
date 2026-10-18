// Authored for Strata; repository LICENSE (MIT) applies.
#include "native16.hpp"
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace xe = strata::experimental::vllm_xe2_native16;

int main(int argc, char **argv) {
  try {
    std::cout << std::unitbuf;
    if (argc != 2 || std::string_view(argv[1]) != "--ownership")
      throw std::invalid_argument("use --ownership");
    // This mode creates no Runtime/queue/allocation and submits no GPU work.
    for (auto layout : {xe::GuLayout::matrix, xe::GuLayout::output_copy})
      for (auto tile : {xe::Tile::compact_16x64, xe::Tile::wide_128x128})
        for (int rows : {1, 3, 17, 33, 129, 2334, 5963, 8191, 8192}) {
          const auto p = xe::inspect_gu_pair_ownership(rows, tile, layout);
          if (p.gu_values != size_t(rows) * 1280 || p.hidden_values != size_t(rows) * 640 ||
              p.gu_missing || p.gu_multiple || p.hidden_missing || p.hidden_multiple ||
              p.checked_pairs != p.hidden_values)
            throw std::runtime_error("GU or hidden element ownership incomplete");
          std::cout << "case gu_ownership rows=" << rows
                    << " tile=" << (tile == xe::Tile::compact_16x64 ? "compact" : "wide")
                    << " layout=" << (layout == xe::GuLayout::matrix ? "matrix" : "output_copy")
                    << " gu_values=" << p.gu_values << " hidden_values=" << p.hidden_values
                    << " masked=" << p.masked << " checked_pairs=" << p.checked_pairs
                    << " partner_mismatch=" << p.partner_mismatch
                    << " lane_xor1_same_index_eligible=" << (p.partner_mismatch == 0)
                    << " status=PASS\n";
        }
    std::cout << "fixture_status=PASS mode=ownership GPU_submissions=0\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
