// Cross-check the compact representation against explicit ordered cell IDs.
#include "group_ids.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace groups = strata::prefill::qsa_groups;
constexpr uint32_t sentinel = 0xfeedfaceu;

int main() {
    constexpr uint32_t cap = 2051;
    uint64_t cases = 0, cells = 0;
    uint32_t min_wide = cap, max_wide = 0;
    auto check = [&](uint32_t blocks, uint32_t cut_index, uint32_t cut_take, uint32_t tail) {
        std::array<uint32_t, groups::row_words(cap) + 2> storage;
        storage.fill(sentinel);
        uint32_t* row = storage.data() + 1;
        uint32_t meta = 0, used = 0;
        std::vector<uint32_t> ids;
        for (uint32_t i = 0; i < blocks + (tail > 0); ++i) {
            const bool physical_tail = i == blocks;
            // Force gaps and include cells in the final physical context page.
            const uint32_t block = physical_tail ? 65535 : 3 + 17 * i;
            const uint32_t take = physical_tail ? tail : (i == cut_index ? cut_take : 4);
            meta |= groups::emit(row, static_cast<uint32_t>(ids.size()), block, take, physical_tail);
            ++used;
            for (uint32_t j = 0; j < take; ++j) ids.push_back(block*4 + j);
        }
        row[0] = meta;
        if (ids.size() > cap || used + 1 > groups::row_words(static_cast<uint32_t>(ids.size())))
            throw std::runtime_error("descriptor bound");
        for (uint32_t rank = 0; rank < ids.size(); ++rank)
            if (groups::cell(row, rank) != ids[rank]) throw std::runtime_error("changed selected ID");
        if (storage.front() != sentinel ||
            !std::all_of(row+1+used, storage.data()+storage.size(), [](uint32_t v){return v == sentinel;}))
            throw std::runtime_error("descriptor padding or guard changed");
        if (ids.size() == cap) {
            min_wide = std::min(min_wide, used);
            max_wide = std::max(max_wide, used);
        }
        ++cases;
        cells += ids.size();
    };
    for (uint32_t blocks = 0; blocks <= 16; ++blocks) {
        for (uint32_t tail = 0; tail < 4; ++tail) {
            check(blocks, blocks, 4, tail);
            for (uint32_t i = 0; i < blocks; ++i)
                for (uint32_t take = 1; take <= 3; ++take) check(blocks, i, take, tail);
        }
    }
    check(512, 512, 4, 3);
    for (uint32_t i = 0; i < 513; ++i) {
        check(513, i, 3, 0);
        check(513, i, 1, 2);
        check(513, i, 2, 1);
    }
    if (min_wide != 513 || max_wide != 514 || groups::row_words(cap) != 515)
        throw std::runtime_error("full-width representation changed");
    std::cout << "cases=" << cases << " cell_ids=" << cells << " wide_groups=" << min_wide << ".."
              << max_wide << " row_words=" << groups::row_words(cap) << '\n';
}
