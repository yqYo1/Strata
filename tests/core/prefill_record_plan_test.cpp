#include "strata/ngram/prefill_record_plan.hpp"

#include <array>
#include <cstdlib>
#include <iostream>

using namespace strata::ngram;

void require(bool test) {
    if (!test) std::abort();
}

int main() {
    constexpr auto record = prefill_ple_record_bytes;
    constexpr auto page = prefill_ple_page_bytes;
    constexpr uint32_t row_bytes = 6;
    constexpr uint64_t table_offset = record - 3;
    constexpr uint64_t file_bytes = table_offset + 3 * row_bytes;
    constexpr uint64_t budget = 160 << 20;
    const std::array<uint32_t, 5> rows = {0, 1, 0, 999, 2};
    const std::array<uint8_t, 5> needs_read = {1, 1, 1, 1, 0};

    auto plan = prefill_ple_record_plan(rows, needs_read, table_offset, 3, file_bytes, row_bytes, 2, budget);
    require(plan.has_value() && plan->reads.size() == 2 && plan->copies.size() == 5);
    require(plan->reads[0].offset == record - page && plan->reads[0].length == page);
    require(plan->reads[1].offset == record && plan->reads[1].required_bytes == 9);
    require(plan->chunk_fragments == std::vector<uint32_t>{3, 2, 0});

    std::vector<uint8_t> input(file_bytes);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<uint8_t>(i % 251);
    std::vector<uint8_t> packed(rows.size() * row_bytes);
    std::copy_n(input.begin() + table_offset + 2 * row_bytes, row_bytes, packed.begin() + 4 * row_bytes);
    auto pending = plan->chunk_fragments;
    // Complete the higher record first. Its file extent ends inside one page.
    for (auto it = plan->reads.rbegin(); it != plan->reads.rend(); ++it) {
        const auto& read = *it;
        const auto received = std::min<uint64_t>(read.length, input.size() - read.offset);
        require(read.required_bytes <= received);
        for (uint32_t i = 0; i < read.copy_count; ++i) {
            const auto& copy = plan->copies[read.copy_begin + i];
            require(copy.source + copy.bytes <= received && copy.destination + copy.bytes <= packed.size());
            std::copy_n(input.begin() + read.offset + copy.source, copy.bytes, packed.begin() + copy.destination);
            auto& count = pending[copy.destination / (row_bytes * 2)];
            require(count > 0);
            --count;
        }
        if (it == plan->reads.rbegin()) require(pending[0] == 1 && pending[1] == 1);
    }
    require(pending == std::vector<uint32_t>{0, 0, 0});
    for (size_t i = 0; i < rows.size(); ++i) {
        for (uint32_t j = 0; j < row_bytes; ++j) {
            const auto expected = rows[i] < 3 ? input[table_offset + rows[i] * row_bytes + j] : 0;
            require(packed[i * row_bytes + j] == expected);
        }
    }

    // Each standalone chunk needs the shared record. The two-chunk
    // window submits that record once and still publishes the first chunk.
    const auto first_only = prefill_ple_record_plan(std::span(rows).first(2), std::span(needs_read).first(2),
        table_offset, 3, file_bytes, row_bytes, 2, budget);
    const auto second_only = prefill_ple_record_plan(std::span(rows).subspan(2, 2), std::span(needs_read).subspan(2, 2),
        table_offset, 3, file_bytes, row_bytes, 2, budget);
    require(first_only && second_only && first_only->reads.size() + second_only->reads.size() == 4);
    require(plan->reads.size() == 2);

    const std::array<uint32_t, 2> order = {record / 4, 0};
    plan = prefill_ple_record_plan(order, {}, 0, (2 * record) / 4, 2 * record, 4, 1, budget);
    require(plan.has_value() && plan->reads.size() == 2);
    require(plan->reads[0].offset == record && plan->reads[0].first_chunk == 0);
    require(plan->reads[1].offset == 0 && plan->reads[1].first_chunk == 1);
    require(!prefill_ple_record_plan(rows, needs_read, table_offset, 3, file_bytes - 1, row_bytes, 2, budget));
    require(!prefill_ple_record_plan(rows, needs_read, table_offset, 3, file_bytes, row_bytes, 2, 1));

    const std::vector<uint32_t> full(262144 * 16, 0);
    plan = prefill_ple_record_plan(full, {}, 0, 1, 90, 90, 8192 * 16, budget);
    require(plan.has_value() && plan->copies.size() == full.size() && plan->reads.size() == 1);
    require(plan->chunk_fragments.size() == 32 && plan->chunk_fragments.back() == 8192 * 16);
    require(plan->copies.back().destination == (full.size() - 1) * 90);

    constexpr uint32_t chunk = 8192 * 16;
    constexpr uint64_t slots = uint64_t{prefill_ple_record_max_inflight} *
        (prefill_ple_record_bytes + sizeof(size_t) + sizeof(uint32_t));
    require(prefill_ple_record_plan_budget(2 * chunk, chunk) == (32ull << 20));
    require(prefill_ple_record_plan_budget(2 * chunk + 1, chunk) == (160ull << 20));
    require(prefill_ple_record_window_host_bytes(2 * chunk, chunk, 90) ==
           2ull * chunk * 95 + (32ull << 20) + slots + 2 * (sizeof(double) + sizeof(uint32_t)));
    require(prefill_ple_record_window_host_bytes(262144 * 16, chunk, 90) ==
           262144ull * 16 * 95 + (160ull << 20) + slots + 32 * (sizeof(double) + sizeof(uint32_t)));
    require(!prefill_ple_record_window_host_bytes(1, 0, 90));
    std::cout << "prefill PLE record planner checks passed\n";
}
