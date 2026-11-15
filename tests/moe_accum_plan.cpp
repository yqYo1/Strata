// Original Strata test; repository LICENSE applies. No GPU/runtime access.
#include "strata/prefill/moe_accum_plan.hpp"
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

using Plan = strata::prefill::detail::MoeAccumPlan;
constexpr uint64_t Legacy = 8192ull * 10240 * 4, Embedding = 8192ull * 2560 * 4;
constexpr uint64_t Region = 1180698624;
void need(bool valid) { if (!valid) throw std::runtime_error("MoE accumulator plan check failed"); }
size_t negative_checks = 0;

template<class Error = std::invalid_argument, class Function> void rejects(Function function) {
    bool rejected = false;
    try { function(); } catch (const Error&) { rejected = true; }
    need(rejected);
    ++negative_checks;
}

void inspect(int64_t rows) {
    const auto plan = Plan::make(rows, Legacy, Embedding, Region);
    need(plan.rows == rows && plan.primary_rows + plan.tail_rows == rows);
    need(plan.primary_bytes <= Legacy && plan.tail_bytes <= Embedding);
    need(plan.region_bytes <= Region);
    uint64_t previous_end = 0;
    for (size_t i = 0; i < size_t(Plan::Plane::count); ++i) {
        const auto& span = plan.span(Plan::Plane(i));
        need(span.offset % 256 == 0 && span.offset == previous_end);
        need(span.reserved >= span.bytes + 256 && span.offset <= Region);
        need(span.reserved <= Region - span.offset);
        previous_end = span.offset + span.reserved;
    }
    need(previous_end == plan.region_bytes);
    // Independent consumer extents: scalar gates are produced once per window
    // row from mixed_bf16, before its 8K preparation buffer is reused.
    need(plan.span(Plan::Plane::shared_gate).bytes >= uint64_t(rows) * sizeof(float));
    need(plan.span(Plan::Plane::injection).bytes >= uint64_t(rows) * 4 * sizeof(float));
    need(plan.span(Plan::Plane::status).bytes >= sizeof(uint32_t));
    uint64_t primary_end = 0, tail_end = 0;
    for (int64_t first = 0; first < rows; first += Plan::chunk) {
        const auto count = std::min(Plan::chunk, rows - first);
        const auto piece = plan.mixed_piece(first, count);
        auto& cursor = piece.tail ? tail_end : primary_end;
        need(piece.offset == cursor);
        cursor += piece.bytes;
    }
    need(primary_end == plan.primary_bytes && tail_end == plan.tail_bytes);
    rejects([&] { (void)Plan::make(rows, Legacy, Embedding, plan.region_bytes - 1); });
    rejects([&] { (void)plan.mixed_piece(rows, 1); });
    rejects([&] { (void)plan.mixed_piece(1, 8192); });
    std::printf("case rows=%lld primary=%lld tail=%lld region=%llu PASS\n",
        static_cast<long long>(rows), static_cast<long long>(plan.primary_rows),
        static_cast<long long>(plan.tail_rows), static_cast<unsigned long long>(plan.region_bytes));
}

int main() {
    try {
        for (int64_t rows : {8193, 16383, 16384, 32768, 49152, 65536, 65537, 78452, 81920})
            inspect(rows);
        rejects([] { (void)Plan::make(8192, Legacy, Embedding, Region); });
        rejects([] { (void)Plan::make(81921, Legacy, Embedding, Region); });
        rejects([] { (void)Plan::make(std::numeric_limits<int64_t>::max(), Legacy, Embedding, Region); });
        rejects([] { (void)Plan::make(81920, Legacy - 1, Embedding, Region); });
        rejects([] { (void)Plan::make(81920, Legacy, Embedding - 1, Region); });
        const auto largest = Plan::make(81920, Legacy, Embedding, Region);
        need(largest.region_bytes == 1059131904);
        need(largest.primary_rows == 65536 && largest.tail_rows == 16384);
        // The 256K request is NOT silently admitted as one GPU window. Its
        // scheduler must retain bounded windows and the existing host tail.
        rejects([] { (void)Plan::make(262144, Legacy, Embedding, Region); });
        rejects<std::overflow_error>([] { (void)Plan::multiply(std::numeric_limits<uint64_t>::max(), 2); });
        need(negative_checks == 34);
        std::printf("negative_checks=%zu PASS\n", negative_checks);
        size_t borrowed_cases = 0;
        for (int64_t rows : {8193, 16384, 65536, 65537, 78452, 81920}) {
            const auto p = Plan::make_scratch_borrowed(rows, Legacy, Region);
            need(p.primary_bytes <= Legacy && p.region_bytes <= Region &&
                 p.tail_bytes <= Region - p.region_bytes);
            need(p.primary_rows == std::min<int64_t>(rows, 65536));
            need(p.primary_bytes + p.tail_bytes == uint64_t(rows) * 5120);
            uint64_t consumed = 0;
            for (int64_t at = 0; at < rows; at += Plan::chunk) {
                const auto s = p.mixed_piece(at, std::min(Plan::chunk, rows - at));
                need(s.tail == (at >= p.primary_rows));
                consumed += s.bytes;
            }
            need(consumed == uint64_t(rows) * 5120);
            ++borrowed_cases;
        }
        const auto borrowed = Plan::make_scratch_borrowed(81920, Legacy, Region);
        need(borrowed.region_bytes == 1059131904 && borrowed.tail_bytes == 83886080);
        need(Region - borrowed.region_bytes == 121566720);
        const auto exact = borrowed.region_bytes + borrowed.tail_bytes;
        (void)Plan::make_scratch_borrowed(81920, Legacy, exact);
        rejects([&] { (void)Plan::make_scratch_borrowed(81920, Legacy, exact - 1); });
        rejects([] { (void)Plan::make_scratch_borrowed(81920, 0, Region); });
        need(borrowed_cases == 6 && negative_checks == 36);
        std::printf("borrowed_cases=%zu borrowed_negative_checks=2 PASS\n", borrowed_cases);
        const size_t previous_negatives = negative_checks;
        size_t split_cases = 0;
        for (int64_t rows : {81921, 90112, 98304, 106496, 114688, 117488}) {
            const auto p = Plan::make_split_scratch(rows, Legacy, Region);
            need(p.accumulator_primary_rows == 65536 &&
                 p.accumulator_tail_rows == rows - 65536);
            need(p.span(Plan::Plane::accumulator).bytes == 671088640);
            need(p.region_bytes + p.tail_bytes <= Region);
            need(p.accumulator_tail_bytes() == uint64_t(rows - 65536) * 10240);
            need(p.accumulator_tail_allocation_bytes() >= p.accumulator_tail_bytes() + 256);
            uint64_t mixed_primary = 0, mixed_tail = 0, acc_primary = 0, acc_tail = 0;
            for (int64_t at = 0; at < rows; at += Plan::chunk) {
                const int64_t count = std::min(Plan::chunk, rows - at);
                const auto mix = p.mixed_piece(at, count), acc = p.accumulator_piece(at, count);
                auto& mix_cursor = mix.tail ? mixed_tail : mixed_primary;
                auto& acc_cursor = acc.tail ? acc_tail : acc_primary;
                need(mix.offset == mix_cursor && acc.offset == acc_cursor);
                mix_cursor += mix.bytes; acc_cursor += acc.bytes;
                need(acc.tail == (at >= 65536));
            }
            need(mixed_primary == p.primary_bytes && mixed_tail == p.tail_bytes);
            need(acc_primary == 671088640 && acc_tail == p.accumulator_tail_bytes());
            // An exact fit is valid; one byte less must be refused, not aliased.
            const auto exact = p.region_bytes + p.tail_bytes;
            (void)Plan::make_split_scratch(rows, Legacy, exact);
            rejects([&] { (void)Plan::make_split_scratch(rows, Legacy, exact - 1); });
            rejects([&] { (void)Plan::make_scratch_borrowed(rows, Legacy, Region); });
            rejects([&] { (void)p.accumulator_piece(1, 8192); });
            rejects([&] { (void)p.accumulator_piece(rows, 1); });
            ++split_cases;
        }
        const auto split = Plan::make_split_scratch(117488, Legacy, Region);
        need(split.region_bytes + split.tail_bytes == 1163756800);
        need(split.accumulator_tail_allocation_bytes() == 531988736);
        rejects([] { (void)Plan::make_split_scratch(81920, Legacy, Region); });
        rejects([] { (void)Plan::make_split_scratch(117489, Legacy, Region); });
        rejects([] { (void)Plan::make_split_scratch(262144, Legacy, Region); });
        rejects([] { (void)Plan::make_split_scratch(117488, 0, Region); });
        need(split_cases == 6 && negative_checks - previous_negatives == 28);
        std::printf("split_cases=%zu split_negative_checks=%zu PASS\n",
                    split_cases, negative_checks - previous_negatives);
        std::puts("moe_accum_plan PASS GPU=UNRUN numeric=UNQUALIFIED lifetime=EXTERNAL_REQUIRED");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "moe_accum_plan failed: %s\n", error.what());
        return 1;
    }
}
