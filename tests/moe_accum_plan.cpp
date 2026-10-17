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
        need(largest.region_bytes == 1058836480);
        need(largest.primary_rows == 65536 && largest.tail_rows == 16384);
        // The 256K request is NOT silently admitted as one GPU window. Its
        // scheduler must retain bounded windows and the existing host tail.
        rejects([] { (void)Plan::make(262144, Legacy, Embedding, Region); });
        rejects<std::overflow_error>([] { (void)Plan::multiply(std::numeric_limits<uint64_t>::max(), 2); });
        need(negative_checks == 34);
        std::printf("negative_checks=%zu PASS\n", negative_checks);
        std::puts("moe_accum_plan PASS GPU=UNRUN numeric=UNQUALIFIED lifetime=EXTERNAL_REQUIRED");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "moe_accum_plan failed: %s\n", error.what());
        return 1;
    }
}
