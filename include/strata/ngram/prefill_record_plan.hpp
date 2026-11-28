#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace strata::ngram {

inline constexpr uint32_t prefill_ple_page_bytes = 4096;
inline constexpr uint32_t prefill_ple_record_bytes = 128 * 1024;
inline constexpr uint32_t prefill_ple_record_max_inflight = 256;

inline constexpr uint64_t prefill_ple_record_plan_budget(uint32_t rows, uint32_t chunk_rows) {
    return (rows > 2 * uint64_t{chunk_rows} ? 160ull : 32ull) << 20;
}

// Reserve the packed rows, row IDs and miss mask, bounded planner storage,
// direct-read slab and its indices. Only one window owns this storage at a time.
inline constexpr std::optional<uint64_t> prefill_ple_record_window_host_bytes(
        uint32_t rows, uint32_t chunk_rows, uint32_t row_bytes) {
    if (rows == 0 || chunk_rows == 0 || row_bytes == 0 || row_bytes > prefill_ple_page_bytes)
        return std::nullopt;
    const uint64_t chunks = 1 + (rows - 1) / chunk_rows;
    return uint64_t{rows} * (row_bytes + sizeof(uint32_t) + sizeof(uint8_t)) +
           prefill_ple_record_plan_budget(rows, chunk_rows) +
           uint64_t{prefill_ple_record_max_inflight} *
               (prefill_ple_record_bytes + sizeof(size_t) + sizeof(uint32_t)) +
           chunks * (sizeof(double) + sizeof(uint32_t));
}

struct PrefillPleCopy {
    uint32_t destination;
    uint32_t source;
    uint32_t bytes;
};

struct PrefillPleRead {
    uint64_t offset;
    uint32_t length;
    uint32_t required_bytes;
    uint32_t copy_begin;
    uint32_t copy_count;
    uint32_t first_chunk;
};

struct PrefillPleRecordPlan {
    std::vector<PrefillPleRead> reads;
    std::vector<PrefillPleCopy> copies;
    std::vector<uint32_t> chunk_fragments;
};

namespace detail {

struct PrefillPleRecordRange {
    uint32_t begin = prefill_ple_record_bytes;
    uint32_t end = 0;
    uint32_t count = 0;
    uint32_t cursor = 0;
    uint32_t first_chunk = 0;
};

template<class Visit>
void prefill_ple_fragments(std::span<const uint32_t> rows, std::span<const uint8_t> needs_read,
                          uint64_t table_offset, uint64_t table_rows, uint32_t row_bytes,
                          uint32_t chunk_rows, Visit visit) {
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] >= table_rows || (!needs_read.empty() && !needs_read[i])) continue;
        const uint64_t offset = table_offset + uint64_t{rows[i]} * row_bytes;
        const uint64_t record = offset / prefill_ple_record_bytes;
        const uint32_t source = offset % prefill_ple_record_bytes;
        const uint32_t first = std::min(row_bytes, prefill_ple_record_bytes - source);
        const auto destination = static_cast<uint32_t>(i * row_bytes);
        const auto chunk = static_cast<uint32_t>(i / chunk_rows);
        visit(record, chunk, PrefillPleCopy{destination, source, first});
        if (first < row_bytes) visit(record + 1, chunk, PrefillPleCopy{destination + first, 0, row_bytes - first});
    }
}

}  // namespace detail

// The caller copies valid cache hits and zeroes out-of-range rows.
// Each fragment writes a disjoint portion of the request's packed output.
// Reads are ordered by first consumer chunk and then by file offset.
inline std::optional<PrefillPleRecordPlan> prefill_ple_record_plan(
        std::span<const uint32_t> rows, std::span<const uint8_t> needs_read,
        uint64_t table_offset, uint64_t table_rows, uint64_t file_bytes,
        uint32_t row_bytes, uint32_t chunk_rows, uint64_t planner_budget_bytes) {
    constexpr uint32_t max_u32 = std::numeric_limits<uint32_t>::max();
    if (row_bytes == 0 || row_bytes > prefill_ple_page_bytes || chunk_rows == 0 ||
        rows.empty() || rows.size() > max_u32 / 2 || rows.size() > max_u32 / row_bytes ||
        (!needs_read.empty() && needs_read.size() != rows.size()) ||
        table_offset > file_bytes || file_bytes > uint64_t{std::numeric_limits<int64_t>::max()} ||
        table_rows == 0 || table_rows > (file_bytes - table_offset) / row_bytes) return std::nullopt;

    const uint64_t first_record = table_offset / prefill_ple_record_bytes;
    const uint64_t table_end = table_offset + table_rows * row_bytes;
    const uint64_t record_count = (table_end - 1) / prefill_ple_record_bytes - first_record + 1;
    const uint64_t chunks = 1 + (rows.size() - 1) / chunk_rows;
    const uint64_t max_copies = 2 * uint64_t{rows.size()};
    const uint64_t budget = record_count * sizeof(detail::PrefillPleRecordRange) +
        std::min(record_count, max_copies) * sizeof(PrefillPleRead) +
        max_copies * sizeof(PrefillPleCopy) + 2 * chunks * sizeof(uint32_t);
    if (record_count > max_u32 || budget > planner_budget_bytes) return std::nullopt;

    using detail::prefill_ple_fragments;
    std::vector<detail::PrefillPleRecordRange> directory(record_count);
    std::vector<uint32_t> next_read(chunks);
    PrefillPleRecordPlan plan;
    plan.chunk_fragments.resize(chunks);
    prefill_ple_fragments(rows, needs_read, table_offset, table_rows, row_bytes, chunk_rows,
                         [&](uint64_t record, uint32_t chunk, PrefillPleCopy copy) {
        auto& range = directory[record - first_record];
        if (range.count == 0) {
            range.first_chunk = chunk;
            ++next_read[chunk];
        }
        ++range.count;
        range.begin = std::min(range.begin, copy.source);
        range.end = std::max(range.end, copy.source + copy.bytes);
        ++plan.chunk_fragments[chunk];
    });

    uint32_t total_reads = 0;
    for (auto& next : next_read) {
        const uint32_t count = next;
        next = total_reads;
        total_reads += count;
    }
    plan.reads.resize(total_reads);
    uint32_t total_copies = 0;
    for (size_t i = 0; i < directory.size(); ++i) {
        auto& range = directory[i];
        if (range.count == 0) continue;
        range.begin = range.begin / prefill_ple_page_bytes * prefill_ple_page_bytes;
        const uint32_t limit = (range.end + prefill_ple_page_bytes - 1) /
                               prefill_ple_page_bytes * prefill_ple_page_bytes;
        range.cursor = total_copies;
        plan.reads[next_read[range.first_chunk]++] = {
            (first_record + i) * prefill_ple_record_bytes + range.begin,
            limit - range.begin, range.end - range.begin, total_copies, range.count, range.first_chunk};
        total_copies += range.count;
    }
    plan.copies.resize(total_copies);
    prefill_ple_fragments(rows, needs_read, table_offset, table_rows, row_bytes, chunk_rows,
                         [&](uint64_t record, uint32_t, PrefillPleCopy copy) {
        auto& range = directory[record - first_record];
        copy.source -= range.begin;
        plan.copies[range.cursor++] = copy;
    });
    return plan;
}

}  // namespace strata::ngram
