#include "strata/ngram/prefill_record_reader.hpp"
#include "strata/ngram/ple_reader.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace strata::ngram;

void require(bool condition) {
    if (!condition) std::abort();
}

void check_ple_reader_window(const std::string& path, uint64_t table_offset, std::string& error) {
    PleReader legacy, candidate;
    constexpr size_t chunk_rows = 2;
    constexpr uint64_t table_rows = 3;
    constexpr uint64_t cache_rows = 64;
    constexpr std::array<uint32_t, 4> ids = {0, 2, 1, 999};
    std::array<uint8_t, ids.size() * ROW_BYTES> expected{}, actual{};
    require(legacy.open(path, table_offset, table_rows, 2, cache_rows, error));
    require(candidate.open(path, table_offset, table_rows, 2, cache_rows, error));
    require(candidate.supports_prefill_record_window());

    const auto old_ticket = legacy.issue(ids.data(), ids.size(), expected.data());
    require(legacy.collect(old_ticket, error));
    std::array<uint8_t, ROW_BYTES> prewarmed{};
    const auto pending = candidate.issue(ids.data() + 1, 1, prewarmed.data());
    const auto unreachable = [](size_t) { std::abort(); return false; };
    const auto no_stop = [] { return false; };
    require(!candidate.read_prefill_record_window(ids, chunk_rows, actual, unreachable, no_stop, error));
    require(error == "PleReader: prefill record window is unavailable");
    require(candidate.collect(pending, error));
    require(std::equal(prewarmed.begin(), prewarmed.end(), expected.begin() + ROW_BYTES));

    size_t published = 0;
    const auto verify_chunk = [&](size_t chunk) {
        require(chunk == published++);
        const auto start = chunk * chunk_rows * ROW_BYTES;
        const auto stop = start + chunk_rows * ROW_BYTES;
        require(std::equal(actual.begin() + start, actual.begin() + stop, expected.begin() + start));
        return true;
    };
    const auto one_warm_row = candidate.snapshot();
    require(candidate.read_prefill_record_window(ids, chunk_rows, actual, verify_chunk, no_stop, error));
    const auto all_warm = candidate.snapshot();
    require(published == 2 && actual == expected && all_warm.reads - one_warm_row.reads == 2
            && all_warm.cache_hits - one_warm_row.cache_hits == 1);

    published = 0;
    actual.fill(0xff);
    require(candidate.read_prefill_record_window(ids, chunk_rows, actual, verify_chunk, no_stop, error));
    const auto after_hits = candidate.snapshot();
    require(published == 2 && actual == expected && after_hits.reads == all_warm.reads
            && after_hits.cache_hits - all_warm.cache_hits == table_rows);

    // The producer can publish more than two chunks, including a ragged tail.
    constexpr std::array<uint32_t, 7> wider_ids = {0, 2, 1, 999, 0, 1, 2};
    std::array<uint8_t, wider_ids.size() * ROW_BYTES> wide_expected{}, wide_actual{};
    const auto wide_ticket = legacy.issue(wider_ids.data(), wider_ids.size(), wide_expected.data());
    require(legacy.collect(wide_ticket, error));
    require(candidate.open(path, table_offset, table_rows, 2, cache_rows, error));
    published = 0;
    require(candidate.read_prefill_record_window(wider_ids, chunk_rows, wide_actual, [&](size_t chunk) {
        require(chunk == published++);
        const size_t start = chunk * chunk_rows * ROW_BYTES;
        const size_t end = std::min(wide_actual.size(), start + chunk_rows * ROW_BYTES);
        require(std::equal(wide_actual.begin() + start, wide_actual.begin() + end, wide_expected.begin() + start));
        return true;
    }, no_stop, error));
    require(published == 4 && wide_actual == wide_expected);

    require(candidate.open(path, table_offset, table_rows, 2, cache_rows, error, true, ROW_BYTES, cache_rows));
    require(candidate.prefill_cache_capacity() == cache_rows);
    published = 0;
    require(candidate.read_prefill_record_window(ids, chunk_rows, actual, verify_chunk, no_stop, error));
    const auto dedicated = candidate.snapshot();
    require(published == 2 && actual == expected);
    require(candidate.collect(candidate.issue(ids.data(), ids.size(), actual.data()), error) && actual == expected);
    const auto ordinary = candidate.snapshot();
    require(ordinary.reads > dedicated.reads && ordinary.cache_hits == dedicated.cache_hits);
    published = 0;
    require(candidate.read_prefill_record_window(ids, chunk_rows, actual, verify_chunk, no_stop, error));
    const auto repeated = candidate.snapshot();
    require(published == 2 && repeated.reads == ordinary.reads
            && repeated.cache_hits - ordinary.cache_hits == table_rows);

    candidate.set_injected_delay_us(1);
    require(!candidate.supports_prefill_record_window());
    require(!candidate.read_prefill_record_window(ids, chunk_rows, actual, unreachable, no_stop, error));
    candidate.set_injected_delay_us(0);

    require(candidate.open(path, table_offset, table_rows, 2, cache_rows, error));
    require(!candidate.read_prefill_record_window(ids, chunk_rows, actual, [](size_t) { return false; },
                                                   no_stop, error));
    require(error == "prefill PLE: record read window cancelled");
    const auto after_cancel = candidate.issue(ids.data(), ids.size(), actual.data());
    require(candidate.collect(after_cancel, error) && actual == expected);

    require(candidate.open(path, table_offset, table_rows, 2, cache_rows, error, false));
    require(!candidate.supports_prefill_record_window());
    require(candidate.open(path, table_offset, table_rows, 257, cache_rows, error));
    require(!candidate.supports_prefill_record_window());
}

int main(int argc, char** argv) {
    require(argc == 2);
    const std::filesystem::path path = std::filesystem::path(argv[1]) / "ple-record-reader-fixture.bin";
    require(!std::filesystem::exists(path));
    constexpr uint32_t record = prefill_ple_record_bytes;
    constexpr uint32_t row_bytes = ROW_BYTES;
    constexpr uint64_t table_offset = record - 3;
    constexpr uint64_t file_size = table_offset + 3 * row_bytes;
    std::vector<uint8_t> source(file_size);
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>(i % 251);
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(source.data()), source.size());
        require(file.good());
    }

    const std::array<uint32_t, 5> rows = {0, 0, 1, 0, 999};
    auto plan = prefill_ple_record_plan(rows, {}, table_offset, 3, file_size, row_bytes, 2, 32 << 20);
    require(plan.has_value() && plan->reads.size() == 2);
    strata::platform::DirectFile source_file, file;
    std::string error;
    require(source_file.open(path.string(), error));
    const auto held_path = path.string() + ".held";
    require(!std::filesystem::exists(held_path));
    std::filesystem::rename(path, held_path);
    require(file.open_prefill_copy(source_file, error));
    source_file.close();

    std::vector<uint8_t> packed(rows.size() * row_bytes);
    size_t published = 0;
    PrefillPleReadStats stats;
    const auto run = [&](auto ready) {
        return prefill_ple_execute_record_plan(file, *plan, packed, row_bytes, 2, 2,
                                               ready, [] { return false; }, stats, error);
    };
    require(run([&](size_t chunk) {
        require(chunk == published++);
        const size_t end = std::min(rows.size(), (chunk + 1) * 2);
        for (size_t row = chunk * 2; row < end; ++row) {
            for (uint32_t j = 0; j < row_bytes; ++j) {
                const auto expected = rows[row] < 3 ? source[table_offset + rows[row] * row_bytes + j] : 0;
                require(packed[row * row_bytes + j] == expected);
            }
        }
        return true;
    }));
    require(file.is_open() && stats.issued == 2 && published == 3);

    const auto first_only = prefill_ple_record_plan(std::array<uint32_t, 2>{1, 0}, {},
        table_offset, 3, file_size, row_bytes, 1, 32 << 20);
    require(first_only.has_value() && first_only->reads.size() == 2);
    require(first_only->reads[0].first_chunk == 0 && first_only->reads[1].first_chunk == 1);
    PrefillPleReadStats first_ready_stats;
    auto two_rows = std::span(packed).first(2 * row_bytes);
    std::fill(two_rows.begin(), two_rows.end(), 0xff);
    published = 0;
    require(prefill_ple_execute_record_plan(file, *first_only, two_rows, row_bytes, 1, 2,
        [&](size_t chunk) {
            require(chunk == published++ && first_ready_stats.issued == chunk + 1);
            const auto row = chunk == 0 ? 1 : 0;
            require(std::equal(two_rows.begin() + chunk * row_bytes,
                               two_rows.begin() + (chunk + 1) * row_bytes,
                               source.begin() + table_offset + row * row_bytes));
            return true;
        }, [] { return false; }, first_ready_stats, error));
    require(published == 2 && first_ready_stats.issued == 2);
    require(!prefill_ple_execute_record_plan(file, *first_only, std::span(packed).first(2 * row_bytes),
        row_bytes, 1, 2, [](size_t) { return false; }, [] { return false; }, stats, error));
    require(error == "prefill PLE: record read window cancelled");
    file.close();
    require(file.open(held_path, error));

    // Accept the first read, then reject the misaligned second submission.
    auto failed_submission = *plan;
    require(failed_submission.reads[0].first_chunk == 0 && failed_submission.reads[1].first_chunk == 0);
    ++failed_submission.reads[1].offset;
    PrefillPleReadStats failed_stats;
    require(!prefill_ple_execute_record_plan(file, failed_submission, packed,
        row_bytes, 2, 2, [](size_t) { std::abort(); return false; },
        [] { return false; }, failed_stats, error));
    require(failed_stats.issued == 1 && !file.is_open() && !error.empty());

    // Fail a completed pread after the first chunk has already been published.
    // The open DirectFile still has the original extent, so this reaches the
    // completion check rather than the submission bounds check.
    require(file.open(held_path, error));
    PrefillPleReadStats short_read;
    published = 0;
    require(!prefill_ple_execute_record_plan(file, *first_only, two_rows, row_bytes, 1, 2,
        [&](size_t chunk) {
            require(chunk == 0 && published++ == 0 && short_read.issued == 1);
            std::filesystem::resize_file(held_path, first_only->reads[1].offset);
            return true;
        }, [] { return false; }, short_read, error));
    require(published == 1 && short_read.issued == 2 &&
            error == "prefill PLE: record read ended before requested row data");
    file.close();
    {
        std::ofstream restored(held_path, std::ios::binary | std::ios::trunc);
        restored.write(reinterpret_cast<const char*>(source.data()),
                       static_cast<std::streamsize>(source.size()));
        restored.close();
        require(!!restored);
    }
    check_ple_reader_window(held_path, table_offset, error);
    require(std::filesystem::remove(held_path));
    std::cout << "prefill PLE DirectFile read window checks passed\n";
}
