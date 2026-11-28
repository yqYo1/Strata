#include "strata/ngram/ple_reader.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

using strata::ngram::PleReader;
using strata::ngram::ROW_BYTES;

void require(bool condition) {
    if (!condition) std::abort();
}

int main(int argc, char** argv) {
    require(argc == 2);
    const auto path = std::filesystem::path(argv[1]) / "ple-phase-cache-fixture.bin";
    const auto replacement_path = std::filesystem::path(argv[1]) / "ple-phase-cache-replacement.bin";
    require(!std::filesystem::exists(path) && !std::filesystem::exists(replacement_path));
    constexpr uint32_t table_rows = 64;
    std::array<uint8_t, table_rows * ROW_BYTES> input{};
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<uint8_t>(i % 251);
    auto replacement = input;
    for (auto& byte : replacement) byte ^= 0xff;
    const auto write = [](const auto& target, const auto& bytes) {
        std::ofstream file(target, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        require(file.good());
    };
    write(path, input);
    write(replacement_path, replacement);

    PleReader reader;
    std::string error;
    using Row = std::array<uint8_t, ROW_BYTES>;
    const auto collect = [&](PleReader::Ticket ticket, uint32_t row, const Row& output, const auto& expected) {
        require(reader.collect(ticket, error));
        require(std::equal(output.begin(), output.end(), expected.begin() + row * ROW_BYTES));
    };
    const auto issue = [&](uint32_t row, bool long_prefill, Row& output) {
        return long_prefill ? reader.issue_long_prefill(&row, 1, output.data())
                            : reader.issue(&row, 1, output.data());
    };
    const auto read = [&](uint32_t row, bool long_prefill, const auto& expected) {
        Row output{};
        const auto before = reader.snapshot().reads;
        collect(issue(row, long_prefill, output), row, output, expected);
        return reader.snapshot().reads - before;
    };
    for (bool threaded : {false, true}) {
        require(reader.open(path.string(), 0, table_rows, 2, 8, error, threaded, ROW_BYTES, 16));
        require(reader.cache_capacity() == 8 && reader.prefill_cache_capacity() == 16);
        require(read(0, true, input) == 1);
        require(read(0, true, input) == 0);
        require(read(0, false, input) == 1);
        require(read(0, false, input) == 0);
        require(read(1, false, input) == 1);
        require(read(1, true, input) == 1);

        // A page-straddling ordinary read is collected before the earlier prefill ticket.
        constexpr uint32_t prefill_row = 2;
        constexpr uint32_t ordinary_row = 45;
        static_assert(ordinary_row * ROW_BYTES < 4096 && (ordinary_row + 1) * ROW_BYTES > 4096);
        Row prefill{}, ordinary{};
        const auto before = reader.snapshot().reads;
        const auto prefill_ticket = issue(prefill_row, true, prefill);
        const auto ordinary_ticket = issue(ordinary_row, false, ordinary);
        collect(ordinary_ticket, ordinary_row, ordinary, input);
        collect(prefill_ticket, prefill_row, prefill, input);
        require(reader.snapshot().reads - before == 2);
        require(read(prefill_row, true, input) == 0);
        require(read(ordinary_row, false, input) == 0);
        require(read(prefill_row, false, input) == 1);
        require(read(ordinary_row, true, input) == 1);

        require(reader.open(replacement_path.string(), 0, table_rows, 2, 8, error, threaded, ROW_BYTES, 16));
        require(read(0, true, replacement) == 1);
        require(read(0, false, replacement) == 1);
        require(read(0, true, replacement) == 0);
        require(read(0, false, replacement) == 0);

        require(reader.open(path.string(), 0, table_rows, 2, 8, error, threaded));
        require(reader.prefill_cache_capacity() == 0);
        require(read(0, true, input) == 1);
        require(read(0, false, input) == 0);
    }
    reader.close();
    require(std::filesystem::remove(path));
    require(std::filesystem::remove(replacement_path));
    std::cout << "PLE phase cache isolation checks passed\n";
}
