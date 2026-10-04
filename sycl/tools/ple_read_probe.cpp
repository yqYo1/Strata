// CPU-only probe of the actual prompt PLE row-gather path. Build from the
// original ngram.cpp, ple_reader.cpp and direct_file.cpp; no GPU or substitute
// decoder. Usage: ple_read_probe SHARD2 TOKENS LENGTHS_CSV direct|mmap REPEATS
#include "strata/kernels/ngram.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

int main(int argc, char** argv) try {
    if (argc != 6) {
        std::fprintf(stderr, "usage: ple_read_probe SHARD2 TOKENS LENGTHS_CSV direct|mmap REPEATS\n");
        return 2;
    }
    std::ifstream input(argv[2]);
    if (!input) throw std::runtime_error("Cannot open token fixture");
    std::vector<int32_t> tokens;
    int32_t token = 0;
    while (input >> token) tokens.push_back(token);
    std::vector<size_t> lengths;
    std::stringstream sizes(argv[3]);
    std::string item;
    while (std::getline(sizes, item, ',')) {
        const auto n = std::stoull(item);
        if (n == 0 || n > tokens.size()) throw std::runtime_error("Length outside token fixture");
        lengths.push_back((size_t) n);
    }
    const std::string mode(argv[4]);
    if (mode != "direct" && mode != "mmap") throw std::runtime_error("Unknown read mode");
    const int repeats = std::stoi(argv[5]);
    if (repeats < 1 || lengths.empty()) throw std::runtime_error("No runs requested");
    const auto constants = k::ple_artifact_consts();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (size_t i = 0; i < lengths.size(); ++i) {
            const size_t n = lengths[repeat % 2 ? lengths.size() - 1 - i : i];
            std::vector<uint32_t> rows(n * k::PLE_N_HEADS);
            int32_t previous[2] = {0, 0};
            for (size_t t = 0; t < n; ++t) {
                k::ngram_rows(tokens.data() + t, previous, 1, constants, rows.data() + t * k::PLE_N_HEADS);
                previous[0] = previous[1]; previous[1] = tokens[t];
            }
            std::vector<float> embedding(n * k::PLE_N_HEADS * k::PLE_HEAD_DIM);
            k::PleTable table;
            k::PleIoOptions options;
            options.mode = mode == "direct" ? k::PleIo::Direct : k::PleIo::Mmap;
            std::string error;
            const auto opened = Clock::now();
            if (!table.open(argv[1], error, options)) throw std::runtime_error(error);
            const double open_ms = elapsed_ms(opened);
            const auto started = Clock::now();
            if (!table.gather_batch(rows.data(), n, embedding.data(), error)) throw std::runtime_error(error);
            const double gather_ms = elapsed_ms(started);
            uint64_t hash = 14695981039346656037ull;
            for (float value : embedding) {
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite PLE embedding");
                uint32_t bits = 0;
                std::memcpy(&bits, &value, sizeof bits);
                // Hash all four bytes in a fixed little-endian order.
                for (int shift = 0; shift < 32; shift += 8) {
                    hash ^= (bits >> shift) & 255u;
                    hash *= 1099511628211ull;
                }
            }
            const char* threads = std::getenv("STRATA_IO_THREADS");
            std::printf("{\"mode\":\"%s\",\"io_threads\":%d,\"tokens\":%zu,\"repeat\":%d,"
                        "\"open_ms\":%.6f,\"gather_ms\":%.6f,\"embedding_floats\":%zu,"
                        "\"fnv1a64\":\"%016llx\",\"row_bytes_read\":%llu}\n",
                        mode.c_str(), threads ? std::atoi(threads) : 16, n, repeat + 1,
                        open_ms, gather_ms, embedding.size(), (unsigned long long) hash,
                        (unsigned long long) table.bytes_read());
            std::fflush(stdout);
            const auto report = table.io_report();
            if (!report.empty()) std::fprintf(stderr, "ple probe io: %s\n", report.c_str());
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "ple probe: %s\n", error.what());
    return 1;
}
