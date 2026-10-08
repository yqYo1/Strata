// CPU-only full-prompt PLE comparison. Uses the original row hash, table
// reader and decoder; compares every output byte with a 1024-token direct
// reference. No GPU calls or substitute decoder.
#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void open(k::PleTable& table, const char* path, bool ram, double& open_ms) {
    k::PleIoOptions options;
    options.mode = ram ? k::PleIo::Mmap : k::PleIo::Direct;
    options.lock = ram; // Same touch fallback as production --ple-io ram.
    std::string error;
    const auto start = Clock::now();
    if (!table.open(path, error, options)) throw std::runtime_error(error);
    open_ms = elapsed(start);
}

int main(int argc, char** argv) try {
    if (argc != 6) {
        std::fprintf(stderr, "usage: ple_full_context_probe SHARD2 TOKENS N CHUNKS_CSV ROUNDS\n");
        return 2;
    }
    const size_t n = std::stoull(argv[3]);
    const int rounds = std::stoi(argv[5]);
    std::ifstream fixture(argv[2]);
    std::vector<int32_t> tokens;
    int32_t value;
    while (fixture >> value) tokens.push_back(value);
    if (n == 0 || n > tokens.size() || rounds < 1) throw std::runtime_error("invalid fixture length or rounds");
    std::vector<size_t> chunks;
    std::stringstream sizes(argv[4]);
    std::string item;
    while (std::getline(sizes, item, ',')) {
        const size_t length = std::stoull(item);
        chunks.push_back(length ? std::min(length, n) : n);
    }
    if (chunks.empty()) throw std::runtime_error("no chunks requested");
    const auto constants = k::ple_artifact_consts();
    std::vector<uint32_t> rows(n * k::PLE_N_HEADS);
    int32_t previous[2] = {0, 0};
    for (size_t t = 0; t < n; ++t) {
        k::ngram_rows(tokens.data() + t, previous, 1, constants, rows.data() + t * k::PLE_N_HEADS);
        previous[0] = previous[1]; previous[1] = tokens[t];
    }
    constexpr size_t per_token = k::PLE_N_HEADS * k::PLE_HEAD_DIM;
    std::vector<float> reference(n * per_token);
    {
        k::PleTable table;
        double open_ms;
        open(table, argv[1], false, open_ms);
        std::printf("{\"kind\":\"reference_start\",\"tokens\":%zu,\"chunk\":1024,\"open_ms\":%.6f,\"table_rows\":%llu,\"format\":\"%s\"}\n",
                    n, open_ms, (unsigned long long) table.rows(), table.format());
        std::fflush(stdout);
        const auto start = Clock::now();
        std::string error;
        for (size_t t = 0; t < n; t += 1024) {
            const size_t count = std::min<size_t>(1024, n - t);
            if (!table.gather_batch(rows.data() + t * k::PLE_N_HEADS, count,
                                    reference.data() + t * per_token, error)) throw std::runtime_error(error);
            if ((t + count) % 16384 == 0 || t + count == n) {
                std::printf("{\"kind\":\"reference_progress\",\"done\":%zu,\"tokens\":%zu,\"elapsed_ms\":%.6f}\n", t + count, n, elapsed(start));
                std::fflush(stdout);
            }
        }
        const double gather_ms = elapsed(start);
        uint64_t hash = 14695981039346656037ull;
        for (float v : reference) {
            if (!std::isfinite(v)) throw std::runtime_error("non-finite reference embedding");
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            for (int shift = 0; shift < 32; shift += 8) {
                hash ^= (bits >> shift) & 255u; hash *= 1099511628211ull;
            }
        }
        std::printf("{\"kind\":\"reference\",\"tokens\":%zu,\"chunk\":1024,\"open_ms\":%.6f,\"gather_ms\":%.6f,\"embedding_bytes\":%zu,\"all_finite\":true,\"fnv1a64\":\"%016llx\"}\n",
                    n, open_ms, gather_ms, reference.size() * sizeof(float), (unsigned long long) hash);
        std::fprintf(stderr, "reference I/O: %s\n", table.io_report().c_str());
        std::fflush(stdout);
    }
    for (int round = 0; round < rounds; ++round) {
        // Reverse both mode and chunk order on alternate rounds. The file cache
        // is not flushed; opening a direct reader starts a fresh row cache.
        for (int policy = 0; policy < 2; ++policy) {
            const bool ram = ((policy + round) % 2) != 0;
            for (size_t i = 0; i < chunks.size(); ++i) {
                const size_t chunk = chunks[round % 2 ? chunks.size() - 1 - i : i];
                k::PleTable table;
                double open_ms;
                open(table, argv[1], ram, open_ms);
                std::vector<float> actual(chunk * per_token);
                std::string error;
                double gather_ms = 0, compare_ms = 0;
                size_t compared = 0;
                const auto wall_start = Clock::now();
                for (size_t t = 0; t < n; t += chunk) {
                    const size_t count = std::min(chunk, n - t);
                    auto start = Clock::now();
                    if (!table.gather_batch(rows.data() + t * k::PLE_N_HEADS, count, actual.data(), error))
                        throw std::runtime_error(error);
                    gather_ms += elapsed(start);
                    start = Clock::now();
                    const size_t bytes = count * per_token * sizeof(float);
                    if (std::memcmp(actual.data(), reference.data() + t * per_token, bytes))
                        throw std::runtime_error("output bytes differ from the direct reference");
                    compared += bytes;
                    compare_ms += elapsed(start);
                }
                std::printf("{\"kind\":\"case\",\"mode\":\"%s\",\"round\":%d,\"tokens\":%zu,\"chunk\":%zu,\"open_ms\":%.6f,\"gather_ms\":%.6f,\"compare_ms\":%.6f,\"wall_after_open_ms\":%.6f,\"compared_bytes\":%zu,\"all_bytes_identical\":true,\"locked\":%s}\n",
                            ram ? "ram" : "direct", round + 1, n, chunk, open_ms, gather_ms, compare_ms,
                            elapsed(wall_start), compared, table.locked() ? "true" : "false");
                std::fprintf(stderr, "case %s round %d chunk %zu I/O: %s\n",
                            ram ? "ram" : "direct", round + 1, chunk, table.io_report().c_str());
                std::fflush(stdout);
            }
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "full PLE probe: %s\n", error.what());
    return 1;
}
