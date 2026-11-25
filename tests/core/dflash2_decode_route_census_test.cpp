#include "strata/program/dflash2_decode_route_census.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

using Census = strata::program::Dflash2DecodeRouteCensus;

void require(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

void observe_window(Census& counters, int width) {
    counters.begin_window(width);
    for (int layer = 0; layer < Census::layers; ++layer) {
        std::array<int32_t, Census::max_width * Census::fanout> ids{};
        for (int i = 0; i < width * Census::fanout; ++i)
            ids[i] = (layer * 13 + i % Census::fanout) % Census::experts;
        counters.observe_layer(layer, ids.data(), width, Census::fanout);
    }
    counters.complete_window();
}

void check_record(std::FILE* file, uint64_t request, uint64_t windows, uint64_t rows) {
    std::array<uint64_t, 3> header{};
    require(std::fread(header.data(), sizeof(uint64_t), header.size(), file) == header.size(),
            "missing census record header");
    require(header[0] == request && header[1] == windows && header[2] == rows, "wrong request or window extent");
    std::array<uint32_t, Census::experts> counts{};
    for (int layer = 0; layer < Census::layers; ++layer) {
        require(std::fread(counts.data(), sizeof(uint32_t), counts.size(), file) == counts.size(),
                "missing layer counts");
        for (int e = 0; e < Census::experts; ++e) {
            const int relative = (e + Census::experts - layer * 13 % Census::experts) % Census::experts;
            require(counts[e] == (relative < Census::fanout ? rows : 0), "routed row count mismatch");
        }
    }
}

} // namespace

int main() {
    std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::tmpfile(), &std::fclose);
    require(file != nullptr, "cannot create test stream");

    {
        Census long_request;
        uint64_t rows = 0;
        for (int i = 0; i < Census::max_windows; ++i) {
            const int width = i % Census::max_width + 1;
            rows += width;
            observe_window(long_request, width);
        }
        require(long_request.write(file.get(), 1), "bounded request write failed");
        long_request.begin_window(1);
        require(!long_request.write(file.get(), 8), "accepted an extra window beyond the cap");

        Census next_request;
        observe_window(next_request, 1);
        require(next_request.write(file.get(), 2), "following short request write failed");
        const Census cancelled_before_decode;
        require(cancelled_before_decode.write(file.get(), 3), "zero-window request write failed");

        std::rewind(file.get());
        check_record(file.get(), 1, Census::max_windows, rows);
        check_record(file.get(), 2, 1, 1);
        check_record(file.get(), 3, 0, 0);
        require(std::fgetc(file.get()) == EOF, "unexpected count payload");
    }
    {
        Census incomplete;
        incomplete.begin_window(4);
        incomplete.complete_window();
        require(!incomplete.write(file.get(), 4), "partial target window became a sample");

        Census split;
        std::array<int32_t, Census::fanout> ids{};
        split.begin_window(2);
        split.observe_layer(0, ids.data(), 1, Census::fanout);
        split.complete_window();
        require(!split.write(file.get(), 5), "unsupported split verifier was accepted");

        Census invalid_id;
        ids.back() = Census::experts;
        invalid_id.begin_window(1);
        invalid_id.observe_layer(0, ids.data(), 1, Census::fanout);
        invalid_id.complete_window();
        require(!invalid_id.write(file.get(), 6), "out-of-range expert was accepted");
    }
    std::puts("DFlash2 decode route census test passed");
    return 0;
}
