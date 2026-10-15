// Main-owned CPU-only bridge from the proposed C++ observer to Python checks.
#include "layermajor-wave-census-v2/proposed/include/strata/prefill/layermajor_shape_census.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using Census = strata::prefill::detail::LayerMajorShapeCensus;
struct Shape { int64_t n, pos, chunk; bool rotate; };

int main() {
    static_assert(sizeof(Census) <= (4u << 20));
    std::printf("wave-host-meta: {\"owner_bytes\":%zu,\"output_cap_bytes\":%zu}\n",
                sizeof(Census), Census::output_cap);
    const Shape shapes[] = {{1, 0, 1, false}, {13, 17, 3, true},
        {32769, 0, 8192, true}, {262144, 0, 8192, false}, {262127, 17, 8192, true}};
    std::array<int32_t, 512> residency{};
    for (int e = 0; e < 512; ++e) residency[e] = e;
    const Census::Format format{23, 42, 1360, 180, 640 * 1360, 1280 * 1360,
                               1280 * 1360 + 2560 * 180};
    for (const auto& shape : shapes) {
        auto census = std::make_unique<Census>(shape.n, shape.pos, shape.chunk);
        for (int layer = 0; layer < 48; ++layer) {
            census->begin(layer, format, residency.data(), true);
            int ordinal = 0;
            for (int64_t off = 0; off < shape.n; off += shape.chunk, ++ordinal) {
                const auto t = std::min(shape.chunk, shape.n - off);
                std::array<int32_t, 512> counts{};
                for (int k = 0; k < 10; ++k) {
                    const int e = shape.rotate ? (layer * 7 + ordinal * 13 + k * 31) % 512 : k;
                    counts[e] = int32_t(t);
                }
                census->counts(layer, t, counts.data(), counts.size());
                for (int e = 511; e >= 0; --e)
                    if (counts[e]) census->issue(true, layer, counts[e], e, e, -1);
                for (int e = 0; e < 512; ++e)
                    if (counts[e]) census->issue(false, layer, counts[e], e, e, -1);
            }
            census->end(layer);
            assert(!census->failure);
        }
        census->complete(stdout);
        assert(!census->failure);
    }
    // A full stdio buffer can defer /dev/full's error until fflush. Exercise
    // both the actual Writer boundary and complete()'s in-memory failure flag.
    // This does not prove durable external acknowledgement of a completion line.
    for (bool via_complete : {false, true}) {
        auto census = std::make_unique<Census>(1, 0, 1);
        std::array<int32_t, 512> counts{};
        for (int e = 0; e < 10; ++e) counts[e] = 1;
        for (int layer = 0; layer < 48; ++layer) {
            census->begin(layer, format, residency.data(), true);
            census->counts(layer, 1, counts.data(), counts.size());
            for (int e = 0; e < 10; ++e) {
                census->issue(true, layer, 1, e, e, -1);
                census->issue(false, layer, 1, e, e, -1);
            }
            census->end(layer);
        }
        assert(!census->failure);
        std::vector<char> buffer(Census::output_cap);
        FILE* bad = std::fopen("/dev/full", "w");
        assert(bad && std::setvbuf(bad, buffer.data(), _IOFBF, buffer.size()) == 0);
        if (via_complete) {
            census->complete(bad);
            assert(census->failure && std::string(census->failure) == "output_io_incomplete");
        } else {
            Census::Writer writer{bad};
            census->serialize(writer);
            assert(writer.ok && writer.bytes < buffer.size() && !std::ferror(bad));
            assert(std::fflush(bad) != 0);
            census->finalized = true;
        }
        std::fclose(bad);
    }
    return std::fflush(stdout) == 0 ? 0 : 1;
}
