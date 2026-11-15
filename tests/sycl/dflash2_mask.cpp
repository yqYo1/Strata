// Independently authored attention-mask discriminator; repository LICENSE.
// Q=K=0 makes the expected attention an arithmetic mean. Only row64
// contains nonzero V: a matching tag gives1, either stale tag must give0.
#include "strata/kernels/dflash2.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
void need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
struct Drain {
    sycl::queue& q;
    ~Drain() noexcept {
        try { q.wait_and_throw(); }
        catch (...) { strata::sycl_backend::cleanup_fail_stop("DFlash2 mask fixture completion"); }
    }
};
}

int main(int argc, char** argv) try {
    const bool supervised = argc == 2 && std::string(argv[1]) == "--supervised";
    need(argc == 1 || supervised, "usage: sycl_dflash2_mask_test [--supervised]");
    auto rt = strata::sycl_backend::runtime_for(0);
    auto& q = rt->compute();
    constexpr int Rows = 8, QueryHeads = 32, KvHeads = 8, Dim = 128;
    constexpr int Slots = 2064, Anchor = 130, Hole = 64, Epoch = 7;
    constexpr int Cols = QueryHeads * Dim, Nq = Rows * Cols, Nd = Nq / 32, Guard = 64;
    constexpr int64_t Live = int64_t(Epoch) << 32;
    std::vector<float> values(size_t(KvHeads) * Slots * Dim, 0);
    std::vector<int64_t> tags(Slots, -1);
    std::vector<int8_t> actual_q(Nq + Guard);
    std::vector<float> actual_d(Nd + Guard);
    const std::array<int32_t, 2> state = {Anchor, Epoch};
    for (int h = 0; h < KvHeads; ++h)
        std::fill_n(values.data() + (size_t(h) * Slots + Hole) * Dim, Dim, float(Anchor + Rows));
    for (int p = 0; p < Anchor; ++p) tags[size_t(p)] = Live | p;
    Allocation query(rt, Nq * 4, MemoryKind::Device), key(rt, values.size() * 4, MemoryKind::Device),
        value(rt, values.size() * 4, MemoryKind::Device), block(rt, Rows * KvHeads * Dim * 4, MemoryKind::Device),
        positions(rt, state.size() * 4, MemoryKind::Device), stamps(rt, tags.size() * 8, MemoryKind::Device),
        partial(rt, strata::kernels::kDf2PartChunks * Rows * QueryHeads * 130 * 4, MemoryKind::Device),
        output_q(rt, actual_q.size(), MemoryKind::Device), output_d(rt, actual_d.size() * 4, MemoryKind::Device);
    Drain drain{q}; // all readback/upload storage and allocations outlive the drain
    q.memset(query.data(), 0, query.size());
    q.memset(key.data(), 0, key.size());
    q.memset(block.data(), 0, block.size());
    q.memcpy(value.data(), values.data(), value.size());
    q.memcpy(positions.data(), state.data(), positions.size()).wait_and_throw();
    int cases = 0;
    for (int variant : {1, 2}) for (int chunk : {64, 128}) {
        // Explicit fixture parameters; never change the process environment.
        strata::kernels::df2_attention_variant(variant);
        strata::kernels::df2_attention_chunk(chunk);
        for (int mode = 0; mode < 3; ++mode) {
            const char* name = mode == 0 ? "stale_epoch" : mode == 1 ? "wrong_position" : "live_control";
            tags[Hole] = mode == 0 ? (int64_t(Epoch - 1) << 32) | Hole :
                         mode == 1 ? Live | (Hole + 1) : Live | Hole;
            std::fill(actual_q.begin(), actual_q.end(), int8_t(0x55));
            std::fill(actual_d.begin(), actual_d.end(), 999.0f);
            q.memcpy(stamps.data(), tags.data(), stamps.size());
            q.memcpy(output_q.data(), actual_q.data(), output_q.size());
            q.memcpy(output_d.data(), actual_d.data(), output_d.size()).wait_and_throw();
            strata::kernels::df2_attention(query.as<float>(), key.as<float>(), value.as<float>(),
                block.as<float>(), block.as<float>(), positions.as<int32_t>(), Slots, 2048,
                partial.as<float>(), output_q.data(), output_d.as<float>(), &q,
                stamps.as<int64_t>(), positions.as<int32_t>() + 1);
            q.memcpy(actual_q.data(), output_q.data(), output_q.size());
            q.memcpy(actual_d.data(), output_d.data(), output_d.size()).wait_and_throw();
            const double expected = mode == 2 ? 1.0 : 0.0;
            for (int i = 0; i < Nq; ++i) {
                const float scale = actual_d[size_t(i) / 32];
                need(std::isfinite(scale) && scale >= 0, "finite nonnegative quantization scale");
                need(std::abs(double(actual_q[size_t(i)]) * scale - expected) < 2e-5,
                     "invalid ring row contributes or live-row control is lost");
            }
            need(std::all_of(actual_q.begin() + Nq, actual_q.end(), [](int8_t v) { return v == 0x55; }) &&
                 std::all_of(actual_d.begin() + Nd, actual_d.end(), [](float v) { return v == 999.0f; }),
                 "attention output guard");
            ++cases;
            std::cout << "dflash2 mask variant=" << variant << " chunk=" << chunk << " " << name << " PASS\n" << std::flush;
        }
    }
    rt->wait();
    std::cout << "dflash2 mask cases=" << cases << " PASS\nDF2_MASK_WAIT\n" << std::flush;
    if (supervised) {
        std::string command;
        need(bool(std::getline(std::cin, command)) && command == "QUIT", "Main runtime-map retirement gate");
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "dflash2 mask FAILED: " << e.what() << '\n'; return 1;
}
