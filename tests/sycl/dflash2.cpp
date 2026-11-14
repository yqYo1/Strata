// Independently authored small DFlash2 kernel contract checks; repository LICENSE.
// Manual only. No model weights, graphs, host-flag polling or timed speed claims.
#include "strata/kernels/dflash2.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include "strata/prefill/dflash2_taps.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
using namespace strata::kernels;
struct Drain {
    sycl::queue& queue;
    ~Drain() noexcept {
        try { queue.wait_and_throw(); }
        catch (...) { strata::sycl_backend::cleanup_fail_stop("DFlash2 fixture completion"); }
    }
};
void need(bool value, const char* what) { if (!value) throw std::runtime_error(what); }

void gemm(const std::shared_ptr<strata::sycl_backend::Runtime>& runtime) {
    constexpr int K = 512, M = 64, Columns = 8, Jobs = 3;
    constexpr size_t Blocks = size_t(K) * M / 32, Packed = Blocks * 18;
    auto& q = runtime->compute();
    std::array<std::vector<uint8_t>, Jobs> weights;
    for (int job = 0; job < Jobs; ++job) {
        weights[job].resize(Packed);
        for (int r = 0; r < M; ++r) for (int block = 0; block < K / 32; ++block) {
            auto* p = weights[job].data() + (r * (K / 32) + block) * 18;
            const uint16_t one = 0x3c00; // exactly representable fp16 scale1
            std::memcpy(p, &one, 2);
            for (int i = 0; i < 16; ++i) {
                const int low = (r + job * 3 + block * 5 + i) & 15;
                const int high = (r * 3 + job + block + i * 7) & 15;
                p[2 + i] = uint8_t(low | high << 4);
            }
        }
    }
    std::vector<float> input(Columns * K), scales(Columns * K / 32);
    std::vector<int8_t> codes(Columns * K);
    std::vector<float> reset(Columns * M + 16, -999999), result(reset.size());
    for (int c = 0; c < Columns; ++c) for (int k = 0; k < K; ++k)
        input[c * K + k] = k % 32 == 0 ? 127.0f : float((k * 19 + c * 11) % 255 - 127);
    Allocation raw(runtime, Packed, MemoryKind::Device);
    Allocation packed(runtime, Jobs * Packed, MemoryKind::Device);
    Allocation x(runtime, input.size() * 4, MemoryKind::Device);
    Allocation xq(runtime, input.size(), MemoryKind::Device);
    Allocation xd(runtime, scales.size() * 4, MemoryKind::Device);
    Allocation output(runtime, Jobs * reset.size() * 4, MemoryKind::Device);
    Drain drain{q}; // before any host owner may unwind after a submitted copy
    for (int job = 0; job < Jobs; ++job) {
        q.memcpy(raw.data(), weights[job].data(), Packed);
        auto* plane = packed.as<uint8_t>() + job * Packed;
        df2_q4_tile(raw.data(), plane, plane + size_t(K) * M / 2, K, Blocks, &q);
        q.wait_and_throw(); // raw is reused only after its consumer completes
    }
    q.memcpy(x.data(), input.data(), input.size() * 4);
    df2_quant_rows(x.as<float>(), K, Columns, xq.data(), xd.as<float>(), &q);
    q.memcpy(codes.data(), xq.data(), codes.size());
    q.memcpy(scales.data(), xd.data(), scales.size() * 4);
    q.wait_and_throw();
    for (size_t i = 0; i < codes.size(); ++i) need(float(codes[i]) == input[i], "planar signed activation code");
    for (float d : scales) need(d == 1.0f, "planar activation scale");
    for (int columns = 1; columns <= Columns; ++columns) {
        std::array<Df2Mat, Jobs> matrices;
        for (int job = 0; job < Jobs; ++job) {
            auto* y = output.as<float>() + job * reset.size();
            q.memcpy(y, reset.data(), reset.size() * 4);
            matrices[job] = {packed.as<uint8_t>() + job * Packed, y, M, true};
        }
        df2_gemm(matrices.data(), Jobs, xq.data(), xd.as<float>(), K, columns, &q);
        for (int job = 0; job < Jobs; ++job) {
            q.memcpy(result.data(), matrices[job].y, result.size() * 4).wait_and_throw();
            for (int c = 0; c < columns; ++c) for (int r = 0; r < M; ++r) {
                int expected = 0;
                for (int k = 0; k < K; ++k) {
                    const auto* block = weights[job].data() + (r * (K / 32) + k / 32) * 18;
                    const int nibble = (block[2 + k % 16] >> (k % 32 >= 16 ? 4 : 0)) & 15;
                    expected += (nibble - 8) * int(input[c * K + k]);
                }
                need(result[c * M + r] == float(expected), "Q4 GEMM signed scalar contract");
            }
            for (size_t i = size_t(columns * M); i < result.size(); ++i)
                need(result[i] == -999999, "Q4 GEMM inactive column/output guard");
        }
        std::cout << "dflash2 gemm columns=" << columns << " matrices=3 PASS\n" << std::flush;
    }
}

void taps(const std::shared_ptr<strata::sycl_backend::Runtime>& runtime) {
    constexpr int N = 2560, HC = 4, Rows = 3, Stride = 5 * N;
    auto& q = runtime->compute();
    std::vector<float> r(Rows * HC * N), bo(Rows * N, 0.5f), inj(Rows * HC, 0),
                       initial(Rows * Stride, -999999), out(initial.size());
    for (size_t i = 0; i < r.size(); ++i) r[i] = float(int(i % 17) - 8) * 0.125f;
    Allocation dr(runtime, r.size() * 4, MemoryKind::Device), db(runtime, bo.size() * 4, MemoryKind::Device),
               di(runtime, inj.size() * 4, MemoryKind::Device), dy(runtime, out.size() * 4, MemoryKind::Device);
    Drain drain{q};
    q.memcpy(dr.data(), r.data(), r.size() * 4); q.memcpy(db.data(), bo.data(), bo.size() * 4);
    q.memcpy(di.data(), inj.data(), inj.size() * 4); q.memcpy(dy.data(), initial.data(), initial.size() * 4);
    df2_tap_rows(dr.as<float>(), dy.as<float>() + 2 * N, Stride, N, HC, Rows, &q);
    df2_tap_gr(dr.as<float>(), HC * N, db.as<float>(), N, di.as<float>(), HC,
               dy.as<float>() + 4 * N, Stride, Rows, N, HC, &q);
    q.memcpy(out.data(), dy.data(), out.size() * 4).wait_and_throw();
    for (int row = 0; row < Rows; ++row) for (int slot = 0; slot < 5; ++slot) for (int i = 0; i < N; ++i) {
        float expected = -999999;
        if (slot == 2 || slot == 4) {
            float sum = 0;
            for (int c = 0; c < HC; ++c) sum += r[(row * HC + c) * N + i];
            expected = sum / HC + (slot == 4 ? 0.5f : 0);
        }
        need(out[row * Stride + slot * N + i] == expected, "tap row/GR value or interleaved guard");
    }
    std::cout << "dflash2 taps rows=3 fp32,pending_GR,interleaved_guards PASS\n" << std::flush;
}

// Test the production tail producer/consumer without loading draft/target
// weights. Readback belongs only to this numerical fixture, never to inference.
void prompt_taps(const std::shared_ptr<strata::sycl_backend::Runtime>& runtime) {
    using Tail = strata::prefill::Dflash2PrefillTaps;
    constexpr int N = int(Tail::kWidth), HC = int(Tail::kStreams), Stride = 5 * N, MaxRows = 12;
    auto& q = runtime->compute();
    Tail tail;
    std::string error;
    Allocation dr(runtime, MaxRows * HC * N * 4, MemoryKind::Device),
               db(runtime, MaxRows * HC * N * 2, MemoryKind::Device),
               dw(runtime, MaxRows * Stride * 4, MemoryKind::Device),
               tiny(runtime, 4, MemoryKind::Device), host(runtime, HC * N * 4, MemoryKind::Host);
    Drain drain{q};
    auto ok = [&](bool value) { if (!value) throw std::runtime_error(error); };
    auto code = [](int64_t pos, int slot, int col, int channel) {
        constexpr uint16_t values[] = {0x3f81, 0xbf82, 0x4001, 0xc002, 0x427f, 0x3a03, 0xba0f, 0x4580};
        return values[(pos * 3 + slot * 5 + col + channel * 7) % 8];
    };
    // Decode BF16 independently of the device's bit-cast. The selected finite
    // normal inputs exercise exponent/sign and rounding with FP32 accumulation.
    auto unpack = [](uint16_t bits) {
        const int exponent = (bits >> 7) & 255;
        const double mantissa = double(128 + (bits & 127));
        return float(std::ldexp((bits & 0x8000) ? -mantissa : mantissa, exponent - 134));
    };
    auto sample = [&](int64_t pos, int slot, int col, int channel, bool bf16) {
        const float value = unpack(code(pos, slot, col, channel));
        return bf16 ? value : value + float(channel + 1) * 0.0003f;
    };
    auto mean = [&](int64_t pos, int slot, int col) {
        float sum = 0;
        for (int c = 0; c < HC; ++c) sum += sample(pos, slot, col, c, slot % 2 == 0);
        return sum / HC;
    };
    auto window_value = [](int64_t pos, int slot, int col) {
        return float((pos % 37) * 17 + slot * 11 + col % 29) * 0.125f;
    };
    auto put = [&](int slot, int64_t pos, int rows) {
        const bool bf16 = slot % 2 == 0;
        std::vector<float> input(size_t(rows) * HC * N);
        std::vector<uint16_t> packed(input.size());
        for (int r = 0; r < rows; ++r) for (int c = 0; c < HC; ++c) for (int i = 0; i < N; ++i) {
            const auto at = (size_t(r) * HC + c) * N + i;
            input[at] = sample(pos + r, slot, i, c, false);
            packed[at] = code(pos + r, slot, i, c);
        }
        if (bf16) q.memcpy(db.data(), packed.data(), packed.size() * 2).wait_and_throw();
        else q.memcpy(dr.data(), input.data(), input.size() * 4).wait_and_throw();
        ok(tail.capture(Tail::layers[slot], pos, rows, bf16 ? nullptr : dr.as<float>(),
                        bf16 ? db.as<uint16_t>() : nullptr, &q, error));
    };
    auto window = [&](int64_t pos, int rows) {
        std::vector<float> input(size_t(rows) * Stride);
        for (int r = 0; r < rows; ++r) for (int slot = 0; slot < 5; ++slot) for (int i = 0; i < N; ++i)
            input[size_t(r) * Stride + slot * N + i] = window_value(pos + r, slot, i);
        q.memcpy(dw.data(), input.data(), input.size() * 4).wait_and_throw();
        ok(tail.capture_window(dw.as<float>(), rows, pos, error));
    };
    auto check = [&](const std::vector<std::pair<int64_t, int64_t>>& runs, auto expected) {
        size_t next = 0;
        int64_t total = 0;
        ok(tail.consume([&](const float* data, int64_t rows, int64_t pos, std::string&) {
            need(next < runs.size() && runs[next] == std::make_pair(pos, rows), "complete row run positions/coverage");
            std::vector<float> actual(size_t(rows) * Stride);
            q.memcpy(actual.data(), data, actual.size() * 4).wait_and_throw();
            for (int64_t r = 0; r < rows; ++r) for (int slot = 0; slot < 5; ++slot) for (int i = 0; i < N; ++i)
                need(actual[size_t(r) * Stride + slot * N + i] == expected(pos + r, slot, i),
                     "prompt mean/window numerical or interleaved layout contract");
            ++next; total += rows; return true;
        }, error));
        need(next == runs.size() && tail.captured_rows() == total, "complete row count");
    };
    auto pass = [](const char* name) { std::cout << "dflash2 prompt " << name << " PASS\n" << std::flush; };

    ok(tail.begin(1, 0, error));
    check({}, mean);
    need(!tail.begin(0, 0, error) && !tail.begin(262145, 0, error), "prompt length admission");
    pass("empty_anchor");

    ok(tail.begin(10, 0, error));
    for (int slot = 0; slot < 5; ++slot) put(slot, 0, MaxRows);
    check({{0, 9}}, mean); // exclude the last prompt anchor and following rows
    pass("mixed_fp32_bf16");

    ok(tail.begin(12, 0, error));
    window(1, 3);
    for (int slot = 0; slot < 4; ++slot) put(slot, 4, 1); // never admit a four-of-five row
    for (int slot = 0; slot < 5; ++slot) put(slot, 7, 2);
    ok(tail.capture(4, 0, 1, nullptr, nullptr, nullptr, error)); // unrelated layer
    need(!tail.consume([](const float*, int64_t, int64_t, std::string& e) {
        e = "intentional consumer refusal"; return false;
    }, error) && error == "intentional consumer refusal", "consumer failure propagation");
    check({{1, 3}, {7, 2}}, [&](int64_t pos, int slot, int i) {
        return pos < 4 ? window_value(pos, slot, i) : mean(pos, slot, i);
    });
    pass("sparse_complete_runs");

    constexpr int64_t Long = 65537, First = Long - Tail::kTail;
    ok(tail.begin(Long, 0, error));
    window(First - 2, 5); // left-clipped pointer must skip two complete rows
    for (int slot = 0; slot < 5; ++slot) put(slot, Long - 2, 2);
    check({{First, 3}, {Long - 2, 1}}, [&](int64_t pos, int slot, int i) {
        return pos == Long - 2 ? mean(pos, slot, i) : window_value(pos, slot, i);
    });
    pass("long_tail_clipping");

    ok(tail.begin(262144, 0, error));
    window(262142, 2);
    ok(tail.capture_window(nullptr, 1, 262143, error)); // decode anchor, not a prompt tap
    ok(tail.capture(5, 262144, 0, nullptr, nullptr, &q, error));
    need(!tail.capture(5, 262144, 1, dr.as<float>(), nullptr, &q, error), "one-past physical extent");
    check({{262142, 1}}, window_value);
    pass("physical_tail_extent"); // not a full target-context execution

    ok(tail.begin(10, 0, error));
    need(!tail.capture(5, -1, 1, dr.as<float>(), nullptr, &q, error), "negative row");
    need(!tail.capture(5, 262143, 2, dr.as<float>(), nullptr, &q, error), "cross physical end");
    need(!tail.capture(5, 0, 1, nullptr, nullptr, &q, error), "missing residual");
    need(!tail.capture(5, 0, 1, dr.as<float>(), db.as<uint16_t>(), &q, error), "ambiguous format");
    need(!tail.capture(5, 0, 1, tiny.as<float>(), nullptr, &q, error), "short registered span");
    need(!tail.capture(5, 0, 1, host.as<float>(), nullptr, &q, error), "host USM rejected");
    need(!tail.capture_window(tiny.as<float>(), 2, 0, error), "short verifier mean span");
    need(!tail.consume({}, error), "missing consumer");
    check({}, mean); // no refused capture is allowed to publish presence
    pass("rejected_spans_formats");

    for (int slot = 0; slot < 5; ++slot) put(slot, 0, 3);
    tail.release(); // complete a queued producer before freeing its tail
    need(!tail.consume([](const float*, int64_t, int64_t, std::string&) { return true; }, error), "released owner");
    ok(tail.begin(3, 0, error));
    check({}, mean); // old row presence must not survive owner reuse
    window(0, 2);
    check({{0, 2}}, window_value);
    ok(tail.begin(1, 0, error));
    check({}, mean);
    pass("release_rebegin");
}
} // namespace

int main(int argc, char** argv) try {
    const bool supervised = argc == 2 && std::string(argv[1]) == "--supervised";
    need(argc == 1 || supervised, "usage: sycl_dflash2_test [--supervised]");
    std::cout << "dflash2 fixture: bounded eager kernels, no model\n" << std::flush;
    auto runtime = strata::sycl_backend::runtime_for(0);
    gemm(runtime);
    taps(runtime);
    prompt_taps(runtime);
    runtime->wait();
    std::cout << "dflash2 fixture cases=16 PASS\n" << std::flush;
    if (supervised) {
        std::cout << "DF2_WAIT\n" << std::flush;
        std::string command;
        need(bool(std::getline(std::cin, command)) && command == "QUIT", "Main runtime-map retirement gate");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "dflash2 fixture FAILED: " << error.what() << '\n'; return 1;
}
