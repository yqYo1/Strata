// Independently authored small DFlash2 kernel contract checks; repository LICENSE.
// Manual only. No model weights, graphs, host-flag polling or timed speed claims.
#include "strata/kernels/dflash2.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
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
} // namespace

int main(int argc, char** argv) try {
    const bool supervised = argc == 2 && std::string(argv[1]) == "--supervised";
    need(argc == 1 || supervised, "usage: sycl_dflash2_test [--supervised]");
    std::cout << "dflash2 fixture: bounded eager kernels, no model\n" << std::flush;
    auto runtime = strata::sycl_backend::runtime_for(0);
    gemm(runtime);
    taps(runtime);
    runtime->wait();
    std::cout << "dflash2 fixture cases=9 PASS\n" << std::flush;
    if (supervised) {
        std::cout << "DF2_WAIT\n" << std::flush;
        std::string command;
        need(bool(std::getline(std::cin, command)) && command == "QUIT", "Main runtime-map retirement gate");
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "dflash2 fixture FAILED: " << error.what() << '\n'; return 1;
}
