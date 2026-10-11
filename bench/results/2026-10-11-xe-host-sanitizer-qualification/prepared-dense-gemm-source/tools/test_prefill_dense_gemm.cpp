// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// No-model parity of the production dense prefill Gemm API, not a benchmark.
#include "strata/core/device.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef STRATA_PARITY_LICENSE
#error "The test target must identify its build mode"
#endif
#ifndef STRATA_PARITY_ONEMATH
#error "The test target must identify the production oneMath compile gate"
#endif

namespace {
namespace gpu = strata::gpu;
using strata::kernels::XmxType;
using strata::prefill::Gemm;

constexpr size_t kGuard = 32;   // at least 64 bytes before and after each logical buffer
constexpr size_t kExtra = 32;   // absolute 64-byte alignment, plus an optional element offset
constexpr size_t kVramBudget = 16u << 20;
constexpr uint64_t kWorkBudget = 50000000;
constexpr uint16_t kInputCanary = 0x55a5;
constexpr float kOutputCanary = -12345.25f;
constexpr int kRepeats = 3;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void checked(bool ok, const char* operation) {
    if (!ok) throw std::runtime_error(std::string(operation) + ": " + gpu::last_error());
}
[[noreturn]] void stop_before_release(const char* operation) noexcept {
    std::fprintf(stderr, "prefill_dense_gemm_parity FAIL: %s: %s; fail-stop before releasing GPU/host storage\n",
                 operation, gpu::last_error());
    std::fflush(nullptr);
    std::_Exit(1);
}
// A failed drain is not proof of completion. Do not unwind through Gemm or host copy storage in that case.
void drain() noexcept {
    if (!gpu::device_sync()) stop_before_release("checked device drain");
}
struct FinalDrain {
    ~FinalDrain() { drain(); }
};

struct Shape {
    int64_t t, n, k, ldy;
    bool offset;
    const char* name;
};
constexpr Shape kShapes[] = {
    {1, 32, 32, 32, false, "single-row"},
    {3, 64, 64, 80, false, "short-rows-stride"},
    {16, 64, 128, 64, false, "aligned"},
    {65, 32, 64, 48, false, "row-tail-stride"},
    {3, 129, 64, 137, false, "column-tail-stride"},
    {3, 32, 33, 39, true, "K-tail-offset-stride"},
};

template<class E>
size_t aligned_offset(E* base, bool offset) {
    const uintptr_t start = reinterpret_cast<uintptr_t>(base + kGuard);
    const uintptr_t aligned = (start + 63u) & ~uintptr_t(63u);
    return (aligned - reinterpret_cast<uintptr_t>(base)) / sizeof(E) + (offset ? 1u : 0u);
}

// All buffers are device USM; vectors are copy staging only. The destructor drains before any member dies,
// including when a synchronous submission throws after earlier commands were accepted by the queue.
struct Fixture {
    Gemm gemm;
    uint16_t* dx = nullptr;
    uint16_t* dw = nullptr;
    float* dy = nullptr;
    size_t xo = 0, wo = 0, yo = 0;
    std::vector<uint16_t> x, w, x_read, w_read;
    std::vector<float> y_seed, y_read;

    ~Fixture() {
        drain();
        gpu::free(dx);
        gpu::free(dw);
        gpu::free(dy);
        // gemm (and its optional BF16-to-FP16 scratch) is destroyed only after this checked drain.
    }
    void init(const Shape& s) {
        const size_t nx = static_cast<size_t>(s.t * s.k) + 2 * kGuard + kExtra;
        const size_t nw = static_cast<size_t>(s.n * s.k) + 2 * kGuard + kExtra;
        const size_t ny = static_cast<size_t>(s.t * s.ldy) + 2 * kGuard + kExtra;
        // Include Gemm's possible FP16 images, not only these three allocations (prepare's tiny trial is separate).
        const size_t bytes = (nx + nw) * sizeof(uint16_t) + ny * sizeof(float) +
                             static_cast<size_t>((s.t + s.n) * s.k) * sizeof(uint16_t);
        require(bytes <= kVramBudget, "fixture exceeds 16 MiB device storage budget");
        x.assign(nx, kInputCanary); w.assign(nw, kInputCanary);
        x_read.resize(nx); w_read.resize(nw);
        y_seed.assign(ny, kOutputCanary); y_read.resize(ny);
        checked(gpu::alloc_device(&dx, nx * sizeof(uint16_t)), "alloc X");
        checked(gpu::alloc_device(&dw, nw * sizeof(uint16_t)), "alloc W");
        checked(gpu::alloc_device(&dy, ny * sizeof(float)), "alloc Y");
        xo = aligned_offset(dx, s.offset); wo = aligned_offset(dw, s.offset); yo = aligned_offset(dy, s.offset);
        require(xo + static_cast<size_t>(s.t * s.k) + kGuard <= nx, "X guards do not fit");
        require(wo + static_cast<size_t>(s.n * s.k) + kGuard <= nw, "W guards do not fit");
        require(yo + static_cast<size_t>(s.t * s.ldy) + kGuard <= ny, "Y guards do not fit");
        std::string error;
        require(gemm.init(nullptr, 0, error), "Gemm init with no dequantization scratch failed");
        std::printf("fixture=%s device_storage_bound_bytes=%zu xmx_gemm_ok=%d\n", s.name, bytes,
                    strata::kernels::xmx_gemm_ok(dx + xo, dw + wo, dy + yo, s.n, s.k, s.ldy));
    }
    void upload() {
        checked(gpu::copy_async(dx, x.data(), x.size() * sizeof(uint16_t), nullptr), "upload X");
        checked(gpu::copy_async(dw, w.data(), w.size() * sizeof(uint16_t), nullptr), "upload W");
        checked(gpu::copy_async(dy, y_seed.data(), y_seed.size() * sizeof(float), nullptr), "upload Y");
        checked(gpu::stream_sync(nullptr), "upload completion before staging reuse");
    }
    void download() {
        checked(gpu::copy_async(x_read.data(), dx, x.size() * sizeof(uint16_t), nullptr), "read X guards");
        checked(gpu::copy_async(w_read.data(), dw, w.size() * sizeof(uint16_t), nullptr), "read W guards");
        checked(gpu::copy_async(y_read.data(), dy, y_read.size() * sizeof(float), nullptr), "read Y");
        checked(gpu::stream_sync(nullptr), "read completion before host access");
    }
};

uint32_t mix(uint32_t v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; return v ^ (v >> 16);
}
uint16_t input_bits(XmxType type, size_t i, uint32_t seed) {
    const uint32_t h = mix(static_cast<uint32_t>(i) + seed);
    const uint16_t sign = static_cast<uint16_t>((h >> 16) & 0x8000u);
    if (h % 31u == 0) return sign;   // signed zeros and finite normal numbers only
    if (type == XmxType::bf16)
        return static_cast<uint16_t>(sign | ((122u + (h >> 8) % 6u) << 7) | (h & 127u));
    return static_cast<uint16_t>(sign | ((10u + (h >> 8) % 6u) << 10) | (h & 1023u));
}

// Independent IEEE binary-format decoder: no production converter, SYCL half, or GPU math is used by the oracle.
double decode(uint16_t bits, XmxType type) {
    const int frac_bits = type == XmxType::bf16 ? 7 : 10;
    const int exp_bits = type == XmxType::bf16 ? 8 : 5;
    const int bias = type == XmxType::bf16 ? 127 : 15;
    const int frac = bits & ((1 << frac_bits) - 1);
    const int ex = (bits >> frac_bits) & ((1 << exp_bits) - 1);
    require(ex != (1 << exp_bits) - 1, "oracle input is NaN or infinity");
    const double magnitude = ex == 0 ? std::ldexp(static_cast<double>(frac), 1 - bias - frac_bits)
        : std::ldexp(static_cast<double>((1 << frac_bits) + frac), ex - bias - frac_bits);
    return (bits & 0x8000u) ? -magnitude : magnitude;
}

void fill_inputs(Fixture& f, const Shape& s, XmxType type, int repeat, bool accumulate) {
    const uint32_t seed = 0x13579bdu + static_cast<uint32_t>(repeat) * 0x2468aceu +
                          (accumulate ? 0x9876543u : 0u);
    for (size_t i = 0; i < static_cast<size_t>(s.t * s.k); ++i) f.x[f.xo + i] = input_bits(type, i, seed);
    for (size_t i = 0; i < static_cast<size_t>(s.n * s.k); ++i) f.w[f.wo + i] = input_bits(type, i, seed ^ 0xdeadbeefu);
    for (int64_t r = 0; r < s.t; ++r)
        for (int64_t n = 0; n < s.n; ++n) {
            const size_t o = f.yo + static_cast<size_t>(r * s.ldy + n);
            // Deliberately nonzero even for beta=0: reading stale Y is an error.
            f.y_seed[o] = static_cast<float>((r * 11 + n * 7 + repeat * 13) % 37 - 18) * 0.125f;
        }
}

struct Oracle {
    std::vector<double> expected, limit;
};
Oracle reference(const Fixture& f, const Shape& s, XmxType type, bool accumulate, bool dp4a_possible) {
    const size_t rows = static_cast<size_t>(s.t), cols = static_cast<size_t>(s.n), k = static_cast<size_t>(s.k);
    std::vector<double> x(rows * k), w(cols * k), xm(rows), wm(cols), xa(rows), wa(cols);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = decode(f.x[f.xo + i], type);
        xm[i / k] = std::max(xm[i / k], std::abs(x[i])); xa[i / k] += std::abs(x[i]);
    }
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = decode(f.w[f.wo + i], type);
        wm[i / k] = std::max(wm[i / k], std::abs(w[i])); wa[i / k] += std::abs(w[i]);
    }
    Oracle result;
    result.expected.resize(rows * cols); result.limit.resize(rows * cols);
    // FP32 roundoff bound, with fourfold allowance for reduction/scale evaluation and a 1e-6 absolute floor.
    // Fixed before running: no observed-error fitting. FP16/BF16 products here have finite normal inputs.
    constexpr double u = 0x1p-24;
    const double steps = static_cast<double>(2 * s.k + 8);
    const double gamma = 4.0 * steps * u / (1.0 - steps * u);
    for (size_t r = 0; r < rows; ++r)
        for (size_t n = 0; n < cols; ++n) {
            const double initial = accumulate ? static_cast<double>(f.y_seed[f.yo + r * static_cast<size_t>(s.ldy) + n]) : 0.0;
            double sum = initial, absolute_products = std::abs(initial);
            for (size_t j = 0; j < k; ++j) {
                const double product = x[r * k + j] * w[n * k + j];
                sum += product; absolute_products += std::abs(product);
            }
            double quantization = 0.0;
            if (dp4a_possible) {
                // The production fallback requantizes to int8. Bound its nearest-rounding error from full-row
                // maxima (conservative for every block), without reproducing its blocks, scales or integer dots.
                const double ex = xm[r] * (1.0 / 254.0 + 8.0 * u), ew = wm[n] * (1.0 / 254.0 + 8.0 * u);
                quantization = ex * wa[n] + ew * xa[r] + static_cast<double>(k) * ex * ew;
            }
            const size_t o = r * cols + n;
            result.expected[o] = sum;
            result.limit[o] = 1e-6 + gamma * (absolute_products + quantization) + quantization;
            require(std::isfinite(sum) && std::isfinite(result.limit[o]), "nonfinite CPU oracle");
        }
    return result;
}

void compare(const Fixture& f, const Shape& s, const Oracle& ref, const char* type, int repeat,
             bool accumulate, bool dp4a_possible) {
    require(f.x == f.x_read && f.w == f.w_read, "input or input guard changed");
    for (size_t i = 0; i < f.y_read.size(); ++i) {
        const bool in_span = i >= f.yo && i - f.yo < static_cast<size_t>(s.t * s.ldy);
        const bool live = in_span && (i - f.yo) % static_cast<size_t>(s.ldy) < static_cast<size_t>(s.n);
        if (!live) require(f.y_read[i] == f.y_seed[i], "output guard or row slack changed");
    }
    double max_abs = 0, max_relative = 0, square_error = 0, max_limit_ratio = 0;
    size_t bad = 0;
    for (int64_t r = 0; r < s.t; ++r)
        for (int64_t n = 0; n < s.n; ++n) {
            const size_t o = static_cast<size_t>(r * s.n + n);
            const double got = f.y_read[f.yo + static_cast<size_t>(r * s.ldy + n)];
            require(std::isfinite(got), "GPU output is NaN or infinity");
            const double error = std::abs(got - ref.expected[o]);
            max_abs = std::max(max_abs, error);
            max_relative = std::max(max_relative, error / std::max(std::abs(ref.expected[o]), 1e-6));
            square_error += error * error;
            max_limit_ratio = std::max(max_limit_ratio, error / ref.limit[o]);
            if (error > ref.limit[o]) {
                if (bad == 0) std::fprintf(stderr, "first mismatch row=%lld col=%lld got=%.9g oracle=%.17g limit=%.9g\n",
                                           static_cast<long long>(r), static_cast<long long>(n), got, ref.expected[o], ref.limit[o]);
                ++bad;
            }
        }
    std::printf("%s case=%s T=%lld N=%lld K=%lld ldy=%lld repeat=%d beta=%d checked=%zu "
                "maxabs=%.9g maxrelative=%.9g RMSE=%.9g max_limit_ratio=%.9g tolerance=%s bad=%zu\n",
                type, s.name, static_cast<long long>(s.t), static_cast<long long>(s.n), static_cast<long long>(s.k),
                static_cast<long long>(s.ldy), repeat, accumulate ? 1 : 0, ref.expected.size(), max_abs, max_relative,
                std::sqrt(square_error / static_cast<double>(ref.expected.size())), max_limit_ratio,
                dp4a_possible ? "FP32+int8-rounding-envelope" : "FP32-roundoff", bad);
    require(bad == 0, "full dense output differs from CPU oracle");
}

void run(const Shape& s, XmxType type) {
    Fixture f;
    f.init(s);
    const char* label = type == XmxType::bf16 ? "BF16" : "FP16";
    const std::string own = strata::kernels::gemm_path(type);
    const bool eligible = strata::kernels::xmx_gemm_ok(f.dx + f.xo, f.dw + f.wo, f.dy + f.yo, s.n, s.k, s.ldy);
    std::printf("type=%s own_gemm_path=%s production_path_before=%s\n", label, own.c_str(), Gemm::path());
    Oracle previous;
    for (int repeat = 0; repeat < kRepeats; ++repeat)
        for (int beta = 0; beta < (type == XmxType::bf16 ? 2 : 1); ++beta) {
            const bool accumulate = beta != 0;
            // Gemm::path does not expose per-type BF16 rejection. Allow the DP4a envelope whenever the
            // hardware-derived own path could take it, even if the global report currently says oneMath.
            const bool dp4a_possible = eligible && own == "DP4a" && !accumulate;
            fill_inputs(f, s, type, repeat, accumulate);
            const Oracle ref = reference(f, s, type, accumulate, dp4a_possible);
            if (!accumulate && !previous.expected.empty()) {
                bool distinguishable = false;
                for (size_t i = 0; i < ref.expected.size(); ++i)
                    distinguishable = distinguishable || std::abs(ref.expected[i] - previous.expected[i]) >
                        ref.limit[i] + previous.limit[i];
                require(distinguishable, "repeated inputs cannot detect stale output within fixed tolerance");
            }
            f.upload();
            if (type == XmxType::bf16)
                f.gemm.bf16(f.dx + f.xo, f.dw + f.wo, f.dy + f.yo, s.t, s.n, s.k, s.ldy, accumulate);
            else f.gemm.f16(f.dx + f.xo, f.dw + f.wo, f.dy + f.yo, s.t, s.n, s.k, s.ldy);
            checked(gpu::stream_sync(nullptr), "production Gemm completion");
            f.download();
            compare(f, s, ref, label, repeat, accumulate, dp4a_possible);
            if (!accumulate) previous = ref;
        }
    std::printf("type=%s production_path_after=%s\n", label, Gemm::path());
}
const char* env(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "<unset>";
}
} // namespace

int main(int argc, char**) try {
    require(argc == 1, "usage: prefill_dense_gemm_parity (no parameters)");
    uint64_t products = 0;
    for (const Shape& s : kShapes) products += static_cast<uint64_t>(s.t * s.n * s.k) * kRepeats * 3;
    require(products < kWorkBudget, "CPU oracle exceeds its scalar-product budget");
    std::printf("prefill_dense_gemm_parity build_mode=%s STRATA_ONEMATH=%d STRATA_NO_BLAS=%s "
                "STRATA_NO_XMX=%s STRATA_NO_BF16_MMA=%s STRATA_BF16_TC=%s "
                "device_storage_budget_bytes=%zu oracle_scalar_products=%llu budget=%llu\n",
                STRATA_PARITY_LICENSE, STRATA_PARITY_ONEMATH, env("STRATA_NO_BLAS"), env("STRATA_NO_XMX"),
                env("STRATA_NO_BF16_MMA"), env("STRATA_BF16_TC"), kVramBudget,
                static_cast<unsigned long long>(products), static_cast<unsigned long long>(kWorkBudget));
    (void) strata::core::Runtime::get();
    FinalDrain final_drain;
    const auto device = strata::core::device_info();
    std::printf("device=%s pci=%s backend=%s driver=%s\n", device.name.c_str(), device.pci.c_str(),
                device.backend.c_str(), device.driver_version.c_str());
    Gemm::prepare();
    Gemm::settle();
    checked(gpu::device_sync(), "prepare/settle completion");
    std::printf("production_path_settled=%s; global path report does not certify each type's BLAS backend\n", Gemm::path());
    for (const Shape& s : kShapes) {
        run(s, XmxType::bf16);
        run(s, XmxType::f16);
    }
    checked(gpu::device_sync(), "final completion");
    std::puts("prefill_dense_gemm_parity OK: synthetic dense BF16/FP16 only; no model, grouped/native quantization, graph, or performance claim");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "prefill_dense_gemm_parity FAIL: %s\n", error.what());
    return 1;
}
