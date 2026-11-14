// SPDX-License-Identifier: MIT
// Bounded execution-location diagnostic, not model performance or task quality.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/runtime.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {
using strata::sycl_backend::Runtime;
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
constexpr size_t Guard = 16;
constexpr float Sentinel = 1234.25f;
void need(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
void drain(const std::shared_ptr<Runtime>& rt) noexcept {
    try { rt->wait(); } catch (...) { std::abort(); }
}
struct Drain {
    std::shared_ptr<Runtime> rt;
    ~Drain() noexcept { drain(rt); }
};
void copy(const std::shared_ptr<Runtime>& rt, void* dst, const void* src, size_t bytes) {
    // A transfer failure cannot unwind while its borrowed host storage is live.
    try { rt->wait(rt->compute().memcpy(dst, src, bytes)); } catch (...) { std::abort(); }
}
uint16_t encode(float value, bool bf16) {
    return bf16 ? uint16_t(std::bit_cast<uint32_t>(value) >> 16)
                : strata::kernels::f16_from_f32(value);
}
uint64_t raw_ns() {
    timespec value{};
    need(clock_gettime(CLOCK_MONOTONIC_RAW, &value) == 0, "raw clock failed");
    return uint64_t(value.tv_sec) * 1000000000ULL + uint64_t(value.tv_nsec);
}
void run(const std::shared_ptr<Runtime>& rt, int rows, bool down, bool bf16, bool hc_once = false) {
    // MTP's HC down-projection width/inner geometry;64 rows bounds the first
    // library-coverage fixture. It is not a model throughput measurement.
    const int width = hc_once ? 320 : down ? 2560 : 1280;
    const int inner = hc_once ? 10240 : down ? 640 : 2560, ld = width + 3;
    const char* role = hc_once ? "HCDown" : down ? "Down" : "GU";
    const size_t nx = size_t(rows) * inner, nw = size_t(width) * inner, ny = size_t(rows) * ld;
    const size_t gpu_bytes = (nx + nw + 4 * Guard) * 2 + (ny + 2 * Guard) * 4;
    need(gpu_bytes < (256u << 20), "bounded device-byte budget");
    std::vector<uint16_t> x(nx + 2 * Guard, 0x3555), w(nw + 2 * Guard, 0x3555);
    std::vector<float> y(ny + 2 * Guard, Sentinel);
    // Small dyadic values are exactly representable in both input types.
    // All rows/columns have an independently computable scalar coefficient.
    for (int t = 0; t < rows; ++t) for (int k = 0; k < inner; ++k)
        x[Guard + size_t(t) * inner + k] = encode(float((t % 7 - 3) * (k % 5 - 2)) / 64.f, bf16);
    for (int n = 0; n < width; ++n) for (int k = 0; k < inner; ++k)
        w[Guard + size_t(n) * inner + k] = encode(float((n % 11 - 5) * (k % 7 - 3)) / 32.f, bf16);
    Allocation dx(rt, x.size() * 2, MemoryKind::Device), dw(rt, w.size() * 2, MemoryKind::Device),
               dy(rt, y.size() * 4, MemoryKind::Device);
    Drain lease{rt}; // Retires before device owners and borrowed host vectors.
    copy(rt, dx.data(), x.data(), x.size() * 2);
    copy(rt, dw.data(), w.data(), w.size() * 2);
    copy(rt, dy.data(), y.data(), y.size() * 4);
    strata::prefill::Gemm gemm;
    std::string error;
    need(gemm.init_external(&rt->compute(), nullptr, 0, nullptr, 0, error), error.c_str());
    std::cout << "BEGIN type=" << (bf16 ? "bf16" : "fp16") << " role=" << role
              << " rows=" << rows << " N=" << width << " K=" << inner << " ld=" << ld << '\n';
    if (hc_once) std::cout << "OP_BEGIN invocation=1 pid=" << getpid() << " tid=" << syscall(SYS_gettid)
                          << " raw_ns=" << raw_ns() << '\n';
    if (bf16) gemm.bf16(dx.as<uint16_t>() + Guard, dw.as<uint16_t>() + Guard,
                        dy.as<float>() + Guard, rows, width, inner, ld);
    else gemm.f16(dx.as<uint16_t>() + Guard, dw.as<uint16_t>() + Guard,
                  dy.as<float>() + Guard, rows, width, inner, ld);
    if (hc_once) std::cout << "OP_RETURN invocation=1 raw_ns=" << raw_ns() << '\n';
    rt->wait();
    if (hc_once) std::cout << "OP_DRAIN invocation=1 raw_ns=" << raw_ns() << " checked=1\n";
    copy(rt, y.data(), dy.data(), y.size() * 4);
    int coefficient = 0;
    for (int k = 0; k < inner; ++k) coefficient += (k % 5 - 2) * (k % 7 - 3);
    for (size_t i = 0; i < Guard; ++i)
        need(y[i] == Sentinel && y[Guard + ny + i] == Sentinel, "output end guards");
    for (int t = 0; t < rows; ++t) for (int n = 0; n < ld; ++n) {
        const float expected = n < width ? float((t % 7 - 3) * (n % 11 - 5) * coefficient) / 2048.f : Sentinel;
        need(std::isfinite(y[Guard + size_t(t) * ld + n]) &&
             y[Guard + size_t(t) * ld + n] == expected, "full analytic output/stride oracle");
    }
    std::vector<uint16_t> xx(x.size()), ww(w.size());
    copy(rt, xx.data(), dx.data(), xx.size() * 2);
    copy(rt, ww.data(), dw.data(), ww.size() * 2);
    need(xx == x && ww == w, "input guards and immutability");
    std::cout << "PASS type=" << (bf16 ? "bf16" : "fp16") << " role=" << role
              << " rows=" << rows << " values=" << size_t(rows) * width
              << " analytic=FULL_EXACT guards=PASS inputs=PASS device_bytes=" << gpu_bytes << '\n';
}
} // namespace
int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    try {
        const bool smoke = argc == 2 && std::string(argv[1]) == "--smoke";
        const bool once = argc == 2 && std::string(argv[1]) == "--one-bf16";
        need(smoke || once || argc == 1, "usage: sycl_gemm_device_probe [--smoke|--one-bf16]");
        auto rt = strata::sycl_backend::runtime_for();
        need(rt->device().is_gpu(), "GPU queue required");
        std::cout << "device=" << rt->device().get_info<sycl::info::device::name>()
                  << " driver=" << rt->device().get_info<sycl::info::device::driver_version>() << '\n';
        if (once) run(rt, 64, false, true, true);
        else for (bool bf16 : {false, true}) for (int rows : {1, 160, 8192}) {
            if (smoke && rows != 1) continue;
            for (bool down : {false, true}) run(rt, rows, down, bf16);
        }
        rt->wait();
        std::cout << "DONE cases=" << (once ? 1 : smoke ? 4 : 12)
                  << " scope=synthetic_location_probe_not_model_performance\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
