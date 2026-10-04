#include "strata/sycl_upstream/mmq_quantize.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "upstream_layout.hpp"
#include <sycl/ext/intel/math.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace strata::sycl_upstream;
namespace {
// Host reference derived from pinned quantize.cu:quantize_mmq_q8_1, not
// from the previous SYCL port. Raw bytes and independently indexed host rows
// avoid sharing the GPU structure, address calculation, or launch geometry.
// CUDA fast-math and NaN-to-char behavior are not emulated (see README).
void reference(const std::vector<float>& x, const std::vector<int32_t>* ids,
               std::vector<unsigned char>& out, ggml_type type, int cols, int ld, int rows) {
    const auto layout = upstream_reference::mmq_get_q8_1_ds_layout(type);
    const bool d4 = layout == upstream_reference::MMQ_Q8_1_DS_LAYOUT_D4;
    const bool d2s6 = layout == upstream_reference::MMQ_Q8_1_DS_LAYOUT_D2S6;
    const int padded = (cols + 511) / 512 * 512;
    auto half_at = [&](size_t at, float v) {
        const uint16_t h = strata::kernels::f16_from_f32(v);
        std::memcpy(out.data() + at, &h, 2);
    };
    for (int row = 0; row < rows; ++row) {
        const int physical = ids ? ids->at(row) : row;
        for (int k = 0; k < padded; k += 128) {
            std::array<float, 128> v{};
            for (int j = 0; j < 128 && k + j < cols; ++j) v[j] = x.at(physical * ld + k + j);
            // First evaluate each lane's sequential four-value partial sum.
            std::array<float, 32> max{}, sum{};
            for (int lane = 0; lane < 32; ++lane) {
                max[lane] = std::fabs(v[4 * lane]);
                for (int j = 1; j < 4; ++j) max[lane] = std::fmax(max[lane], std::fabs(v[4 * lane + j]));
                sum[lane] = ((v[4 * lane] + v[4 * lane + 1]) + v[4 * lane + 2]) + v[4 * lane + 3];
            }
            for (int delta = d2s6 ? 8 : 4; delta; delta /= 2) {
                const auto before = max;
                for (int lane = 0; lane < 32; ++lane) max[lane] = std::fmax(before[lane], before[lane ^ delta]);
            }
            if (!d4) for (int delta = d2s6 ? 2 : 4; delta; delta /= 2) {
                const auto before = sum;
                for (int lane = 0; lane < 32; ++lane) sum[lane] = before[lane] + before[lane ^ delta];
            }
            const size_t base = (size_t(k / 128) * rows + row) * 144;
            for (int lane = 0; lane < 32; ++lane) {
                const float inv = 127.f / max[lane], scale = 1.f / inv;
                for (int j = 0; j < 4; ++j) {
                    const int8_t code = max[lane] == 0 ? 0 : int8_t(std::round(v[4 * lane + j] * inv));
                    std::memcpy(out.data() + base + 16 + 4 * lane + j, &code, 1);
                }
                if (d2s6) {
                    if (lane % 4 == 0 && lane < 24) {
                        half_at(base + 2 * (2 + lane / 4), sum[lane]);
                        if (lane % 16 == 0) half_at(base + 2 * (lane / 16), scale);
                    }
                } else if (lane % 8 == 0) {
                    if (d4) std::memcpy(out.data() + base + lane / 8 * 4, &scale, 4);
                    else {
                        half_at(base + lane / 8 * 4, scale);
                        half_at(base + lane / 8 * 4 + 2, sum[lane]);
                    }
                }
            }
        }
    }
}
float value(int row, int col, int pattern) {
    if (pattern == 0) return 0;
    if (pattern == 1) {
        uint32_t bits = (uint32_t(row) * 1664525u + uint32_t(col) * 1013904223u) ^ 0xa6712345u;
        return std::ldexp(float(int(bits % 4093) - 2046) / 137.f, int((bits >> 16) % 20) - 10);
    }
    if (pattern == 2) {
        // Exact and adjacent float half-way values with d_inv == 1.
        if (col % 32 == 31) return 127.f;
        float mid = float((col + row * 7) % 253 - 126) + 0.5f;
        return col % 3 == 0 ? mid : std::nextafter(mid, col % 3 == 1 ? INFINITY : -INFINITY);
    }
    if (pattern == 4) {
        // Regression: a real FP32 product rounds to exactly +/-2.5 before roundf.
        // Ordinary multiply + round + int8 conversion produced +/-2 in the
        // original SYCL expression on B570 despite -ffp-contract=off.
        if (col % 32 == 31) return 0x1.daa0b4p+12f;
        return (col % 2 ? -1.f : 1.f) * 0x1.2afa64p+7f;
    }
    const float cancellation[4] = {10000.f, .001f, -10000.f, -.0001f};
    return cancellation[col % 4] * float(row + 1);
}
}
int main() try {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << '\n';
    int mapped_types = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        int expected = -1, actual = -1;
        try { expected = int(upstream_reference::mmq_get_q8_1_ds_layout(ggml_type(t))); }
        catch (const std::invalid_argument&) {}
        try { actual = int(mmq_layout(ggml_type(t))); }
        catch (const std::invalid_argument&) {}
        if (expected != actual) throw std::runtime_error("type-layout selector mismatch: " + std::to_string(t));
        mapped_types += expected >= 0;
    }
    const std::array types = {GGML_TYPE_IQ3_S, GGML_TYPE_Q4_K, GGML_TYPE_Q2_K};
    int cases = 0;
    size_t compared = 0;
    for (int cols : {4, 32, 128, 512, 1280, 2560})
    for (int rows : {1, 7, 17, 4007})
    for (int mapped : {0, 1})
    for (int pattern : {0, 1, 2, 3, 4})
    for (ggml_type type : types) {
        if (rows == 4007 && (cols < 1280 || mapped != 1 || pattern != 1)) continue;
        const int ld = cols + 4;
        std::vector<float> x(size_t(rows) * ld, 999999.f);
        for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) x[r * ld + c] = value(r, c, pattern);
        std::vector<int32_t> ids(rows);
        for (int r = 0; r < rows; ++r) ids[r] = (r * 7 + 1) % rows;
        // Extra leading guard and upstream's 128 trailing guard blocks stay untouched.
        const size_t bytes = mmq_q8_bytes(rows, cols);
        std::vector<unsigned char> expected(bytes, 0xa5), actual(bytes + 144, 0xa5);
        reference(x, mapped ? &ids : nullptr, expected, type, cols, ld, rows);
        float* dx = sycl::malloc_device<float>(x.size(), q);
        int32_t* di = sycl::malloc_device<int32_t>(ids.size(), q);
        auto* raw = sycl::malloc_device<unsigned char>(actual.size(), q);
        if (!dx || !di || !raw) throw std::runtime_error("allocation failed");
        q.memcpy(dx, x.data(), x.size() * sizeof(float));
        q.memcpy(di, ids.data(), ids.size() * sizeof(int32_t));
        q.memset(raw, 0xa5, actual.size());
        mmq_quantize(q, dx, mapped ? di : nullptr, reinterpret_cast<MmqBlock*>(raw + 144), type, cols, ld, rows);
        q.memcpy(actual.data(), raw, actual.size()).wait_and_throw();
        sycl::free(dx, q); sycl::free(di, q); sycl::free(raw, q);
        if (!std::all_of(actual.begin(), actual.begin() + 144, [](auto v) { return v == 0xa5; }))
            throw std::runtime_error("leading guard overwritten");
        for (size_t i = 0; i < bytes; ++i) if (actual[i + 144] != expected[i]) {
            std::cerr << "cols=" << cols << " rows=" << rows << " mapped=" << mapped << " pattern=" << pattern
                      << " type=" << int(type) << " byte=" << i << " expected=" << int(expected[i])
                      << " actual=" << int(actual[i + 144]) << '\n';
            const size_t block = i / 144, lane_byte = i % 144;
            if (lane_byte >= 16) {
                const int r = block % rows, c = int(block / rows) * 128 + int(lane_byte) - 16;
                const int pr = mapped ? ids[r] : r;
                const int width = type == GGML_TYPE_Q2_K ? 64 : 32;
                float mx = 0;
                for (int cc = c / width * width; cc < (c / width + 1) * width && cc < cols; ++cc)
                    mx = std::fmax(mx, std::fabs(x[pr * ld + cc]));
                const float v = c < cols ? x[pr * ld + c] : 0;
                float* probe = sycl::malloc_shared<float>(4, q);
                q.single_task([=] {
                    const float inv = sycl::ext::intel::math::fdiv_rn(127.f, mx);
                    probe[0] = inv; probe[1] = v * inv; probe[2] = sycl::round(v * inv);
                    probe[3] = sycl::ext::intel::math::fdiv_rn(1.f, inv);
                }).wait_and_throw();
                std::cerr << std::hexfloat << "v=" << v << " max=" << mx
                          << " cpu_inv=" << 127.f / mx << " cpu_product=" << v * (127.f / mx)
                          << " cpu_round=" << std::round(v * (127.f / mx))
                          << " gpu_inv=" << probe[0] << " gpu_product=" << probe[1]
                          << " gpu_round=" << probe[2] << " gpu_scale=" << probe[3] << std::defaultfloat << '\n';
                sycl::free(probe, q);
            }
            throw std::runtime_error("source-reference byte mismatch");
        }
        ++cases;
        compared += bytes;
    }
    std::cout << "PASS source_reference_cases=" << cases << " layout_types=" << mapped_types
              << " compared_bytes_including_guards=" << compared << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
