#include "strata/prefill/native16.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::prefill::native16 {
namespace {
using namespace sycl_backend;
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, FF = 640;
size_t count(int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0)
        throw std::invalid_argument("native16 prefill: nonpositive shape");
    const size_t n = checked_count(rows, cols);
    if (n > SIZE_MAX / 4)
        throw std::invalid_argument("native16 prefill: byte overflow");
    return n;
}
float load(uint16_t v) { return strata::kernels::f32_from_bf16(v); }
uint16_t store(float v) { return strata::kernels::bf16_from_f32(v); }
float sigmoid(float v) { return 1.f / (1.f + sycl::exp(-v)); }
float sum32(sycl::sub_group sg, float v) {
    for (int d = 16; d; d >>= 1) v += sycl::permute_group_by_xor(sg, v, d);
    return v;
}
// Preserve the existing expert hidden activation's saturation contract.
// BF16 GEMM outputs and the HC residual itself are never saturated here.
uint16_t hidden(float v) {
    return strata::kernels::f16_from_f32(sycl::isnan(v) ? v : sycl::clamp(v, -65504.f, 65504.f));
}
}

bool requested() {
    static const bool value = [] {
        const char* v = std::getenv("STRATA_SYCL_PREFILL_NATIVE16");
        if (!v || std::strcmp(v, "0") == 0) return false;
        if (std::strcmp(v, "bf16") == 0) return true;
        throw std::invalid_argument("STRATA_SYCL_PREFILL_NATIVE16 must be 0 or bf16");
    }();
    return value;
}
void restore(const uint16_t* x, float* y, int64_t n, void* stream) {
    const size_t z = count(n, 1);
    validate_spans({{y, z * 4}}, {{x, z * 2}});
    for_each(z, stream, [=](size_t i) { y[i] = load(x[i]); });
}
void pack(const float* x, uint16_t* y, int64_t n, void* stream) {
    const size_t z = count(n, 1);
    validate_spans({{y, z * 2}}, {{x, z * 4}});
    for_each(z, stream, [=](size_t i) { y[i] = store(x[i]); });
}
void broadcast(const float* x, uint16_t* r, int64_t rows, void* stream) {
    const size_t n = count(rows, D);
    validate_spans({{r, n * 2}}, {{x, size_t(rows) * N * 4}});
    for_each(n, stream, [=](size_t i) { r[i] = store(x[(i / D) * N + i % N]); });
}
void norm(const uint16_t* r, const float* w, float eps, float* rs,
          uint16_t* xn, int64_t rows, void* stream) {
    const size_t n = count(rows, D);
    if (!std::isfinite(eps) || eps < 0)
        throw std::invalid_argument("native16 norm: invalid epsilon");
    validate_spans({{rs, size_t(rows) * HC * 4}, {xn, n * 2}}, {{r, n * 2}, {w, D * 4}});
    auto e = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(8, h);
        h.parallel_for(sycl::nd_range<1>(size_t(rows) * HC * 256, 256),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const size_t row = it.get_group_linear_id();
                const int tid = it.get_local_linear_id(), lane = tid % 32;
                float ss = 0;
                for (int d = tid; d < N; d += 256) {
                    const float a = load(r[row * N + d]);
                    ss = sycl::fma(a, a, ss);
                }
                ss = sum32(it.get_sub_group(), ss);
                if (!lane) partial[tid / 32] = ss;
                it.barrier(sycl::access::fence_space::local_space);
                ss = sum32(it.get_sub_group(), lane < 8 ? partial[lane] : 0.f);
                const float scale = sycl::rsqrt(ss / float(N) + eps);
                if (!tid) rs[row] = scale;
                for (int d = tid; d < N; d += 256) {
                    const size_t i = row * N + d;
                    xn[i] = store((load(r[i]) * scale) * w[(row % HC) * N + d]);
                }
            });
    });
    finish(stream, e);
}
void silu(const uint16_t* x, uint16_t* y, int64_t rows, void* stream) {
    const size_t n = count(rows, LR);
    validate_spans({{y, n * 2}}, {{x, n * 2}});
    for_each(n, stream, [=](size_t i) {
        const float a = load(x[i]) / float(HC);
        y[i] = store(a * sigmoid(a));
    });
}
void mix(const uint16_t* r, const float* rs, const float* w, const uint16_t* g,
         uint16_t* y, uint16_t* yh, int64_t rows, void* stream) {
    const size_t n = count(rows, N);
    validate_spans({{y, n * 2}, {yh, n * 2}},
                   {{r, n * HC * 2}, {g, n * HC * 2}, {rs, size_t(rows) * HC * 4}, {w, D * 4}});
    for_each(n, stream, [=](size_t i) {
        const size_t t = i / N, d = i % N;
        float v = 0;
        for (int c = 0; c < HC; ++c) {
            const size_t j = t * D + c * N + d;
            const float x = (load(r[j]) * rs[t * HC + c]) * w[c * N + d];
            v = sycl::fma(x, sigmoid(load(g[j])), v);
        }
        v /= float(HC);
        y[i] = store(v);
        yh[i] = strata::kernels::f16_from_f32(v);
    });
}
void write(uint16_t* r, const uint16_t* bo, const float* inj, int64_t rows, void* stream) {
    const size_t n = count(rows, D);
    validate_spans({{r, n * 2}}, {{bo, size_t(rows) * N * 2}, {inj, size_t(rows) * HC * 4}});
    for_each(n, stream, [=](size_t i) {
        const size_t t = i / D, c = i % D / N, d = i % N;
        const float gate = 2.f * sigmoid(inj[t * HC + c] / float(HC));
        r[i] = store(sycl::fma(load(bo[t * N + d]), gate, load(r[i])));
    });
}
void swiglu(const uint16_t* gu, uint16_t* y, int64_t rows, void* stream) {
    const size_t n = count(rows, FF);
    validate_spans({{y, n * 2}}, {{gu, n * 4}});
    for_each(n, stream, [=](size_t i) {
        const float g = load(gu[2 * i]);
        y[i] = hidden((g * sigmoid(g)) * load(gu[2 * i + 1]));
    });
}
void swiglu_pair(const uint16_t* g, const uint16_t* u, uint16_t* y, int64_t rows, void* stream) {
    const size_t n = count(rows, FF);
    validate_spans({{y, n * 2}}, {{g, n * 2}, {u, n * 2}});
    for_each(n, stream, [=](size_t i) {
        const float v = load(g[i]);
        y[i] = hidden((v * sigmoid(v)) * load(u[i]));
    });
}
void combine(const uint16_t* d, const int32_t* slot, const float* w, const uint16_t* shared,
             const float* sg, uint16_t* bo, int64_t rows, int64_t down_rows, void* stream) {
    const size_t n = count(rows, N), dn = count(down_rows, N);
    if (down_rows != rows * 10)
        throw std::invalid_argument("native16 combine: incorrect routed-row count");
    validate_spans({{bo, n * 2}}, {{d, dn * 2}, {slot, size_t(rows) * 10 * 4},
        {w, size_t(rows) * 10 * 4}, {shared, n * 2}, {sg, size_t(rows) * 4}});
    for_each(n, stream, [=](size_t i) {
        const size_t t = i / N, col = i % N;
        float v = 0;
        for (int k = 0; k < 10; ++k) {
            const int row = slot[t * 10 + k];
            if (row >= down_rows) v = sycl::nan(0u);
            else if (row >= 0)
                v = sycl::fma(w[t * 10 + k], load(d[size_t(row) * N + col]), v);
        }
        bo[i] = store(v + load(shared[i]) * sigmoid(sg[t]));
    });
}
} // namespace strata::prefill::native16
