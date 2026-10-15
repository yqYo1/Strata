#include "strata/prefill/native16.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;
constexpr int N = 2560, HC = 4, D = N * HC, FF = 640, LR = 320;
void need(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
void check(cudaError_t v) { need(v == cudaSuccess, cudaGetErrorString(v)); }
template<class T> struct Buffer {
    T* base = nullptr; size_t size;
    static constexpr size_t guard = 128;
    explicit Buffer(size_t n): size(n) {
        check(cudaMalloc(reinterpret_cast<void**>(&base), (n + guard * 2) * sizeof(T)));
        check(cudaMemset(base, 0xa5, (n + guard * 2) * sizeof(T)));
    }
    ~Buffer() { if (base) (void)cudaFree(base); }
    T* ptr() const { return base + guard; }
    void put(const std::vector<T>& x) {
        need(x.size() == size, "fixture put size");
        check(cudaMemcpy(ptr(), x.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> v(size); check(cudaMemcpy(v.data(), ptr(), size * sizeof(T), cudaMemcpyDeviceToHost)); return v;
    }
    void guards() const {
        std::vector<unsigned char> head(guard * sizeof(T)), tail(head.size());
        check(cudaMemcpy(head.data(), base, head.size(), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(tail.data(), ptr() + size, tail.size(), cudaMemcpyDeviceToHost));
        need(std::all_of(head.begin(), head.end(), [](auto c){return c == 0xa5;}) &&
             std::all_of(tail.begin(), tail.end(), [](auto c){return c == 0xa5;}), "fixture guard overwritten");
    }
    Buffer(const Buffer&) = delete;
};
double sigmoid(double x) { return 1. / (1. + std::exp(-x)); }
void near_bf16(uint16_t actual, double want, const char* message) {
    const float a = f32_from_bf16(actual);
    // An independent double oracle; up to two BF16 ulps covers FP32 reduction
    // order and exp/rsqrt rounding, not an arbitrary model-quality tolerance.
    const float rounded = f32_from_bf16(bf16_from_f32(float(want)));
    const double tolerance = std::max(1.e-7, std::abs(double(rounded)) / 64.);
    need(std::isfinite(a) && std::abs(double(a) - rounded) <= tolerance, message);
}
void operations(int T, cudaStream_t stream) {
    namespace p = strata::prefill::native16;
    std::vector<float> e(T * N), weight(D);
    for (size_t i = 0; i < e.size(); ++i) e[i] = float(int(i % 67) - 33) / 16.f;
    for (int i = 0; i < D; ++i) weight[i] = .5f + float(i % 37) / 32.f;
    Buffer<float> emb(e.size()), gamma(D), scales(T * HC), restored(T * D);
    Buffer<uint16_t> residual(T * D), normalized(T * D), gate(T * D), mixed(T * N), mixed_h(T * N);
    Buffer<uint16_t> lo(T * LR), lo_out(T * LR), bo(T * N);
    Buffer<float> inj(T * HC);
    emb.put(e); gamma.put(weight);
    p::broadcast(emb.ptr(), residual.ptr(), T, stream);
    check(cudaStreamSynchronize(stream));
    auto r = residual.get();
    for (int t = 0; t < T; ++t) for (int c = 0; c < HC; ++c) for (int d = 0; d < N; ++d)
        need(r[t * D + c * N + d] == bf16_from_f32(e[t * N + d]), "broadcast bits");
    // Give every HC branch distinct data. A one-stream shortcut must fail.
    for (size_t i = 0; i < r.size(); ++i) r[i] = bf16_from_f32(float(int(i % 113) - 56) / 32.f);
    residual.put(r);
    std::vector<uint16_t> g(T * D), l(T * LR), b(T * N);
    std::vector<float> injection(T * HC);
    for (size_t i = 0; i < g.size(); ++i) g[i] = bf16_from_f32(float(int(i % 23) - 11) / 8.f);
    for (size_t i = 0; i < l.size(); ++i) l[i] = bf16_from_f32(float(int(i % 41) - 20) / 2.f);
    for (size_t i = 0; i < b.size(); ++i) b[i] = bf16_from_f32(float(int(i % 17) - 8) / 16.f);
    for (size_t i = 0; i < injection.size(); ++i) injection[i] = float(int(i % 13) - 6) + .0001234f;
    gate.put(g); lo.put(l); bo.put(b); inj.put(injection);
    p::norm(residual.ptr(), gamma.ptr(), 1.e-6f, scales.ptr(), normalized.ptr(), T, stream);
    p::mix(residual.ptr(), scales.ptr(), gamma.ptr(), gate.ptr(), mixed.ptr(), mixed_h.ptr(), T, stream);
    p::silu(lo.ptr(), lo_out.ptr(), T, stream);
    check(cudaStreamSynchronize(stream));
    auto xn = normalized.get(), m = mixed.get(), mh = mixed_h.get(), slo = lo_out.get();
    auto rs = scales.get();
    std::vector<double> oracle_rs(T * HC);
    for (int t = 0; t < T; ++t) for (int c = 0; c < HC; ++c) {
        double ss = 0;
        for (int d = 0; d < N; ++d) { double v = f32_from_bf16(r[t * D + c * N + d]); ss += v * v; }
        const double scale = 1. / std::sqrt(ss / N + 1.e-6);
        oracle_rs[t * HC + c] = scale;
        need(std::abs(rs[t * HC + c] - scale) <= 3.e-6 * scale, "norm scale oracle");
        for (int d = 0; d < N; ++d) {
            const size_t i = t * D + c * N + d;
            near_bf16(xn[i], f32_from_bf16(r[i]) * scale * weight[c * N + d], "norm oracle");
        }
    }
    for (int t = 0; t < T; ++t) for (int d = 0; d < N; ++d) {
        double sum = 0;
        for (int c = 0; c < HC; ++c) {
            const size_t j = t * D + c * N + d;
            sum += f32_from_bf16(r[j]) * oracle_rs[t * HC + c] * weight[c * N + d] * sigmoid(f32_from_bf16(g[j]));
        }
        sum /= HC;
        const size_t i = t * N + d;
        near_bf16(m[i], sum, "HC mix oracle");
        need(std::isfinite(f32_from_f16(mh[i])) && std::abs(f32_from_f16(mh[i]) - sum) <= 1.e-5 + std::abs(sum) / 512., "FP16 mixed oracle");
    }
    for (size_t i = 0; i < l.size(); ++i) {
        double v = f32_from_bf16(l[i]) / HC;
        near_bf16(slo[i], v * sigmoid(v), "HC silu oracle");
    }
    p::write(residual.ptr(), bo.ptr(), inj.ptr(), T, stream);
    p::restore(residual.ptr(), restored.ptr(), T * D, stream);
    check(cudaStreamSynchronize(stream));
    const auto updated = residual.get(); const auto expanded = restored.get();
    for (int t = 0; t < T; ++t) for (int c = 0; c < HC; ++c) for (int d = 0; d < N; ++d) {
        const size_t i = t * D + c * N + d;
        const double want = f32_from_bf16(r[i]) + f32_from_bf16(b[t * N + d]) * 2 * sigmoid(injection[t * HC + c] / double(HC));
        near_bf16(updated[i], want, "HC residual write oracle");
        need(expanded[i] == f32_from_bf16(updated[i]), "final restore exact");
    }
    need(emb.get() == e && gamma.get() == weight && gate.get() == g && lo.get() == l && bo.get() == b && inj.get() == injection, "HC inputs mutated");
    residual.guards(); normalized.guards(); gate.guards(); mixed.guards(); mixed_h.guards();
    lo.guards(); lo_out.guards(); bo.guards(); inj.guards(); scales.guards(); restored.guards();
}
void experts(cudaStream_t stream) {
    namespace p = strata::prefill::native16;
    constexpr int T = 3, rows = T * 10;
    std::vector<uint16_t> gu(rows * FF * 2), g(rows * FF), u(rows * FF), down(rows * N), shared(T * N);
    for (size_t i = 0; i < g.size(); ++i) {
        g[i] = bf16_from_f32(float(int(i % 73) - 36) / 8.f);
        u[i] = bf16_from_f32(float(int(i % 41) - 20) / 8.f);
        gu[2 * i] = g[i]; gu[2 * i + 1] = u[i];
    }
    for (size_t i = 0; i < down.size(); ++i) down[i] = bf16_from_f32(float(int(i % 157) - 78) / 16.f);
    for (size_t i = 0; i < shared.size(); ++i) shared[i] = bf16_from_f32(float(int(i % 31) - 15) / 16.f);
    std::vector<int32_t> slot(rows); std::vector<float> w(rows), sg(T);
    for (int i = 0; i < rows; ++i) { slot[i] = (i * 7) % rows; w[i] = float(i % 10 + 1) / 55.f; }
    slot[4] = -1; sg = {-2.f, 0.f, 3.f};
    Buffer<uint16_t> dgu(gu.size()), dg(g.size()), du(u.size()), h(g.size()), pair(g.size()), dd(down.size()), sh(shared.size()), bo(T * N);
    Buffer<int32_t> slots(rows); Buffer<float> weights(rows), gates(T);
    dgu.put(gu); dg.put(g); du.put(u); dd.put(down); sh.put(shared); slots.put(slot); weights.put(w); gates.put(sg);
    p::swiglu(dgu.ptr(), h.ptr(), rows, stream);
    p::swiglu_pair(dg.ptr(), du.ptr(), pair.ptr(), rows, stream);
    p::combine(dd.ptr(), slots.ptr(), weights.ptr(), sh.ptr(), gates.ptr(), bo.ptr(), T, rows, stream);
    check(cudaStreamSynchronize(stream));
    const auto result = h.get(), result_pair = pair.get(), out = bo.get();
    need(result == result_pair, "interleaved/separate activation disagreement");
    for (size_t i = 0; i < result.size(); ++i) {
        double x = f32_from_bf16(g[i]); double want = x * sigmoid(x) * f32_from_bf16(u[i]);
        need(std::isfinite(f32_from_f16(result[i])) && std::abs(f32_from_f16(result[i]) - want) <= 1.e-6 + std::abs(want) / 512., "SwiGLU oracle");
    }
    for (int t = 0; t < T; ++t) for (int d = 0; d < N; ++d) {
        double v = f32_from_bf16(shared[t * N + d]) * sigmoid(sg[t]);
        for (int k = 0; k < 10; ++k) if (slot[t * 10 + k] >= 0)
            v += double(w[t * 10 + k]) * f32_from_bf16(down[slot[t * 10 + k] * N + d]);
        near_bf16(out[t * N + d], v, "MoE combine oracle");
    }
    need(dgu.get() == gu && dg.get() == g && du.get() == u && dd.get() == down && sh.get() == shared && slots.get() == slot && weights.get() == w && gates.get() == sg, "MoE inputs mutated");
    dgu.guards(); dg.guards(); du.guards(); h.guards(); pair.guards(); dd.guards(); sh.guards(); bo.guards(); slots.guards(); weights.guards(); gates.guards();
    // Corrupt table must remain finite-budget and propagate a visible NaN,
    // never read outside the allocation or silently ignore the invalid slot.
    slot[0] = rows; slots.put(slot);
    p::combine(dd.ptr(), slots.ptr(), weights.ptr(), sh.ptr(), gates.ptr(), bo.ptr(), T, rows, stream);
    check(cudaStreamSynchronize(stream));
    need(std::isnan(f32_from_bf16(bo.get()[0])), "invalid slot must be visible");
}
}
int main(int argc, char** argv) {
    cudaStream_t stream = nullptr;
    try {
        need(argc == 1 || (argc == 2 && std::string(argv[1]) == "--smoke"), "usage: native16_prefill [--smoke]");
        check(cudaStreamCreate(&stream));
        operations(1, stream);
        if (argc == 1) { operations(3, stream); operations(17, stream); experts(stream); }
        check(cudaStreamSynchronize(stream));
        check(cudaStreamDestroy(stream)); stream = nullptr;
        std::puts("native16_prefill PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "native16_prefill FAIL: %s\n", e.what());
        if (stream && cudaStreamSynchronize(stream) != cudaSuccess) std::abort();
        if (stream) (void)cudaStreamDestroy(stream);
        return 1;
    }
}
