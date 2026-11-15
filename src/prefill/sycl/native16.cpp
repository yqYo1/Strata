#include "strata/prefill/native16.hpp"
#include "strata/prefill/moe_window_routes.hpp"
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

namespace {
void accumulator_spans(void* stream, std::initializer_list<Span> outputs,
                       std::initializer_list<Span> inputs) {
    validate_spans(outputs, inputs);
    const auto& q = queue_for(stream);
    for (auto span : outputs) validate_device_span(span.pointer, span.bytes, q);
    for (auto span : inputs) validate_device_span(span.pointer, span.bytes, q);
}
size_t accumulator_routes(const detail::MoeWindowRoutes& plan, void* stream) {
    const size_t routes = size_t(plan.rows()) * detail::MoeWindowRoutes::top_k;
    const auto limit = plan.split_accumulator() ? detail::MoeWindowRoutes::max_split_accum_rows
                                               : detail::MoeWindowRoutes::max_accum_rows;
    if (plan.rows() < 1 || plan.rows() > limit ||
        plan.original_slots().size() != routes || plan.source_rows().size() != routes ||
        !queue_for(stream).has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("MoE accumulator requires an inverse plan and in-order queue");
    return routes;
}
void accumulator_error(uint32_t* status, uint32_t bit) {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*status).fetch_or(bit);
}
}

void moe_job_weights(const detail::MoeWindowRoutes& plan, const float* weights,
                     const int32_t* original, float* job_weights, uint32_t* status, void* stream) {
    const size_t routes = accumulator_routes(plan, stream);
    accumulator_spans(stream, {{job_weights, routes * 4}, {status, 4}},
                   {{weights, routes * 4}, {original, routes * 4}});
    for_each(routes, stream, [=](size_t i) {
        const int32_t slot = original[i];
        if (slot < 0 || size_t(slot) >= routes) {
            accumulator_error(status, 1); job_weights[i] = 0.f;
        } else job_weights[i] = weights[size_t(slot)];
    });
}

void moe_gather_mixed(const detail::MoeWindowRoutes& plan, size_t job_index,
                      const uint16_t* primary, const uint16_t* tail, int64_t primary_rows,
                      const int32_t* source, uint16_t* gathered, uint32_t* status, void* stream) {
    const size_t routes = accumulator_routes(plan, stream);
    const auto& job = plan.jobs().at(job_index);
    const int64_t rows = plan.rows();
    if (primary_rows <= 0 || primary_rows > rows || job.rows < 1 || job.rows > 8192 ||
        job.global_row < 0 || size_t(job.global_row) > routes - size_t(job.rows))
        throw std::invalid_argument("MoE mixed gather bank/job bounds");
    const size_t n = count(job.rows, N), offset = size_t(job.global_row);
    // Always validate complete logical banks, not just the first input row.
    accumulator_spans(stream, {{gathered, n * 2}, {status, 4}},
        {{primary, count(primary_rows, N) * 2}, {source, routes * 4}});
    if (primary_rows < rows) {
        accumulator_spans(stream, {{gathered, n * 2}, {status, 4}},
            {{tail, count(rows - primary_rows, N) * 2}});
        validate_spans({{primary, count(primary_rows, N) * 2}},
                       {{tail, count(rows - primary_rows, N) * 2}});
    } else if (tail) throw std::invalid_argument("MoE mixed gather unexpected tail");
    for_each(n, stream, [=](size_t i) {
        const int32_t row = source[offset + i / N];
        const size_t col = i % N;
        if (row < 0 || int64_t(row) >= rows) {
            if (!col) accumulator_error(status, 4);
            gathered[i] = 0;
            return;
        }
        gathered[i] = int64_t(row) < primary_rows ? primary[size_t(row) * N + col]
            : tail[size_t(int64_t(row) - primary_rows) * N + col];
    });
}

void moe_accumulate(const detail::MoeWindowRoutes& plan, size_t job_index,
                    const uint16_t* down, const int32_t* source, const int32_t* original,
                    const float* job_weights, float* accumulator, uint32_t* status, void* stream) {
    if (plan.split_accumulator())
        throw std::invalid_argument("MoE ordinary accumulator cannot consume a split plan");
    const size_t routes = accumulator_routes(plan, stream);
    const auto& job = plan.jobs().at(job_index);
    if (job.rows < 1 || job.rows > 8192 || size_t(job.rows) > routes || job.global_row < 0 ||
        size_t(job.global_row) > routes - size_t(job.rows))
        throw std::invalid_argument("MoE accumulator job span");
    const size_t rows = size_t(plan.rows()), offset = size_t(job.global_row);
    const size_t n = count(job.rows, N);
    accumulator_spans(stream, {{accumulator, rows * N * 4}, {status, 4}},
                   {{down, n * 2}, {source, routes * 4}, {original, routes * 4}, {job_weights, routes * 4}});
    for_each(n, stream, [=](size_t i) {
        const size_t routed = offset + i / N, col = i % N;
        const int32_t token = source[routed], slot = original[routed];
        if (slot < 0 || size_t(slot) >= routes) { accumulator_error(status, 1); return; }
        if (token < 0 || size_t(token) >= rows || token != slot / 10) {
            accumulator_error(status, 2); return;
        }
        const size_t out = size_t(token) * N + col;
        accumulator[out] = sycl::fma(job_weights[routed], load(down[i]), accumulator[out]);
    });
}

void moe_accumulate_split(const detail::MoeWindowRoutes& plan, size_t job_index,
                    const uint16_t* down, const int32_t* source, const int32_t* original,
                    const float* job_weights, float* primary, float* tail,
                    int64_t primary_rows, uint32_t* status, void* stream) {
    if (!plan.split_accumulator() || primary_rows < 1 || primary_rows >= plan.rows())
        throw std::invalid_argument("MoE split accumulator requires two nonempty logical banks");
    const size_t routes = accumulator_routes(plan, stream);
    const auto& job = plan.jobs().at(job_index);
    if (job.rows < 1 || job.rows > 8192 || size_t(job.rows) > routes || job.global_row < 0 ||
        size_t(job.global_row) > routes - size_t(job.rows))
        throw std::invalid_argument("MoE split accumulator job span");
    const size_t rows = size_t(plan.rows()), first_rows = size_t(primary_rows);
    const size_t offset = size_t(job.global_row), n = count(job.rows, N);
    accumulator_spans(stream, {{primary, first_rows * N * 4},
        {tail, (rows - first_rows) * N * 4}, {status, 4}},
        {{down, n * 2}, {source, routes * 4}, {original, routes * 4}, {job_weights, routes * 4}});
    for_each(n, stream, [=](size_t i) {
        const size_t routed = offset + i / N, col = i % N;
        const int32_t token = source[routed], slot = original[routed];
        if (slot < 0 || size_t(slot) >= routes) { accumulator_error(status, 1); return; }
        if (token < 0 || size_t(token) >= rows || token != slot / 10) {
            accumulator_error(status, 2); return;
        }
        const bool in_tail = size_t(token) >= first_rows;
        float* bank = in_tail ? tail : primary;
        const size_t out = (size_t(token) - (in_tail ? first_rows : 0)) * N + col;
        bank[out] = sycl::fma(job_weights[routed], load(down[i]), bank[out]);
    });
}

void moe_finish(const float* accumulator, const uint16_t* shared, const float* sg,
                uint16_t* output, int64_t rows, void* stream) {
    if (rows > 8192) throw std::invalid_argument("MoE accumulator epilogue exceeds operator capacity");
    const size_t n = count(rows, N);
    accumulator_spans(stream, {{output, n * 2}}, {{accumulator, n * 4}, {shared, n * 2}, {sg, size_t(rows) * 4}});
    for_each(n, stream, [=](size_t i) {
        output[i] = store(accumulator[i] + load(shared[i]) * sigmoid(sg[i / N]));
    });
}
} // namespace strata::prefill::native16
