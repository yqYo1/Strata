// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Niko1221 and the Strata contributors
#include "strata/prefill/qsa_native16.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl/launch.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <sycl/ext/intel/math.hpp>

namespace strata::prefill::native16::qsa {
namespace {
using namespace strata::sycl_backend;
constexpr int HD = 256, NH = 24, KH = 2;
void need(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
size_t bytes(int64_t rows, int64_t ld, size_t element_bytes) {
    const size_t n = checked_count(rows, ld);
    need(n <= SIZE_MAX / element_bytes, "QSA16 byte overflow");
    return n * element_bytes;
}
void domains(sycl::queue& queue, std::initializer_list<Span> outputs,
             std::initializer_list<Span> inputs, size_t alignment = 2) {
    validate_spans(outputs, inputs, alignment);
    for (auto s : outputs) validate_device_span(s.pointer, s.bytes, queue);
    for (auto s : inputs) validate_device_span(s.pointer, s.bytes, queue);
}
float load(uint16_t x) { return strata::kernels::f32_from_bf16(x); }
uint16_t store(float x) { return strata::kernels::bf16_from_f32(x); }
void shape(int64_t tokens, int64_t ld, int64_t width) {
    need(tokens > 0 && tokens <= 8192 && ld >= width,
         "QSA16 unsupported tokens or stride");
}
} // namespace

bool requested() {
    static const bool result = [] {
        const char* value = std::getenv("STRATA_SYCL_PREFILL_NATIVE16_QSA");
        if (!value || !std::strcmp(value, "0")) return false;
        if (!std::strcmp(value, "1")) return true;
        throw std::invalid_argument("STRATA_SYCL_PREFILL_NATIVE16_QSA must be 0 or 1");
    }();
    return result;
}
AttentionMath attention_math() {
    static const AttentionMath result = [] {
        const char* value = std::getenv("STRATA_SYCL_PREFILL_QSA_MATH");
        if (!value || !std::strcmp(value, "existing")) return AttentionMath::existing;
        if (!std::strcmp(value, "bf16")) return AttentionMath::bf16;
        if (!std::strcmp(value, "bf16-prefix")) return AttentionMath::bf16_prefix;
        if (!std::strcmp(value, "bf16-prefix-optimized")) return AttentionMath::bf16_prefix_optimized;
        if (!std::strcmp(value, "bf16-prefix-qcache")) return AttentionMath::bf16_prefix_qcache;
        if (!std::strcmp(value, "bf16-prefix-kvec4")) return AttentionMath::bf16_prefix_kvec4;
        if (!std::strcmp(value, "bf16-prefix-kvvec4")) return AttentionMath::bf16_prefix_kvvec4;
        if (!std::strcmp(value, "bf16-prefix-ordered-exp")) return AttentionMath::bf16_prefix_ordered_exp;
        if (!std::strcmp(value, "bf16-prefix-pair")) return AttentionMath::bf16_prefix_pair;
        if (!std::strcmp(value, "bf16-prefix-tilepack")) return AttentionMath::bf16_prefix_tilepack;
        if (!std::strcmp(value, "bf16-prefix-subgroup")) return AttentionMath::bf16_prefix_subgroup;
        throw std::invalid_argument("STRATA_SYCL_PREFILL_QSA_MATH must be existing, bf16, bf16-prefix, bf16-prefix-optimized, bf16-prefix-qcache, bf16-prefix-kvec4, bf16-prefix-kvvec4, bf16-prefix-ordered-exp, bf16-prefix-pair, bf16-prefix-tilepack or bf16-prefix-subgroup");
    }();
    return result;
}
void validate_device(void* stream) {
    auto& queue = queue_for(stream);
    const auto device = queue.get_device();
    const auto subgroups = device.get_info<sycl::info::device::sub_group_sizes>();
    need(device.is_gpu() && device.has(sycl::aspect::fp16) &&
         device.get_info<sycl::info::device::max_work_group_size>() >= HD &&
         device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0] >= HD &&
         std::find(subgroups.begin(), subgroups.end(), 32) != subgroups.end(),
         "QSA16 requires GPU, FP16, WG256 and SG32");
}
void norm_rope_split(const uint16_t* x, const float* w, uint16_t* y,
                     int64_t tokens, int64_t heads, int64_t head_stride,
                     int64_t input_ld, int64_t output_ld, int64_t pos0,
                     int64_t table_rows, const float* cosine, const float* sine,
                     float eps, void* stream) {
    need((heads == KH || heads == NH) && head_stride >= HD &&
         head_stride <= INT64_MAX / heads, "QSA16 unsupported head geometry");
    shape(tokens, input_ld, heads * head_stride);
    shape(tokens, output_ld, heads * HD);
    need(pos0 >= 0 && table_rows > 0 && table_rows <= 262144 &&
         pos0 <= table_rows && tokens <= table_rows - pos0 &&
         std::isfinite(eps) && eps >= 0, "QSA16 invalid RoPE extent/epsilon");
    validate_device(stream);
    auto& queue = queue_for(stream);
    domains(queue, {{y, bytes(tokens, output_ld, 2)}},
        {{x, bytes(tokens, input_ld, 2)}, {w, HD * 4},
         {cosine, bytes(table_rows, 32, 4)}, {sine, bytes(table_rows, 32, 4)}});
    // Float table/weight alignment is checked independently of BF16 views.
    validate_spans({}, {{w, HD * 4}, {cosine, bytes(table_rows, 32, 4)},
                        {sine, bytes(table_rows, 32, 4)}}, 4);
    auto event = queue.parallel_for(
        sycl::nd_range<1>(size_t(tokens) * heads * HD, HD),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const size_t row = it.get_group_linear_id(), t = row / heads, h = row % heads;
            const int d = it.get_local_linear_id();
            const size_t begin = t * input_ld + h * head_stride;
            const float value = load(x[begin + d]);
            const int bad = sycl::reduce_over_group(it.get_group(),
                sycl::isfinite(value) ? 0 : 1, sycl::maximum<int>());
            const float maximum = sycl::reduce_over_group(it.get_group(),
                sycl::isfinite(value) ? sycl::fabs(value) : 0.f, sycl::maximum<float>());
            // Scale before squaring: BF16's full finite range must not overflow
            // the FP32 RMS reduction. This is local FP32, not a tensor restore.
            const float scale = sycl::fmax(maximum, sycl::sqrt(eps));
            const float a = scale > 0.f ? value / scale : 0.f;
            const float ss = sycl::reduce_over_group(it.get_group(), a * a, sycl::plus<float>());
            const float reciprocal = sycl::rsqrt(ss / float(HD) +
                (scale > 0.f ? (eps / scale) / scale : 0.f));
            float result = (a * reciprocal) * w[d];
            if (d < 64) {
                const int pair = d % 32;
                const float u = ((load(x[begin + pair]) / scale) * reciprocal) * w[pair];
                const float v = ((load(x[begin + pair + 32]) / scale) * reciprocal) * w[pair + 32];
                const float c = cosine[(pos0 + t) * 32 + pair], s = sine[(pos0 + t) * 32 + pair];
                result = d < 32 ? u * c - v * s : u * s + v * c;
            }
            // Diagnose nonfinite source for the entire normalized head.
            y[t * output_ld + h * HD + d] = bad ? uint16_t(0x7fc0) : store(result);
        });
    finish(stream, event);
}
void gate(const uint16_t* a, const uint16_t* q, uint16_t* y, int64_t tokens,
          int64_t attention_ld, int64_t query_ld, int64_t output_ld, void* stream) {
    shape(tokens, attention_ld, NH * HD); shape(tokens, query_ld, NH * 2 * HD);
    shape(tokens, output_ld, NH * HD);
    validate_device(stream);
    auto& queue = queue_for(stream);
    domains(queue, {{y, bytes(tokens, output_ld, 2)}},
        {{a, bytes(tokens, attention_ld, 2)}, {q, bytes(tokens, query_ld, 2)}});
    auto event = queue.parallel_for(sycl::range<1>(size_t(tokens) * NH * HD),
        [=](sycl::id<1> id) {
            const size_t i = id[0], t = i / (NH * HD), h = i / HD % NH, d = i % HD;
            const float z = load(q[t * query_ld + h * 2 * HD + HD + d]);
            const float sigmoid = 1.f / (1.f + sycl::exp(-z));
            const float result = load(a[t * attention_ld + h * HD + d]) * sigmoid;
            y[t * output_ld + h * HD + d] = strata::kernels::f16_from_f32(result);
        });
    finish(stream, event);
}
void append_q8(const uint16_t* k, const uint16_t* v, int64_t tokens,
               int64_t input_ld, int64_t pos0,
               const Q8WritePools& p,
               const strata::kernels::QsaShapes& s,
               const strata::kernels::QsaPromptBf16Bounds& bounds, void* stream) {
    shape(tokens, input_ld, KH * HD);
    need(s.n_head == NH && s.n_head_kv == KH && s.head_dim == HD &&
         s.page_size > 0 && s.page_size <= 262144 &&
         bounds.logical_pages > 0 && bounds.logical_pages <= 262144 &&
         bounds.physical_pages > 0 && bounds.physical_pages <= 262144,
         "QSA16 append requires resident INT8 pools");
    const int64_t cells = int64_t(checked_count(bounds.logical_pages, s.page_size));
    need(cells <= 262144 && pos0 >= 0 && pos0 <= cells && tokens <= cells - pos0,
         "QSA16 append exceeds logical table");
    validate_device(stream);
    auto& queue = queue_for(stream);
    const int64_t rows = int64_t(checked_count(checked_count(bounds.physical_pages, KH), s.page_size));
    const Span outputs[] = {{p.k_q, bytes(rows, HD, 1)}, {p.v_q, bytes(rows, HD, 1)},
        {p.k_scale, bytes(rows, 4, 2)}, {p.v_scale, bytes(rows, 4, 2)}};
    const Span inputs[] = {{k, bytes(tokens, input_ld, 2)}, {v, bytes(tokens, input_ld, 2)},
        {p.page_table, bytes(bounds.logical_pages, 1, 4)}};
    // Identical K/V or scale outputs are not a folded-pool opt-in.
    domains(queue, {outputs[0], outputs[1], outputs[2], outputs[3]},
        {inputs[0], inputs[1], inputs[2]}, 1);
    validate_spans({}, {{k, inputs[0].bytes}, {v, inputs[1].bytes}, outputs[2], outputs[3]}, 2);
    validate_spans({}, {inputs[2]}, 4);
    const auto pools = p; const auto shapes = s; const auto b = bounds;
    auto event = queue.parallel_for(
        sycl::nd_range<1>(size_t(tokens) * 2 * KH * 4 * 64, 64),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const size_t group = it.get_group_linear_id();
            const int lane = it.get_local_linear_id(), g = group % 4;
            const size_t h = group / 4 % KH, which = group / (4 * KH) % 2,
                         t = group / (4 * KH * 2);
            const float x = load((which ? v : k)[t * input_ld + h * HD + g * 64 + lane]);
            const int bad = sycl::reduce_over_group(it.get_group(),
                sycl::isfinite(x) ? 0 : 1, sycl::maximum<int>());
            const float maximum = sycl::reduce_over_group(it.get_group(),
                sycl::isfinite(x) ? sycl::fabs(x) : 0.f, sycl::maximum<float>());
            const uint16_t scale_bits = bad ? uint16_t(0x7e00) :
                strata::kernels::f16_from_f32(sycl::ext::intel::math::fdiv_rn(maximum, 127.f));
            const float scale = strata::kernels::f32_from_f16(scale_bits);
            // Never convert NaN/Inf to int. Preserve their diagnostic in scale.
            const int code = !bad && sycl::isfinite(scale) && scale > 0.f ?
                int(sycl::clamp(sycl::rint(sycl::ext::intel::math::fdiv_rn(x, scale)), -127.f, 127.f)) : 0;
            const int64_t pos = pos0 + t, logical = pos / shapes.page_size;
            const int64_t physical = pools.page_table[logical];
            // Uniform per workgroup, after all group functions. No divergent
            // barriers or collectives, even for absent/out-of-range pages.
            if (physical >= 0 && physical < b.physical_pages) {
                const size_t row = (physical * KH + h) * shapes.page_size + pos % shapes.page_size;
                (which ? pools.v_q : pools.k_q)[row * HD + g * 64 + lane] = int8_t(code);
                if (!lane) (which ? pools.v_scale : pools.k_scale)[row * 4 + g] = scale_bits;
            }
        });
    finish(stream, event);
}
} // namespace strata::prefill::native16::qsa
