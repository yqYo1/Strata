#include "mmq_test_api.hpp"
#include "strata/sycl_upstream/mmq_product.hpp"
#include "upstream_cpu_dequant.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata::sycl_upstream;
void check(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
template<class T> struct Buffer {
    sycl::queue& q; T* p; size_t n;
    Buffer(sycl::queue& q, size_t n): q(q), p(sycl::malloc_shared<T>(n, q)), n(n) { check(p, "allocation"); }
    ~Buffer() { sycl::free(p, q); }
    void fill(T v) { std::fill(p, p + n, v); }
};
struct Kind { ggml_type type; const char* name; int qk, bytes; void (*decode)(const void*, float*, int64_t); };
#define KIND(T,N,Q) Kind{T,#N,Q,sizeof(block_##N), [](const void* p,float* f,int64_t n){ upstream_cpu::dequantize_row_##N(static_cast<const block_##N*>(p),f,n); }}
const Kind kinds[] = {
    KIND(GGML_TYPE_Q2_0,q2_0,QK2_0), KIND(GGML_TYPE_IQ2_XXS,iq2_xxs,QK_K),
    KIND(GGML_TYPE_IQ2_XS,iq2_xs,QK_K), KIND(GGML_TYPE_IQ2_S,iq2_s,QK_K),
    KIND(GGML_TYPE_IQ3_XXS,iq3_xxs,QK_K), KIND(GGML_TYPE_IQ3_S,iq3_s,QK_K),
    KIND(GGML_TYPE_IQ4_NL,iq4_nl,QK4_NL), KIND(GGML_TYPE_IQ4_XS,iq4_xs,QK_K),
    KIND(GGML_TYPE_Q8_0,q8_0,QK8_0)};
#undef KIND
int main() try {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << '\n';
    size_t bytes_checked = 0, numeric_checked = 0;
    double max_swiglu = 0, max_product_l1 = 0;
    // Each pointer's alignment independently selects the byte fallback.
    for (int mode = 0; mode < 7; ++mode) {
        const size_t na = mode == 6 ? 277 : 288, nc = mode == 6 ? 143 : 160;
        Buffer<uint8_t> a(q, 640), b(q, 640), c(q, 640), gu(q, 1024), dn(q, 1024);
        for (size_t j = 0; j < 640; ++j) { a.p[j] = j * 17; b.p[j] = j * 13 + 3; c.p[j] = j * 7 + 11; }
        gu.fill(0xce); dn.fill(0xce);
        const int ao = mode == 1, bo = mode == 2, co = mode == 3, go = 32 + (mode == 4), dno = 32 + (mode == 5);
        tested::mmq_gather_native(q, a.p + ao, b.p + bo, na, c.p + co, nc, gu.p + go, dn.p + dno).wait_and_throw();
        for (size_t j = 0; j < gu.n; ++j) {
            const uint8_t expected = j < go || j >= go + 2 * na ? 0xce :
                j < go + na ? a.p[ao + j - go] : b.p[bo + j - go - na];
            check(gu.p[j] == expected, "native gather/guard"); ++bytes_checked;
        }
        for (size_t j = 0; j < dn.n; ++j) {
            check(dn.p[j] == (j < dno || j >= dno + nc ? 0xce : c.p[co + j - dno]), "down gather/guard"); ++bytes_checked;
        }
    }
    {
        constexpr size_t na = 288, nc = 160, up = 320, down = 640, gs = 640, ds = 192;
        Buffer<uint8_t> blobs(q, 16 * 1024), gu(q, 16 * gs + 64), dn(q, 16 * ds + 64);
        for (size_t j = 0; j < blobs.n; ++j) blobs.p[j] = uint8_t(j * 31 + j / 1024);
        strata::prefill::mmq::GatherGroup g;
        for (int e = 0; e < 16; ++e) g.blob[e] = blobs.p + e * 1024;
        g.first = 3; g.n = 16; gu.fill(0xce); dn.fill(0xce);
        check(tested::mmq_gather_native_group(q, g, up, na, down, nc, gu.p + 32, gs, dn.p + 32, ds), "group rejected");
        q.wait_and_throw();
        for (int out = 0; out < 2; ++out) {
            const auto& buf = out ? dn : gu; const size_t stride = out ? ds : gs, active = out ? nc : 2 * na;
            for (size_t j = 0; j < buf.n; ++j) {
                uint8_t expected = 0xce;
                if (j >= 32) { size_t e = (j - 32) / stride, k = (j - 32) % stride;
                    if (e >= 3 && e < 16 && k < active) expected = g.blob[e][out ? down + k : k < na ? k : up + k - na]; }
                check(buf.p[j] == expected, "group indexed stride/guard"); ++bytes_checked;
            }
        }
        // Every alignment field and pointer, plus all invalid range categories.
        for (int bad = 0; bad < 13; ++bad) {
            gu.fill(0xce); dn.fill(0xce); auto invalid = g;
            if (bad == 0) invalid.first = -1;
            if (bad == 1) invalid.n = invalid.first;
            if (bad == 2) invalid.n = 17;
            if (bad == 3) invalid.blob[9]++;
            bool accepted = tested::mmq_gather_native_group(q, invalid, up + (bad == 4), na + (bad == 5),
                down + (bad == 6), nc + (bad == 7), gu.p + (bad == 8), gs + (bad == 9),
                dn.p + (bad == 10), ds + (bad == 11));
            if (bad == 12) { check(accepted, "valid group"); q.wait_and_throw(); continue; }
            check(!accepted, "invalid group accepted"); q.wait_and_throw();
            check(std::all_of(gu.p, gu.p + gu.n, [](auto b){ return b == 0xce; }) &&
                  std::all_of(dn.p, dn.p + dn.n, [](auto b){ return b == 0xce; }), "rejected group wrote output");
        }
    }
    {
        constexpr size_t gc = 1280 * 640, dc = 2560 * 160, ng = 1280 * 40, nd = 2560 * 10;
        Buffer<uint8_t> blob(q, gc + dc + (ng + nd) * 2), gu(q, ng * 18 + 64), dn(q, nd * 18 + 64);
        for (size_t j = 0; j < blob.n; ++j) blob.p[j] = uint8_t((j * 17) ^ (j >> 8));
        gu.fill(0xce); dn.fill(0xce);
        tested::mmq_gather_strata_q2(q, blob.p, gu.p + 32, dn.p + 32).wait_and_throw();
        for (int out = 0; out < 2; ++out) {
            const auto& buf = out ? dn : gu;
            for (size_t j = 0; j < buf.n; ++j) {
                uint8_t expected = 0xce;
                if (j >= 32 && j < buf.n - 32) { size_t b = (j - 32) / 18, k = (j - 32) % 18;
                    expected = k < 2 ? blob.p[gc + dc + (out ? ng * 2 : 0) + b * 2 + k] : blob.p[(out ? gc : 0) + b * 16 + k - 2]; }
                check(buf.p[j] == expected, "Q2 repack/guard"); ++bytes_checked;
            }
        }
    }
    for (bool interleaved : {false, true}) {
        constexpr int n = 3 * 257;
        Buffer<float> gu(q, 2 * n), h(q, n + 32); h.fill(-1234567.f);
        for (int i = 0; i < n; ++i) {
            const float g = float(i % 241 - 120), u = float(i % 29 - 14) / 7;
            const int base = i / 257 * 514, k = i % 257;
            gu.p[base + (interleaved ? 2 * k : k)] = g;
            gu.p[base + (interleaved ? 2 * k + 1 : 257 + k)] = u;
        }
        tested::mmq_swiglu(q, gu.p, h.p, 3, 257, interleaved).wait_and_throw();
        for (int i = 0; i < n; ++i) {
            const double g = i % 241 - 120, u = float(i % 29 - 14) / 7;
            const double ref = g / (1 + std::exp(-g)) * u, error = std::abs(h.p[i] - ref);
            check(std::isfinite(h.p[i]) && error <= 3e-6 * std::abs(ref) + 1e-6, "SwiGLU exp/reference");
            max_swiglu = std::max(max_swiglu, error); ++numeric_checked;
        }
        for (int i = n; i < n + 32; ++i) check(h.p[i] == -1234567.f, "SwiGLU guard");
        tested::mmq_swiglu(q, nullptr, nullptr, 0, 257, interleaved);
    }
    for (int n : {0, 1, 255, 256, 257, 4007}) {
        Buffer<int32_t> ids(q, n + 32); ids.fill(-12345);
        tested::mmq_iota(q, ids.p, n); q.wait_and_throw();
        for (int i = 0; i < n + 32; ++i) check(ids.p[i] == (i < n ? i : -12345), "iota/guard");
    }
    uint32_t rng = 0x489275;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
    for (int fixture = 0; fixture < 10; ++fixture) {
        const bool packed = fixture == 9;
        const auto& kind = kinds[packed ? 0 : fixture];
        const int cols = packed ? 2560 : 512, ff = packed ? 640 : 512;
        const int total = packed ? 1 : 7, experts = packed ? 1 : 3;
        const size_t half_bytes = mmq_matrix_bytes(kind.type, ff, cols);
        const size_t down_bytes = mmq_matrix_bytes(kind.type, cols, ff);
        const size_t gs = 2 * half_bytes + 4096, ds = down_bytes + 4096;
        Buffer<uint8_t> guw(q, experts * gs + 4096), dw(q, experts * ds + 4096);
        // Native blobs have a gap between each matrix; packed Q2 has separate planes.
        const size_t up_off = half_bytes + 64, down_off = up_off + half_bytes + 64;
        const size_t blob_bytes = packed ? 1382400 : down_off + down_bytes + 64;
        Buffer<uint8_t> blob(q, experts * blob_bytes);
        blob.fill(0); guw.fill(0); dw.fill(0);
        if (packed) {
            constexpr size_t codes = 1280 * 640 + 2560 * 160;
            for (size_t j = 0; j < codes; ++j) blob.p[j] = next();
            for (size_t j = codes; j < blob_bytes; j += 2) {
                const uint16_t scale = 0x1400 + next() % 0x800;
                std::memcpy(blob.p + j, &scale, 2);
            }
        } else {
            for (int e = 0; e < experts; ++e) for (int mat = 0; mat < 3; ++mat) {
                const size_t size = mat == 2 ? down_bytes : half_bytes;
                auto* base = blob.p + e * blob_bytes + (mat == 2 ? down_off : mat == 1 ? up_off : 0);
                for (size_t j = 0; j < size; j += kind.bytes) {
                    const uint16_t scale = 0x1400 + next() % 0x800;
                    std::memcpy(base + j, &scale, 2);
                    for (int k = 2; k < kind.bytes; ++k) base[j + k] = next();
                }
            }
        }
        Buffer<float> x(q, total * (cols + 12)), gu(q, total * 2 * ff + 32), h(q, total * ff + 32), dst(q, total * (cols + 3) + 32);
        Buffer<int32_t> identity(q, total), input_ids(q, total), out_ids(q, total), bounds(q, experts + 1);
        Buffer<MmqBlock> xq(q, mmq_q8_bytes(total, cols) / sizeof(MmqBlock)), hq(q, mmq_q8_bytes(total, ff) / sizeof(MmqBlock));
        for (size_t j = 0; j < x.n; ++j) x.p[j] = float(int(next() % 2001) - 1000) / 701.f;
        for (int i = 0; i < total; ++i) { input_ids.p[i] = total - i - 1; out_ids.p[i] = total - i - 1; }
        bounds.p[0] = 0;
        if (packed) bounds.p[1] = total;
        else { bounds.p[1] = 2; bounds.p[2] = 2; bounds.p[3] = total; }
        gu.fill(-1234567.f); h.fill(-1234567.f); dst.fill(-1234567.f);
        MmqProduct pg{guw.p, kind.type, 2 * ff, cols, gs, experts, xq.p, bounds.p, identity.p, total, total, gu.p, 2 * ff};
        MmqProduct pd{dw.p, kind.type, cols, ff, ds, experts, hq.p, bounds.p, out_ids.p, total, total, dst.p, cols + 3};
        const int cu = q.get_device().get_info<sycl::info::device::max_compute_units>();
        const auto gp = mmq_plan(pg, cu), dp = mmq_plan(pd, cu);
        Buffer<float> scratch(q, std::max(gp.scratch_bytes, dp.scratch_bytes) / 4 + 1);
        // No wait/readback between gather, quantizer, products and activation.
        if (packed) tested::mmq_gather_strata_q2(q, blob.p, guw.p, dw.p);
        else {
            strata::prefill::mmq::GatherGroup group; group.n = experts;
            for (int e = 0; e < experts; ++e) group.blob[e] = blob.p + e * blob_bytes;
            check(tested::mmq_gather_native_group(q, group, up_off, half_bytes, down_off, down_bytes, guw.p, gs, dw.p, ds), "chain group alignment");
        }
        tested::mmq_iota(q, identity.p, total);
        tested::mmq_quantize(q, x.p, input_ids.p, xq.p, kind.type, cols, cols + 12, total);
        tested::mmq_product(q, pg, gp, scratch.p);
        tested::mmq_swiglu(q, gu.p, h.p, total, ff, packed);
        tested::mmq_quantize(q, h.p, nullptr, hq.p, kind.type, ff, ff, total);
        tested::mmq_product(q, pd, dp, scratch.p).wait_and_throw();
        // Independent pinned CPU dequantizer + double dot for every output of
        // both products. Actual quantized intermediates are stage inputs, so
        // each failure is attributable without cascading quantization thresholds.
        for (int stage = 0; stage < 2; ++stage) {
            const int kcols = stage ? ff : cols, wr = stage ? cols : 2 * ff;
            const auto* w = stage ? dw.p : guw.p; const size_t stride = stage ? ds : gs;
            const auto* quant = stage ? hq.p : xq.p;
            const auto* result = stage ? dst.p : gu.p;
            std::vector<float> decoded(kcols);
            for (int e = 0; e < experts; ++e) for (int r = 0; r < wr; ++r) {
                kind.decode(w + e * stride + r * size_t(kcols / kind.qk) * kind.bytes, decoded.data(), kcols);
                for (int token = bounds.p[e]; token < bounds.p[e + 1]; ++token) {
                    double ref = 0, l1 = 0;
                    for (int k = 0; k < kcols; ++k) {
                        const auto& b = quant[(k / 128) * total + token];
                        const double term = double(decoded[k]) * b.qs[k % 128] * b.d4[(k % 128) / 32];
                        ref += term; l1 += std::abs(term);
                    }
                    const float actual = result[(stage ? out_ids.p[token] : token) * (stage ? cols + 3 : 2 * ff) + r];
                    const double error = std::abs(actual - ref);
                    check(std::isfinite(actual) && error <= 3e-6 * l1 + 1e-6, "chain CPU product");
                    max_product_l1 = std::max(max_product_l1, error / (l1 + 1e-30)); ++numeric_checked;
                }
            }
        }
        for (int t = 0; t < total; ++t) for (int k = 0; k < ff; ++k) {
            const float g = gu.p[t * 2 * ff + (packed ? 2 * k : k)], u = gu.p[t * 2 * ff + (packed ? 2 * k + 1 : ff + k)];
            const double ref = double(g) / (1 + std::exp(-double(g))) * u;
            const double error = std::abs(h.p[t * ff + k] - ref);
            check(std::isfinite(h.p[t * ff + k]) && error <= 3e-6 * std::abs(ref) + 1e-6, "chain SwiGLU");
            max_swiglu = std::max(max_swiglu, error); ++numeric_checked;
        }
        for (int i = 0; i < 32; ++i) check(gu.p[total * 2 * ff + i] == -1234567.f && h.p[total * ff + i] == -1234567.f && dst.p[total * (cols + 3) + i] == -1234567.f, "chain tail guard");
        for (int t = 0; t < total; ++t) for (int i = cols; i < cols + 3; ++i) check(dst.p[t * (cols + 3) + i] == -1234567.f, "chain stride guard");
        std::cout << "PASS chain=" << kind.name << " packed=" << packed << " gate_groups=" << gp.groups << " down_groups=" << dp.groups << '\n';
    }
    std::cout << "PASS byte_checks=" << bytes_checked << " numeric_checks=" << numeric_checked
              << " max_swiglu_absolute_error=" << max_swiglu << " max_product_error_over_L1=" << max_product_l1 << '\n';
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
