#include "strata/sycl_upstream/mmq_product.hpp"
#include "upstream_cpu_dequant.hpp"
#include "upstream_mmq_loaders.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace strata::sycl_upstream;
struct Kind {
    ggml_type type;
    const char* name;
    int qk, bytes;
    void (*decode)(const void*, float*, int64_t);
};
#define KIND(TYPE, name, QK) Kind{TYPE, #name, QK, sizeof(block_##name), \
    [](const void* p, float* out, int64_t n) { upstream_cpu::dequantize_row_##name(static_cast<const block_##name*>(p), out, n); }}
const Kind kinds[] = {
    KIND(GGML_TYPE_Q2_0, q2_0, QK2_0), KIND(GGML_TYPE_IQ2_XXS, iq2_xxs, QK_K),
    KIND(GGML_TYPE_IQ2_XS, iq2_xs, QK_K), KIND(GGML_TYPE_IQ2_S, iq2_s, QK_K),
    KIND(GGML_TYPE_IQ3_XXS, iq3_xxs, QK_K), KIND(GGML_TYPE_IQ3_S, iq3_s, QK_K),
    KIND(GGML_TYPE_IQ4_NL, iq4_nl, QK4_NL), KIND(GGML_TYPE_IQ4_XS, iq4_xs, QK_K),
    KIND(GGML_TYPE_Q8_0, q8_0, QK8_0)};
#undef KIND
int main() try {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << '\n';
    uint32_t rng = 0x97382981;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
    size_t checked = 0, guards = 0;
    double max_relative_l1 = 0;
    int cases = 0;
    for (const auto& kind : kinds)
    for (int wr : {1, 16, 19, 1280, 2560})
    for (int cols : {256, 640, 1280, 2560}) {
        if (cols % kind.qk) continue;
        if (wr == 1280 && (cols != 2560 ||
            (kind.type != GGML_TYPE_IQ3_S && kind.type != GGML_TYPE_IQ2_S && kind.type != GGML_TYPE_IQ4_XS))) continue;
        if (wr == 2560 && (cols != 640 ||
            (kind.type != GGML_TYPE_Q2_0 && kind.type != GGML_TYPE_IQ4_NL))) continue;
        const int total = wr > 19 ? 4007 : 75;
        constexpr int experts = 4;
        constexpr float sentinel = -1234567.f;
        const int ld = wr + 5;
        const size_t row_bytes = size_t(cols / kind.qk) * kind.bytes;
        const size_t expert_bytes = mmq_matrix_bytes(kind.type, wr, cols) + 4096;
        std::vector<uint8_t> weights(experts * expert_bytes + 4096, 0);
        for (int e = 0; e < experts; ++e) for (int r = 0; r < wr; ++r) for (int b = 0; b < cols / kind.qk; ++b) {
            auto* block = weights.data() + e * expert_bytes + r * row_bytes + b * kind.bytes;
            for (int j = 2; j < kind.bytes; ++j) block[j] = uint8_t(next());
            uint16_t scale = uint16_t(0x2000 + next() % 0x1c00);
            if ((b + r + e) % 7 == 0) scale = 0;
            if (e % 2) scale |= 0x8000;
            std::memcpy(block, &scale, 2);
        }
        std::vector<float> x(size_t(total) * cols);
        for (auto& v : x) v = float(int(next() % 20001) - 10000) / 731.f;
        // Empty expert, one row, 65 rows (crosses J), four rows; leading and
        // trailing unused activation rows test that bounds are respected.
        const std::vector<int32_t> bounds = {2, 2, 3, total - 7, total - 3};
        std::vector<int32_t> ids(total);
        for (int i = 0; i < total; ++i) ids[i] = wr == 16 ? i : i * 37 % total;
        std::vector<float> result(size_t(total) * ld + 32, sentinel);
        std::vector<MmqBlock> quantized(mmq_q8_bytes(total, cols) / sizeof(MmqBlock));
        auto* dw = sycl::malloc_device<uint8_t>(weights.size(), q);
        auto* dx = sycl::malloc_device<float>(x.size(), q);
        auto* dq = sycl::malloc_device<MmqBlock>(quantized.size(), q);
        auto* db = sycl::malloc_device<int32_t>(bounds.size(), q);
        auto* di = sycl::malloc_device<int32_t>(ids.size(), q);
        auto* dst = sycl::malloc_device<float>(result.size(), q);
        if (!dw || !dx || !dq || !db || !di || !dst) throw std::runtime_error("allocation failed");
        q.memcpy(dw, weights.data(), weights.size());
        q.memcpy(dx, x.data(), x.size() * 4);
        q.memset(dq, 0, quantized.size() * sizeof(MmqBlock));
        q.memcpy(db, bounds.data(), bounds.size() * 4);
        q.memcpy(di, ids.data(), ids.size() * 4);
        q.memcpy(dst, result.data(), result.size() * 4);
        mmq_quantize(q, dx, nullptr, dq, kind.type, cols, cols, total);
        MmqProduct p{dw, kind.type, wr, cols, expert_bytes, experts, dq, db, di, total, total - 10, dst, ld};
        if (kind.type == GGML_TYPE_IQ2_XS && wr == 1 && cols == 1280) {
            auto* diagnostic = sycl::malloc_shared<float>(cols, q);
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<int, 1> wt(16 * 84, h);
                h.parallel_for(sycl::nd_range<1>(size_t(cols / 256) * 128, 128), [=](sycl::nd_item<1> it) {
                    const int tid = int(it.get_local_linear_id()), block = int(it.get_group_linear_id());
                    auto* w = wt.template get_multi_ptr<sycl::access::decorated::no>().get();
                    original_loaders::ggml_cuda_mmq_load_tiles_iq2_xs<GGML_TYPE_IQ2_XS, 64, true>(
                        reinterpret_cast<const char*>(dw + expert_bytes), w, block, 0, cols / 256, {tid % 32, tid / 32});
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int k = tid; k < 256; k += 128)
                        diagnostic[block * 256 + k] = float(reinterpret_cast<const int8_t*>(w)[k]) *
                            reinterpret_cast<const float*>(w + 64)[k / 16];
                });
            }).wait_and_throw();
            std::vector<float> cpu(cols);
            kind.decode(weights.data() + expert_bytes, cpu.data(), cols);
            int mismatches = 0;
            for (int k = 0; k < cols; ++k) if (cpu[k] != diagnostic[k]) {
                if (mismatches++ < 6) std::cerr << "loader k=" << k << " gpu=" << diagnostic[k] << " cpu=" << cpu[k] << '\n';
            }
            if (mismatches) throw std::runtime_error("IQ2_XS loader regression");
            sycl::free(diagnostic, q);
        }
        mmq_product(q, p);
        q.memcpy(result.data(), dst, result.size() * 4);
        q.memcpy(quantized.data(), dq, quantized.size() * sizeof(MmqBlock)).wait_and_throw();
        std::vector<uint8_t> after(weights.size());
        q.memcpy(after.data(), dw, after.size()).wait_and_throw();
        if (after != weights) throw std::runtime_error("weight input modified");
        sycl::free(dw, q); sycl::free(dx, q); sycl::free(dq, q);
        sycl::free(db, q); sycl::free(di, q); sycl::free(dst, q);
        std::vector<uint8_t> written(result.size(), 0);
        std::vector<float> dequant(cols);
        for (int e = 0; e < experts; ++e) for (int r = 0; r < wr; ++r) {
            kind.decode(weights.data() + e * expert_bytes + r * row_bytes, dequant.data(), cols);
            for (int token = bounds[e]; token < bounds[e + 1]; ++token) {
                const size_t at = size_t(ids[token]) * ld + r;
                written[at] = 1;
                if (!std::isfinite(result[at]) || result[at] == sentinel)
                    throw std::runtime_error("active output unwritten/nonfinite");
                // Complete comparison on all small cases. Full model dimensions
                // additionally sample across channel/token tiles and boundaries.
                if (wr > 19 && ((r > 1 && r < wr - 2 && r % 31 != 0) ||
                    (token != bounds[e] && token != bounds[e + 1] - 1 && token % 127 != 0))) continue;
                double ref = 0, l1 = 0;
                for (int k = 0; k < cols; ++k) {
                    const auto& block = quantized[size_t(k / 128) * total + token];
                    const double xv = double(block.qs[k % 128]) * block.d4[k % 128 / 32];
                    const double term = double(dequant[k]) * xv;
                    ref += term; l1 += std::abs(term);
                }
                const double error = std::abs(result[at] - ref);
                if (!std::isfinite(result[at]) || error > 3e-6 * l1 + 1e-6) {
                    std::cerr << kind.name << " wr=" << wr << " cols=" << cols << " expert=" << e
                              << " token=" << token << " weight_row=" << r << " got=" << result[at]
                              << " ref=" << ref << " error=" << error << " l1=" << l1 << '\n';
                    throw std::runtime_error("MMQ product disagrees with independent CPU dequantization");
                }
                max_relative_l1 = std::max(max_relative_l1, error / (l1 + 1e-30));
                written[at] = 1; ++checked;
            }
        }
        for (size_t i = 0; i < result.size(); ++i) if (!written[i]) {
            if (result[i] != sentinel) throw std::runtime_error("output gap/guard overwritten");
            ++guards;
        }
        ++cases;
        std::cout << "PASS " << kind.name << " wr=" << wr << " cols=" << cols << '\n';
    }
    std::cout << "PASS cases=" << cases << " products=" << checked << " output_guards=" << guards
              << " max_error_over_l1=" << max_relative_l1 << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
