#include "strata/sycl_upstream/mmq_adapter.hpp"
#include "strata/sycl_upstream/mmq_quantize.hpp"
#include "upstream_cpu_dequant.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <vector>
namespace api = strata::prefill::mmq;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid input was accepted");
}
int main() try {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    sycl::queue other(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
    sycl::queue alias = q;
    sycl::queue unordered(q.get_context(), q.get_device());
    std::cout << "device=" << q.get_device().get_info<sycl::info::device::name>() << '\n';
    check(api::built(), "built");
    int supported_count = 0, sizes_checked = 0;
    for (int t = -1; t <= GGML_TYPE_COUNT; ++t) {
        const bool expected = t == GGML_TYPE_Q2_0 || t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ2_XS ||
            t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ3_XXS || t == GGML_TYPE_IQ3_S ||
            t == GGML_TYPE_IQ4_NL || t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_Q8_0;
        check(api::supported(t) == expected, "supported format set");
        check(api::fits(t, 19) == expected, "visible GPU fits");
        check(strata::sycl_upstream::mmq_device_fits(q.get_device(), t, 19) == expected, "device tile fits");
        supported_count += expected;
    }
    check(!api::fits(GGML_TYPE_Q8_0, 0) && !api::fits(GGML_TYPE_Q8_0, int64_t(INT_MAX) + 1), "fits geometry");
    // Generic GGML type-size API must not silently shrink to the nine MMQ formats.
    for (auto t : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K,
                   GGML_TYPE_Q5_K, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XS}) {
        const int64_t block = ggml_blck_size(t);
        for (int rows : {0, 1, 19}) {
            check(api::matrix_bytes(t, rows, block * 3) == size_t(rows) * 3 * ggml_type_size(t), "matrix byte count");
            ++sizes_checked;
        }
    }
    check(api::q8_bytes(7, 640) == 7 * 1024 / 128 * 144 + 128 * 144, "q8 padded bytes");
    constexpr int cols = 1024, wr = 19, total = 5, ld = 23, experts = 2, runs = 24;
    const size_t expert_bytes = api::matrix_bytes(GGML_TYPE_Q8_0, wr, cols);
    auto* w = sycl::malloc_shared<block_q8_0>(experts * wr * cols / 32 + 128, q);
    auto* x = sycl::malloc_shared<float>((runs + 3) * total * cols, q);
    auto* xq = sycl::malloc_shared<uint8_t>((runs + 3) * api::q8_bytes(total, cols), q);
    auto* bounds = sycl::malloc_shared<int32_t>(experts + 1, q);
    auto* ids = sycl::malloc_shared<int32_t>(total, q);
    auto* dst = sycl::malloc_shared<float>((runs + 8) * total * ld, q);
    check(w && x && xq && bounds && ids && dst, "allocation");
    for (int b = 0; b < experts * wr * cols / 32 + 128; ++b) {
        w[b].d = sycl::half(0.0625f);
        for (int j = 0; j < 32; ++j) w[b].qs[j] = int8_t((b + j) % 7 - 3);
    }
    for (int run = 0; run < runs + 3; ++run) for (int r = 0; r < total; ++r) for (int k = 0; k < cols; ++k)
        x[(run * total + r) * cols + k] = float(k % 32 == 31 ? 127 : k % 61 - 30) * std::ldexp(1.f, r + run);
    bounds[0] = 0; bounds[1] = 2; bounds[2] = total;
    for (int r = 0; r < total; ++r) ids[r] = total - r - 1;
    std::fill(dst, dst + (runs + 8) * total * ld, -1234567.f);
    for (int run = 0; run < runs + 3; ++run)
        api::quantize(x + run * total * cols, nullptr, xq + run * api::q8_bytes(total, cols), GGML_TYPE_Q8_0, cols, cols, total, &q);
    q.wait_and_throw(); // Initial immutable tensors are shared by two independent queues.
    const api::Product base{w,GGML_TYPE_Q8_0,wr,cols,expert_bytes,experts,xq,bounds,ids,total,3,dst,ld};
    {
        api::Context context;
        api::Product empty; context.run(empty, nullptr); // Upstream no-op guards precede stream lookup.
        api::quantize(nullptr,nullptr,nullptr,GGML_TYPE_Q8_0,cols,cols,0,nullptr);
        api::swiglu(nullptr,nullptr,0,512,false,nullptr); api::iota(nullptr,0,nullptr);
        api::GatherGroup rejected;
        check(!api::gather_native_group(rejected,0,16,32,16,nullptr,32,nullptr,16,nullptr), "empty group no-op");
        rejected.n = 1;
        check(!api::gather_native_group(rejected,1,16,32,16,nullptr,32,nullptr,16,nullptr), "unaligned group no-op");
        rejects([&]{ context.run(base, nullptr); });
        rejects([&]{ context.run(base, &unordered); });
        auto bad = base; bad.w_rows = int64_t(INT_MAX) + 1;
        rejects([&]{ context.run(bad, &q); });
        bad = base; bad.type = GGML_TYPE_Q4_K;
        rejects([&]{ context.run(bad, &q); });
        for (int r = 0; r < runs / 2; ++r) {
            auto p = base; p.dst = dst + r * total * ld; p.xq = xq + r * api::q8_bytes(total, cols);
            context.run(p, r % 3 == 0 ? &q : r % 3 == 1 ? &other : &alias);
        }
        // Two host submitters must not interleave main/fixup pairs sharing one
        // queue's scratch. Their differently scaled inputs expose cross-use.
        auto submit = [&](int parity) {
            for (int r = runs / 2 + parity; r < runs; r += 2) {
                auto p = base; p.dst = dst + r * total * ld; p.xq = xq + r * api::q8_bytes(total, cols);
                context.run(p, &q);
            }
        };
        auto first = std::async(std::launch::async, submit, 0);
        auto second = std::async(std::launch::async, submit, 1);
        first.get(); second.get();
        // No caller wait: destructor must retain scratch through the final fixup.
    }
    size_t comparisons = 0;
    auto verify = [&](int run) {
        for (int e = 0; e < experts; ++e) for (int token = bounds[e]; token < bounds[e + 1]; ++token)
            for (int r = 0; r < wr; ++r) {
                double ref = 0;
                for (int k = 0; k < cols; ++k) {
                    const auto& b = w[(e * wr + r) * cols / 32 + k / 32];
                    ref += double(float(b.d)) * b.qs[k % 32] * x[(run * total + token) * cols + k];
                }
                // All input scales are exact powers of two; quantization and
                // accumulation are exact for this fixture (not a loose tolerance).
                const float actual = dst[run * total * ld + ids[token] * ld + r];
                if (actual != float(ref)) {
                    std::cerr << "run=" << run << " expert=" << e << " token=" << token << " row=" << r
                              << " actual=" << actual << " reference=" << ref << " error=" << double(actual) - ref << '\n';
                    throw std::runtime_error("context product/fixup");
                }
                ++comparisons;
            }
        for (int t = 0; t < total; ++t) for (int j = wr; j < ld; ++j)
            check(dst[run * total * ld + t * ld + j] == -1234567.f, "context output stride guard");
    };
    for (int r = 0; r < runs; ++r) verify(r);
    {
        api::Context context;
        auto p = base; p.dst = dst + runs * total * ld; p.xq = xq + runs * api::q8_bytes(total, cols);
        context.run(p, &q); q.wait_and_throw(); // Warm the retained scratch and device code.
        std::promise<void> release;
        const auto gate = release.get_future().share();
        q.submit([&](sycl::handler& h){ h.host_task([gate]{ gate.wait(); }); });
        auto submitted = std::async(std::launch::async, [&]{
            auto pending = base; pending.dst = dst + (runs + 1) * total * ld; pending.xq = xq + (runs + 1) * api::q8_bytes(total, cols);
            context.run(pending, &q);
        });
        const bool asynchronous = submitted.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        release.set_value(); submitted.get(); q.wait_and_throw();
        check(asynchronous, "run waited for previous queue work");
    }
    verify(runs); verify(runs + 1);
    {
        api::Context context;
        { // Context's retained queue copy must outlive this host queue wrapper.
            sycl::queue temporary(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
            auto p = base; p.dst = dst + (runs + 2) * total * ld; p.xq = xq + (runs + 2) * api::q8_bytes(total, cols);
            context.run(p, &temporary);
        }
    }
    verify(runs + 2);
    sycl::free(w,q); sycl::free(x,q); sycl::free(xq,q); sycl::free(bounds,q); sycl::free(ids,q); sycl::free(dst,q);
    std::cout << "PASS supported=" << supported_count << " generic_matrix_size_cases=" << sizes_checked
              << " exact_products=" << comparisons << " queue_lifetime_runs=" << runs + 3
              << " concurrent_host_submissions=12 blocked_queue_nonblocking=1\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
