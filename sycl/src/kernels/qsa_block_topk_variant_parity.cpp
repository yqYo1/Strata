// Selected IDs must match an independent stable CPU sort and the original GPU selector.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/qsa_select_variant.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
namespace k = strata::kernels;
uint32_t key(float x) {
    if (std::isnan(x))
        return 0;
    // Signed zero is one tie, as required by the existing selection contract.
    if (x == 0)
        return 0x80000000u;
    auto u = std::bit_cast<uint32_t>(x);
    return (u >> 31) ? ~u : u | 0x80000000u;
}
std::vector<int32_t> oracle(const float *scores, int n, int width) {
    std::vector<int32_t> cells(n);
    std::iota(cells.begin(), cells.end(), 0);
    std::stable_sort(cells.begin(), cells.end(), [&](int a, int b) { return key(scores[a / 4]) > key(scores[b / 4]); });
    cells.resize(width);
    std::sort(cells.begin(), cells.end());
    return cells;
}
double median(std::vector<double> a) {
    std::sort(a.begin(), a.end());
    return a[a.size() / 2];
}
struct Result {
    bool ok;
    double original, tuned;
};
Result run(int ctx, int nq, int capacity, int mode, int reps, bool counted) {
    const auto s = k::qsa_real_shapes();
    const int stride = capacity / 4 + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<float> scores(size_t(nq) * stride, -12345.f);
    std::vector<int32_t> steps(size_t(nq) * k::kStepCount), host_ref(size_t(nq) * cap), host_new(host_ref.size());
    std::mt19937 rng(84931 + ctx + mode);
    std::normal_distribution<float> dist(0, 2);
    for (int i = 0; i < nq; i++) {
        const int n = ctx - nq + i + 1;
        auto *st = steps.data() + i * k::kStepCount;
        st[k::kStepPos] = n - 1;
        st[k::kStepNKv] = n;
        st[k::kStepNBid] = n / 4;
        st[k::kStepWidth] = k::qsa_selection_width(n, s);
        for (int b = 0; b <= n / 4; b++) {
            float x = dist(rng);
            if (mode == 1)
                x = std::floor(x * 4) / 4;
            if (mode == 2)
                x = .25f;
            if (mode == 3) {
                const float edge[] = {-INFINITY, INFINITY, NAN, -0.f,
                                      0.f,       -1.f,     1.f, std::numeric_limits<float>::denorm_min()};
                x = edge[b % 8];
            }
            scores[size_t(i) * stride + b] = x;
        }
        if (n % 4 && mode < 3)
            scores[size_t(i) * stride + n / 4] += 1e9f;
    }
    auto &q = *strata::q_of(nullptr);
    float *ds = sycl::malloc_device<float>(scores.size(), q);
    int32_t *dt = sycl::malloc_device<int32_t>(steps.size(), q);
    int32_t *dr = sycl::malloc_device<int32_t>(host_ref.size(), q);
    int32_t *dn = sycl::malloc_device<int32_t>(host_ref.size(), q);
    q.memcpy(ds, scores.data(), scores.size() * 4).wait();
    q.memcpy(dt, steps.data(), steps.size() * 4).wait();
    const int64_t active = counted ? ctx / 4 + 1 : -1;
    auto old_run = [&] { k::qsa_block_topk(ds, dt, nq, stride, cap, s, dr, nullptr, active); };
    bool launched = false;
    auto new_run = [&] {
        launched = k::qsa_block_topk_prompt_variant(ds, dt, nq, stride, cap, s, dn, nullptr, active);
        if (launched != (counted ? ctx / 4 + 1 <= 256 * 9 : stride <= 256 * 9)) {
            std::fprintf(stderr, "FAIL variant eligibility ctx=%d capacity=%d counted=%d\n", ctx, capacity, counted);
            std::exit(2);
        }
        if (!launched)
            k::qsa_block_topk(ds, dt, nq, stride, cap, s, dn, nullptr, active);
    };
    old_run();
    new_run();
    q.wait_and_throw();
    q.memcpy(host_ref.data(), dr, host_ref.size() * 4).wait();
    q.memcpy(host_new.data(), dn, host_new.size() * 4).wait();
    bool ok = true;
    for (int i = 0; i < nq; i++) {
        const auto *st = steps.data() + i * k::kStepCount;
        auto expect = oracle(scores.data() + size_t(i) * stride, st[k::kStepNKv], st[k::kStepWidth]);
        const auto a = host_ref.begin() + size_t(i) * cap, b = host_new.begin() + size_t(i) * cap;
        if (!std::equal(expect.begin(), expect.end(), a) || !std::equal(expect.begin(), expect.end(), b)) {
            std::fprintf(stderr, "FAIL ctx=%d capacity=%d mode=%d query=%d\n", ctx, capacity, mode, i);
            ok = false;
            break;
        }
    }
    auto time_run = [&](auto fn) {
        q.wait_and_throw();
        auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++)
            fn();
        q.wait_and_throw();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / reps;
    };
    std::vector<double> old_times, new_times;
    for (int round = 0; round < 3; round++) {
        if (round % 2) {
            new_times.push_back(time_run(new_run));
            old_times.push_back(time_run(old_run));
        } else {
            old_times.push_back(time_run(old_run));
            new_times.push_back(time_run(new_run));
        }
    }
    sycl::free(ds, q);
    sycl::free(dt, q);
    sycl::free(dr, q);
    sycl::free(dn, q);
    const double old = median(old_times), candidate = median(new_times);
    std::printf("%s ctx=%d nq=%d capacity=%d mode=%d counted=%d launched=%d original=%.6fms variant=%.6fms "
                "IDs=CPU_sort_and_original_exact\n",
                ok ? "PASS" : "FAIL", ctx, nq, capacity, mode, counted, launched, old, candidate);
    std::fflush(stdout);
    return {ok, old, candidate};
}
int main(int argc, char **argv) {
    int ctx = argc > 1 ? std::atoi(argv[1]) : 8192, nq = argc > 2 ? std::atoi(argv[2]) : 129,
        reps = argc > 3 ? std::atoi(argv[3]) : 3;
    if (ctx < nq || nq < 1 || reps < 1)
        return 2;
    for (int mode = 0; mode < 4; mode++)
        if (!run(ctx, nq, ctx, mode, reps, true).ok)
            return 1;
    // Identity selection, long capacity with/without a live bound, and both sides
    // of the nine-key eligibility boundary. Padding is outside the API contract.
    for (int mode = 0; mode < 4; mode++) {
        if (!run(1500, 65, 8192, mode, reps, false).ok)
            return 1;
        if (!run(8192, 129, 262144, mode, reps, true).ok)
            return 1;
        if (!run(8192, 65, 262144, mode, reps, false).ok)
            return 1;
        if (!run(9215, 65, 9215, mode, reps, true).ok)
            return 1;
        if (!run(9216, 65, 9216, mode, reps, true).ok)
            return 1;
        if (!run(32768, 65, 32768, mode, reps, false).ok)
            return 1;
    }
    std::puts("PASS all 28 cases: original and tuned match independent CPU sort");
}
