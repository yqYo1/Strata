#include <sycl/sycl.hpp>
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using I = int64_t;
constexpr I N = 2560, F = 640, E = 512, K = 10, GU = 2 * F, WEIGHT_SLOTS = 8;
constexpr I G = 128;
constexpr float SENTINEL = -98765.f;
constexpr uint16_t HSENTINEL = 0x7bff;
float hfloat(uint16_t v) { return float(std::bit_cast<sycl::half>(v)); }
uint16_t hbits(float v) { return std::bit_cast<uint16_t>(sycl::half(v)); }
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <class T> struct Buffer {
    sycl::queue &q;
    I n;
    T *allocation;
    T *data;
    T sentinel;
    Buffer(sycl::queue &queue, I count, T value) : q(queue), n(count), sentinel(value) {
        allocation = sycl::malloc_device<T>(n + 2 * G, q);
        if (!allocation)
            throw std::bad_alloc();
        data = allocation + G;
        q.fill(allocation, sentinel, n + 2 * G);
    }
    ~Buffer() { sycl::free(allocation, q); }
    Buffer(const Buffer &) = delete;
    std::vector<T> read() {
        std::vector<T> v(n + 2 * G);
        q.memcpy(v.data(), allocation, v.size() * sizeof(T)).wait_and_throw();
        return v;
    }
    bool guarded(const std::vector<T> &v) const {
        for (I i = 0; i < G; ++i)
            if (v[i] != sentinel || v[G + n + i] != sentinel)
                return false;
        return true;
    }
};

bool run(sycl::queue &q, I T) {
    std::mt19937 rng(98451 + T);
    std::uniform_real_distribution<float> dist(-0.125f, 0.125f);
    std::vector<uint16_t> x(T * N), wg(WEIGHT_SLOTS * GU * N), wd(WEIGHT_SLOTS * N * F);
    for (auto &v : x)
        v = hbits(dist(rng));
    for (auto &v : wg)
        v = hbits(dist(rng));
    for (auto &v : wd)
        v = hbits(dist(rng));
    std::vector<std::vector<int32_t>> grouped(E);
    for (I t = 0; t < T; ++t) {
        std::vector<int32_t> chosen;
        while (chosen.size() < K) {
            const int32_t e = rng() % E;
            if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) {
                chosen.push_back(e);
                grouped[e].push_back(int32_t(t));
            }
        }
    }
    std::vector<int32_t> src;
    std::vector<I> offset(E + 1);
    I nonempty = 0, max_rows = 0;
    for (I e = 0; e < E; ++e) {
        offset[e] = I(src.size());
        src.insert(src.end(), grouped[e].begin(), grouped[e].end());
        nonempty += !grouped[e].empty();
        max_rows = std::max(max_rows, I(grouped[e].size()));
    }
    offset[E] = src.size();
    if (I(src.size()) != T * K || max_rows > T)
        throw std::logic_error("routing bounds");
    auto dx = sycl::malloc_device<uint16_t>(x.size(), q);
    auto dwg = sycl::malloc_device<uint16_t>(wg.size(), q);
    auto dwd = sycl::malloc_device<uint16_t>(wd.size(), q);
    auto ds = sycl::malloc_device<int32_t>(src.size(), q);
    if (!dx || !dwg || !dwd || !ds)
        throw std::bad_alloc();
    q.memcpy(dx, x.data(), x.size() * 2);
    q.memcpy(dwg, wg.data(), wg.size() * 2);
    q.memcpy(dwd, wd.data(), wd.size() * 2);
    q.memcpy(ds, src.data(), src.size() * 4);
    Buffer<uint16_t> ox(q, T * K * N, HSENTINEL), oh(q, T * K * F, HSENTINEL);
    Buffer<float> og(q, T * K * GU, SENTINEL), od(q, T * K * N, SENTINEL);
    Buffer<uint16_t> cx(q, T * N, HSENTINEL), ch(q, T * F, HSENTINEL);
    Buffer<float> cg(q, T * GU, SENTINEL), cd(q, T * K * N, SENTINEL);
    Buffer<uint16_t> checkx(q, T * K * N, HSENTINEL), checkh(q, T * K * F, HSENTINEL);
    Buffer<float> checkg(q, T * K * GU, SENTINEL);
    strata::prefill::Gemm engine;
    std::string err;
    if (!engine.init_external(&q, nullptr, 0, nullptr, 0, err))
        throw std::runtime_error(err);
    q.wait_and_throw();
    auto original = [&] {
        strata::prefill::gather_rows16(dx, ds, ox.data, T * K, N, &q);
        for (I e = 0; e < E; ++e) {
            const I rows = offset[e + 1] - offset[e], at = offset[e], slot = e % WEIGHT_SLOTS;
            if (!rows)
                continue;
            engine.f16(ox.data + at * N, dwg + slot * GU * N, og.data + at * GU, rows, GU, N);
            strata::prefill::swiglu_interleaved(og.data + at * GU, oh.data + at * F, rows, &q);
            engine.f16(oh.data + at * F, dwd + slot * N * F, od.data + at * N, rows, N, F);
        }
    };
    auto compact = [&](bool capture) {
        for (I e = 0; e < E; ++e) {
            const I rows = offset[e + 1] - offset[e], at = offset[e], slot = e % WEIGHT_SLOTS;
            if (!rows)
                continue;
            strata::prefill::gather_rows16(dx, ds + at, cx.data, rows, N, &q);
            engine.f16(cx.data, dwg + slot * GU * N, cg.data, rows, GU, N);
            strata::prefill::swiglu_interleaved(cg.data, ch.data, rows, &q);
            if (capture) {
                q.memcpy(checkx.data + at * N, cx.data, rows * N * 2);
                q.memcpy(checkg.data + at * GU, cg.data, rows * GU * 4);
                q.memcpy(checkh.data + at * F, ch.data, rows * F * 2);
            }
            engine.f16(ch.data, dwd + slot * N * F, cd.data + at * N, rows, N, F);
        }
    };
    original();
    compact(true);
    q.wait_and_throw();
    bool guards = true, finite = true;
    size_t unequal = 0;
    auto compare = [&](auto &a, auto &b) {
        auto av = a.read(), bv = b.read();
        guards = guards && a.guarded(av) && b.guarded(bv);
        using V = typename decltype(av)::value_type;
        for (I i = 0; i < a.n; ++i) {
            unequal += std::memcmp(&av[G + i], &bv[G + i], sizeof(V)) != 0;
            if constexpr (sizeof(V) == 4)
                finite = finite && std::isfinite(av[G + i]) && std::isfinite(bv[G + i]);
            else
                finite = finite && std::isfinite(hfloat(av[G + i])) && std::isfinite(hfloat(bv[G + i]));
        }
    };
    compare(ox, checkx);
    compare(og, checkg);
    compare(oh, checkh);
    compare(od, cd);
    guards = guards && cx.guarded(cx.read()) && cg.guarded(cg.read()) && ch.guarded(ch.read());
    const auto gu_host = og.read();
    const auto h_host = oh.read();
    const auto out_host = od.read();
    double gu_l1_error = 0, down_l1_error = 0;
    for (I e = 0; e < E; e += 7) {
        const I rows = offset[e + 1] - offset[e], slot = e % WEIGHT_SLOTS;
        if (!rows)
            continue;
        for (I local : {I(0), rows - 1}) {
            const I row = offset[e] + local;
            for (I c = 0; c < GU; c += 64) {
                double sum = 0, l1 = 0;
                for (I k = 0; k < N; ++k) {
                    double product = double(hfloat(x[src[row] * N + k])) * hfloat(wg[slot * GU * N + c * N + k]);
                    sum += product;
                    l1 += std::abs(product);
                }
                gu_l1_error =
                    std::max(gu_l1_error, std::abs(double(gu_host[G + row * GU + c]) - sum) / std::max(1e-30, l1));
            }
            for (I c = 0; c < N; c += 64) {
                double sum = 0, l1 = 0;
                for (I k = 0; k < F; ++k) {
                    double product = double(hfloat(h_host[G + row * F + k])) * hfloat(wd[slot * N * F + c * F + k]);
                    sum += product;
                    l1 += std::abs(product);
                }
                down_l1_error =
                    std::max(down_l1_error, std::abs(double(out_host[G + row * N + c]) - sum) / std::max(1e-30, l1));
            }
        }
    }
    original();
    compact(false);
    q.wait_and_throw();
    auto timing = [&](auto fn) {
        q.wait_and_throw();
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 2; ++i)
            fn();
        q.wait_and_throw();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / 2;
    };
    std::vector<double> old_times, new_times;
    for (int r = 0; r < 3; ++r) {
        if (r % 2) {
            new_times.push_back(timing([&] { compact(false); }));
            old_times.push_back(timing(original));
        } else {
            old_times.push_back(timing(original));
            new_times.push_back(timing([&] { compact(false); }));
        }
    }
    const bool okay = !unequal && guards && finite && gu_l1_error < 1e-6 && down_l1_error < 1e-6;
    const I old_scratch_bytes = T * K * (N * 2 + GU * 4 + F * 2);
    const I compact_scratch_bytes = T * (N * 2 + GU * 4 + F * 2);
    std::printf("%s T=%lld experts=%lld max_rows=%lld unequal=%zu guards=%d finite=%d gu_L1=%.9g down_L1=%.9g "
                "original=%.6fms compact=%.6fms original_scratch_bytes=%lld compact_scratch_bytes=%lld\n",
                okay ? "PASS" : "FAIL", (long long)T, (long long)nonempty, (long long)max_rows, unequal, guards, finite,
                gu_l1_error, down_l1_error, median(old_times), median(new_times), (long long)old_scratch_bytes,
                (long long)compact_scratch_bytes);
    std::fflush(stdout);
    for (void *p :
         {static_cast<void *>(dx), static_cast<void *>(dwg), static_cast<void *>(dwd), static_cast<void *>(ds)})
        sycl::free(p, q);
    return okay;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    std::printf("device=%s queue_profiling=1\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    bool okay = true;
    for (I T : {I(1), I(17), I(257), I(1024), I(4096), I(8192)})
        okay = run(q, T) && okay;
    return okay ? 0 : 2;
}
