// Compare the full state and both outputs against the original prompt recurrence.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/gdn_variant.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
namespace {
constexpr size_t guard = 32;
constexpr float sentinel = -98765.f;
constexpr uint16_t half_sentinel = 0x5a5a;
double median(std::vector<double> a) {
    std::sort(a.begin(), a.end());
    return a[a.size() / 2];
}
bool run(sycl::queue &q, int64_t T) {
    const size_t ns = 128 * 48 * 128, nh = T * 10240, ny = T * 48 * 128, ng = T * 48;
    std::mt19937 rng(5927 + T);
    std::uniform_real_distribution<float> dist(-1, 1);
    std::vector<float> state(ns), h(nh), gate(ng), beta(ng), z(ny), gamma(128);
    for (auto &v : state)
        v = dist(rng) * .01f;
    for (auto &v : h)
        v = dist(rng) * .06f;
    for (auto &v : gate)
        v = -.1f - std::abs(dist(rng)) * .1f;
    for (auto &v : beta)
        v = .1f + std::abs(dist(rng)) * .5f;
    for (auto &v : z)
        v = dist(rng);
    for (auto &v : gamma)
        v = 1 + dist(rng) * .1f;
    auto astate = sycl::malloc_device<float>(ns + 2 * guard, q);
    auto bstate = sycl::malloc_device<float>(ns + 2 * guard, q);
    auto dh = sycl::malloc_device<float>(nh, q);
    auto dg = sycl::malloc_device<float>(ng, q);
    auto db = sycl::malloc_device<float>(ng, q);
    auto dz = sycl::malloc_device<float>(ny, q);
    auto dgamma = sycl::malloc_device<float>(128, q);
    auto ay = sycl::malloc_device<float>(ny + 2 * guard, q);
    auto by = sycl::malloc_device<float>(ny + 2 * guard, q);
    auto ah = sycl::malloc_device<uint16_t>(ny + 2 * guard, q);
    auto bh = sycl::malloc_device<uint16_t>(ny + 2 * guard, q);
    q.fill(astate, sentinel, ns + 2 * guard);
    q.fill(bstate, sentinel, ns + 2 * guard);
    q.fill(ay, sentinel, ny + 2 * guard);
    q.fill(by, sentinel, ny + 2 * guard);
    q.fill(ah, half_sentinel, ny + 2 * guard);
    q.fill(bh, half_sentinel, ny + 2 * guard);
    q.memcpy(astate + guard, state.data(), ns * 4);
    q.memcpy(bstate + guard, state.data(), ns * 4);
    q.memcpy(dh, h.data(), nh * 4);
    q.memcpy(dg, gate.data(), ng * 4);
    q.memcpy(db, beta.data(), ng * 4);
    q.memcpy(dz, z.data(), ny * 4);
    q.memcpy(dgamma, gamma.data(), 128 * 4);
    q.wait_and_throw();
    auto original = [&](float *st, float *y, uint16_t *half) {
        strata::prefill::gdn_recurrence(st + guard, dh, dg, db, dz, dgamma, 1e-6f, y + guard, half + guard, T, &q);
    };
    auto before = [&] { original(astate, ay, ah); };
    bool launched = false;
    auto after = [&] {
        launched = strata::prefill::gdn_recurrence_keyhead_variant(bstate + guard, dh, dg, db, dz, dgamma, 1e-6f,
                                                                   by + guard, bh + guard, T, &q);
        if (launched != (T >= 256)) {
            std::fprintf(stderr, "FAIL eligibility T=%lld\n", (long long)T);
            std::exit(2);
        }
        if (!launched)
            original(bstate, by, bh);
    };
    before();
    after();
    q.wait_and_throw();
    size_t different_state = 0, different_y = 0, different_half = 0, guard_errors = 0;
    bool finite = true;
    auto floats = [&](float *a, float *b, size_t count, size_t &different) {
        std::vector<float> ha(count + 2 * guard), hb(ha.size());
        q.memcpy(ha.data(), a, ha.size() * 4);
        q.memcpy(hb.data(), b, hb.size() * 4);
        q.wait_and_throw();
        for (size_t i = 0; i < ha.size(); ++i) {
            if (i < guard || i >= count + guard) {
                guard_errors += ha[i] != sentinel || hb[i] != sentinel;
            } else {
                different += std::bit_cast<uint32_t>(ha[i]) != std::bit_cast<uint32_t>(hb[i]);
                finite = finite && std::isfinite(ha[i]) && std::isfinite(hb[i]);
            }
        }
    };
    floats(astate, bstate, ns, different_state);
    floats(ay, by, ny, different_y);
    std::vector<uint16_t> ha(ny + 2 * guard), hb(ha.size());
    q.memcpy(ha.data(), ah, ha.size() * 2);
    q.memcpy(hb.data(), bh, hb.size() * 2);
    q.wait_and_throw();
    for (size_t i = 0; i < ha.size(); ++i) {
        if (i < guard || i >= ny + guard) {
            guard_errors += ha[i] != half_sentinel || hb[i] != half_sentinel;
        } else {
            different_half += ha[i] != hb[i];
            finite = finite && (ha[i] & 0x7c00) != 0x7c00 && (hb[i] & 0x7c00) != 0x7c00;
        }
    }
    if (launched && T == 256) {
        for (const auto &id : sycl::get_kernel_ids()) {
            if (std::string(id.get_name()).find("gdn_rec_kh_tuned_kernel") == std::string::npos)
                continue;
            auto bundle =
                sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context(), {q.get_device()}, {id});
            auto kernel = bundle.get_kernel(id);
            std::printf(
                "resources requested_SG=16 requested_GRF=256 private=%zu spill=%zu bytes\n",
                kernel.get_info<sycl::info::kernel_device_specific::private_mem_size>(q.get_device()),
                kernel.get_info<sycl::ext::intel::info::kernel_device_specific::spill_memory_size>(q.get_device()));
        }
    }
    auto time_run = [&](auto fn) {
        q.wait_and_throw();
        auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < 5; ++r)
            fn();
        q.wait_and_throw();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 5;
    };
    std::vector<double> old_times, new_times;
    for (int r = 0; r < 3; ++r) {
        if (r % 2) {
            new_times.push_back(time_run(after));
            old_times.push_back(time_run(before));
        } else {
            old_times.push_back(time_run(before));
            new_times.push_back(time_run(after));
        }
    }
    const bool ok = finite && different_state == 0 && different_y == 0 && different_half == 0 && guard_errors == 0;
    std::printf("%s T=%lld launched=%d state_unequal=%zu output_unequal=%zu half_unequal=%zu guards=%zu finite=%d "
                "original=%.6fms variant=%.6fms\n",
                ok ? "PASS" : "FAIL", (long long)T, launched, different_state, different_y, different_half,
                guard_errors, finite, median(old_times), median(new_times));
    std::fflush(stdout);
    for (void *p : std::vector<void *>{astate, bstate, dh, dg, db, dz, dgamma, ay, by, ah, bh})
        sycl::free(p, q);
    return ok;
}
} // namespace
int main() {
    auto enabled = [](const char *name) {
        const char *v = std::getenv(name);
        return v && std::atoi(v) != 0;
    };
    const char *pipe = std::getenv("STRATA_GDN_PIPELINE");
    if (enabled("STRATA_GDN_KEYHEAD_TUNED") || enabled("STRATA_GDN_KEYHEAD") || std::getenv("STRATA_GDN_REC_HEADS") ||
        (pipe && std::atoi(pipe) == 0)) {
        std::fprintf(stderr, "GDN parity needs the original pipeline reference; unset other GDN selectors\n");
        return 2;
    }
    auto &q = *strata::q_of(nullptr);
    for (int64_t t : {1, 17, 255, 256, 257, 1024, 4096, 8192})
        if (!run(q, t))
            return 1;
    std::puts("PASS all 8 cases: full state, FP32 output, FP16 output and guards");
}
