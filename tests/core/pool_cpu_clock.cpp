// CPU-only accounting qualification of the real ExpertPool, with synthetic
// valid Q2_0 and Q8_0 experts. No model, device runtime or benchmark timing.
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace c = strata::kernels::cpu;
using strata::core::CpuTimeCounter;
using strata::core::cpu_counter_delta;
using strata::core::cpu_time_delta;

namespace {
void require(bool yes, const char* why) {
    if (!yes) throw std::runtime_error(why);
}

using Profile = c::ExpertPool::CpuProfile;
using Counts = std::array<uint64_t, 8>;

Profile settled(c::ExpertPool& pool, bool enabled) {
    // Startup is not a completed dispatch; allow workers to reach their first
    // park before taking the initial snapshot. No publisher runs concurrently.
    for (int i = 0; i < 200; ++i) {
        const auto p = pool.cpu_profile();
        if (!enabled || p.quiescent) return p;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("workers did not reach initial quiescence");
}

uint64_t check(const Profile& a, const Profile& b, bool enabled,
               const Counts& expected, uint64_t quant_calls, bool no_host) {
    require(a.enabled == enabled && b.enabled == enabled, "observer selection");
    if (!enabled) {
        require(!a.main_thread.valid && !b.process.valid && !b.worker_thread_ns,
                "disabled observer read clocks");
    } else {
        require(a.quiescent && b.quiescent && !a.read_errors && !b.read_errors,
                "snapshot not quiescent or clock failure");
    }
    CpuTimeCounter workers, host;
    for (size_t mode = 0; mode < expected.size(); ++mode) {
        const auto w = cpu_counter_delta(a.worker_jobs[mode], b.worker_jobs[mode]);
        const auto h = cpu_counter_delta(a.host_jobs[mode], b.host_jobs[mode]);
        require(!w.errors && !h.errors && !w.unwound && !h.unwound, "invalid task counter");
        require(w.calls + h.calls == (enabled ? expected[mode] : 0), "task count not conserved");
        if (no_host) require(!h.calls, "disabled host drain claimed a task");
        workers.merge(w); host.merge(h);
    }
    const auto q = cpu_counter_delta(a.native_quant, b.native_quant);
    require(!q.errors && !q.unwound && q.calls == (enabled ? quant_calls : 0),
            "native quantization batch count");
    if (enabled) {
        const auto main = cpu_time_delta(a.main_thread, b.main_thread);
        const auto process = cpu_time_delta(a.process, b.process);
        require(main.valid && process.valid && b.worker_thread_ns >= a.worker_thread_ns,
                "invalid lifetime CPU clock delta");
        require(workers.ns <= b.worker_thread_ns - a.worker_thread_ns,
                "worker task time exceeds worker CPU");
        require(host.ns + q.ns <= main.ns, "main subsets exceed main CPU");
        require(!workers.errors && !host.errors && (!workers.calls || workers.ns) &&
                (!host.calls || host.ns) && (!q.calls || q.ns), "missing CPU duration");
    }
    return host.calls;
}

void finite_equal(const float* actual, const float* expected, size_t n) {
    for (size_t i = 0; i < n; ++i)
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite output");
    require(std::memcmp(actual, expected, n * sizeof(float)) == 0, "pool output differs from serial");
}

struct Inputs {
    const bool q2_supported = c::cpu_features().usable();
    std::vector<uint8_t> q2 = std::vector<uint8_t>(c::BLOB);
    c::NativeFmt fmt;
    std::vector<uint8_t> q8;
    c::ActQ acts[c::MAXT];
    alignas(64) uint8_t native[c::MAXT][c::kNativeActBytes]{};
    std::vector<float> single = std::vector<float>(c::MAXT * c::H);
    std::vector<float> native_single = std::vector<float>(c::MAXT * c::H);

    Inputs() {
        // Deterministic code bytes and finite, small FP16 scales for this
        // repository's Q2_0 layout. Distinct rows/activations expose indexing.
        for (size_t i = 0; i < c::O_GU_SCALES; ++i) q2[i] = uint8_t(i * 37 + i / 19);
        for (size_t i = c::O_GU_SCALES; i < q2.size(); i += 2) {
            const uint16_t scale = uint16_t(0x1800 + (i / 2) % 127);
            std::memcpy(q2.data() + i, &scale, 2);
        }
        std::string err;
        const bool valid_fmt = c::native_fmt(8, 8, c::H, c::FF, fmt, err); // ggml Q8_0
        require(valid_fmt, err.c_str());
        require(fmt.gu_row == size_t(c::H / 32 * 34) &&
                fmt.d_row == size_t(c::FF / 32 * 34), "Q8_0 layout changed");
        q8.resize(fmt.bytes);
        for (size_t i = 0; i < q8.size(); i += 34) {
            const uint16_t scale = 0x1800;
            std::memcpy(q8.data() + i, &scale, 2);
            for (size_t j = 0; j < 32; ++j)
                q8[i + 2 + j] = uint8_t(int((i / 34 + j * 11) % 31) - 15);
        }
        c::ExpertScratch scratch;
        for (int t = 0; t < c::MAXT; ++t) {
            std::vector<float> x(c::H);
            for (int i = 0; i < c::H; ++i) x[i] = float((i * 17 + t * 13) % 127 - 63) / 80;
            c::native_quant_act(fmt, x.data(), native[t]);
            if (q2_supported) {
                c::act_quant_q8_1(x.data(), c::H, acts[t]);
                c::s2_expert_vnni_q(q2.data(), acts[t], single.data() + t * c::H, scratch);
            }
            float ff[c::FF];
            alignas(64) uint8_t h[c::kNativeHBytes]{};
            const void* ap[] = {native[t]}; float* fp[] = {ff};
            c::native_gu_rows(fmt, q8.data(), ap, 1, fp, 0, c::FF);
            c::native_quant_h(fmt, ff, h);
            const void* hp[] = {h}; float* op[] = {native_single.data() + t * c::H};
            c::native_down_rows(fmt, q8.data(), hp, 1, op, 0, c::H);
        }
    }
};

void exercise(const Inputs& data, bool enabled, int workers, bool host_works) {
    c::ExpertPool pool(workers, false, host_works);
    auto before = settled(pool, enabled);
    pool.run(nullptr, 0);
    check(before, pool.cpu_profile(), enabled, {}, 0, true);
    uint64_t observed_host = 0;
    for (int pass = 0; data.q2_supported && pass < 2; ++pass) {
        // Cross the real park-to-sleep threshold, then exercise a tiny batch.
        std::this_thread::sleep_for(c::ExpertPool::kSpinBeforeSleep + std::chrono::milliseconds(5));
        const int n = pass ? 3 : 1;
        std::vector<float> out(n * c::H, NAN);
        std::vector<c::ExpertJob> jobs(n);
        for (int e = 0; e < n; ++e)
            jobs[e] = {data.q2.data(), &data.acts[e], out.data() + e * c::H};
        before = pool.cpu_profile();
        pool.run(jobs.data(), n);
        Counts expected{}; expected[0] = n;
        observed_host += check(before, pool.cpu_profile(), enabled, expected, 0, workers > 1 && !host_works);
        for (int e = 0; e < n; ++e) finite_equal(out.data() + e * c::H, data.single.data() + e * c::H, c::H);
        if (workers == 1 && enabled)
            require(pool.cpu_profile().worker_jobs[0].calls == 0, "inline path used worker slot");
        std::fill(out.begin(), out.end(), NAN);
        before = pool.cpu_profile();
        pool.run_split(jobs.data(), n);
        expected = {};
        if (workers == 1) expected[0] = n;
        else expected[1] = expected[2] = n * std::max(1, (3 * (workers + int(host_works)) + n - 1) / n);
        observed_host += check(before, pool.cpu_profile(), enabled, expected, 0, workers > 1 && !host_works);
        for (int e = 0; e < n; ++e) finite_equal(out.data() + e * c::H, data.single.data() + e * c::H, c::H);
    }
    for (bool native : {false, true}) for (int nt : {1, 4}) {
        if (!native && !data.q2_supported) continue;
        // Native n=97 exercises two quantization batches; Q2 multi and its
        // fallback both run with the same bounded total output allocation.
        for (int n : {2, c::ExpertPool::kMaxSplitMulti + 1}) {
            std::this_thread::sleep_for(c::ExpertPool::kSpinBeforeSleep + std::chrono::milliseconds(5));
            std::vector<float> out(size_t(n) * nt * c::H, NAN);
            std::vector<c::ExpertJobMulti> jobs(n);
            for (int e = 0; e < n; ++e) {
                jobs[e].blob = native ? data.q8.data() : data.q2.data();
                jobs[e].nt = nt;
                for (int t = 0; t < nt; ++t) {
                    jobs[e].act[t] = &data.acts[t]; jobs[e].nact[t] = data.native[t];
                    jobs[e].out[t] = out.data() + (size_t(e) * nt + t) * c::H;
                }
            }
            before = pool.cpu_profile();
            if (native) pool.run_split_multi_native(data.fmt, jobs.data(), n);
            else pool.run_split_multi(jobs.data(), n);
            Counts expected{};
            const uint64_t tasks = 3 * (workers + int(host_works));
            const uint64_t batches = (n + c::ExpertPool::kMaxSplitMulti - 1) / c::ExpertPool::kMaxSplitMulti;
            if (native) expected[5] = expected[6] = tasks * batches;
            else if (n <= c::ExpertPool::kMaxSplitMulti) expected[3] = expected[4] = tasks;
            else if (workers == 1) expected[0] = n * nt;
            else {
                for (int off = 0; off < n * nt; off += c::ExpertPool::kMaxSplit) {
                    const int count = std::min(c::ExpertPool::kMaxSplit, n * nt - off);
                    expected[1] += count * std::max(1, int((tasks + count - 1) / count));
                }
                expected[2] = expected[1];
            }
            observed_host += check(before, pool.cpu_profile(), enabled, expected, native ? batches : 0,
                                   workers > 1 && !host_works);
            for (int e = 0; e < n; ++e) for (int t = 0; t < nt; ++t)
                finite_equal(jobs[e].out[t], (native ? data.native_single : data.single).data() + t * c::H, c::H);
        }
    }
    // Host participation is observed, not inferred from a scheduling option.
    std::printf("pool_cpu_clock: workers=%d host_works=%d observed_host_calls=%llu\n",
                workers, int(host_works), (unsigned long long) observed_host);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "on") && std::strcmp(argv[1], "off"))) return 2;
    try {
        const bool enabled = !std::strcmp(argv[1], "on");
        require(strata::core::decode_cpu_profile_enabled() == enabled, "launch-time selection");
        const auto data = std::make_unique<Inputs>();
        std::printf("pool_cpu_clock: native modes5/6 enabled; Q2 modes0..4=%s\n",
                    data->q2_supported ? "enabled" : "unsupported AVX512, not qualified");
        exercise(*data, enabled, 1, false);
        exercise(*data, enabled, 2, false);
        exercise(*data, enabled, 2, true);
        std::puts("pool_cpu_clock: passed");
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "pool_cpu_clock: FAIL %s\n", ex.what());
        return 1;
    }
}
