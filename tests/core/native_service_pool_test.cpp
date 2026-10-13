// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/kernels/cpu/pool.hpp"
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#if defined(STRATA_NATIVE_EXPERTS)
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#endif

namespace cpu = strata::kernels::cpu;
#if defined(STRATA_NATIVE_EXPERTS)
namespace {
// Checks remain active in Release, including when NDEBUG is defined.
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "native_service_pool_test: %s\n", message); std::abort(); }
}
constexpr size_t kGuard = 8;
constexpr float kSentinel = 12345.25f;
struct alignas(64) Activation { std::array<uint8_t, cpu::kNativeActBytes> data{}; };
using Output = std::array<float, (size_t)cpu::H + 2*kGuard>;
using Outputs = std::array<Output, 3>;

void reset_outputs(Outputs& out) {
    for (auto& row : out) {
        row.fill(kSentinel);
        for (size_t i = kGuard; i < kGuard + (size_t)cpu::H; ++i)
            row[i] = std::numeric_limits<float>::quiet_NaN();
    }
}
void check_outputs(const Outputs& out) {
    bool nonzero = false;
    for (const auto& row : out) for (size_t i = 0; i < row.size(); ++i) {
        if (i < kGuard || i >= kGuard + (size_t)cpu::H)
            require(std::bit_cast<uint32_t>(row[i]) == std::bit_cast<uint32_t>(kSentinel), "output guard overwritten");
        else {
            require(std::isfinite(row[i]), "missing or nonfinite output");
            nonzero = nonzero || row[i] != 0.f;
        }
    }
    require(nonzero, "degenerate all-zero synthetic output");
}
void parity(const Outputs& out, const Outputs& reference) {
    check_outputs(out);
    for (size_t r = 0; r < out.size(); ++r) for (size_t i = 0; i < out[r].size(); ++i)
        require(std::bit_cast<uint32_t>(out[r][i]) == std::bit_cast<uint32_t>(reference[r][i]), "OFF/ON or partition bit parity");
}
std::vector<uint8_t> make_blob(const cpu::NativeFmt& f, uint64_t seed) {
    require(f.up_off == f.gu_row*(size_t)cpu::FF && f.down_off == 2*f.up_off &&
            f.bytes == f.down_off + f.d_row*(size_t)cpu::H, "native geometry offsets");
    require(f.gu_row % sizeof(block_iq3_s) == 0 && f.d_row % sizeof(block_iq4_nl) == 0,
            "native block geometry");
    std::vector<uint8_t> blob(f.bytes);
    // IQ3_S indexes/signs/scales and IQ4_NL packed nibble indexes accept these bit patterns.
    for (auto& byte : blob) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        byte = static_cast<uint8_t>(seed);
    }
    // Every block header is a finite FP16 scale, rather than randomized FP16 bits.
    for (size_t i = 0; i < f.down_off; i += sizeof(block_iq3_s)) {
        const ggml_half scale = ggml_fp32_to_fp16((i/sizeof(block_iq3_s) % 2 ? -1.f : 1.f)*0.0005f);
        std::memcpy(blob.data() + i, &scale, sizeof scale);
    }
    for (size_t i = f.down_off; i < f.bytes; i += sizeof(block_iq4_nl)) {
        const ggml_half scale = ggml_fp32_to_fp16((i/sizeof(block_iq4_nl) % 2 ? -1.f : 1.f)*0.003f);
        std::memcpy(blob.data() + i, &scale, sizeof scale);
    }
    return blob;
}
bool same_counter(const cpu::ServiceCounter& a, const cpu::ServiceCounter& b) {
    return a.calls == b.calls && a.cpu_ns == b.cpu_ns && a.wall_ns == b.wall_ns &&
           a.reads == b.reads && a.invalid == b.invalid && a.overflow == b.overflow;
}
void unchanged(const cpu::NativeServiceBudget& a, const cpu::NativeServiceBudget& b) {
    require(a.invalid == b.invalid, "retired request validity changed");
    for (int ph = 0; ph < 2; ++ph) {
        require(a.planned_tasks[ph] == b.planned_tasks[ph], "retired request plan changed");
        for (int boundary = 0; boundary < 3; ++boundary)
            require(same_counter(a.phase[ph][boundary], b.phase[ph][boundary]), "retired request phase changed");
    }
    for (int i = 0; i <= cpu::NativeServiceBudget::kWorkers; ++i) {
        require(a.slots[i].overflow == b.slots[i].overflow, "retired request overflow changed");
        for (int ph = 0; ph < 2; ++ph)
            require(a.slots[i].tasks[ph] == b.slots[i].tasks[ph], "retired request task changed");
        for (int bucket = 0; bucket < 6; ++bucket)
            require(same_counter(a.slots[i].call[bucket], b.slots[i].call[bucket]), "retired request call changed");
    }
}
uint64_t expected_ranges(int grain, int rows_per_expert) {
    // A task spanning the boundary between the two experts makes two native calls.
    const int64_t rows = 2*(int64_t)rows_per_expert;
    uint64_t count = 0;
    for (int task = 0; task < grain; ++task) {
        const int64_t begin = rows*task/grain, end = rows*(task+1)/grain;
        require(end > begin, "empty synthetic task");
        count += (uint64_t)((end-1)/rows_per_expert - begin/rows_per_expert + 1);
    }
    return count;
}
void receipt(const cpu::NativeServiceBudget& budget, int workers, bool host, int grain, bool available) {
    require(budget.complete(workers) == available, "actual pool clock/task completion");
    std::array<cpu::ServiceCounter, 6> total{};
    std::array<uint64_t, 2> tasks{};
    for (int i = 0; i <= workers; ++i) {
        for (int ph = 0; ph < 2; ++ph) tasks[ph] += budget.slots[i].tasks[ph];
        for (int b = 0; b < 6; ++b) {
            const auto& c = budget.slots[i].call[b];
            require(c.reads == 2*c.calls && !c.overflow, "call clock receipt accounting");
            if (c.calls) require(c.complete() == available, "call CPU-clock availability");
            total[b].merge(c);
        }
    }
    for (int ph = 0; ph < 2; ++ph) {
        require(tasks[ph] == (uint64_t)grain && budget.planned_tasks[ph] == tasks[ph], "planned/claimed task equality");
        for (const auto& c : budget.phase[ph])
            require(c.calls == 1 && c.reads == 2 && c.complete() == available, "phase boundary receipts");
    }
    require(total[0].calls == expected_ranges(grain, cpu::FF), "native GU invocation/range count");
    require(total[2].calls == expected_ranges(grain, cpu::H), "native Down invocation/range count");
    require(total[1].calls == 3 && budget.slots[0].call[1].calls == 3, "three coalesced activation-row quant calls");
    require(total[1].calls != 2, "activation rows are not distinct jobs");
    for (int b = 3; b < 6; ++b) require(total[b].calls == 0, "unexpected Q2 fallback");
    for (int i = 1; i <= workers; ++i)
        require(budget.slots[i].call[1].calls == 0, "quantization moved off issuer");
    if (!host) for (int ph = 0; ph < 2; ++ph)
        require(budget.slots[0].tasks[ph] == 0, "disabled inline issuer claimed a task");
    // Out-of-envelope slots must remain untouched, not silently acquire work.
    const cpu::ServiceCounter zero;
    for (int i = workers+1; i <= cpu::NativeServiceBudget::kWorkers; ++i) {
        require(!budget.slots[i].overflow && budget.slots[i].tasks[0] == 0 && budget.slots[i].tasks[1] == 0,
                "inactive worker task slot");
        for (const auto& c : budget.slots[i].call) require(same_counter(c, zero), "inactive worker call slot");
    }
}
}
#endif

int main() {
    if (!cpu::native_experts_available()) {
        std::puts("native_service_pool_test: native experts unavailable"); return 77;
    }
#if !defined(STRATA_NATIVE_EXPERTS)
    std::fputs("native_service_pool_test: native availability/build mismatch\n", stderr); return 1;
#else
    cpu::NativeFmt fmt;
    std::string error;
    if (!cpu::native_fmt(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, cpu::H, cpu::FF, fmt, error)) {
        std::fprintf(stderr, "native_service_pool_test: native_fmt: %s\n", error.c_str()); return 1;
    }
    const std::array<std::vector<uint8_t>, 2> blobs{make_blob(fmt, 0x123456789abcdefULL),
                                                 make_blob(fmt, 0x3456789abcdef12ULL)};
    std::array<Activation, 3> activation{};
    std::array<float, (size_t)cpu::H> x{};
    for (size_t row = 0; row < activation.size(); ++row) {
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = 0.75f*std::sin((float)(i + 19*row)*0.031f) + (float)((int)(i%7)-3)*0.05f;
        cpu::native_quant_act(fmt, x.data(), activation[row].data.data());
    }
    Outputs out{}, reference{};
    std::array<cpu::ExpertJobMulti, 2> jobs{};
    jobs[0].blob = blobs[0].data(); jobs[0].nt = 1;
    jobs[1].blob = blobs[1].data(); jobs[1].nt = 2;
    size_t row = 0;
    uint64_t activation_rows = 0;
    for (auto& job : jobs) {
        require(cpu::service_activation_rows_add(activation_rows, job.nt, cpu::MAXT), "activation-row denominator");
        for (int t = 0; t < job.nt; ++t, ++row) {
            job.nact[t] = activation[row].data.data(); job.out[t] = out[row].data() + kGuard;
        }
    }
    require(row == 3 && activation_rows == 3, "two jobs must have three disjoint activation/output rows");
    int bad = -1;
    require(cpu::ExpertPool::check_multi_jobs(fmt, jobs.data(), 2, bad) == nullptr, "valid native job envelope");
    uint64_t clock_ns = 0;
    const bool available = cpu::service_cpu_now(clock_ns);
    bool reference_ready = false;
    uint64_t inline_gu = 0, inline_down = 0, worker_gu = 0, worker_down = 0;
    int configurations = 0, requests = 0;
    for (int workers = 1; workers <= 5; ++workers) for (int grain : {1, 17}) for (bool host : {false, true}) {
        cpu::ExpertPool pool(workers, false, host, 0, grain);
        require(pool.workers() == workers && pool.host_works() == host, "pool worker/issuer configuration");
        // With CTest's STRATA_POOL_SPIN_US=0 this permits sleep before publication.
        // There is no injected worker delay or scheduling override.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        reset_outputs(out);
        pool.run_split_multi_native(fmt, jobs.data(), 2); // diagnostic OFF
        check_outputs(out);
        if (!reference_ready) { reference = out; reference_ready = true; }
        else parity(out, reference);
        auto first = std::make_unique<cpu::NativeServiceBudget>();
        auto second = std::make_unique<cpu::NativeServiceBudget>();
        const cpu::NativeServiceBudget empty;
        unchanged(*first, empty); unchanged(*second, empty);
        std::unique_ptr<cpu::NativeServiceBudget> first_snapshot;
        for (int request = 0; request < 2; ++request) {
            auto& budget = request == 0 ? *first : *second;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            reset_outputs(out);
            // Single issuer binding precedes publication; return includes final worker park.
            pool.service_budget(&budget);
            pool.run_split_multi_native(fmt, jobs.data(), 2);
            pool.service_budget(nullptr);
            receipt(budget, workers, host, grain, available);
            parity(out, reference);
            inline_gu += budget.slots[0].call[0].calls; inline_down += budget.slots[0].call[2].calls;
            for (int i = 1; i <= workers; ++i) {
                worker_gu += budget.slots[i].call[0].calls; worker_down += budget.slots[i].call[2].calls;
            }
            if (request == 0) first_snapshot = std::make_unique<cpu::NativeServiceBudget>(budget);
            else unchanged(*first, *first_snapshot);
            ++requests;
        }
        const auto second_snapshot = std::make_unique<cpu::NativeServiceBudget>(*second);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        reset_outputs(out);
        pool.run_split_multi_native(fmt, jobs.data(), 2); // unbound OFF after two different requests
        parity(out, reference);
        unchanged(*first, *first_snapshot); unchanged(*second, *second_snapshot);
        ++configurations;
    }
    require(inline_gu > 0 && inline_down > 0, "no observed inline issuer native GU/Down coverage");
    require(worker_gu > 0 && worker_down > 0, "no observed worker native GU/Down coverage");
    const char* spin = std::getenv("STRATA_POOL_SPIN_US");
    std::printf("native_service_pool_test: configurations=%d requests=%d jobs=2 activation_rows=3 clock_available=%d "
                "inline_gu=%llu inline_down=%llu worker_gu=%llu worker_down=%llu sleep_knob_zero=%d\n",
                configurations, requests, available, (unsigned long long)inline_gu, (unsigned long long)inline_down,
                (unsigned long long)worker_gu, (unsigned long long)worker_down, spin && std::strcmp(spin, "0") == 0);
    return 0;
#endif
}
