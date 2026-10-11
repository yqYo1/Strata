// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// SOURCE ONLY: contingent on Main's comprehensive byte/guard qualification.
#include "strata/core/runtime.hpp"
#include "strata/core/gpu.hpp"
#include "strata/kernels/iq_kernels.hpp"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

namespace k = strata::kernels;
namespace gpu = strata::gpu;
namespace {
constexpr int64_t ff = 640, emb = 2560;
constexpr size_t slots = 16, matrix = size_t(ff) * size_t(emb), guard = 64, MiB = 1024 * 1024;
constexpr size_t gu_stride = 2 * matrix + 24, down_stride = matrix + 24;
constexpr size_t gu_capacity = slots * gu_stride * 2, down_capacity = slots * down_stride * 2;
constexpr size_t gu_offset = guard, down_offset = gu_offset + gu_capacity + guard;
constexpr size_t output_bytes = down_offset + down_capacity + guard;
constexpr unsigned char fill = 0xa5;
constexpr int repetitions = 4, blocks = 3, samples_per_arm = 6;
struct Pair { int gu, down; };
constexpr Pair pairs[] = {{18,20},{18,42},{21,20},{21,42},{22,20},{22,42},{23,20}};
constexpr bool abba[] = {false,true,true,false}; // A = individual, B = group
using Clock = std::chrono::steady_clock;
uint64_t samples = 0, individual_returns = 0, group_returns = 0, expert_pairs = 0;
uint64_t readbacks = 0, read_bytes = 0, oracle_pairs = 0;

[[noreturn]] void fatal(const char* phase, const char* reason) noexcept {
    std::fprintf(stderr, "FAIL phase=%s reason=%s completion_unconfirmed=1 release_skipped=1\n", phase, reason);
    std::fflush(stderr);
    std::_Exit(74);
}
void require(bool condition, const char* reason) noexcept { if (!condition) fatal("invariant", reason); }
template<class F> void checked(const char* phase, F f) noexcept {
    // Catch before the caller's owning buffers can unwind while a copy/kernel is pending.
    try { f(); }
    catch (const std::exception& e) { fatal(phase, e.what()); }
    catch (...) { fatal(phase, "nonstandard exception"); }
}
void api(bool accepted, const char* phase) noexcept { if (!accepted) fatal(phase, gpu::last_error()); }
uint64_t mix(uint64_t x) noexcept {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
// Native-endian uint64-word FNV-style supplementary identity, not byte FNV-1a.
uint64_t hash(const unsigned char* p, size_t n) noexcept {
    uint64_t h=UINT64_C(14695981039346656037);size_t i=0;
    for(;n-i>=sizeof(uint64_t);i+=sizeof(uint64_t)) {
        uint64_t word=0;std::memcpy(&word,p+i,sizeof(word));h=(h^word)*UINT64_C(1099511628211);
    }
    for(;i<n;++i)h=(h^p[i])*UINT64_C(1099511628211);
    return h;
}
struct Format { size_t block, scale; };
Format format(int t) noexcept {
    switch (t) {
        case 18: return {sizeof(block_iq3_xxs), offsetof(block_iq3_xxs,d)};
        case 20: return {sizeof(block_iq4_nl), offsetof(block_iq4_nl,d)};
        case 21: return {sizeof(block_iq3_s), offsetof(block_iq3_s,d)};
        case 22: return {sizeof(block_iq2_s), offsetof(block_iq2_s,d)};
        case 23: return {sizeof(block_iq4_xs), offsetof(block_iq4_xs,d)};
        case 42: return {sizeof(block_q2_0), offsetof(block_q2_0,d)};
        default: fatal("format", "unsupported fixture type");
    }
}
void native(unsigned char* p, size_t bytes, int type, int expert, int projection) noexcept {
    const Format f = format(type);
    require(bytes % f.block == 0, "native block extent");
    for (size_t b = 0; b < bytes / f.block; ++b) {
        auto* q = p + b * f.block;
        const uint64_t seed = mix((uint64_t(expert + 1) << 48) ^ (uint64_t(projection + 1) << 40) ^ b ^ uint64_t(type));
        for (size_t j = 0; j < f.block; ++j) q[j] = static_cast<unsigned char>(mix(seed + j));
        // Signed finite small normal scale; all index/code bits stay native integer fields.
        // No infinity/NaN edge set in a timing workload. Reference output finiteness is checked.
        const uint16_t scale = static_cast<uint16_t>(0x1400 | (seed & 0x3ff) | (((seed >> 12) & 1) << 15));
        q[f.scale] = static_cast<unsigned char>(scale);
        q[f.scale + 1] = static_cast<unsigned char>(scale >> 8);
    }
}
void compare(const std::vector<unsigned char>& got, const std::vector<unsigned char>& reference,
             const char* label) noexcept {
    require(got.size() == reference.size(), "compare extent");
    size_t differences = 0, first = 0;
    if (std::memcmp(got.data(),reference.data(),got.size())!=0)
    for (size_t i = 0; i < got.size(); ++i) if (got[i] != reference[i]) {
        if (!differences) first = i;
        ++differences;
    }
    std::printf("BYTE_COMPARE label=%s bytes=%zu differences=%zu first_difference=%zu reference_word_fnv64_native_endian=%016llx actual_word_fnv64_native_endian=%016llx\n",
                label, got.size(), differences, first, (unsigned long long)hash(reference.data(), reference.size()),
                (unsigned long long)hash(got.data(), got.size()));
    std::fflush(stdout);
    require(!differences, "complete byte comparison failed");
}
void reference_guards_and_finite(const std::vector<unsigned char>& v) noexcept {
    size_t live = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        const bool gu = i >= gu_offset && i < gu_offset + gu_capacity &&
                        (i - gu_offset) % (gu_stride * 2) < matrix * 4;
        const bool down = i >= down_offset && i < down_offset + down_capacity &&
                          (i - down_offset) % (down_stride * 2) < matrix * 2;
        if (!gu && !down) require(v[i] == fill, "individual full reference guard/padding");
        else {
            ++live;
            if (!(i % 2)) {
                const uint16_t word = uint16_t(v[i]) | (uint16_t(v[i + 1]) << 8);
                require((word & 0x7c00) != 0x7c00, "fixture output must be finite fp16");
            }
        }
    }
    require(live == slots * matrix * 6, "reference live extent");
}
struct Sample {
    bool grouped = false;
    int arm_index = 0, block = 0, position = 0;
    double submit_ms = 0, submit_drain_ms = 0, mark_enclosure_ms = 0;
    double begin_mark_submit_ms = 0, end_mark_submit_ms = 0;
    float queue_interval_ms = -1;
};
double ms(Clock::time_point a, Clock::time_point b) noexcept {
    return std::chrono::duration<double,std::milli>(b - a).count();
}
void pair_run(strata::core::Runtime& rt, sycl::queue& q, Pair pair, bool tags, bool smoke) noexcept {
    const size_t gu_native = k::iq_row_bytes(pair.gu, int64_t(matrix));
    const size_t down_native = k::iq_row_bytes(pair.down, int64_t(matrix));
    require(gu_native && down_native, "supported native extents");
    const size_t up_offset = (gu_native + 15) / 16 * 16 + guard;
    const size_t source_down_offset = up_offset + (gu_native + 15) / 16 * 16 + guard;
    const size_t source_stride = source_down_offset + (down_native + 15) / 16 * 16 + guard;
    const size_t source_bytes = 2 * guard + slots * source_stride;
    const size_t device_payload = source_bytes + output_bytes;
    const size_t host_payload = source_bytes * 2 + output_bytes * 3;
    require(device_payload <= 512 * MiB && host_payload <= 768 * MiB, "allocation payload budgets");
    std::printf("PAIR_BEGIN gu_type=%d down_type=%d ff=640 emb=2560 device_payload=%zu host_payload=%zu gu_stride_elements=%zu down_stride_elements=%zu gu_capacity_bytes=%zu down_capacity_bytes=%zu source_projection_gu_bytes=%zu source_projection_down_bytes=%zu source_stride_bytes=%zu\n",
                pair.gu, pair.down, device_payload, host_payload, gu_stride, down_stride, gu_capacity, down_capacity,
                gu_native, down_native, source_stride);
    std::fflush(stdout);
    std::vector<unsigned char> input, source_read, reference, read, expected;
    checked("host_vectors", [&] {
        input.assign(source_bytes, fill); source_read.resize(source_bytes);
        reference.resize(output_bytes); read.resize(output_bytes); expected.resize(output_bytes);
    });
    for (size_t e = 0; e < slots; ++e) {
        auto* p = input.data() + guard + e * source_stride;
        native(p, gu_native, pair.gu, int(e), 0);
        native(p + up_offset, gu_native, pair.gu, int(e), 1);
        native(p + source_down_offset, down_native, pair.down, int(e), 2);
    }
    std::printf("INPUT bytes=%zu word_fnv64_native_endian=%016llx identity=deterministic_native_finite_small_scale\n", source_bytes,
                (unsigned long long)hash(input.data(), input.size()));
    std::fflush(stdout);
    void* src = nullptr; void* out = nullptr;
    checked("device_allocations", [&] {
        src = sycl::aligned_alloc_device(16, source_bytes, q);
        out = sycl::aligned_alloc_device(16, output_bytes, q);
        require(src && out, "device allocation refusal");
        for (void* p : {src,out}) {
            require(reinterpret_cast<uintptr_t>(p) % 16 == 0, "real allocation alignment");
            require(sycl::get_pointer_type(p, rt.context()) == sycl::usm::alloc::device &&
                    strata::core::Runtime::owner(p) == &rt, "allocation Runtime provenance");
        }
    });
    auto drain = [&] { checked("checked_compute_drain", [&] { rt.finish(q); }); };
    auto copy = [&](void* dest, const void* from, size_t bytes) {
        checked("checked_copy", [&] { rt.wait(q.memcpy(dest, from, bytes), "native group component copy"); });
    };
    auto reset = [&] {
        checked("checked_output_reset", [&] { rt.wait(q.memset(out, fill, output_bytes), "native group component reset"); });
        drain();
    };
    auto readback = [&](std::vector<unsigned char>& v, const void* p) {
        require(read_bytes + v.size() <= UINT64_C(1099511627776), "finite 1TiB readback budget");
        drain(); copy(v.data(), p, v.size()); ++readbacks; read_bytes += v.size();
    };
    copy(src, input.data(), source_bytes); drain();
    auto* gu_out = reinterpret_cast<uint16_t*>(static_cast<unsigned char*>(out) + gu_offset);
    auto* down_out = reinterpret_cast<uint16_t*>(static_cast<unsigned char*>(out) + down_offset);
    k::IqDequantGroup group{};
    for (size_t e = 0; e < slots; ++e) {
        auto* p = static_cast<unsigned char*>(src) + guard + e * source_stride;
        group.gate[e] = p; group.up[e] = p + up_offset; group.down[e] = p + source_down_offset;
        group.gate_bytes[e] = group.up_bytes[e] = gu_native; group.down_bytes[e] = down_native;
    }
    void* stream = &q;
    // Refusal is fatal, with no change of work or retry. Logical normal returns
    // below do NOT prove low-level submission counts or GPU completion.
    auto enqueue = [&](bool grouped, int n, int reps) {
        checked("GU_Down_submit", [&] {
            group.first = 0; group.n = n;
            for (int r = 0; r < reps; ++r) {
                if (grouped) {
                    require(k::iq_dequant_gu_group_f16(pair.gu, group, ff, emb, gu_out, gu_stride, gu_capacity, stream), "admitted GU group refused");
                    ++group_returns;
                    require(k::iq_dequant_down_group_f16(pair.down, group, ff, emb, down_out, down_stride, down_capacity, stream), "admitted Down group refused");
                    ++group_returns;
                } else for (int e = 0; e < n; ++e) {
                    k::iq_dequant_gu_f16(pair.gu, group.gate[e], group.up[e], ff, emb, gu_out + size_t(e) * gu_stride, stream);
                    ++individual_returns;
                    k::iq_dequant_f16(pair.down, group.down[e], int64_t(matrix), down_out + size_t(e) * down_stride, stream);
                    ++individual_returns;
                }
                expert_pairs += uint64_t(n);
            }
        });
    };
    // One complete per-pair old-API reference versus the optional group API,
    // using all sixteen distinct experts and ALL allocation bytes before timing.
    reset(); enqueue(false, 16, 1); drain(); readback(reference, out);
    reference_guards_and_finite(reference);
    readback(source_read, src); compare(source_read, input, "individual_full_source_guards");
    reset(); enqueue(true, 16, 1); drain(); readback(read, out);
    compare(read, reference, "group_full_output_guard_padding_oracle");
    readback(source_read, src); compare(source_read, input, "group_full_source_guards");
    ++oracle_pairs;
    gpu::Event* begin = nullptr; gpu::Event* end = nullptr;
    if (tags) checked("production_timing_events", [&] {
        api(gpu::event_create(&begin, true), "event_create_begin");
        api(gpu::event_create(&end, true), "event_create_end");
        require(begin && end, "production timing event handles");
    });
    for (int n : {1,8,16}) {
        if (smoke && n != 1) continue;
        std::fill(expected.begin(), expected.end(), fill);
        for (int e = 0; e < n; ++e) {
            const size_t go = gu_offset + size_t(e) * gu_stride * 2;
            const size_t d = down_offset + size_t(e) * down_stride * 2;
            std::copy_n(reference.data() + go, matrix * 4, expected.data() + go);
            std::copy_n(reference.data() + d, matrix * 2, expected.data() + d);
        }
        std::array<Sample, blocks * 4> records{};
        int arm_count[2] = {0,0};
        std::printf("TIMING_BEGIN gu_type=%d down_type=%d n=%d tags=%d repetitions=4 samples_per_arm=6 order=ABBA_ABBA_ABBA warmup_pairs_per_sample=%d\n",
                    pair.gu, pair.down, n, int(tags), n);
        std::fflush(stdout);
        // Quiet, fixed samples. No copies, hashing, scans or logging inside this loop.
        for (int b = 0; b < blocks; ++b) for (int pos = 0; pos < 4; ++pos) {
            const bool grouped = abba[pos];
            reset(); enqueue(grouped, n, 1); drain();
            Sample& s = records[size_t(b * 4 + pos)];
            s.grouped = grouped; s.arm_index = arm_count[int(grouped)]++; s.block = b; s.position = pos;
            const auto mark_start = Clock::now();
            if (tags) api(gpu::event_record(begin, stream), "event_record_begin");
            const auto submit_start = Clock::now();
            enqueue(grouped, n, repetitions);
            const auto submit_end = Clock::now();
            if (tags) api(gpu::event_record(end, stream), "event_record_end");
            const auto marks_end = Clock::now();
            drain();
            const auto completed = Clock::now();
            s.submit_ms = ms(submit_start, submit_end);
            s.submit_drain_ms = ms(submit_start, completed);
            s.mark_enclosure_ms = ms(mark_start, completed);
            s.begin_mark_submit_ms = ms(mark_start, submit_start);
            s.end_mark_submit_ms = ms(submit_end, marks_end);
            if (tags) {
                api(gpu::event_elapsed_ms(&s.queue_interval_ms, begin, end), "completed_production_event_elapsed");
                require(std::isfinite(s.queue_interval_ms) && s.queue_interval_ms > 0, "positive finite completed stream tag interval");
            }
            require(s.submit_ms >= 0 && s.submit_drain_ms >= s.submit_ms, "steady host clock order");
            ++samples;
        }
        require(arm_count[0] == samples_per_arm && arm_count[1] == samples_per_arm, "independent arm sample count");
        // Last timed sample is A (ABBA), so check its full result immediately.
        readback(read, out); compare(read, expected, "last_timed_individual_full_output_inactive_guards");
        readback(source_read, src); compare(source_read, input, "timed_arms_full_source_guards");
        // Separate post-timing B check at this n; no numerical host decoder.
        reset(); enqueue(true, n, 1); drain(); readback(read, out);
        compare(read, expected, "post_timing_group_full_output_inactive_guards");
        readback(source_read, src); compare(source_read, input, "post_timing_group_full_source_guards");
        for (const Sample& s : records) {
            const int experts = n * repetitions;
            const int calls = (s.grouped ? 2 : 2 * n) * repetitions;
            std::printf("SAMPLE gu_type=%d down_type=%d n=%d tags=%d arm=%s index=%d block=%d position=%d repetitions=4 expert_pairs=%d logical_normal_return_api_calls=%d logical_output_bytes=%llu host_submit_ms=%.9f host_submit_per_expert_us=%.9f host_submit_checked_drain_ms=%.9f host_submit_checked_drain_per_expert_us=%.9f host_mark_enclosure_ms=%.9f begin_mark_submit_ms=%.9f end_mark_submit_ms=%.9f completed_stream_tag_ms=%.9g completed_stream_tag_per_expert_us=%.9g\n",
                        pair.gu, pair.down, n, int(tags), s.grouped ? "group" : "individual", s.arm_index, s.block, s.position,
                        experts, calls, (unsigned long long)(uint64_t(experts) * matrix * 6), s.submit_ms,
                        s.submit_ms * 1000 / experts, s.submit_drain_ms, s.submit_drain_ms * 1000 / experts,
                        s.mark_enclosure_ms, s.begin_mark_submit_ms, s.end_mark_submit_ms,
                        double(s.queue_interval_ms), tags ? double(s.queue_interval_ms) * 1000 / experts : -1.0);
        }
        std::fflush(stdout);
    }
    drain();
    checked("free_after_checked_terminal_drain", [&] {
        if (tags) { gpu::event_destroy(begin); gpu::event_destroy(end); }
        sycl::free(out, rt.context()); sycl::free(src, rt.context());
    });
    std::puts("PAIR_PASS scope=component_bytes_and_samples low_level_submission_count_proved=0 model_quality_qualified=0");
    std::fflush(stdout);
}
}
int main(int argc, char** argv) {
    const bool smoke_host = argc == 2 && std::strcmp(argv[1], "--smoke-host") == 0;
    const bool full_host = argc == 2 && std::strcmp(argv[1], "--full-host") == 0;
    const bool smoke_tags = argc == 2 && std::strcmp(argv[1], "--smoke-tags") == 0;
    const bool full_tags = argc == 2 && std::strcmp(argv[1], "--full-tags") == 0;
    if (!smoke_host && !full_host && !smoke_tags && !full_tags) {
        std::fputs("Use --smoke-host, --full-host, --smoke-tags or --full-tags\n", stderr); return 2;
    }
    const bool smoke = smoke_host || smoke_tags, tags = smoke_tags || full_tags;
    std::puts("SCOPE standalone_same_work_component prerequisite=Main_all_format_full_byte_guards host_clock=steady completed_tags=production_gpu_helper service=submit_plus_checked_drain exclusive_kernel_service=0 one_submission_proof=0 model32K_fairness=0");
    std::fflush(stdout);
    checked("all_pairs", [&] {
        auto& rt = strata::core::Runtime::get(); auto& q = rt.compute();
        require(rt.device().is_gpu() && rt.device().has(sycl::aspect::usm_device_allocations) &&
                q.has_property<sycl::property::queue::in_order>() && q.get_context() == rt.context() &&
                q.get_device() == rt.device() && strata::core::Runtime::owner_of_stream(&q) == &rt,
                "owned in-order Runtime compute GPU stream");
        const auto& d = rt.device();
        require(d.get_info<sycl::info::device::max_work_group_size>() >= 256 &&
                d.get_info<sycl::info::device::max_work_item_sizes<3>>()[2] >= 256 &&
                d.get_info<sycl::info::device::max_work_item_sizes<2>>()[1] >= 256,
                "group local work shape capability required; refusal has no benchmark fallback");
        // Production experimental profiling tags are separately qualified on this unprofiled Runtime queue.
        // Ordinary enable_profiling is unnecessary; checked event_record/elapsed handles actual capability.
        for (const Pair pair : pairs) {
            pair_run(rt, q, pair, tags, smoke);
            if (smoke) break;
        }
        rt.finish(); // checked by the outer scope; no owning data buffers remain.
        require(samples == (smoke ? 12 : 252) && oracle_pairs == (smoke ? 1 : 7), "finite completion counts");
        // Includes oracle, warmup and post-timing checks, not only timing calls.
        require(individual_returns == (smoke ? 92 : 10724) && group_returns == (smoke ? 64 : 1316) &&
                expert_pairs == (smoke ? 93 : 10899) && readbacks == (smoke ? 8 : 112), "finite logical command counts");
        std::printf("PASS samples=%llu oracle_pairs=%llu individual_normal_returns=%llu group_normal_returns=%llu total_logical_expert_pairs=%llu readbacks=%llu read_bytes=%llu component_executed=1 model_executed=0 low_level_submission_count_proved=0\n",
                    (unsigned long long)samples, (unsigned long long)oracle_pairs, (unsigned long long)individual_returns,
                    (unsigned long long)group_returns, (unsigned long long)expert_pairs, (unsigned long long)readbacks,
                    (unsigned long long)read_bytes);
        std::fflush(stdout);
    });
    std::_Exit(0); // avoids unchecked Runtime destructor waits; this is not a GPU fence.
}
