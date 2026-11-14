#pragma once

#include "strata/kernels/cpu/native_expert.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <thread>
#include <tuple>

namespace strata::core {

// Request-owned, host-only diagnostic. The verifier's serialized caller records
// FINAL jobs after pool completion. Workers and GPU commands never own this
// object. Wrong-thread calls only latch an atomic error; TLS is not identity.
// Main-authored from this fork's dispatcher contract; no external kernel code.
class CpuWorkCensus {
public:
    using Format = strata::kernels::cpu::NativeFmt;
    static constexpr int kLayers = 48, kExperts = 512, kNT = 8, kEntries = 128;
    struct Bin {
        Format format{};
        std::array<uint64_t, kNT> jobs{};
        std::array<uint64_t, 4> entries{}; // CPU, resident, PCIe, peer
        uint64_t calls = 0, pool_ns = 0;
    };

    CpuWorkCensus(uint64_t request, int layers, int experts, const int32_t* slots, const Format* formats)
        : owner_(std::this_thread::get_id()), request_(request), layers_(layers), experts_(experts) {
        if (!request || layers < 1 || layers > kLayers || experts < 1 || experts > kExperts || !slots || !formats) {
            invalidate("request_geometry"); return;
        }
        for (int l = 0; l < layers; ++l) {
            const auto& f = formats[l];
            if (f.gu_type < 0 || f.d_type < 0 || f.n_embd <= 0 || f.n_ff <= 0 || !f.bytes) {
                invalidate("native_format"); return;
            }
            int b = 0;
            while (b < bins_used_ && !same_format(bins_[b].format, f)) ++b;
            if (b == bins_used_) bins_[bins_used_++].format = f;
            layer_bin_[l] = b;
            for (int e = 0; e < experts; ++e) {
                const int32_t slot = slots[l * experts + e];
                if (slot < -1) { invalidate("residency_slot"); return; }
                slots_[l * kExperts + e] = slot;
            }
        }
    }
    CpuWorkCensus(const CpuWorkCensus&) = delete;
    CpuWorkCensus& operator=(const CpuWorkCensus&) = delete;

    static bool checked_add(uint64_t& dst, uint64_t value) noexcept {
        if (value > std::numeric_limits<uint64_t>::max() - dst) return false;
        dst += value; return true;
    }
    void invalidate(const char* why) noexcept { if (!error_) error_ = why; }
    bool ok() const noexcept { return !error_ && !wrong_thread_.load(std::memory_order_relaxed); }
    bool closed() const noexcept { return closed_; }
    const auto& bins() const noexcept { return bins_; }
    int bins_used() const noexcept { return bins_used_; }
    uint64_t windows() const noexcept { return windows_; }
    uint64_t cpu_jobs() const noexcept { return cpu_jobs_; }
    uint64_t cpu_entries() const noexcept { return entries_[0]; }

    void write_begin(FILE* out, int64_t prompt_tokens) const {
        std::fprintf(out, "strata cpu work: begin request=%llu phase=decode prompt_tokens=%lld "
            "layers=%d experts=%d cache=restored_decode_snapshot_immutable\n",
            (unsigned long long)request_, (long long)prompt_tokens, layers_, experts_);
        if (!ok()) return;
        // The exact CPU-vs-resident set is small and recoverable without keeping
        // route IDs, tensor dumps, or pretending profile rank is live residency.
        for (int l = 0; l < layers_; ++l) {
            std::fprintf(out, "strata cpu work: cache request=%llu layer=%d bin=%d resident_bits=",
                (unsigned long long)request_, l, layer_bin_[l]);
            for (int block = 0; block < (experts_ + 63) / 64; ++block) {
                uint64_t bits = 0;
                for (int bit = 0; bit < 64 && block * 64 + bit < experts_; ++bit)
                    if (slots_[l * kExperts + block * 64 + bit] >= 0) bits |= uint64_t{1} << bit;
                std::fprintf(out, "%016llx", (unsigned long long)bits);
            }
            std::fputc('\n', out);
        }
    }

    void begin_window(uint64_t window, int64_t position, int nt) noexcept {
        if (!on_owner() || !ok()) return;
        if (closed_ || in_window_ || window == 0 || window - 1 != windows_ ||
            position < 0 || (windows_ && position <= position_) || nt < 1 || nt > kNT) {
            invalidate("window_sequence"); return;
        }
        position_ = position; nt_ = nt; next_layer_ = 0; in_window_ = true;
    }

    template<class Job>
    void record(int64_t layer, int64_t nt, int64_t k, int64_t experts, const int32_t* slots,
                const Format& format, const int32_t* ids, const int32_t* kind,
                const int16_t* job_of, const Job* jobs, int njobs, int64_t pool_ns) noexcept {
        if (!on_owner() || !ok()) return;
        if (!in_window_ || layer != next_layer_ || layer < 0 || layer >= layers_ || nt != nt_ ||
            k < 1 || k > kEntries / nt_ || experts != experts_ || !slots || !ids || !kind || !job_of ||
            njobs < 0 || njobs > nt * k || (njobs && !jobs) || pool_ns < 0) {
            invalidate("callback_geometry_or_sequence"); return;
        }
        const int b = layer_bin_[layer];
        if (fanout_ && fanout_ != k) { invalidate("fanout_changed"); return; }
        fanout_ = (int)k;
        if (!same_format(format, bins_[b].format)) { invalidate("format_changed"); return; }
        for (int e = 0; e < experts_; ++e)
            if (slots[layer * experts_ + e] != slots_[layer * kExperts + e]) {
                invalidate("residency_changed"); return;
            }
        std::array<int, kEntries> counts{};
        std::array<uint64_t, 4> entries{};
        uint64_t pcie_jobs = 0;
        for (int i = 0; i < nt * k; ++i) {
            const int e = ids[i];
            if (kind[i] < -1 || kind[i] > 2) { invalidate("route_kind"); return; }
            const int route = kind[i] + 1;
            if (e < 0 || e >= experts_ ||
                ((slots_[layer * kExperts + e] >= 0) != (route == 1))) {
                invalidate("route_or_residency"); return;
            }
            for (int q = 0; q < i; ++q)
                if (ids[q] == e && kind[q] != kind[i]) { invalidate("expert_route_changed"); return; }
            // Duplicate IDs within a token violate the MAXT job contract.
            for (int j = (i / k) * k; j < i; ++j)
                if (ids[j] == e) { invalidate("duplicate_token_expert"); return; }
            ++entries[route];
            if (route == 0) {
                const int j = job_of[e];
                if (j < 0 || j >= njobs) { invalidate("job_map"); return; }
                // Different expert IDs cannot alias one final CPU job.
                for (int q = 0; q < i; ++q)
                    if (kind[q] < 0 && job_of[ids[q]] == j && ids[q] != e) {
                        invalidate("job_alias"); return;
                    }
                ++counts[j];
            } else if (route == 2) {
                bool first = true;
                for (int j = 0; j < i; ++j) if (ids[j] == e) first = false;
                if (first) ++pcie_jobs;
            }
        }
        std::array<uint64_t, kNT> histogram{};
        for (int j = 0; j < njobs; ++j) {
            const int n = jobs[j].nt;
            if (n < 1 || n > nt_ || n != counts[j]) { invalidate("job_nt"); return; }
            ++histogram[n - 1];
        }
        auto& bin = bins_[b];
        for (int n = 0; n < kNT; ++n) add(bin.jobs[n], histogram[n]);
        for (int route = 0; route < 4; ++route) {
            add(bin.entries[route], entries[route]); add(entries_[route], entries[route]);
        }
        add(bin.calls, 1); add(bin.pool_ns, (uint64_t)pool_ns);
        add(cpu_jobs_, (uint64_t)njobs); add(pcie_jobs_, pcie_jobs); add(callbacks_, 1);
        ++next_layer_;
    }

    void end_window() noexcept {
        if (!on_owner() || !ok()) return;
        if (!in_window_ || next_layer_ != layers_) { invalidate("missing_layer"); return; }
        in_window_ = false; add(windows_, 1); add(rows_, (uint64_t)nt_);
    }

    void finish(FILE* out, int64_t windows, int64_t rows, int64_t jobs, int64_t entries,
                int64_t hits, int64_t pcie_jobs) noexcept {
        if (!on_owner()) return;
        if (in_window_ || closed_ || windows <= 0 || rows < 0 || jobs < 0 || entries < 0 || hits < 0 || pcie_jobs < 0 ||
            windows_ != (uint64_t)windows || rows_ != (uint64_t)rows || cpu_jobs_ != (uint64_t)jobs ||
            entries_[0] != (uint64_t)entries || entries_[1] != (uint64_t)hits || entries_[3] != 0 ||
            pcie_jobs_ != (uint64_t)pcie_jobs) invalidate("request_conservation");
        closed_ = true;
        for (int b = 0; b < bins_used_; ++b) {
            const auto& bin = bins_[b]; const auto& f = bin.format;
            std::fprintf(out, "strata cpu work: bin request=%llu bin=%d gu=%d down=%d H=%lld FF=%lld "
                "blob_bytes=%llu gu_act=%d down_act=%d gu_row=%llu down_row=%llu act_bytes=%llu h_bytes=%llu "
                "calls=%llu pool_ns=%llu CPU=%llu VRAM=%llu PCIe=%llu peer=%llu jobs_NT=",
                (unsigned long long)request_, b, f.gu_type, f.d_type, (long long)f.n_embd, (long long)f.n_ff,
                (unsigned long long)f.bytes, f.gu_act, f.d_act, (unsigned long long)f.gu_row,
                (unsigned long long)f.d_row, (unsigned long long)f.act_bytes, (unsigned long long)f.h_bytes,
                (unsigned long long)bin.calls, (unsigned long long)bin.pool_ns,
                (unsigned long long)bin.entries[0], (unsigned long long)bin.entries[1],
                (unsigned long long)bin.entries[2], (unsigned long long)bin.entries[3]);
            for (int n = 0; n < kNT; ++n) std::fprintf(out, "%s%llu", n ? "," : "", (unsigned long long)bin.jobs[n]);
            std::fputc('\n', out);
        }
        std::fprintf(out, "strata cpu work: end request=%llu phase=decode valid=%d windows=%llu rows=%llu "
            "callbacks=%llu k=%d cpu_jobs=%llu CPU=%llu VRAM=%llu PCIe=%llu peer=%llu pcie_jobs=%llu error=%s\n",
            (unsigned long long)request_, (int)ok(), (unsigned long long)windows_, (unsigned long long)rows_,
            (unsigned long long)callbacks_, fanout_, (unsigned long long)cpu_jobs_, (unsigned long long)entries_[0],
            (unsigned long long)entries_[1], (unsigned long long)entries_[2], (unsigned long long)entries_[3],
            (unsigned long long)pcie_jobs_, wrong_thread_.load() ? "wrong_thread" : error_ ? error_ : "none");
        std::fflush(out);
    }
    void abort(FILE* out) noexcept {
        invalidate("request_aborted"); finish(out, 0, 0, 0, 0, 0, 0);
    }

private:
    static bool same_format(const Format& a, const Format& b) noexcept {
        auto key = [](const Format& f) { return std::tie(f.gu_type, f.d_type, f.gu_act, f.d_act, f.n_embd,
            f.n_ff, f.gu_row, f.d_row, f.up_off, f.down_off, f.bytes, f.act_bytes, f.h_bytes); };
        return key(a) == key(b);
    }
    bool on_owner() noexcept {
        if (std::this_thread::get_id() == owner_) return true;
        wrong_thread_.store(true, std::memory_order_relaxed); return false;
    }
    void add(uint64_t& dst, uint64_t value) noexcept { if (!checked_add(dst, value)) invalidate("overflow"); }
    const std::thread::id owner_;
    const uint64_t request_;
    const int layers_, experts_;
    std::atomic<bool> wrong_thread_{false};
    const char* error_ = nullptr;
    std::array<int32_t, kLayers * kExperts> slots_{};
    std::array<int, kLayers> layer_bin_{};
    std::array<Bin, kLayers> bins_{};
    int bins_used_ = 0, nt_ = 0, next_layer_ = 0, fanout_ = 0;
    int64_t position_ = -1;
    uint64_t windows_ = 0, rows_ = 0, callbacks_ = 0, cpu_jobs_ = 0, pcie_jobs_ = 0;
    std::array<uint64_t, 4> entries_{};
    bool in_window_ = false, closed_ = false;
};
} // namespace strata::core
