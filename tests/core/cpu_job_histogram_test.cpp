// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// CPU actual-dispatch fixture. Main compiles/runs; no model/GPU/driver/runtime linkage.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace cpu = strata::kernels::cpu;
namespace core = strata::core;
namespace {
static_assert(cpu::MAXT == 8 && cpu::H == 2560 && cpu::FF == 640, "fixture production native geometry changed");
void check(bool yes, const char* reason) { if (!yes) throw std::runtime_error(reason); }
uint64_t mix(uint64_t& seed) {
    seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27;
    return seed * 2685821657736338717ull;
}
// Same valid block construction as iq_avx2_parity.cpp: all packed code bits, finite fp16 scale.
void blocks(uint8_t* p, size_t bytes, ggml_type type, uint64_t& seed) {
    for (size_t i = 0; i < bytes; i += 8) {
        const auto value = mix(seed);
        std::memcpy(p + i, &value, std::min<size_t>(8, bytes - i));
    }
    const size_t size = ggml_type_size(type);
    check(size > 0 && bytes % size == 0, "complete native block extent");
    for (size_t i = 0; i < bytes; i += size) {
        const float value = (0.5f + static_cast<float>(mix(seed) >> 40) / 16777216.f) * 1e-3f;
        const ggml_fp16_t half = ggml_fp32_to_fp16(value);
        std::memcpy(p + i, &half, sizeof half);
    }
}
class Source final : public core::ExpertSource {
public:
    std::vector<uint8_t> bytes;
    size_t extent;
    explicit Source(const cpu::NativeFmt& fmt) : bytes(fmt.bytes * 16), extent(fmt.bytes) {
        uint64_t seed = 0x6a09e667f3bcc909ull;
        for (size_t i = 0; i < 16; ++i) {
            blocks(bytes.data() + i * extent, fmt.down_off, GGML_TYPE_IQ3_S, seed);
            blocks(bytes.data() + i * extent + fmt.down_off, fmt.bytes - fmt.down_off, GGML_TYPE_IQ4_NL, seed);
        }
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        check(layer >= 0 && layer < 2 && expert >= 0 && expert < 8, "exact layer/expert source guard");
        return bytes.data() + static_cast<size_t>(layer * 8 + expert) * extent;
    }
};
class Capture {
    std::FILE* file_ = nullptr;
    int saved_ = -1;
public:
    Capture() {
        std::fflush(stderr);
        file_ = std::tmpfile();
        check(file_ != nullptr, "capture allocation");
        saved_ = dup(STDERR_FILENO);
        check(saved_ >= 0 && dup2(fileno(file_), STDERR_FILENO) >= 0, "capture descriptor");
    }
    ~Capture() {
        std::fflush(stderr);
        if (saved_ >= 0) { dup2(saved_, STDERR_FILENO); close(saved_); }
        if (file_) std::fclose(file_);
    }
    std::string read() {
        std::fflush(stderr);
        check(std::fseek(file_, 0, SEEK_END) == 0, "capture seek");
        const long size = std::ftell(file_);
        check(size >= 0 && size <= 32768, "request output<=32KiB");
        check(std::fseek(file_, 0, SEEK_SET) == 0, "capture rewind");
        std::string text(static_cast<size_t>(size), '\0');
        check(std::fread(text.data(), 1, text.size(), file_) == text.size(), "capture read");
        return text;
    }
};
using Counts = std::array<uint64_t, 8>;
std::map<std::string, std::string> fields(const std::string& line) {
    check(line.rfind("strata cpu jobs: ", 0) == 0, "only histogram output");
    std::istringstream in(line.substr(17));
    std::map<std::string, std::string> out;
    std::string word;
    while (in >> word) {
        const auto equal = word.find('=');
        check(equal != std::string::npos && out.emplace(word.substr(0, equal), word.substr(equal + 1)).second,
              "unique histogram fields");
    }
    return out;
}
void verify(const std::string& text, bool enabled, const char* status, uint64_t jobs, uint64_t rows,
            const Counts& nt, int parts, bool full) {
    if (!enabled) { check(text.empty(), "defaultOFF emits no histogram"); return; }
    std::istringstream in(text);
    std::string line;
    check(static_cast<bool>(std::getline(in, line)), "one request header");
    const auto header = fields(line);
    check(header.at("status") == status && header.at("eligible") == "1" && header.at("invalid") == "0" &&
          header.at("overflow") == "0" && header.at("serial_parts") == std::to_string(parts), "request status/coverage");
    const bool accepted = header.at("status") == "prompt_complete" && header.at("invalid") == "0" &&
                          header.at("overflow") == "0";
    check(accepted == full, "R255 full acceptance gate");
    for (int layer = 0; layer < 2; ++layer) {
        check(static_cast<bool>(std::getline(in, line)), "actual layer row");
        const auto row = fields(line);
        check(row.at("layer") == std::to_string(layer) && row.at("gu") == std::to_string(GGML_TYPE_IQ3_S) &&
              row.at("down") == std::to_string(GGML_TYPE_IQ4_NL), "exact native fmt/layer attribution");
        check(row.at("jobs") == std::to_string(jobs) && row.at("rows") == std::to_string(rows), "literal expected jobs/rows");
        for (int bucket = 1; bucket <= 8; ++bucket)
            check(row.at("nt" + std::to_string(bucket)) == std::to_string(nt[static_cast<size_t>(bucket - 1)]),
                  "literal expected NT bucket");
    }
    check(!std::getline(in, line), "one header+two rows, no per-job trace or repeated finish");
}
std::vector<float> dispatch(core::ExpertDispatch& d, const std::vector<int32_t>& ids, int tokens, int k) {
    check(d.plan == nullptr && d.remote_count == 0 && d.cache == nullptr && d.host_res == nullptr &&
          d.pcie_num == 0, "CPU-only dispatcher eligibility");
    std::vector<float> x(static_cast<size_t>(tokens) * cpu::H);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.25f * std::sin(static_cast<float>(i % 127) * 0.07f);
    std::vector<float> result;
    for (int layer = 0; layer < 2; ++layer) {
        std::vector<float> output(static_cast<size_t>(tokens * k) * cpu::H, -999.f);
        d.layers = layer; d.failed = false;
        core::expert_pool_dispatch_multi(d, x.data(), ids.data(), tokens, k, output.data());
        check(!d.failed && d.layers == layer + 1, "actual production dispatch and callback layer");
        for (float value : output) check(std::isfinite(value) && value != -999.f, "complete finite actual pool output");
        result.insert(result.end(), output.begin(), output.end());
    }
    return result;
}
void same(const std::vector<float>& a, const std::vector<float>& b) {
    check(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
          "unbound/bound actual CPU float arrays bitwise equal");
}
}  // namespace

int main(int argc, char** argv) try {
    check(argc == 2 && (std::string(argv[1]) == "--off" || std::string(argv[1]) == "--on"), "--off or --on required");
    const bool enabled = std::string(argv[1]) == "--on";
    check(std::getenv("STRATA_POOL_TRACE") == nullptr, "no per-dispatch trace in bounded fixture");
    check(setenv("STRATA_PREFILL_CPU_JOB_HISTOGRAM", enabled ? "1" : "0", 1) == 0, "observer flag before first use");
    if (!cpu::cpu_avx2_ok()) return 77;
    cpu::NativeFmt fmt;
    std::string error;
    check(cpu::native_fmt(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, cpu::H, cpu::FF, fmt, error), "actual native format");
    char path[] = "/tmp/strata-cpu-job-histogram-XXXXXX";
    check(mkdtemp(path) != nullptr, "fresh CPU metadata directory");
    {
        std::ofstream out(std::string(path) + "/native_experts.txt");
        out << "# strata native experts v3 (n_expert 8)\n";
        for (int layer = 0; layer < 2; ++layer)
            out << layer << ' ' << GGML_TYPE_IQ3_S << ' ' << GGML_TYPE_IQ4_NL << ' '
                << static_cast<uint64_t>(layer) * 8 * fmt.bytes << ' ' << fmt.bytes << '\n';
        check(static_cast<bool>(out), "bounded metadata write");
    }
    check(cpu::expert_layout_load(path, 2, 8, error), "production metadata loader");
    const auto& layout = cpu::expert_layout();
    check(layout.native && layout.n_layers == 2 && layout.n_expert == 8 && layout.fmt.size() == 2,
          "exact native layer geometry");
    check(fmt.n_embd == cpu::H && fmt.n_ff == cpu::FF && fmt.bytes > 0 && fmt.bytes <= (16u << 20) &&
          fmt.down_off > 0 && fmt.down_off < fmt.bytes, "bounded real native format shape");
    for (const auto& layer : layout.fmt)
        check(layer.gu_type == GGML_TYPE_IQ3_S && layer.d_type == GGML_TYPE_IQ4_NL &&
              layer.n_embd == fmt.n_embd && layer.n_ff == fmt.n_ff && layer.bytes == fmt.bytes &&
              layer.down_off == fmt.down_off, "loaded exact native format/layer source guards");
    Source source(fmt);
    cpu::ExpertPool pool(2, false, true);
    core::ExpertDispatch d;
    d.pool = &pool; d.src = &source; d.n_expert = 8;
    const std::vector<int32_t> single{0, 1}, repeated(8, 0);
    const std::vector<int32_t> mixed{0,1, 0,2, 0,2, 0,3, 0,3, 0,3, 0,4, 0,4};
    const auto reference1 = dispatch(d, single, 1, 2);
    const auto reference8 = dispatch(d, repeated, 8, 1);
    const auto reference_mix = dispatch(d, mixed, 8, 2);
    // Request1: two read_windows-equivalent bindings aggregate, no counter-reset arithmetic.
    {
        Capture capture;
        {
            core::CpuJobHistogramScope scope(d, 2, 10, 0, true);
            { core::CpuJobHistogramScope::Binding binding(scope); same(reference1, dispatch(d, single, 1, 2)); }
            { core::CpuJobHistogramScope::Binding binding(scope); same(reference8, dispatch(d, repeated, 8, 1)); }
            scope.finish("prompt_complete"); scope.finish("prompt_complete");
        }
        verify(capture.read(), enabled, "prompt_complete", 3, 10, Counts{2,0,0,0,0,0,0,1}, 2, true);
    }
    // Request2: literal mixed NT distribution; must not contain request1's earlier counters.
    {
        Capture capture;
        {
            core::CpuJobHistogramScope scope(d, 2, 9, 0, true);
            { core::CpuJobHistogramScope::Binding binding(scope); same(reference_mix, dispatch(d, mixed, 8, 2)); }
            scope.finish("prompt_complete");
        }
        verify(capture.read(), enabled, "prompt_complete", 5, 16, Counts{1,2,1,0,0,0,0,1}, 1, true);
    }
    // Actual dispatch guard failure after valid math: partial state is never full histogram evidence.
    {
        Capture capture;
        {
            core::CpuJobHistogramScope scope(d, 2, 10, 0, true);
            core::CpuJobHistogramScope::Binding binding(scope);
            same(reference1, dispatch(d, single, 1, 2));
            float x[cpu::H]{}, output[cpu::H]{};
            core::expert_pool_dispatch_multi(d, x, single.data(), 9, 1, output);
            check(d.failed, "actual production MAXT guard failure");
        }
        verify(capture.read(), enabled, "error_or_early_exit", 2, 2, Counts{2,0,0,0,0,0,0,0}, 1, false);
    }
    std::printf("PASS actual CPU dispatch/pool float parity, NT1/NT8/mixed, request reset, partial status; metadata=%s\n", path);
    // Retain the tiny owned synthetic metadata for Main's serialized retention decision; no cleanup here.
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL CPU histogram fixture: %s\n", error.what());
    return 1;
}
