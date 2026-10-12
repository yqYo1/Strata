// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
namespace cpu = strata::kernels::cpu;
namespace {
bool same_bits(const std::array<float, 642>& a, const std::array<float, 642>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) return false;
    return true;
}
}

int main() {
    if (!cpu::cpu_avx2_ok()) return 77;
    constexpr size_t row = 1100, up = 640 * row;
    cpu::NativeFmt fmt;
    std::string error;
    if (!cpu::native_fmt(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, 2560, 640, fmt, error)) return 1;
    // The fixture uses only the GU range; Down is never read by either tested kernel.
    // All IQ3_S index/sign/scale bits are valid. Replace random headers with finite FP16 scales.
    std::vector<uint8_t> storage(2 * up + 1);
    uint8_t* blob = storage.data() + 1;
    uint64_t state = 0x123456789abcdefULL;
    for (size_t i = 0; i < 2 * up; ++i) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        blob[i] = static_cast<uint8_t>(state);
    }
    for (size_t i = 0; i < 2 * up; i += 110) {
        const ggml_half d = ggml_fp32_to_fp16((i / 110 % 2 ? -1.f : 1.f) * 0.001f);
        std::memcpy(blob + i, &d, sizeof(d));
    }
    std::array<block_q8_K, 10> activation{};
    std::array<float, 2560> x{};
    const void* act[] = {activation.data()};
    std::array<float, 642> baseline{}, candidate{};
    float* base[] = {baseline.data() + 1};
    float* out[] = {candidate.data() + 1};
    const std::array<std::array<int, 2>, 3> ranges{{{0, 1}, {1, 639}, {639, 640}}};
    for (int pattern = 0; pattern < 4; ++pattern) {
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = pattern == 0 ? 0.f : pattern == 1 ? std::sin(static_cast<float>(i)) :
                   pattern == 2 ? (i % 2 ? -4.f : 4.f) : static_cast<float>(static_cast<int>(i % 31) - 15) / 8.f;
        cpu::q8k_quant_avx2(x.data(), activation.data(), 2560);
        for (int variant : {0, cpu::kIq256Gather}) {
            baseline.fill(123.f); candidate.fill(123.f);
            cpu::iq256_gu_rows_v(variant, 21, blob, row, up, 2560, act, 1, base, 0, 640);
            for (const auto& range : ranges)
                if (!cpu::iq256_gu_rows_paired_v(variant, 21, blob, row, up, 2560, act, 1, out, range[0], range[1])) return 1;
            for (size_t r = 0; r < baseline.size(); ++r) {
                if (!std::isfinite(candidate[r]) || std::bit_cast<uint32_t>(baseline[r]) != std::bit_cast<uint32_t>(candidate[r])) {
                    std::fprintf(stderr, "parity pattern=%d variant=%d row=%zu\n", pattern, variant, r);
                    return 1;
                }
            }
            const auto saved = candidate;
            for (int nt : {0, 2, 8})
                if (cpu::iq256_gu_rows_paired_v(variant, 21, blob, row, up, 2560, act, nt, out, 0, 640)) return 1;
            if (cpu::iq256_gu_rows_paired_v(cpu::kIq256Vnni, 21, blob, row, up, 2560, act, 1, out, 0, 640) ||
                cpu::iq256_gu_rows_paired_v(variant, 20, blob, row, up, 2560, act, 1, out, 0, 640) ||
                cpu::iq256_gu_rows_paired_v(variant, 21, blob, row, up, 2560, act, 1, out, 0, 641) ||
                !same_bits(saved, candidate)) return 1;
        }
    }
    baseline.fill(123.f); candidate.fill(123.f);
    cpu::iq256_gu_rows_v(cpu::iq256_variant(), 21, blob, row, up, 2560, act, 1, base, 0, 640);
    cpu::iq256_gu_rows(21, blob, row, up, 2560, act, 1, out, 0, 640);
    if (!same_bits(baseline, candidate)) return 1;
    std::puts("IQ3_S paired GU actual-kernel parity complete");
    return 0;
}
