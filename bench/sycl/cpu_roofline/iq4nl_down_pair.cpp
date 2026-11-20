// Strata component benchmark for the Zen 3 decode IQ4_NL Down path.
// The reference is the linked GGML CPU dot used by native_down_rows.
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kInput = 640, kRows = 2560, kBlocks = kInput / QK4_NL;
constexpr int kExperts = 128;
constexpr size_t kExpertBlocks = size_t(kRows) * kBlocks;
constexpr float kCanary = 123.f;
static_assert(QK4_NL == QK8_0 && kBlocks == 20);
static_assert(sizeof(block_iq4_nl) == 18 && sizeof(block_q8_0) == 34);

using Activation = std::array<block_q8_0, kBlocks>;
using Output = std::array<float, kRows + 2>;
using Dot = ggml_vec_dot_t;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

float half_to_float(uint16_t bits) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
}

// One activation load and scale conversion serve both adjacent rows. Each
// row keeps GGML's even/odd block accumulators and its horizontal sum tree.
template<int Rows>
void accumulate_rows(const block_iq4_nl* weights, int block, const block_q8_0& x,
                     const __m128i lookup, __m256 (&sums)[Rows]) {
    const auto input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs));
    const float input_scale = half_to_float(x.d);
    const auto mask = _mm_set1_epi8(15);
    const auto ones = _mm256_set1_epi16(1);
    for (int row = 0; row < Rows; ++row) {
        const auto& w = weights[row * kBlocks + block];
        const auto nibbles = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w.qs));
        const auto lo = _mm_shuffle_epi8(lookup, _mm_and_si128(nibbles, mask));
        const auto hi = _mm_shuffle_epi8(lookup, _mm_and_si128(_mm_srli_epi16(nibbles, 4), mask));
        const auto q = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        const auto products = _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(input, q));
        const auto lanes = _mm256_cvtepi32_ps(_mm256_madd_epi16(products, ones));
        sums[row] = _mm256_fmadd_ps(_mm256_set1_ps(input_scale * half_to_float(w.d)), lanes, sums[row]);
    }
}

float finish_row(__m256 even, __m256 odd) {
    const auto v = _mm256_add_ps(even, odd);
    auto sum = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    return _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum)));
}

template<int Rows>
void pair_rows(const block_iq4_nl* weights, const Activation& input, float* result) {
    const auto lookup = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    __m256 even[Rows]{}, odd[Rows]{};
    for (int block = 0; block < kBlocks; block += 2) {
        accumulate_rows(weights, block, input[block], lookup, even);
        accumulate_rows(weights, block + 1, input[block + 1], lookup, odd);
    }
    for (int row = 0; row < Rows; ++row) result[row] = finish_row(even[row], odd[row]);
}

void decode_iq4nl_down_pair(const block_iq4_nl* weights, const Activation& input,
                            float* output, int first, int last) {
    for (; first + 1 < last; first += 2)
        pair_rows<2>(weights + first * kBlocks, input, output + first);
    if (first < last) pair_rows<1>(weights + first * kBlocks, input, output + first);
}

void reference(Dot dot, const block_iq4_nl* weights, const Activation& input,
               float* output, int first, int last) {
    for (int row = first; row < last; ++row)
        dot(kInput, output + row, 0, weights + row * kBlocks, 0, input.data(), 0, 1);
}

void check_rows(Dot dot, const block_iq4_nl* weights, const Activation& input, int first, int last) {
    Output want, got;
    want.fill(kCanary);
    got.fill(kCanary);
    reference(dot, weights, input, want.data() + 1, first, last);
    decode_iq4nl_down_pair(weights, input, got.data() + 1, first, last);
    for (size_t i = 0; i < want.size(); ++i) {
        if (!std::isfinite(want[i]) || std::bit_cast<uint32_t>(want[i]) != std::bit_cast<uint32_t>(got[i])) {
            std::cerr << "IQ4_NL mismatch in " << first << ".." << last << " at guarded index " << i << '\n';
            throw std::runtime_error("GGML parity or output guard");
        }
    }
}

void synthetic(Dot dot) {
    constexpr auto scales = std::to_array<uint16_t>({0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff,
        0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff});
    constexpr std::array bounds{std::array{0, kRows}, std::array{1, kRows},
        std::array{73, 714}, std::array{kRows - 1, kRows}, std::array{kRows, kRows}};
    std::vector<block_iq4_nl> weights(kExpertBlocks);
    for (unsigned seed = 0; seed < scales.size(); ++seed) {
        Activation input{};
        for (int b = 0; b < kBlocks; ++b) {
            input[b].d = scales[(b + seed) % scales.size()];
            for (int i = 0; i < QK8_0; ++i)
                input[b].qs[i] = int8_t(int((37 * i + 113 * b + 47 * seed) & 255) - 128);
        }
        for (size_t i = 0; i < weights.size(); ++i) {
            auto& w = weights[i];
            w.d = scales[(i + 5 * seed) % scales.size()];
            for (size_t j = 0; j < sizeof w.qs; ++j) w.qs[j] = uint8_t(i + 17 * j + 13 * seed);
        }
        for (auto [first, last] : bounds) check_rows(dot, weights.data(), input, first, last);
    }
    std::cout << "IQ4_NL Down synthetic exact cases " << scales.size() * bounds.size() << '\n';
}

uint64_t offset_from(std::string_view text) {
    uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    require(error == std::errc{} && end == text.data() + text.size(), "Down tensor byte offset");
    return value;
}

void time_rows(Dot dot, const char* model_path, uint64_t offset, bool rotating, bool reverse) {
    constexpr int kIterations = 48 * kExperts;
    const int experts = rotating ? kExperts : 1;
    std::vector<block_iq4_nl> weights(experts * kExpertBlocks);
    std::ifstream tensor(model_path, std::ios::binary | std::ios::ate);
    const std::streamoff extent = tensor.tellg();
    const auto bytes = weights.size() * sizeof(weights[0]);
    require(extent >= 0 && offset <= uint64_t(extent) && bytes <= uint64_t(extent) - offset,
        "actual contiguous IQ4_NL Down tensor extent");
    tensor.seekg(static_cast<std::streamoff>(offset));
    tensor.read(reinterpret_cast<char*>(weights.data()), static_cast<std::streamsize>(bytes));
    require(bool(tensor), "read actual model Down rows");

    strata::kernels::cpu::NativeFmt format;
    std::string error;
    require(strata::kernels::cpu::native_fmt(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, kRows, kInput, format, error),
        "GGML IQ4_NL Down geometry");
    require(format.d_act == GGML_TYPE_Q8_0 && format.h_bytes == sizeof(Activation), "Q8_0 activation geometry");
    std::array<Activation, 4> inputs{};
    uint32_t random = 47113;
    for (auto& input : inputs) {
        std::array<float, kInput> values;
        for (auto& value : values) {
            random = 1664525u * random + 1013904223u;
            value = (int(random >> 8) - 8388608) * 0x1p-23f;
        }
        strata::kernels::cpu::native_quant_h(format, values.data(), input.data());
    }
    auto check_all = [&] {
        for (int expert = 0; expert < experts; ++expert)
            for (const auto& input : inputs)
                check_rows(dot, weights.data() + expert * kExpertBlocks, input, 0, kRows);
    };
    check_all();

    auto run = [&](bool candidate) {
        using Clock = std::chrono::steady_clock;
        Output out;
        out.fill(kCanary);
        uint64_t fingerprint = 0;
        const auto begin = Clock::now();
        for (int i = 0; i < kIterations; ++i) {
            const auto* w = weights.data() + (i % experts) * kExpertBlocks;
            const auto& x = inputs[(i + i / kExperts) % inputs.size()];
            if (candidate) decode_iq4nl_down_pair(w, x, out.data() + 1, 0, kRows);
            else reference(dot, w, x, out.data() + 1, 0, kRows);
            fingerprint = fingerprint * 1099511628211ull + std::bit_cast<uint32_t>(out[1 + (53 * i) % kRows]);
        }
        const double elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
        require(out.front() == kCanary && out.back() == kCanary, "timed output guard");
        return std::pair{elapsed, fingerprint};
    };
    for (int repeat = 0; repeat < 5; ++repeat) {
        const bool candidate_first = reverse != ((repeat & 1) != 0);
        const auto first = run(candidate_first);
        const auto second = run(!candidate_first);
        require(first.second == second.second, "timed output fingerprints differ");
        std::cout << std::setprecision(12) << (candidate_first ? "BA" : "AB") << ' ' << kIterations << ' '
            << (candidate_first ? second.first : first.first) << ' '
            << (candidate_first ? first.first : second.first) << ' ' << first.second << '\n';
    }
    check_all();
}

}  // namespace

int main(int argc, char** argv) try {
    ggml_cpu_init();
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
    require(traits && traits->vec_dot_type == GGML_TYPE_Q8_0 && traits->vec_dot, "GGML reference dot");
    if (argc == 2 && std::string_view(argv[1]) == "check") {
        synthetic(traits->vec_dot);
        return 0;
    }
    require(argc == 6 && std::string_view(argv[1]) == "time", "check | time FILE OFFSET reuse|rotate AB|BA");
    const std::string_view pattern = argv[4], order = argv[5];
    require((pattern == "reuse" || pattern == "rotate") && (order == "AB" || order == "BA"), "timing case");
    time_rows(traits->vec_dot, argv[2], offset_from(argv[3]), pattern == "rotate", order == "BA");
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
