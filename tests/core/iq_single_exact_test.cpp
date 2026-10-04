// Compare the tuned single-token kernels with the linked GGML CPU oracle.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

int main() {
    using strata::kernels::cpu::iq256_single_gu_rows;
    ggml_cpu_init();
    float untouched = 123.f;
    for (int type : {0, 16, 17, 19, 20, 21, 23}) {
        if (iq256_single_gu_rows(type, nullptr, 0, 0, 256, nullptr, &untouched, 0, 1) ||
            untouched != 123.f) return 1;
    }
    if (!iq256_single_gu_rows(18, nullptr, 0, 0, 256, nullptr, &untouched, 0, 0)) {
        std::puts("single-token tuning unavailable for this compiler/target");
        return 77;
    }
    unsigned cases = 0;
    for (int type : {18, 22}) for (int n : {256, 512, 768, 2560}) {
        const size_t block = type == 18 ? sizeof(block_iq3_xxs) : sizeof(block_iq2_s);
        const size_t row = n / 256 * block + 16; // Exercise padded row strides.
        constexpr int rows = 9;
        const size_t up = rows * row + 32;
        std::vector<uint8_t> weights(2 * up);
        std::vector<block_q8_K> act(n / 256);
        for (unsigned seed = 1; seed <= 8; ++seed) {
            uint32_t state = seed;
            auto random = [&] { state = state * 1664525u + 1013904223u; return state; };
            for (auto& byte : weights) byte = random() >> 24;
            // All IQ codebook indices are valid; limit FP16 scales to finite small values.
            for (size_t half : {size_t(0), up}) for (int r = 0; r < rows; ++r)
                for (int b = 0; b < n / 256; ++b) {
                    const uint16_t scale = uint16_t(0x1400 + (random() & 0x3ff));
                    std::memcpy(weights.data() + half + r * row + b * block, &scale, 2);
                }
            for (auto& a : act) {
                a.d = seed == 1 ? 0.f : (seed % 2 ? .002f : -.001f);
                for (int j = 0; j < 256; ++j)
                    a.qs[j] = seed == 2 ? (j % 2 ? -127 : 127) : int(random() % 255) - 127;
                for (int j = 0; j < 16; ++j) {
                    a.bsums[j] = 0;
                    for (int k = 0; k < 16; ++k) a.bsums[j] += a.qs[j * 16 + k];
                }
            }
            const auto dot = ggml_get_type_traits_cpu(ggml_type(type))->vec_dot;
            for (const auto bounds : {std::pair{0, rows}, std::pair{1, 8},
                                      std::pair{8, 9}, std::pair{4, 4}}) {
                std::vector<float> ref(rows + 2, 123.f), got = ref;
                for (int r = bounds.first; r < bounds.second; ++r) {
                    float g, u;
                    dot(n, &g, 0, weights.data() + r * row, 0, act.data(), 0, 1);
                    dot(n, &u, 0, weights.data() + up + r * row, 0, act.data(), 0, 1);
                    ref[r + 1] = (g / (1.f + std::exp(-g))) * u;
                }
                if (!iq256_single_gu_rows(type, weights.data(), row, up, n, act.data(),
                                          got.data() + 1, bounds.first, bounds.second)) return 1;
                for (size_t i = 0; i < got.size(); ++i) {
                    if (!std::isfinite(got[i]) || std::memcmp(&got[i], &ref[i], sizeof(float))) {
                        std::fprintf(stderr, "mismatch type=%d n=%d seed=%u rows=%d:%d at=%zu\n",
                                     type, n, seed, bounds.first, bounds.second, i);
                        return 1;
                    }
                }
                ++cases;
            }
        }
    }
    std::printf("IQ single exact PASS: %u cases, bitwise GGML parity and row canaries\n", cases);
}
