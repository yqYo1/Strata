// Compare the tuned single-token kernels with the linked GGML CPU oracle.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {
bool pair_cases(unsigned& cases, int dispatch_arm = -1) {
    namespace cpu = strata::kernels::cpu;
    constexpr int rows = 640, n = 2560;
    constexpr float canary = 123.f;
    constexpr uint16_t edge_scales[] = {0x0000, 0x8000, 0x0001, 0x8001,
        0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    for (int padding : {0, 17}) {
        std::array<bool, 128> sign_seen{};
        std::array<bool, 16> nibble_seen{};
        std::array<bool, 256> grid_seen{};
        std::array<bool, sizeof(edge_scales) / sizeof(edge_scales[0])> edge_seen{};
        cpu::NativeFmt f;
        std::string error;
        if (!cpu::native_fmt(18, 20, n, rows, f, error)) return false;
        f.gu_row += padding;
        f.up_off = rows * f.gu_row + (padding ? 33 : 0);
        f.down_off = f.up_off + rows * f.gu_row;
        f.bytes = f.down_off + n * f.d_row;
        // The paired component accepts packed byte inputs, including odd row
        // strides and unaligned scales. Give the GGML oracle aligned copies.
        std::vector<uint8_t> storage(f.bytes + 2), raw_act(f.act_bytes + 2);
        auto* weights = storage.data() + (padding ? 1 : 0);
        std::vector<block_q8_K> act(n / 256);
        const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_XXS)->vec_dot;
        for (unsigned seed = 1; seed <= 8; ++seed) {
            uint32_t state = seed;
            const auto random = [&] { state = state * 1664525u + 1013904223u; return state; };
            for (auto& byte : storage) byte = random() >> 24;
            for (size_t offset : {size_t(0), f.up_off}) for (int r = 0; r < rows; ++r)
                for (int b = 0; b < n / 256; ++b) {
                    auto* block = weights + offset + r * f.gu_row + b * sizeof(block_iq3_xxs);
                    uint16_t scale = uint16_t(0x1400 + (random() & 1023) +
                        (seed == 5 ? 0x8000 : 0));
                    if (seed >= 6) {
                        const size_t edge = (r * 10 + b + seed + (offset != 0)) % edge_seen.size();
                        scale = edge_scales[edge];
                        edge_seen[edge] = true;
                        auto* packed = block + offsetof(block_iq3_xxs, qs);
                        for (unsigned group = 0; group < 8; ++group) {
                            const unsigned ordinal = (r * 10 + b) * 8 + group + seed + (offset != 0);
                            const unsigned nibble = (ordinal / 128) % 16;
                            uint32_t gas = nibble << 28;
                            nibble_seen[nibble] = true;
                            for (unsigned j = 0; j < 4; ++j) {
                                const unsigned sign = (ordinal + 31 * j) % 128;
                                gas |= sign << (7 * j);
                                sign_seen[sign] = true;
                            }
                            std::memcpy(packed + 64 + 4 * group, &gas, sizeof gas);
                            for (unsigned j = 0; j < 8; ++j) {
                                const uint8_t grid = uint8_t(8 * ordinal + j);
                                packed[8 * group + j] = grid;
                                grid_seen[grid] = true;
                            }
                        }
                    }
                    std::memcpy(block, &scale, sizeof scale);
                }
            for (auto& a : act) {
                a.d = seed == 1 ? 0.f : (seed % 2 ? -.002f : .001f);
                for (int j = 0; j < 256; ++j)
                    a.qs[j] = seed == 2 ? (j % 2 ? -127 : 127) :
                        seed == 3 ? (j % 2 ? -128 : 127) : int(random() % 255) - 127;
                for (int j = 0; j < 16; ++j) {
                    a.bsums[j] = 0;
                    for (int k = 0; k < 16; ++k) a.bsums[j] += a.qs[j * 16 + k];
                }
            }
            auto* a = raw_act.data() + (padding ? 1 : 0);
            std::memcpy(a, act.data(), f.act_bytes);
            std::vector<float> expected(rows);
            for (int r = 0; r < rows; ++r) {
                block_iq3_xxs gate[n / 256], up[n / 256];
                std::memcpy(gate, weights + r * f.gu_row, sizeof gate);
                std::memcpy(up, weights + f.up_off + r * f.gu_row, sizeof up);
                float g, u;
                dot(n, &g, 0, gate, 0, act.data(), 0, 1);
                dot(n, &u, 0, up, 0, act.data(), 0, 1);
                expected[r] = (g / (1.f + std::exp(-g))) * u;
            }
            for (const auto bounds : {std::pair{0, rows}, std::pair{1, 639},
                                      std::pair{639, 640}, std::pair{319, 320},
                                      std::pair{640, 640}}) {
                std::vector<float> got(rows + 2, canary);
                if (!cpu::iq3xxs_pair_gu_rows(f, weights, a, got.data() + 1,
                                            bounds.first, bounds.second)) return false;
                for (int r = -1; r <= rows; ++r) {
                    const float value = r >= bounds.first && r < bounds.second ? expected[r] : canary;
                    if (!std::isfinite(got[r + 1]) || std::memcmp(&got[r + 1], &value, sizeof value)) {
                        std::fprintf(stderr, "pair mismatch padding=%d seed=%u rows=%d:%d at=%d\n",
                            padding, seed, bounds.first, bounds.second, r);
                        return false;
                    }
                }
                ++cases;
                if (dispatch_arm >= 0 && padding == 0) {
                    for (auto phase : {cpu::NativeGuPhase::Prefill, cpu::NativeGuPhase::Decode}) {
                        std::fill(got.begin(), got.end(), canary);
                        const void* ap[]{a}; float* fp[]{got.data() + 1};
                        const auto before = cpu::iq3_gu_audit_snapshot();
                        cpu::native_gu_rows(f, weights, ap, 1, fp, bounds.first, bounds.second, phase);
                        const auto after = cpu::iq3_gu_audit_snapshot();
                        const uint64_t active = bounds.second - bounds.first;
                        const uint64_t selected = phase == cpu::NativeGuPhase::Decode && dispatch_arm ? active : 0;
                        if (!before.enabled || !after.enabled || before.overflow || after.overflow ||
                            after.calls - before.calls != 1 || after.nt1_rows - before.nt1_rows != active ||
                            after.pair_rows - before.pair_rows != selected) return false;
                        for (int r = -1; r <= rows; ++r) {
                            const float value = r >= bounds.first && r < bounds.second ? expected[r] : canary;
                            if (!std::isfinite(got[r + 1]) || std::memcmp(&got[r + 1], &value, sizeof value))
                                return false;
                        }
                    }
                }
            }
        }
        const auto complete = [](const auto& seen) {
            for (bool covered : seen) if (!covered) return false;
            return true;
        };
        if (!complete(sign_seen) || !complete(nibble_seen) || !complete(grid_seen) ||
            !complete(edge_seen)) return false;
        std::vector<float> untouched(rows + 2, canary);
        const auto weights_before = storage;
        const auto act_before = raw_act;
        const auto refused = [&](const cpu::NativeFmt& fmt, const uint8_t* w, const void* a,
                                 float* out, int begin, int end) {
            if (cpu::iq3xxs_pair_gu_rows(fmt, w, a, out, begin, end)) return false;
            // An aliased output points into these actual input buffers, rather
            // than untouched. Check both complete targets on every refusal.
            if (storage != weights_before || raw_act != act_before) return false;
            for (float value : untouched) if (value != canary) return false;
            return true;
        };
        const auto reject = [&](const cpu::NativeFmt& fmt, int begin, int end) {
            return refused(fmt, weights, raw_act.data(), untouched.data(), begin, end);
        };
        for (int kind = 0; kind < 11; ++kind) {
            auto bad = f;
            switch (kind) {
                case 0: bad.gu_type = 21; break;
                case 1: bad.gu_act = 8; break;
                case 2: bad.n_embd = 256; break;
                case 3: bad.n_ff = rows - 1; break;
                case 4: bad.gu_row = 10 * sizeof(block_iq3_xxs) - 1; break;
                case 5: --bad.act_bytes; break;
                case 6: bad.up_off = rows * f.gu_row - 1; break;
                case 7: bad.up_off = f.bytes + 1; break;
                case 8: bad.down_off = f.up_off + rows * f.gu_row - 1; break;
                case 9: bad.bytes = f.down_off - 1; break;
                case 10: bad.gu_row = std::numeric_limits<size_t>::max(); break;
            }
            if (!reject(bad, 0, rows)) return false;
        }
        const auto max = std::numeric_limits<uintptr_t>::max();
        if (!reject(f, -1, rows) || !reject(f, 2, 1) || !reject(f, 0, rows + 1) ||
            !refused(f, nullptr, raw_act.data(), untouched.data(), 0, rows) ||
            !refused(f, weights, nullptr, untouched.data(), 0, rows) ||
            !refused(f, weights, raw_act.data(), nullptr, 0, rows) ||
            !refused(f, weights, weights, untouched.data(), 0, rows) ||
            !refused(f, weights, raw_act.data(), reinterpret_cast<float*>(weights), 0, rows) ||
            !refused(f, weights, raw_act.data(), reinterpret_cast<float*>(raw_act.data()), 0, rows) ||
            !refused(f, weights, raw_act.data(), reinterpret_cast<float*>(
                reinterpret_cast<uint8_t*>(untouched.data()) + 1), 0, rows) ||
            !refused(f, reinterpret_cast<const uint8_t*>(max - 16), raw_act.data(), untouched.data(), 0, rows) ||
            !refused(f, weights, reinterpret_cast<const void*>(max - 16), untouched.data(), 0, rows) ||
            !refused(f, weights, raw_act.data(), reinterpret_cast<float*>(max - 16), 0, rows))
            return false;
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    using strata::kernels::cpu::iq256_single_gu_rows;
    int dispatch_arm = -1;
    if (argc != 1) {
        const char* arm = std::getenv("STRATA_IQ3_PAIR");
        const char* audit = std::getenv("STRATA_IQ3_PAIR_AUDIT");
        const char* single = std::getenv("STRATA_IQ_SINGLE_EXACT");
        if (argc != 2 || std::strcmp(argv[1], "--dispatch") || !arm ||
            (std::strcmp(arm, "0") && std::strcmp(arm, "1")) || !audit ||
            std::strcmp(audit, "1") || !single || std::strcmp(single, "0") ||
            std::getenv("STRATA_IQ_MT_MIN") || std::getenv("STRATA_NO_IQ256")) return 1;
        dispatch_arm = arm[0] - '0';
    }
    ggml_cpu_init();
    float untouched = 123.f;
    for (int type : {0, 16, 17, 19, 20, 23}) {
        if (iq256_single_gu_rows(type, nullptr, 0, 0, 256, nullptr, &untouched, 0, 1) ||
            untouched != 123.f) return 1;
    }
    if (!iq256_single_gu_rows(18, nullptr, 0, 0, 256, nullptr, &untouched, 0, 0)) {
        std::puts("single-token tuning unavailable for this compiler/target");
        return 77;
    }
    unsigned cases = 0;
    for (int type : {18, 21, 22}) for (int n : {256, 512, 768, 2560}) {
        const size_t block = type == 18 ? sizeof(block_iq3_xxs) :
                             type == 21 ? sizeof(block_iq3_s) : sizeof(block_iq2_s);
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
    unsigned paired = 0;
    if (!pair_cases(paired, dispatch_arm)) return 1;
    std::printf("IQ single exact PASS: %u single and %u paired cases, bitwise GGML parity and row canaries\n",
        cases, paired);
    if (dispatch_arm >= 0) {
        const auto audit = strata::kernels::cpu::iq3_gu_audit_snapshot();
        if (!audit.enabled || audit.overflow || audit.threads != 1 || audit.calls != 80 ||
            audit.nt1_rows != 20480 || audit.pair_rows != uint64_t(dispatch_arm ? 10240 : 0)) return 1;
        std::printf("IQ3 production dispatch PASS arm=%d calls=80 nt1_rows=20480 pair_rows=%llu prefill_pair_rows=0\n",
            dispatch_arm, static_cast<unsigned long long>(audit.pair_rows));
    }
}
