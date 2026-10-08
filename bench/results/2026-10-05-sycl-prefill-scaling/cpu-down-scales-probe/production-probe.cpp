#include "strata/artifact/gguf_reader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sched.h>
#include <vector>

namespace strata::kernels::cpu_original {
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
}
namespace strata::kernels::cpu {
void iq4nl256_down_rows_scales(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
}
using Function = void (*)(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int);
static Function original = strata::kernels::cpu_original::iq4nl256_down_rows;
static Function alternate = strata::kernels::cpu::iq4nl256_down_rows_scales;
static uint64_t checked = 0;
static void require(bool ok, const char* reason) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", reason); std::exit(1); }
}
static double milliseconds() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Activations {
    std::vector<uint8_t> bytes;
    const void* pointers[8];
    Activations(int n, std::mt19937& rng, bool edges) {
        const size_t size = ggml_row_size(GGML_TYPE_Q8_0, n);
        bytes.resize(8 * size);
        std::vector<float> x(n);
        std::normal_distribution<float> normal(0.f, .5f);
        const auto* quant = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        for (int t = 0; t < 8; ++t) {
            for (int i = 0; i < n; ++i)
                x[i] = edges && t == 1 ? 0.f : edges && t == 2 ? .125f :
                       edges && t == 3 ? (i % 2 ? -.25f : .25f) : normal(rng);
            pointers[t] = bytes.data() + t * size;
            quant->from_float(x.data(), bytes.data() + t * size, n);
        }
    }
};

static void verify(const uint8_t* w, size_t stride, int n, int rows, const Activations& act) {
    for (int nt = 1; nt <= 8; ++nt) {
        std::vector<float> a(8 * (rows + 2), .125f), b(a);
        float* ap[8]; float* bp[8];
        for (int t = 0; t < 8; ++t) { ap[t] = a.data() + t * (rows + 2) + 1; bp[t] = b.data() + t * (rows + 2) + 1; }
        original(w, stride, n, act.pointers, nt, ap, 2, rows - 1);
        alternate(w, stride, n, act.pointers, nt, bp, 2, rows - 1);
        for (size_t i = 0; i < a.size(); ++i) {
            require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite down output");
            if (std::memcmp(a.data() + i, b.data() + i, sizeof(float))) {
                std::fprintf(stderr, "n=%d nt=%d index=%zu original=%.9g alternate=%.9g\n", n, nt, i, a[i], b[i]);
                require(false, "down output bits differ");
            }
            ++checked;
        }
        for (int t = 0; t < 8; ++t) for (int r = 0; r < rows + 2; ++r)
            if (t >= nt || r < 3 || r >= rows)
                require(a[t * (rows + 2) + r] == .125f, "original guard overwritten");
    }
}

int main(int argc, char** argv) {
    require(argc == 2, "provide real GGUF shard");
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    cpu_set_t cpus; CPU_ZERO(&cpus); CPU_SET(2, &cpus);
    require(sched_setaffinity(0, sizeof(cpus), &cpus) == 0, "cannot pin CPU 2");
    ggml_cpu_init();
    std::mt19937 rng(1127);
    for (int n : {32, 96, 640, 768, 2560, 4096}) for (int offset : {0, 1, 3, 7, 15}) for (int padding : {0, 1, 13}) {
        const int rows = 19;
        const size_t payload = n / 32 * sizeof(block_iq4_nl), stride = payload + padding;
        std::vector<uint8_t> storage(rows * stride + offset + 512);
        uint8_t* weights = storage.data() + offset;
        for (auto& value : storage) value = rng();
        for (int r = 0; r < rows; ++r) for (int k = 0; k < n / 32; ++k) {
            const uint16_t scale = rng() & 0xfbff; // finite half values, including signs/subnormals
            std::memcpy(weights + r * stride + k * sizeof(block_iq4_nl), &scale, sizeof(scale));
        }
        Activations act(n, rng, true);
        verify(weights, stride, n, rows, act);
    }
    std::printf("{\"kind\":\"synthetic\",\"checked_floats\":%llu,\"all_bits_equal\":true}\n", (unsigned long long)checked);
    strata::GgufFile file(argv[1]);
    std::vector<uint8_t> streaming;
    int real_cases = 0, n = 0, rows = 0;
    size_t stride = 0, expert_bytes = 0;
    for (int layer = 0; layer < 48; ++layer) {
        const strata::TensorInfo* down = nullptr;
        const auto name = "blk." + std::to_string(layer) + ".ffn_down_exps.weight";
        for (const auto& t : file.tensors()) if (t.name == name) down = &t;
        require(down != nullptr, "missing down tensor");
        if (down->type != GGML_TYPE_IQ4_NL) continue;
        const int width = down->shape[0], height = down->shape[1];
        const size_t row_size = ggml_row_size(GGML_TYPE_IQ4_NL, width), bytes = row_size * height;
        Activations act(width, rng, true);
        for (int e : {0, 7, 31}) { verify(file.tensor_data(*down) + e * bytes, row_size, width, height, act); ++real_cases; }
        if (streaming.empty()) {
            n = width; rows = height; stride = row_size; expert_bytes = bytes;
            streaming.resize(96 * bytes);
            std::memcpy(streaming.data(), file.tensor_data(*down), streaming.size());
        }
    }
    require(!streaming.empty(), "no IQ4_NL down tensors");
    std::printf("{\"kind\":\"real\",\"expert_cases\":%d,\"checked_floats\":%llu,\"all_bits_equal\":true}\n", real_cases, (unsigned long long)checked);
    Activations act(n, rng, false);
    std::vector<float> result(8 * rows);
    float* out[8]; for (int t = 0; t < 8; ++t) out[t] = result.data() + t * rows;
    auto run = [&](Function function, int nt, int experts, int repeats) {
        const double start = milliseconds();
        for (int z = 0; z < repeats; ++z) for (int e = 0; e < experts; ++e)
            function(streaming.data() + e * expert_bytes, stride, n, act.pointers, nt, out, 0, rows);
        return milliseconds() - start;
    };
    for (int nt : {1, 2, 3, 4, 8}) for (int experts : {1, 96}) {
        const int repeats = experts == 1 ? 128 : 2;
        run(original, nt, experts, 1); run(alternate, nt, experts, 1);
        for (int round = 0; round < 9; ++round) {
            double a, b;
            if (round % 2 == 0) { a = run(original, nt, experts, repeats); b = run(alternate, nt, experts, repeats); }
            else { b = run(alternate, nt, experts, repeats); a = run(original, nt, experts, repeats); }
            std::printf("{\"kind\":\"timing\",\"nt\":%d,\"experts\":%d,\"repeats\":%d,\"round\":%d,\"original_first\":%s,\"original_ms\":%.9f,\"candidate_ms\":%.9f,\"speed_ratio\":%.9f}\n",
                        nt, experts, repeats, round, round % 2 == 0 ? "true" : "false", a, b, a / b);
        }
    }
    std::printf("{\"kind\":\"completed\",\"cpu\":2,\"streaming_bytes\":%zu,\"n_ff\":%d,\"n_embd\":%d,\"all_bits_equal\":true}\n", streaming.size(), n, rows);
}
