// Main-authored for Strata. Repository LICENSE applies.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using strata::sycl_backend::Allocation;
using strata::sycl_backend::MemoryKind;
using strata::sycl_backend::runtime_for;
using namespace strata::kernels;
namespace compat = strata::sycl_backend::compat;

constexpr int tokens = 4, experts = 10, columns = 2560;
constexpr int rows = tokens * experts, parts = rows * columns, outputs = tokens * columns;
constexpr int guard = 16, batches = 4, replays = 512;
constexpr float sentinel = -7103.0f;

void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
void check(cudaError_t code, const char* action) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(action) + " " + cudaGetErrorString(code));
}
void retire(cudaError_t code) noexcept {
    if (code != cudaSuccess) std::abort();
}

class Graphs {
    cudaGraph_t recording_ = nullptr;
    bool capturing_ = false;
    std::array<cudaGraphExec_t, 2> executable_{};
    cudaEvent_t completion_ = nullptr;
public:
    cudaStream_t stream = nullptr;
    void open() {
        check(cudaStreamCreate(&stream), "stream");
        check(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming), "completion");
    }
    ~Graphs() {
        if (capturing_) retire(cudaStreamEndCapture(stream, &recording_));
        if (stream) retire(cudaStreamSynchronize(stream));
        for (auto graph : executable_)
            if (graph) retire(cudaGraphExecDestroy(graph));
        if (recording_) retire(cudaGraphDestroy(recording_));
        if (completion_) retire(cudaEventDestroy(completion_));
        if (stream) retire(cudaStreamDestroy(stream));
    }
    template<class Record> void capture(int arm, size_t expected_nodes, Record record) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        capturing_ = true;
        record(stream);
        check(cudaStreamEndCapture(stream, &recording_), "end capture");
        capturing_ = false;
        size_t actual = 0;
        check(cudaGraphGetNodes(recording_, nullptr, &actual), "graph nodes");
        require(actual == expected_nodes, "ordinary graph topology changed");
        check(cudaGraphInstantiate(&executable_[arm], recording_, 0ull), "instantiate graph");
        check(cudaGraphDestroy(recording_), "retire source graph");
        recording_ = nullptr;
    }
    void replay(int arm) {
        check(compat::graph_launch_with_completion(executable_[arm], stream, completion_), "launch");
        check(cudaEventSynchronize(completion_), "replay completion");
    }
};

float part_value(size_t index, int seed) {
    if ((index + seed) % 31 == 0) return -0.0f;
    return float(int((index * 37 + seed * 17) % 379) - 189) / 43.0f;
}
float weight_value(int row) {
    return float(row % 9 - 4) / 23.0f;
}

void run(std::string_view mode) {
    auto runtime = runtime_for();
    Allocation host(runtime, parts * sizeof(float), MemoryKind::Host);
    Allocation gpu(runtime, parts * sizeof(float), MemoryKind::Device);
    Allocation assembled(runtime, parts * sizeof(float), MemoryKind::Device);
    Allocation meta(runtime, (rows + 1) * sizeof(int32_t), MemoryKind::Device);
    Allocation weights(runtime, rows * sizeof(float), MemoryKind::Device);
    Allocation shared(runtime, outputs * sizeof(float), MemoryKind::Device);
    Allocation result(runtime, (outputs + 2 * guard) * sizeof(float), MemoryKind::Device);
    Graphs graphs; // Retire all graph/stream owners before freeing any pointer.
    graphs.open();
    float* cpu = host.as<float>();
    float* device = gpu.as<float>();
    float* stage = assembled.as<float>();
    int32_t* hits = meta.as<int32_t>();
    float* out = result.as<float>() + guard;
    graphs.capture(0, 3, [&](void* stream) {
        copy_rows_from_mapped(stage, cpu, rows, columns, hits + 1, hits, stream);
        moe_hit_add(stage, device, hits + 1, hits, rows, columns, stream);
        native_moe_combine_multi(stage, weights.as<float>(), shared.as<float>(), out, columns, experts, tokens, stream);
    });
    graphs.capture(1, 1, [&](void* stream) {
        decode_moe_combine_mapped_rows(cpu, device, hits + 1, hits, weights.as<float>(), shared.as<float>(), out, stream);
    });

    std::vector<float> gpu_input(parts), shared_input(outputs), output(outputs + 2 * guard, sentinel);
    std::array<int32_t, rows + 1> metadata{};
    std::array<float, rows> weight_input{};
    for (int row = 0; row < rows; ++row) weight_input[row] = weight_value(row);
    for (int i = 0; i < outputs; ++i) shared_input[i] = part_value(i, 17);
    auto upload = [&](void* dst, const void* src, size_t bytes) {
        check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, graphs.stream), "fixture upload");
    };
    upload(weights.data(), weight_input.data(), weights.size());
    upload(shared.data(), shared_input.data(), shared.size());
    for (int count : {0, 4, 10, 20, 39, rows}) {
        if (mode != "validate" && (count == 0 || count > 20)) continue;
        metadata.fill(-1);
        metadata[0] = count;
        std::array<bool, rows> resident{};
        for (int hit = 0; hit < count; ++hit) {
            const int row = (hit * 7 + count) % rows;
            resident[row] = true;
            metadata[hit + 1] = row;
        }
        for (size_t i = 0; i < parts; ++i) {
            const float value = part_value(i, count);
            const float poison = std::numeric_limits<float>::quiet_NaN();
            cpu[i] = resident[i / columns] ? poison : value;
            gpu_input[i] = resident[i / columns] ? value : poison;
        }
        upload(meta.data(), metadata.data(), meta.size());
        upload(gpu.data(), gpu_input.data(), gpu.size());
        check(cudaStreamSynchronize(graphs.stream), "publish case");

        for (int arm : {0, 1}) {
            std::fill(output.begin(), output.end(), sentinel);
            upload(result.data(), output.data(), result.size());
            graphs.replay(arm);
            check(cudaMemcpyAsync(output.data(), result.data(), result.size(),
                                  cudaMemcpyDeviceToHost, graphs.stream), "readback");
            check(cudaStreamSynchronize(graphs.stream), "readback completion");
            for (int i = 0; i < guard; ++i)
                require(output[i] == sentinel && output[guard + outputs + i] == sentinel, "output guard moved");
            for (int i = 0; i < outputs; ++i) {
                const int first = i / columns * experts;
                const int column = i % columns;
                auto part = [&](int row) {
                    const float value = part_value(row * columns + column, count);
                    return resident[row] ? 0.0f + value : value;
                };
                float sum = part(first) * weight_value(first);
                for (int row = first + 1; row < first + experts; ++row)
                    sum = std::fma(part(row), weight_value(row), sum);
                const float expected = sum + shared_input[i];
                require(std::bit_cast<uint32_t>(expected) == std::bit_cast<uint32_t>(output[guard + i]),
                        "ordered FP32 combine or row source differs");
            }
        }

        if (mode == "validate") continue;
        const int start = mode == "time-reverse" ? 1 : 0;
        for (int batch = 0; batch < batches; ++batch) {
            const int first = start ^ (batch % 2);
            for (int arm : {first, 1 - first}) {
                const auto begin = std::chrono::steady_clock::now();
                for (int i = 0; i < replays; ++i) graphs.replay(arm);
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                require(ms >= 10.0, "ineligible timing batch");
                std::cout << "decode MoE combine hits=" << count << " batch=" << batch
                          << " arm=" << arm << " replays=" << replays << " ms="
                          << std::fixed << std::setprecision(6) << ms << '\n';
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc == 2 ? argv[1] : "";
        require(mode == "validate" || mode == "time-forward" || mode == "time-reverse",
                "select validate, time-forward or time-reverse");
        run(mode);
        std::cout << "decode MoE combine numerical=PASS guards=PASS nodes=3/1 completed=1\n";
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
