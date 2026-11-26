// Main-authored for Strata. Repository LICENSE applies.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
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
using Clock = std::chrono::steady_clock;
enum class Comparison { stock_to_mapped, mapped_to_copy, all_three };
constexpr int stock_hit_counts[] = {0, 1, 2, 3, 4, 5, 10, 39, rows};
constexpr int legacy_hit_counts[] = {0, 4, 10, 20, 39, rows};
constexpr std::array<std::array<int, 3>, 6> three_arm_orders{{
    {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
}};

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
    std::array<cudaGraphExec_t, 3> executable_{};
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

void run(std::string_view mode, Comparison comparison) {
    const bool all_three = comparison == Comparison::all_three;
    const bool mapped_to_copy = comparison == Comparison::mapped_to_copy;
    const int arm_count = all_three ? 3 : 2;
    const int copy_arm = all_three ? 2 : 1;
    const bool include_copy = all_three || mapped_to_copy;
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
    auto capture_fused = [&](int arm, const float* cpu_input) {
        graphs.capture(arm, 1, [&](void* stream) {
            decode_moe_combine_mapped_rows(cpu_input, device, hits + 1, hits,
                                           weights.as<float>(), shared.as<float>(), out, stream);
        });
    };
    if (mapped_to_copy) {
        capture_fused(0, cpu);
    } else {
        graphs.capture(0, 3, [&](void* stream) {
            copy_rows_from_mapped(stage, cpu, rows, columns, hits + 1, hits, stream);
            moe_hit_add(stage, device, hits + 1, hits, rows, columns, stream);
            native_moe_combine_multi(stage, weights.as<float>(), shared.as<float>(), out, columns, experts, tokens, stream);
        });
    }
    capture_fused(1, mapped_to_copy ? stage : cpu);
    if (all_three) capture_fused(2, stage);

    std::vector<float> gpu_input(parts), shared_input(outputs), output(outputs + 2 * guard, sentinel);
    std::array<std::vector<float>, 2> cpu_inputs{std::vector<float>(parts), std::vector<float>(parts)};
    std::array<int32_t, rows + 1> metadata{};
    std::array<float, rows> weight_input{};
    for (int row = 0; row < rows; ++row) weight_input[row] = weight_value(row);
    for (int i = 0; i < outputs; ++i) shared_input[i] = part_value(i, 17);
    auto upload = [&](void* dst, const void* src, size_t bytes) {
        check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, graphs.stream), "fixture upload");
    };
    auto replay = [&](int arm) {
        // The copy and graph share the ordered stream. Replay waits before
        // the caller overwrites the host publication for the next iteration.
        if (include_copy && arm == copy_arm) upload(stage, cpu, host.size());
        graphs.replay(arm);
    };
    upload(weights.data(), weight_input.data(), weights.size());
    upload(shared.data(), shared_input.data(), shared.size());
    const auto hit_counts = all_three ? std::span<const int>(stock_hit_counts) : std::span<const int>(legacy_hit_counts);
    for (int count : hit_counts) {
        if (mode != "validate" && ((!all_three && count == 0) || count > 20)) continue;
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
            cpu_inputs[0][i] = value;
            cpu_inputs[1][i] = part_value(i, count + 71);
        }
        upload(meta.data(), metadata.data(), meta.size());
        upload(gpu.data(), gpu_input.data(), gpu.size());
        check(cudaStreamSynchronize(graphs.stream), "publish case");

        auto publish_cpu_rows = [&](int version) {
            for (int row = 0; row < rows; ++row)
                if (!resident[row])
                    std::memcpy(cpu + row * columns, cpu_inputs[version].data() + row * columns,
                                columns * sizeof(float));
            std::atomic_thread_fence(std::memory_order_seq_cst);
        };
        auto verify = [&](int arm, int version) {
            publish_cpu_rows(version);
            std::fill(output.begin(), output.end(), sentinel);
            upload(result.data(), output.data(), result.size());
            replay(arm);
            check(cudaMemcpyAsync(output.data(), result.data(), result.size(),
                                  cudaMemcpyDeviceToHost, graphs.stream), "readback");
            check(cudaStreamSynchronize(graphs.stream), "readback completion");
            for (int i = 0; i < guard; ++i)
                require(output[i] == sentinel && output[guard + outputs + i] == sentinel, "output guard moved");
            for (int i = 0; i < outputs; ++i) {
                const int first = i / columns * experts;
                const int column = i % columns;
                auto part = [&](int row) {
                    const int seed = count + (resident[row] ? 0 : 71 * version);
                    const float value = part_value(row * columns + column, seed);
                    return resident[row] ? 0.0f + value : value;
                };
                float sum = part(first) * weight_value(first);
                for (int row = first + 1; row < first + experts; ++row)
                    sum = std::fma(part(row), weight_value(row), sum);
                const float expected = sum + shared_input[i];
                require(std::bit_cast<uint32_t>(expected) == std::bit_cast<uint32_t>(output[guard + i]),
                        "ordered FP32 combine or row source differs");
            }
        };
        for (int arm = 0; arm < arm_count; ++arm)
            for (int version : {0, 1}) verify(arm, version);

        if (mode == "validate") continue;
        const int start = mode == "time-reverse" ? 1 : 0;
        const int order_start = mode == "time-rotated" ? 2 : 3 * start;
        const int batch_count = all_three ? static_cast<int>(three_arm_orders.size()) : batches;
        for (int batch = 0; batch < batch_count; ++batch) {
            const int first = start ^ (batch % 2);
            const std::array<int, 3> pair_order{first, 1 - first, 2};
            const auto& order = all_three ? three_arm_orders[(order_start + batch) % batch_count] : pair_order;
            for (int index = 0; index < arm_count; ++index) {
                const int arm = order[index];
                Clock::duration elapsed{}, full{}, empty{};
                for (int i = 0; i < replays; ++i) {
                    const auto publish_begin = Clock::now();
                    publish_cpu_rows(i % 2);
                    const auto begin = Clock::now();
                    replay(arm);
                    const auto end = Clock::now();
                    elapsed += end - begin;
                    full += end - publish_begin;
                    const auto empty_begin = Clock::now();
                    empty += Clock::now() - empty_begin;
                }
                const double ms = std::chrono::duration<double, std::milli>(elapsed).count();
                const double full_ms = std::chrono::duration<double, std::milli>(full).count();
                const double empty_ms = std::chrono::duration<double, std::milli>(empty).count();
                require(ms >= 10.0 && empty_ms <= ms * 0.01, "ineligible timing batch or timer overhead");
                std::cout << "decode MoE combine hits=" << count << " batch=" << batch
                          << " arm=" << arm << " replays=" << replays << " ms="
                          << std::fixed << std::setprecision(6) << ms << " full_ms=" << full_ms
                          << " empty_ms=" << empty_ms;
                if (include_copy)
                    std::cout << " cpu_copy_bytes_per_replay=" << (arm == copy_arm ? host.size() : 0);
                std::cout << '\n';
            }
        }
    }
    if (mode == "validate") {
        // The regular dispatcher publishes each output slot once. Exercise
        // a malformed list separately, where the old scatter has no contract.
        metadata[0] = 2;
        metadata[1] = metadata[2] = 0;
        std::fill_n(cpu, parts, 1.0f);
        upload(meta.data(), metadata.data(), meta.size());
        for (int arm = 1; arm < arm_count; ++arm) {
            replay(arm);
            check(cudaMemcpyAsync(output.data(), result.data(), result.size(),
                                  cudaMemcpyDeviceToHost, graphs.stream), "malformed list readback");
            check(cudaStreamSynchronize(graphs.stream), "malformed list completion");
            for (int i = 0; i < outputs; ++i)
                require(i < columns ? std::isnan(output[guard + i]) : std::isfinite(output[guard + i]),
                        "duplicate destination was not rejected for its token");
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool compare_copy = argc == 3 && std::string_view(argv[2]) == "--compare-copy";
        const bool compare_stock = argc == 3 && std::string_view(argv[2]) == "--compare-stock";
        const std::string_view mode = (argc == 2 || compare_copy || compare_stock) ? argv[1] : "";
        require(mode == "validate" || mode == "time-forward" || mode == "time-reverse"
                || (compare_stock && mode == "time-rotated"), "select a valid combine comparison and mode");
        Comparison comparison = Comparison::stock_to_mapped;
        if (compare_copy) comparison = Comparison::mapped_to_copy;
        if (compare_stock) comparison = Comparison::all_three;
        run(mode, comparison);
        std::cout << "decode MoE combine numerical=PASS guards=PASS nodes="
                  << (compare_stock ? "3/1/1" : compare_copy ? "1/1" : "3/1") << " completed=1\n";
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
