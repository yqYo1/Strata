// Main-authored production-adapter collector fixture; repository LICENSE.
// Uses this fork's CUDA-named SYCL adapter, not a CUDA/HIP device backend.
// The external owner changes collector state only at fully drained gates.
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

class StrataCompatWindowProduce;
class StrataCompatWindowConsume;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

void gate(const char* phase) {
    std::cout << "COMPONENT_READY " << phase << '\n' << std::flush;
    require(std::cin.get() == 'X', "Main collector-window gate");
}

struct Fixture {
    static constexpr size_t n = 257, guard = 17, total = n + 2 * guard;
    static constexpr uint32_t sentinel = 0xa5a5a5a5u;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t completion = nullptr;
    uint32_t* device = nullptr;
    std::array<uint32_t, total> input{}, output{};
    int completed = 0;

    void validate() {
        check(cudaMemcpyAsync(output.data(), device, sizeof(output),
                              cudaMemcpyDeviceToHost, stream), "readback");
        check(cudaStreamSynchronize(stream), "readback drain");
        for (size_t i = 0; i < total; ++i) {
            uint32_t expected = input[i];
            if (i >= guard && i < guard + n)
                for (int r = 0; r < completed; ++r)
                    expected = (expected * 3u + 1u) * 5u - 2u;
            require(output[i] == expected, "numerical result or allocation guard");
        }
    }

    void replay(const char* phase, int count) {
        for (int r = 0; r < count; ++r) {
            check(strata::sycl_backend::compat::graph_launch_with_completion(
                      executable, stream, completion), "graph replay");
            check(cudaEventSynchronize(completion), "graph completion");
            check(cudaEventQuery(completion), "completed event query");
            ++completed;
            validate();
        }
        std::cout << "case phase=" << phase << " replays=" << count
                  << " kernels=" << 2 * count << " completed=" << completed
                  << " numerical=PASS guards=PASS\n";
    }

    void run() {
        check(cudaStreamCreate(&stream), "create stream");
        const auto name = stream->get_device().get_info<sycl::info::device::name>();
        require(name.find("B570") != std::string::npos, "unexpected GPU");
        std::cout << "device=" << name << '\n';
        gate("runtime-ready");
        check(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(input)), "device allocation");
        check(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming), "completion event");
        input.fill(sentinel);
        for (size_t i = 0; i < n; ++i) input[guard + i] = static_cast<uint32_t>(i);
        check(cudaMemcpyAsync(device, input.data(), sizeof(input),
                              cudaMemcpyHostToDevice, stream), "initial upload");
        check(cudaStreamSynchronize(stream), "initial upload drain");

        // Record while collection is active. Unsigned arithmetic defines wrap
        // on both devices and host, including the tenth cumulative replay.
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        auto* const data = device;
        stream->parallel_for<StrataCompatWindowProduce>(sycl::range<1>(n), [=](sycl::id<1> i) {
            data[guard + i[0]] = data[guard + i[0]] * 3u + 1u;
        });
        stream->parallel_for<StrataCompatWindowConsume>(sycl::range<1>(n), [=](sycl::id<1> i) {
            data[guard + i[0]] = data[guard + i[0]] * 5u - 2u;
        });
        check(cudaStreamEndCapture(stream, &graph), "end capture");
        check(cudaGraphInstantiate(&executable, graph, 0ull), "instantiate graph");
        check(cudaGraphDestroy(graph), "destroy recorded graph");
        graph = nullptr;

        replay("prefix", 2);       // Active construction and four recorded kernels.
        gate("prefix-drained");   // Main pauses collection before releasing.
        replay("gap", 3);          // Six executed kernels must be absent from trace.
        gate("gap-drained");      // Main resumes collection before releasing.
        replay("window", 3);       // Six resumed kernels must retain names and joins.
        gate("window-drained");   // Main pauses collection before releasing.
        replay("suffix", 2);       // Four more executed kernels must be absent.
        gate("suffix-drained");

        check(cudaStreamSynchronize(stream), "final drain");
        check(cudaGraphExecDestroy(executable), "destroy executable");
        executable = nullptr;
        check(cudaEventDestroy(completion), "destroy event");
        completion = nullptr;
        check(cudaFree(device), "free device allocation");
        device = nullptr;
        check(cudaStreamDestroy(stream), "destroy stream");
        stream = nullptr;
        gate("before-exit");
        std::cout << "fixture_status=PASS mode=compat-window replays=10 kernels=20\n";
    }
};
}  // namespace

int main(int argc, char**) {
    std::cout << std::unitbuf;
    try {
        require(argc == 1, "compat-window fixture takes no arguments");
        gate("before-runtime");
        // On failure, keep all asynchronous operands and owners alive until the
        // supervised process exits. Never unwind and free uncertain GPU work.
        auto* fixture = new Fixture;
        fixture->run();
        delete fixture;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixture_status=FAIL " << error.what() << '\n' << std::flush;
        std::_Exit(1);
    }
}
