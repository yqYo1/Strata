// Main-authored collector qualification; repository LICENSE.
// Exercise the production compatibility graph API, not a second graph adapter.
// Main controls collection at the gates; process environment stays immutable.
#include "strata/sycl/compat/cuda_runtime.h"
#include "strata/sycl/launch.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace compat = strata::sycl_backend::compat;
class StrataCompatWindowProduce;
class StrataCompatWindowConsume;

namespace {
void need(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
void check(cudaError_t value, const char* why) {
    if (value != cudaSuccess)
        throw std::runtime_error(std::string(why) + ": " + cudaGetErrorString(value));
}
void gate(const char* phase) {
    std::cout << "COMPAT_PROFILE_GATE " << phase << '\n' << std::flush;
    need(std::cin.get() == 'X', "Main gate token");
}

struct Fixture {
    static constexpr size_t n = 257, guard = 17, total = n + 2 * guard;
    static constexpr int sentinel = -1234567;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t completion = nullptr;
    int* device = nullptr;
    std::array<int, total> input{}, output{};

    void replay(int count) {
        check(compat::graph_launch_with_completion(executable, stream, completion), "graph launch");
        check(cudaEventSynchronize(completion), "actual graph completion");
        compat::GraphEnvelopeTimestamps timestamps{};
        need(compat::graph_completion_timestamps(completion, &timestamps) == cudaErrorNotSupported &&
                 timestamps.available_fields == 0,
             "ordinary graph exposed profiling timestamps");
        check(cudaMemcpyAsync(output.data(), device, sizeof(output), cudaMemcpyDeviceToHost, stream),
              "guarded readback");
        check(cudaStreamSynchronize(stream), "readback completion");
        for (size_t i = 0; i < total; ++i) {
            int expected = input[i];
            if (i >= guard && i < guard + n)
                for (int r = 0; r < count; ++r) expected = (expected * 3 + 1) * 5 - 2;
            need(output[i] == expected, "numerical result or allocation guard");
        }
    }

    void run() {
        const char* capture = std::getenv("STRATA_SYCL_VERIFY_NATIVE_CAPTURE");
        need(capture && std::strcmp(capture, "0") == 0, "fixed ordinary production route required");
        check(cudaSetDevice(0), "device select");
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, 0), "device properties");
        need(std::string(props.name).find("B570") != std::string::npos, "unexpected GPU");
        check(cudaStreamCreate(&stream), "owned stream create");
        check(cudaEventCreateWithFlags(&completion, cudaEventDisableTiming), "completion create");
        std::cout << "device=" << props.name << " route=compat-ordinary profile=0\n";
        gate("runtime-ready");
        check(cudaMalloc(&device, sizeof(input)), "guarded device allocation");
        input.fill(sentinel);
        for (size_t i = 0; i < n; ++i) input[guard + i] = int(i) - 128;
        check(cudaMemcpyAsync(device, input.data(), sizeof(input), cudaMemcpyHostToDevice, stream),
              "initial upload");
        check(cudaStreamSynchronize(stream), "initial upload completion");

        auto& queue = strata::sycl_backend::queue_for(stream);
        strata::sycl_backend::validate_device_span(device, sizeof(input), queue);
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "ordinary capture begin");
        int* const data = device;
        auto produce = queue.parallel_for<StrataCompatWindowProduce>(sycl::range<1>(n), [=](sycl::id<1> i) {
            data[guard + i[0]] = data[guard + i[0]] * 3 + 1;
        });
        strata::sycl_backend::finish(stream, produce);
        auto consume = queue.parallel_for<StrataCompatWindowConsume>(sycl::range<1>(n), [=](sycl::id<1> i) {
            data[guard + i[0]] = data[guard + i[0]] * 5 - 2;
        });
        strata::sycl_backend::finish(stream, consume);
        check(cudaStreamEndCapture(stream, &graph), "ordinary capture end");
        size_t nodes = 0;
        check(cudaGraphGetNodes(graph, nullptr, &nodes), "ordinary graph node query");
        need(nodes == 2, "unexpected graph topology or native recording");
        check(cudaGraphInstantiate(&executable, graph, 0ull), "ordinary graph instantiate");
        std::cout << "case graph_nodes=2 native_recording=0 profiled_instantiate=0\n";
        gate("captured");

        for (int count = 1; count <= 3; ++count) replay(count);
        std::cout << "case clean_replays=3 kernels=6 readbacks=3 numerical=PASS guards=PASS\n";
        gate("window-drained");
        // A separate post-window replay checks final-flush attribution. It is
        // not a test of pausing with pending device work or of late completion.
        replay(4);
        std::cout << "case post_window_replays=1 kernels=2 readbacks=1 numerical=PASS guards=PASS\n";

        check(cudaStreamSynchronize(stream), "final stream drain");
        check(cudaGraphExecDestroy(executable), "executable retirement"); executable = nullptr;
        check(cudaGraphDestroy(graph), "source graph retirement"); graph = nullptr;
        check(cudaEventDestroy(completion), "completion retirement"); completion = nullptr;
        check(cudaFree(device), "allocation retirement"); device = nullptr;
        check(cudaStreamDestroy(stream), "stream retirement"); stream = nullptr;
        gate("before-exit");
        std::cout << "fixture_status=PASS mode=compat-profile-window checked_teardown=PASS\n";
    }
};
} // namespace

int main(int argc, char**) {
    std::cout << std::unitbuf;
    try {
        need(argc == 1, "compat-profile-window takes no arguments");
        gate("before-runtime");
        // Do not destroy a graph/queue/allocation after uncertain asynchronous
        // failure. The external finite owner terminates and reaps this process.
        auto* fixture = new Fixture;
        fixture->run();
        delete fixture;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fixture_status=FAIL " << error.what() << '\n' << std::flush;
        std::_Exit(1);
    }
}
