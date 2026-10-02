#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#define CHECK(x)                                                               \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess)                                                      \
      throw std::runtime_error(std::string(#x) + ": " +                        \
                               cudaGetErrorString(e));                         \
  } while (0)
void require(bool v, const char *s) {
  if (!v)
    throw std::runtime_error(s);
}
int main() {
  try {
    constexpr size_t n = 257, bytes = n * sizeof(int);
    cudaStream_t a{}, b{};
    CHECK(cudaStreamCreate(&a));
    CHECK(cudaStreamCreate(&b));
    int *in{}, *out{}, *device{};
    CHECK(cudaMallocHost((void **)&in, bytes));
    CHECK(cudaMallocHost((void **)&out, bytes));
    CHECK(cudaMalloc((void **)&device, bytes));
    void *alias{};
    CHECK(cudaHostGetDevicePointer(&alias, in, 0));
    require(alias == in, "host alias");
    int ordinary = 0;
    require(cudaHostRegister(&ordinary, sizeof ordinary, 0) ==
                cudaErrorNotSupported,
            "host registration must fail");
    require(cudaPeekAtLastError() == cudaErrorNotSupported,
            "sticky last error");
    require(cudaGetLastError() == cudaErrorNotSupported &&
                cudaGetLastError() == cudaSuccess,
            "clear last error");
    cudaEvent_t begin{}, done{}, captured{};
    CHECK(cudaEventCreate(&begin));
    CHECK(cudaEventCreate(&done));
    CHECK(cudaEventCreateWithFlags(&captured, cudaEventDisableTiming));
    CHECK(cudaEventQuery(done));
    for (size_t i = 0; i < n; ++i)
      in[i] = int(i);
    CHECK(cudaEventRecord(begin, a));
    CHECK(cudaMemcpyAsync(device, in, bytes, cudaMemcpyHostToDevice, a));
    a->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { device[i] *= 3; });
    CHECK(cudaEventRecord(done, a));
    CHECK(cudaStreamWaitEvent(b, done));
    CHECK(cudaMemcpyAsync(out, device, bytes, cudaMemcpyDeviceToHost, b));
    CHECK(cudaStreamSynchronize(b));
    for (size_t i = 0; i < n; ++i)
      require(out[i] == int(i) * 3, "cross stream event");
    float ms = -1;
    CHECK(cudaEventElapsedTime(&ms, begin, done));
    require(std::isfinite(ms) && ms >= 0, "profiling");
    // Startup PCIe calibration uses the default stream, not an explicit one.
    // Its events must carry device timestamps as well as ordering semantics.
    CHECK(cudaEventRecord(begin));
    for (int repeat = 0; repeat < 32; ++repeat)
      CHECK(cudaMemcpyAsync(device, in, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(done));
    CHECK(cudaEventSynchronize(done));
    CHECK(cudaEventElapsedTime(&ms, begin, done));
    require(std::isfinite(ms) && ms > 0, "default-stream profiling");
    CHECK(cudaStreamBeginCapture(a, cudaStreamCaptureModeThreadLocal));
    CHECK(cudaMemcpyAsync(device, in, bytes, cudaMemcpyHostToDevice, a));
    a->parallel_for(sycl::range<1>(n),
                    [=](sycl::id<1> i) { device[i] = device[i] * 7 + 2; });
    CHECK(cudaEventRecord(captured, a));
    CHECK(cudaMemcpyAsync(out, device, bytes, cudaMemcpyDeviceToHost, a));
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    CHECK(cudaStreamEndCapture(a, &graph));
    size_t count = 0;
    CHECK(cudaGraphGetNodes(graph, nullptr, &count));
    require(count >= 3, "graph nodes");
    std::vector<cudaGraphNode_t> nodes(count);
    CHECK(cudaGraphGetNodes(graph, nodes.data(), &count));
    int kernels = 0;
    for (auto node : nodes) {
      cudaGraphNodeType type;
      CHECK(cudaGraphNodeGetType(node, &type));
      kernels += type == cudaGraphNodeTypeKernel;
    }
    require(kernels == 1, "kernel graph node");
    CHECK(cudaGraphInstantiate(&exec, graph, 0ull));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaGraphUpload(exec, b));
    for (int iteration = 0; iteration < 12; ++iteration) {
      for (size_t i = 0; i < n; ++i)
        in[i] = int(i) + iteration * 1000;
      CHECK(cudaGraphLaunch(exec, iteration % 2 ? a : b));
      CHECK(cudaEventSynchronize(
          captured)); // Must include D2H after the captured event.
      CHECK(cudaEventQuery(captured));
      for (size_t i = 0; i < n; ++i)
        require(out[i] == (int(i) + iteration * 1000) * 7 + 2,
                "graph replay host handoff");
    }
    // A verifier segment publishes routed inputs, then the CPU writes a plan
    // and results before another segment may consume them. Exercise the same
    // external-event boundary with changing payloads and alternating queues.
    CHECK(cudaStreamBeginCapture(a, cudaStreamCaptureModeThreadLocal));
    CHECK(cudaMemcpyAsync(device, out, bytes, cudaMemcpyHostToDevice, a));
    a->parallel_for(sycl::range<1>(n),
                    [=](sycl::id<1> i) { device[i] = device[i] * 5 - 3; });
    CHECK(cudaMemcpyAsync(out, device, bytes, cudaMemcpyDeviceToHost, a));
    cudaGraph_t consume_graph{};
    cudaGraphExec_t consume{};
    CHECK(cudaStreamEndCapture(a, &consume_graph));
    CHECK(cudaGraphInstantiate(&consume, consume_graph, 0ull));
    CHECK(cudaGraphDestroy(consume_graph));
    for (int iteration = 0; iteration < 12; ++iteration) {
      for (size_t i = 0; i < n; ++i)
        in[i] = int(i) - iteration * 77;
      auto producer = iteration % 2 ? a : b;
      auto consumer = iteration % 2 ? b : a;
      CHECK(cudaGraphLaunch(exec, producer));
      CHECK(cudaEventRecord(done, producer));
      CHECK(cudaEventSynchronize(done));
      for (size_t i = 0; i < n; ++i) {
        require(out[i] == in[i] * 7 + 2, "segment publication");
        out[i] += iteration + int(i % 3);
      }
      CHECK(cudaGraphLaunch(consume, consumer));
      CHECK(cudaEventRecord(done, consumer));
      CHECK(cudaEventSynchronize(done));
      for (size_t i = 0; i < n; ++i)
        require(out[i] == (in[i] * 7 + 2 + iteration + int(i % 3)) * 5 - 3,
                "completed segment consumes current host results");
    }
    CHECK(cudaGraphExecDestroy(consume));
    CHECK(cudaEventDestroy(
        captured)); // Executable retains only weak event references.
    CHECK(cudaGraphLaunch(exec, a));
    CHECK(cudaGraphExecDestroy(exec));
    std::vector<unsigned char> src(35, 0x31), dst(49, 0x7f);
    CHECK(cudaMemcpy2DAsync(dst.data(), 7, src.data(), 5, 3, 7,
                            cudaMemcpyHostToHost, b));
    CHECK(cudaStreamSynchronize(b));
    for (size_t row = 0; row < 7; ++row)
      for (size_t col = 0; col < 7; ++col)
        require(dst[row * 7 + col] == (col < 3 ? 0x31 : 0x7f),
                "pitched copy padding");
    int callback = 0;
    CHECK(cudaLaunchHostFunc(
        b, [](void *p) { *static_cast<int *>(p) = 42; }, &callback));
    CHECK(cudaStreamSynchronize(b));
    require(callback == 42, "host callback");
    CHECK(cudaMemsetAsync(device, 0, bytes, b));
    CHECK(cudaMemcpyAsync(out, device, bytes, cudaMemcpyDeviceToHost, b));
    CHECK(cudaFree(device)); // Free must drain all registered streams.
    for (size_t i = 0; i < n; ++i)
      require(out[i] == 0, "free synchronization");
    CHECK(cudaFreeHost(in));
    CHECK(cudaFreeHost(out));
    CHECK(cudaEventDestroy(begin));
    CHECK(cudaEventDestroy(done));
    CHECK(cudaStreamDestroy(a));
    CHECK(cudaStreamDestroy(b));
    std::cout << "SYCL host adapter: events, profiling, graph replay, copies "
                 "and lifecycle passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
