// SPDX-License-Identifier: MIT
// Authored for Strata. Diagnostic graph support, not a model-speed benchmark.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace compat = strata::sycl_backend::compat;
namespace {
void check(cudaError_t status, const char* where) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
// Drain before releasing any graph, event or allocation, including exceptions.
// An uncertain drain retains owners until process termination.
struct Owners {
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  cudaEvent_t done{}, internal{};
  int* data{};
  bool capturing = false;
  static void close_check(cudaError_t status) noexcept {
    if (status != cudaSuccess) {
      std::fprintf(stderr, "graph envelope: teardown uncertain: %s\n", cudaGetErrorString(status));
      std::fflush(stderr);
      std::abort();
    }
  }
  void close() noexcept {
    if (capturing) {
      close_check(cudaStreamEndCapture(stream, &graph));
      capturing = false;
    }
    if (stream) close_check(cudaStreamSynchronize(stream));
    if (executable) { close_check(cudaGraphExecDestroy(executable)); executable = nullptr; }
    if (graph) { close_check(cudaGraphDestroy(graph)); graph = nullptr; }
    if (internal) { close_check(cudaEventDestroy(internal)); internal = nullptr; }
    if (done) { close_check(cudaEventDestroy(done)); done = nullptr; }
    if (data) { close_check(cudaFree(data)); data = nullptr; }
    if (stream) { close_check(cudaStreamDestroy(stream)); stream = nullptr; }
  }
  ~Owners() { close(); }
};
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2 && (std::string(argv[1]) == "--profile=0" ||
                         std::string(argv[1]) == "--profile=1"),
            "usage: sycl_graph_envelope_test --profile=0|1");
    const bool profile = std::string(argv[1]) == "--profile=1";
    std::cout << std::unitbuf;
    std::cerr << "graph envelope: create owners, ordinary capture only\n";
    Owners owner;
    check(cudaStreamCreate(&owner.stream), "stream create");
    check(cudaEventCreateWithFlags(&owner.done, cudaEventDisableTiming), "completion create");
    check(cudaEventCreateWithFlags(&owner.internal, cudaEventDisableTiming), "internal create");
    constexpr size_t guard = 17, n = 257, total = n + guard * 2;
    constexpr int sentinel = -1234567;
    std::vector<int> expected(total, sentinel), output(total);
    for (size_t i = 0; i < n; ++i) expected[guard + i] = int(i);
    check(cudaMalloc(&owner.data, total * sizeof(int)), "device allocate");
    check(cudaMemcpyAsync(owner.data, expected.data(), total * sizeof(int),
                          cudaMemcpyHostToDevice, owner.stream), "input copy");
    check(cudaStreamSynchronize(owner.stream), "input completion");
    check(cudaStreamBeginCapture(owner.stream, cudaStreamCaptureModeThreadLocal), "capture begin");
    owner.capturing = true;
    int* data = owner.data;
    owner.stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { data[guard + i[0]] += 1; });
    check(cudaEventRecord(owner.internal, owner.stream), "internal capture");
    owner.stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { data[guard + i[0]] *= 3; });
    check(cudaStreamEndCapture(owner.stream, &owner.graph), "capture end");
    owner.capturing = false;
    const auto instantiated = profile
        ? compat::graph_instantiate_profiled(&owner.executable, owner.graph)
        : cudaGraphInstantiate(&owner.executable, owner.graph, 0ull);
    if (instantiated != cudaSuccess) {
      std::cout << "graph envelope: profile=" << profile << " support=UNKNOWN execution=NOT_RUN status="
                << int(instantiated) << " error=" << cudaGetErrorString(instantiated) << '\n';
      owner.close();
      return profile ? 77 : 1;
    }
    check(cudaGraphDestroy(owner.graph), "source graph destroy");
    owner.graph = nullptr;
    check(cudaGraphUpload(owner.executable, owner.stream), "explicit graph upload");
    check(cudaStreamSynchronize(owner.stream), "upload completion");
    bool unavailable = false;
    compat::GraphEnvelopeTimestamps first{};
    for (int replay = 0; replay < 2; ++replay) {
      check(compat::graph_launch_with_completion(owner.executable, owner.stream, owner.done), "launch");
      check(cudaStreamSynchronize(owner.stream), "graph completion");
      compat::GraphEnvelopeTimestamps sample{};
      const auto begin = std::chrono::steady_clock::now();
      const auto status = compat::graph_completion_timestamps(owner.done, &sample);
      const auto query_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin).count();
      // Submit comes from a separately estimated clock on this adapter. Keep
      // inversions without discarding a positive ordered start/end envelope.
      const bool valid = status == cudaSuccess && sample.available_fields == 7 &&
          sample.timer_resolution > 0 && sample.end > sample.start &&
          sample.end - sample.start >= sample.timer_resolution && query_ns < 1000000000;
      if (profile) {
        std::cout << "graph sample: replay=" << replay << " status=" << int(status)
                  << " fields=" << sample.available_fields << " submit=" << sample.submit
                  << " start=" << sample.start << " end=" << sample.end
                  << " timer_resolution=" << sample.timer_resolution << " query_ns=" << query_ns
                  << " envelope_qualified=" << valid
                  << " queue_delay_qualified=" << (valid && sample.start >= sample.submit) << '\n';
        if (status != cudaSuccess)
          std::cerr << "graph timestamp query: " << cudaGetErrorString(status) << '\n';
        unavailable = !valid;
        if (valid && replay) require(sample.start >= first.end, "replay timestamp order");
        if (valid && !replay) first = sample;
      } else {
        require(status == cudaErrorNotSupported && sample.available_fields == 0,
                "unprofiled completion exposed graph timestamps");
      }
      check(cudaMemcpy(output.data(), owner.data, total * sizeof(int), cudaMemcpyDeviceToHost), "readback");
      for (size_t i = 0; i < n; ++i) expected[guard + i] = (expected[guard + i] + 1) * 3;
      require(output == expected, "graph replay numerical or guard mismatch");
      compat::GraphEnvelopeTimestamps alias{};
      require(compat::graph_completion_timestamps(owner.internal, &alias) == cudaErrorInvalidValue,
              "captured internal alias exposed a node clock");
      (void)cudaGetLastError();
      if (unavailable) break; // no retry of unsupported/invalid graph profiling
    }
    // A completion handle recorded again as an ordinary event loses its graph
    // provenance. Its previous replay must never be returned under the new use.
    check(cudaEventRecord(owner.done, owner.stream), "ordinary rerecord");
    check(cudaStreamSynchronize(owner.stream), "rerecord completion");
    compat::GraphEnvelopeTimestamps stale{};
    require(compat::graph_completion_timestamps(owner.done, &stale) == cudaErrorNotSupported &&
                stale.available_fields == 0, "stale graph timestamps survived rerecord");
    owner.close();
    std::cout << "graph envelope: profile=" << profile << " numerical=PASS guards=PASS aliases=PASS"
              << " checked_teardown=PASS support=" << (unavailable ? "UNKNOWN" : profile ? "SUPPORTED" : "NOT_ENABLED")
              << " arithmetic_service=UNMEASURED hardware_saturation=UNMEASURED\n";
    return unavailable ? 77 : 0;
  } catch (const std::exception& error) {
    std::cerr << "graph envelope: FAIL " << error.what() << '\n';
    return 1;
  }
}
