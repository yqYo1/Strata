// Main-authored qualification fixture; repository LICENSE applies.
// Reuse the checked graph/event/allocation owner without changing the engine.
#define main strata_graph_envelope_fixture_main
#include "../../tests/sycl/graph_envelope.cpp"
#undef main

#include <cerrno>
#include <poll.h>
#include <time.h>
#include <unistd.h>

namespace {
uint64_t raw_ns() {
  timespec value{};
  require(clock_gettime(CLOCK_MONOTONIC_RAW, &value) == 0, "raw clock");
  return uint64_t(value.tv_sec) * 1000000000ULL + uint64_t(value.tv_nsec);
}

// Both checkpoints follow checked drains. The controller changes collector
// recording state, never the native process environment or a live GPU owner.
void checkpoint(const char* phase) {
  std::cout << "graph_window phase=" << phase << " pid=" << getpid()
            << " raw_ns=" << raw_ns() << '\n' << std::flush;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  std::string input;
  for (;;) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now()).count();
    require(remaining > 0, "control deadline");
    pollfd fd{STDIN_FILENO, POLLIN, 0};
    const int status = poll(&fd, 1, static_cast<int>(remaining));
    if (status < 0 && errno == EINTR) continue;
    require(status > 0 && (fd.revents & POLLIN), "control unavailable");
    char byte = 0;
    require(read(STDIN_FILENO, &byte, 1) == 1, "control EOF");
    if (byte == '\n') break;
    require(input.size() < 16, "control extent");
    input += byte;
  }
  require(input == "continue", "control command");
}

class RecordWindowAdd;
class RecordWindowMultiply;
} // namespace

int main() {
  try {
    std::cout << std::unitbuf;
    std::cerr << "graph window: begin ordinary unprofiled capture\n";
    Owners owner;
    check(cudaStreamCreate(&owner.stream), "stream create");
    check(cudaEventCreateWithFlags(&owner.done, cudaEventDisableTiming), "event create");
    constexpr size_t guard = 17, n = 257, total = n + guard * 2;
    std::vector<int> expected(total, -1234567), actual(total);
    for (size_t i = 0; i < n; ++i) expected[guard + i] = int(i);
    check(cudaMalloc(&owner.data, total * sizeof(int)), "allocate");
    check(cudaMemcpyAsync(owner.data, expected.data(), total * sizeof(int),
                          cudaMemcpyHostToDevice, owner.stream), "input copy");
    check(cudaStreamSynchronize(owner.stream), "input drain");
    check(cudaStreamBeginCapture(owner.stream, cudaStreamCaptureModeThreadLocal), "capture begin");
    owner.capturing = true;
    int* data = owner.data;
    owner.stream->parallel_for<RecordWindowAdd>(sycl::range<1>(n),
        [=](sycl::id<1> i) { data[guard + i[0]] += 1; });
    owner.stream->parallel_for<RecordWindowMultiply>(sycl::range<1>(n),
        [=](sycl::id<1> i) { data[guard + i[0]] *= 3; });
    check(cudaStreamEndCapture(owner.stream, &owner.graph), "capture end");
    owner.capturing = false;
    check(cudaGraphInstantiate(&owner.executable, owner.graph, 0ull), "instantiate");
    check(cudaGraphDestroy(owner.graph), "source graph destroy");
    owner.graph = nullptr;
    check(cudaGraphUpload(owner.executable, owner.stream), "upload");
    check(cudaStreamSynchronize(owner.stream), "upload drain");
    const auto replay = [&] {
      check(compat::graph_launch_with_completion(owner.executable, owner.stream, owner.done), "replay");
      check(cudaStreamSynchronize(owner.stream), "replay drain");
      for (size_t i = 0; i < n; ++i) expected[guard + i] = (expected[guard + i] + 1) * 3;
    };
    replay();
    replay();
    checkpoint("ready"); // Capture, upload and two warm replays were paused.
    replay();
    replay();
    replay();
    checkpoint("drained"); // Three recorded replays are complete, owners live.
    check(cudaMemcpy(actual.data(), owner.data, total * sizeof(int), cudaMemcpyDeviceToHost), "readback");
    require(actual == expected, "five-replay numerical/guard mismatch");
    owner.close();
    std::cout << "graph_window PASS warm_replays=2 window_replays=3 kernels_per_replay=2"
              << " numerical=PASS guards=PASS checked_teardown=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "graph window: FAIL " << error.what() << '\n';
    return 1;
  }
}
