// Main-authored real-copy lifetime regression; repository LICENSE.
// No invalid device address, device fault, fork or runtime environment change.
#include "strata/core/expert_cache.hpp"
#include "strata/sycl/checked_worker.hpp"
#include "strata/sycl/runtime.hpp"
#include <cuda_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

namespace sb = strata::sycl_backend;
namespace {
constexpr size_t Bytes = 4096;
void need(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void marker(const char *message) {
  std::puts(message);
  std::fflush(stdout);
}
void checkpoint(const char *name) {
  std::printf("COMPONENT_READY %s\n", name);
  std::fflush(stdout);
  need(std::getchar() == 'X', "owner gate");
}
struct Gate {
  std::mutex mutex;
  std::condition_variable cv;
  bool released = false, timed_out = false;
  void release() {
    std::lock_guard lock(mutex);
    released = true;
    cv.notify_all();
  }
  void wait() {
    std::unique_lock lock(mutex);
    if (!cv.wait_for(lock, std::chrono::seconds(10), [&] { return released; })) {
      timed_out = true;
      released = true;
    }
  }
  bool clean() {
    std::lock_guard lock(mutex);
    return released && !timed_out;
  }
};
struct Owners {
  bool &finished;
  strata::core::ExpertCache cache;
  sb::Allocation input, output;
  std::shared_ptr<Gate> gate = std::make_shared<Gate>();
  Owners(bool &done, std::shared_ptr<sb::Runtime> runtime, bool tracked)
      : finished(done), input(runtime, Bytes, sb::MemoryKind::Host),
        output(runtime, Bytes, sb::MemoryKind::Host) {
    std::string error;
    need(cache.open(1, 1, 1, Bytes, error), error.c_str());
    if (tracked) cache.enable_scratch_loan_tracking();
    need(cache.admit(0, 0) == 0, "cache slot admission");
    for (size_t i = 0; i < Bytes; ++i) input.as<unsigned char>()[i] = (i * 53 + 19) & 255;
    std::memset(output.data(), 0, Bytes);
  }
  ~Owners() noexcept {
    if (!finished) sb::cleanup_fail_stop("device fixture owner unwound before join");
    cache.close(); // Actual production close, with scratch tracking both on/off.
    marker("DEVICE_OWNERS_RELEASED");
  }
};
} // namespace

int main(int argc, char **argv) {
  try {
    need(argc == 2, "use exception|tracked|failed-drain");
    const std::string mode = argv[1];
    need(mode == "exception" || mode == "tracked" || mode == "failed-drain", "invalid mode");
    checkpoint("before-runtime");
    auto runtime = sb::runtime_for();
    auto &queue = runtime->compute();
    need(queue.has_property<sycl::property::queue::in_order>(), "ordered queue required");
    need(queue.get_device().get_info<sycl::info::device::name>().find("B570") != std::string::npos,
         "unexpected device");
    checkpoint("runtime-ready");
    bool finished = false, result = true, caught = false;
    try {
      Owners owners(finished, runtime, mode == "tracked");
      std::thread worker;
      sb::CheckedWorkerJoin join(worker);
      worker = std::thread([&] {
        result = sb::checked_worker([&]() -> bool {
          const auto gate = owners.gate;
          queue.submit([gate](sycl::handler &handler) {
            handler.host_task([gate] { gate->wait(); });
          });
          need(cudaMemcpyAsync(owners.cache.device_slot(0), owners.input.data(), Bytes,
                               cudaMemcpyHostToDevice, nullptr) == cudaSuccess, "real H2D submission");
          const auto pending = queue.ext_oneapi_submit_barrier();
          need(pending.get_info<sycl::info::event::command_execution_status>() !=
                   sycl::info::event_command_status::complete, "copy was not held pending");
          marker("DEVICE_COPY_ACCEPTED_PENDING");
          throw std::bad_alloc();
        }, [&] {
          owners.gate->release();
          need(cudaDeviceSynchronize() == cudaSuccess, "production all-queue drain");
          need(owners.gate->clean(), "host gate timed out");
          need(cudaMemcpy(owners.output.data(), owners.cache.device_slot(0), Bytes,
                          cudaMemcpyDeviceToHost) == cudaSuccess, "read completed payload");
          need(std::memcmp(owners.input.data(), owners.output.data(), Bytes) == 0, "copy payload mismatch");
          marker("DEVICE_DRAIN_PAYLOAD_PASS");
          // Inject a software error after known completion to exercise fail-stop
          // without deliberately leaving live device work at process termination.
          if (mode == "failed-drain") {
            checkpoint("before-failstop");
            throw std::runtime_error("injected unverified drain status");
          }
        }, "device accepted-copy fixture");
        finished = true;
      });
      throw std::runtime_error("main commit/draft exception");
    } catch (const std::runtime_error &error) {
      need(std::string(error.what()) == "main commit/draft exception", "unexpected main exception");
      caught = true;
    }
    need(caught && finished && !result && mode != "failed-drain", "worker did not retire correctly");
    checkpoint("before-exit");
    marker("checked_worker_device=PASS pending_copy=1 payload=1 main_unwind_join=1");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "checked worker device: %s\n", error.what());
    return 2;
  }
}
