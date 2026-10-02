#pragma once

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

namespace strata::sycl_backend {

// Both queues share a context. Ordering between them is always expressed by
// events. Host allocations are accessed by the CPU only after the producing
// event completes.
class Runtime {
public:
  explicit Runtime(int ordinal = 0);
  static std::vector<sycl::device> devices();
  sycl::queue &compute() { return compute_; }
  sycl::queue &transfer() { return transfer_; }
  // Additional engine queues share this runtime's context and sticky errors.
  sycl::queue make_queue(bool profiling = false);
  const sycl::device &device() const { return device_; }
  const sycl::context &context() const { return context_; }
  void check();
  void wait();
  void wait(sycl::event event);
  // Timeout does not cancel work or release its buffers. The caller decides
  // whether to keep waiting or terminate the process after a device hang.
  bool wait_for(sycl::event event, std::chrono::milliseconds timeout);
  sycl::event copy(void *dst, const void *src, size_t bytes,
                   const std::vector<sycl::event> &dependencies = {});

private:
  struct Errors {
    std::mutex mutex;
    std::exception_ptr first;
    void capture(sycl::exception_list errors);
    void rethrow();
  };
  std::shared_ptr<Errors> errors_;
  sycl::device device_;
  sycl::context context_;
  sycl::queue compute_, transfer_;
};

enum class MemoryKind { Device, Host };

// The engine uses one runtime per device so its arenas and launches share a
// context.
std::shared_ptr<Runtime> runtime_for(int ordinal = 0);

// Keeps the runtime alive and completes its queues before releasing memory.
// Bulk expert weights remain ordinary host memory; Host is for bounded DMA
// staging.
class Allocation {
public:
  Allocation() = default;
  Allocation(std::shared_ptr<Runtime> runtime, size_t bytes, MemoryKind kind);
  ~Allocation();
  Allocation(const Allocation &) = delete;
  Allocation &operator=(const Allocation &) = delete;
  Allocation(Allocation &&other) noexcept;
  Allocation &operator=(Allocation &&other) noexcept;
  void *data() const { return data_; }
  size_t size() const { return bytes_; }
  void *at(size_t offset, size_t bytes) const;
  template <typename T> T *as() const { return static_cast<T *>(data_); }

private:
  void release() noexcept;
  std::shared_ptr<Runtime> runtime_;
  void *data_ = nullptr;
  size_t bytes_ = 0;
};

} // namespace strata::sycl_backend
