#include "strata/sycl/runtime.hpp"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <thread>
#include <utility>

namespace strata::sycl_backend {
namespace {
sycl::device select_device(int ordinal) {
  auto candidates = Runtime::devices();
  if (ordinal < 0 || static_cast<size_t>(ordinal) >= candidates.size())
    throw std::runtime_error(
        "SYCL GPU ordinal out of range (check ONEAPI_DEVICE_SELECTOR)");
  auto device = candidates[ordinal];
  if (!device.has(sycl::aspect::usm_device_allocations) ||
      !device.has(sycl::aspect::usm_host_allocations))
    throw std::runtime_error(
        "SYCL backend requires device and host USM allocations");
  return device;
}
} // namespace

std::vector<sycl::device> Runtime::devices() {
  return sycl::device::get_devices(sycl::info::device_type::gpu);
}

std::shared_ptr<Runtime> runtime_for(int ordinal) {
  static std::mutex mutex;
  static std::map<int, std::shared_ptr<Runtime>> runtimes;
  std::lock_guard lock(mutex);
  auto &runtime = runtimes[ordinal];
  if (!runtime)
    runtime = std::make_shared<Runtime>(ordinal);
  return runtime;
}

void Runtime::Errors::capture(sycl::exception_list errors) {
  std::lock_guard lock(mutex);
  for (const auto &error : errors)
    if (!first)
      first = error;
}

void Runtime::Errors::rethrow() {
  std::exception_ptr error;
  {
    std::lock_guard lock(mutex);
    error = first;
  }
  // A runtime with an asynchronous failure stays failed; subsequent submissions
  // must not silently continue after a caller has caught the first exception.
  if (error)
    std::rethrow_exception(error);
}

Runtime::Runtime(int ordinal)
    : errors_(std::make_shared<Errors>()), device_(select_device(ordinal)),
      context_(device_), compute_(
                             context_, device_,
                             [errors = errors_](sycl::exception_list list) {
                               errors->capture(list);
                             },
                             sycl::property::queue::in_order{}),
      transfer_(
          context_, device_,
          [errors = errors_](sycl::exception_list list) {
            errors->capture(list);
          },
          sycl::property::queue::in_order{}) {}

sycl::queue Runtime::make_queue(bool profiling) {
  check();
  const auto errors = errors_;
  const sycl::property_list properties =
      profiling ? sycl::property_list{sycl::property::queue::in_order{},
                                      sycl::property::queue::enable_profiling{}}
                : sycl::property_list{sycl::property::queue::in_order{}};
  return sycl::queue(
      context_, device_,
      [errors](sycl::exception_list list) { errors->capture(list); },
      properties);
}

void Runtime::check() { errors_->rethrow(); }

void Runtime::wait() {
  // Drain both queues even when the first queue reports an error.
  std::exception_ptr error;
  try {
    compute_.wait_and_throw();
  } catch (...) {
    error = std::current_exception();
  }
  try {
    transfer_.wait_and_throw();
  } catch (...) {
    if (!error)
      error = std::current_exception();
  }
  check();
  if (error)
    std::rethrow_exception(error);
}

void Runtime::wait(sycl::event event) {
  event.wait_and_throw();
  check();
}

bool Runtime::wait_for(sycl::event event, std::chrono::milliseconds timeout) {
  if (timeout.count() < 0)
    throw std::invalid_argument("negative SYCL wait timeout");
  const auto start = std::chrono::steady_clock::now();
  while (event.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
    check();
    if (std::chrono::steady_clock::now() - start >= timeout)
      return false;
    std::this_thread::yield();
  }
  wait(event);
  return true;
}

sycl::event Runtime::copy(void *dst, const void *src, size_t bytes,
                          const std::vector<sycl::event> &dependencies) {
  check();
  if (bytes && (!dst || !src))
    throw std::invalid_argument("SYCL copy: null pointer");
  return transfer_.submit([&](sycl::handler &h) {
    h.depends_on(dependencies);
    if (bytes)
      h.memcpy(dst, src, bytes);
    else
      h.single_task([] {});
  });
}

Allocation::Allocation(std::shared_ptr<Runtime> runtime, size_t bytes,
                       MemoryKind kind)
    : runtime_(std::move(runtime)), bytes_(bytes) {
  if (!runtime_)
    throw std::invalid_argument("SYCL allocation requires a runtime");
  runtime_->check();
  if (!bytes)
    return;
  if (kind == MemoryKind::Device &&
      bytes >
          runtime_->device().get_info<sycl::info::device::max_mem_alloc_size>())
    throw std::length_error(
        "SYCL allocation exceeds device maximum allocation size");
  data_ = kind == MemoryKind::Device
              ? sycl::malloc_device(bytes, runtime_->compute())
              : sycl::malloc_host(bytes, runtime_->context());
  if (!data_)
    throw std::bad_alloc();
}

Allocation::~Allocation() { release(); }
Allocation::Allocation(Allocation &&other) noexcept {
  *this = std::move(other);
}
Allocation &Allocation::operator=(Allocation &&other) noexcept {
  if (this != &other) {
    release();
    runtime_ = std::move(other.runtime_);
    data_ = std::exchange(other.data_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

void Allocation::release() noexcept {
  if (!data_)
    return;
  try {
    runtime_->wait();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "SYCL allocation cleanup: %s\n", error.what());
  } catch (...) {
    std::fprintf(stderr,
                 "SYCL allocation cleanup: unknown asynchronous error\n");
  }
  try {
    sycl::free(data_, runtime_->context());
  } catch (...) {
    std::fprintf(stderr, "SYCL allocation cleanup: free failed\n");
  }
  data_ = nullptr;
  bytes_ = 0;
}

void *Allocation::at(size_t offset, size_t bytes) const {
  if (offset > bytes_ || bytes > bytes_ - offset)
    throw std::out_of_range("SYCL allocation subrange out of bounds");
  return data_ ? static_cast<std::byte *>(data_) + offset : nullptr;
}
} // namespace strata::sycl_backend
