#include "strata/sycl/cpu_wait_profile.hpp"
#include "strata/sycl/runtime.hpp"
#include "strata/sycl/checked_cleanup.hpp"

#include <cstdio>
#include <map>
#include <iterator>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>

namespace strata::sycl_backend {
namespace {
// This registry covers both engine arenas and CUDA-adapter device buffers.
// Entries refer to their Allocation's retained Runtime; moves keep addresses.
struct DeviceSpan { size_t bytes; std::shared_ptr<Runtime> runtime; };
struct DeviceRegistry {
  std::mutex mutex;
  std::map<uintptr_t, DeviceSpan> spans;
};
DeviceRegistry &device_registry() {
  // compat.cpp's allocations map is constructed before main and may release
  // owners AFTER function statics have been destroyed. Keep the registry object
  // for the process lifetime; entries still retire with their Allocation.
  // A destructible local static here causes use-after-free at ordinary exit.
  static DeviceRegistry* const registry = new DeviceRegistry;
  return *registry;
}
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
  (void)device_registry();
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
                             // This is also the engine's default stream.
                             // Its ordinary events support elapsed-time queries.
                             sycl::property_list{
                                 sycl::property::queue::in_order{},
                                 sycl::property::queue::enable_profiling{}}),
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
    strata::sycl_backend::cpu_wait_profile::blocking(
        strata::sycl_backend::cpu_wait_profile::compute_queue, [&] { compute_.wait_and_throw(); });
  } catch (...) {
    error = std::current_exception();
  }
  try {
    strata::sycl_backend::cpu_wait_profile::blocking(
        strata::sycl_backend::cpu_wait_profile::transfer_queue, [&] { transfer_.wait_and_throw(); });
  } catch (...) {
    if (!error)
      error = std::current_exception();
  }
  check();
  if (error)
    std::rethrow_exception(error);
}

void Runtime::wait(sycl::event event) {
  strata::sycl_backend::cpu_wait_profile::blocking(
        strata::sycl_backend::cpu_wait_profile::event, [&] { event.wait_and_throw(); });
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

std::shared_ptr<Runtime> runtime_for_device_span(const void *pointer, size_t bytes,
                                               const sycl::queue &queue) {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  if (!pointer || !bytes || bytes > UINTPTR_MAX - address)
    throw std::invalid_argument("invalid registered SYCL device span");
  auto &registry = device_registry();
  std::lock_guard lock(registry.mutex);
  auto next = registry.spans.upper_bound(address);
  if (next == registry.spans.begin())
    throw std::invalid_argument("unregistered SYCL device span");
  const auto &[base, entry] = *std::prev(next);
  const size_t offset = address - base;
  if (offset > entry.bytes || bytes > entry.bytes - offset ||
      queue.get_context() != entry.runtime->context() ||
      queue.get_device() != entry.runtime->device())
    throw std::invalid_argument("SYCL device span exceeds allocation or queue domain");
  return entry.runtime;
}

void validate_device_span(const void *pointer, size_t bytes,
                          const sycl::queue &queue) {
  (void)runtime_for_device_span(pointer, bytes, queue);
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
  if (kind == MemoryKind::Device) {
    try {
      auto &registry = device_registry();
      std::lock_guard lock(registry.mutex);
      if (!registry.spans.emplace(reinterpret_cast<uintptr_t>(data_),
                                  DeviceSpan{bytes_, runtime_}).second)
        throw std::runtime_error("duplicate SYCL device allocation");
    } catch (...) {
      sycl::free(data_, runtime_->context());
      data_ = nullptr;
      throw;
    }
  }
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
  checked_cleanup([&] { runtime_->wait(); },
                  [&] {
                    auto &registry = device_registry();
                    std::lock_guard lock(registry.mutex);
                    sycl::free(data_, runtime_->context());
                    registry.spans.erase(reinterpret_cast<uintptr_t>(data_));
                  },
                  "allocation completion check failed");
  data_ = nullptr;
  bytes_ = 0;
}

void *Allocation::at(size_t offset, size_t bytes) const {
  if (offset > bytes_ || bytes > bytes_ - offset)
    throw std::out_of_range("SYCL allocation subrange out of bounds");
  return data_ ? static_cast<std::byte *>(data_) + offset : nullptr;
}
} // namespace strata::sycl_backend
