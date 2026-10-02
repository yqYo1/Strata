#include "strata/core/device.hpp"
#include "strata/sycl/runtime.hpp"

#include <limits>
#include <sstream>

namespace strata::core {

const char *compiled_gpu_archs() { return "SYCL SPIR-V (JIT)"; }
int device_count() {
  try {
    return static_cast<int>(sycl_backend::Runtime::devices().size());
  } catch (const sycl::exception &) {
    return 0;
  }
}

bool device_summary(int ordinal, std::string &name, std::string &detail) {
  try {
    auto devices = sycl_backend::Runtime::devices();
    if (ordinal < 0 || static_cast<size_t>(ordinal) >= devices.size())
      return false;
    const auto &device = devices[ordinal];
    name = device.get_info<sycl::info::device::name>();
    std::ostringstream out;
    out << "SYCL, "
        << device.get_info<sycl::info::device::global_mem_size>() /
               (1024.0 * 1024 * 1024)
        << " GiB, driver "
        << device.get_info<sycl::info::device::driver_version>();
    detail = out.str();
    return true;
  } catch (const sycl::exception &) {
    return false;
  }
}

std::string gpu_arch_problem(int ordinal) {
  try {
    sycl_backend::runtime_for(ordinal);
    return {};
  } catch (const std::exception &error) {
    return error.what();
  }
}

std::string device_code_error() {
  try {
    auto runtime = sycl_backend::runtime_for();
    runtime->wait(runtime->compute().single_task([] {}));
    return {};
  } catch (const std::exception &error) {
    return error.what();
  }
}

DeviceInfo device_info(int ordinal) {
  const auto &device = sycl_backend::runtime_for(ordinal)->device();
  DeviceInfo info;
  info.ordinal = ordinal;
  info.name = device.get_info<sycl::info::device::name>();
  info.arch = compiled_gpu_archs();
  info.total_bytes = device.get_info<sycl::info::device::global_mem_size>();
  // A total-memory fallback would overstate the available expert-cache budget.
  if (!device.has(sycl::aspect::ext_intel_free_memory))
    throw std::runtime_error(
        "SYCL memory planning requires the Intel free-memory query");
  info.free_bytes =
      device.get_info<sycl::ext::intel::info::device::free_memory>();
  info.multi_processor_count =
      device.get_info<sycl::info::device::max_compute_units>();
  return info;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
  if (!bytes || bytes > std::numeric_limits<size_t>::max())
    throw CudaError("invalid SYCL DeviceArena size", -1);
  auto runtime = sycl_backend::runtime_for(ordinal);
  sycl_allocation_ = std::make_shared<sycl_backend::Allocation>(
      runtime, static_cast<size_t>(bytes), sycl_backend::MemoryKind::Device);
  base_ = sycl_allocation_->data();
  if (poison_) {
    runtime->wait(runtime->compute().fill(static_cast<uint32_t *>(base_),
                                          uint32_t{0x7fc00000},
                                          bytes / sizeof(uint32_t)));
  }
}

DeviceArena::~DeviceArena() = default;

void *DeviceArena::alloc(uint64_t bytes, uint64_t align) {
  if (!bytes)
    return nullptr;
  if (!align || (align & (align - 1)))
    throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
  // Align the actual pointer, not only its offset: SYCL does not promise a
  // 4 KiB-aligned malloc_device base. Subtraction checks avoid uint64 wrap.
  const uint64_t address = reinterpret_cast<uintptr_t>(base_) + used_;
  const uint64_t padding = (uint64_t{0} - address) & (align - 1);
  if (padding > capacity_ - used_ || bytes > capacity_ - used_ - padding)
    throw CudaError("SYCL DeviceArena exhausted", -1);
  const uint64_t start = used_ + padding;
  used_ = start + bytes;
  return static_cast<std::byte *>(base_) + start;
}
} // namespace strata::core
