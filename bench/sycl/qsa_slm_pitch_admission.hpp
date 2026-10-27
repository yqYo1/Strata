// Main-authored admission-only helper; repository LICENSE applies.
// Read compiled properties through the already loaded qualified Level Zero
// loader. No kernel submission, argument mutation or USM allocation occurs here.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>

struct QsaSlmPitchAdmission {
  std::size_t maximum_workgroup;
  uint32_t subgroup, arguments, reported_local_bytes, spill_descriptor;
  uint32_t per_group_local_max;
  uint64_t declared_local_bytes, conservative_local_budget;
};

inline std::array<QsaSlmPitchAdmission, 3> qsa_slm_pitch_admit(
    sycl::queue &queue, const std::array<std::string, 3> &names,
    const char *qualified_loader_path) {
  auto require = [](bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  };
  require(queue.get_backend() == sycl::backend::ext_oneapi_level_zero,
          "pitch32 admission needs the qualified Level Zero queue");
  require(qualified_loader_path && qualified_loader_path[0] == '/',
          "pitch32 admission needs the pinned absolute loader path");
  const auto device = queue.get_device();
  require(device.is_gpu() && device.get_info<sycl::info::device::vendor_id>() == 0x8086,
          "pitch32 admission needs the pinned Intel device");
  for (std::size_t arm = 0; arm < names.size(); ++arm) {
    require(!names[arm].empty(), "empty compiled kernel identity");
    for (std::size_t earlier = 0; earlier < arm; ++earlier)
      require(names[arm] != names[earlier], "duplicate compiled arm identity");
  }
  struct LoaderReference {
    void *handle;
    ~LoaderReference() { if (handle) dlclose(handle); }
  } loader{dlopen(qualified_loader_path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD)};
  require(loader.handle != nullptr, "qualified loader is not already loaded");
  const auto compute_properties = reinterpret_cast<decltype(&zeDeviceGetComputeProperties)>(
      dlsym(loader.handle, "zeDeviceGetComputeProperties"));
  const auto kernel_properties = reinterpret_cast<decltype(&zeKernelGetProperties)>(
      dlsym(loader.handle, "zeKernelGetProperties"));
  require(compute_properties && kernel_properties, "qualified property API missing");
  ze_device_compute_properties_t compute{};
  compute.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  require(compute_properties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device),
                             &compute) == ZE_RESULT_SUCCESS,
          "Level Zero compute property query failed");
  require(compute.maxTotalGroupSize >= 128 && compute.maxGroupSizeX >= 128 &&
              compute.maxSharedLocalMemory > 0, "device group limits refuse pitch32");
  const auto ids = sycl::get_kernel_ids();
  const std::array<uint64_t, 3> declared{20036, 20036, 20548};
  std::array<QsaSlmPitchAdmission, 3> result{};
  for (std::size_t arm = 0; arm < names.size(); ++arm) {
    require(std::count_if(ids.begin(), ids.end(), [&](const auto &id) {
      return names[arm] == id.get_name();
    }) == 1, "expected exactly one compiled attention kernel identity");
    const auto id = *std::find_if(ids.begin(), ids.end(), [&](const auto &value) {
      return names[arm] == value.get_name();
    });
    const auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        queue.get_context(), {device}, {id});
    const auto kernel = bundle.get_kernel(id);
    ze_kernel_properties_t properties{};
    properties.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    require(kernel_properties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(kernel),
                              &properties) == ZE_RESULT_SUCCESS,
            "Level Zero kernel property query failed");
    const auto maximum = kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(device);
    const auto subgroup = kernel.get_info<sycl::info::kernel_device_specific::compile_sub_group_size>(device);
    require(maximum >= 128 && subgroup == 16 && properties.requiredSubgroupSize == 16 &&
                properties.numKernelArgs == 18 && properties.maxNumSubgroups >= 8,
            "compiled workgroup/subgroup/argument contract refuses pitch32");
    require((!properties.requiredGroupSizeX || properties.requiredGroupSizeX == 128) &&
                (!properties.requiredGroupSizeY || properties.requiredGroupSizeY == 1) &&
                (!properties.requiredGroupSizeZ || properties.requiredGroupSizeZ == 1) &&
                (!properties.requiredNumSubGroups || properties.requiredNumSubGroups == 8),
            "compiled required group shape differs");
    // This query precedes local-argument binding. Add the complete declared
    // accessor payload and reserve conservatively; this is not an occupancy
    // measurement or proof of later argument sizes. The traced first fixture
    // must independently account its actual local arguments before acceptance.
    const uint64_t budget = uint64_t(properties.localMemSize) + declared[arm] + 4096;
    require(budget <= compute.maxSharedLocalMemory, "per-group local budget exceeded");
    result[arm] = {maximum, subgroup, properties.numKernelArgs, properties.localMemSize,
                  properties.spillMemSize, compute.maxSharedLocalMemory, declared[arm], budget};
  }
  return result;
}
