// Main-authored diagnostic from this project's qsa_topk_prefill_entry contract.
// Repository LICENSE. Outside timed batches;no compiler/resource override.
#pragma once
#include "strata/kernels/qsa_prefill_workgroup256.hpp"
#include "strata/sycl/runtime.hpp"
#include <iostream>
#include <stdexcept>
inline void admit_prefill_workgroup256(sycl::queue &queue) {
  using strata::kernels::prefill::QsaPrefillWorkgroup256Kernel;
  const auto device=queue.get_device();
  const sycl::kernel_id ids[]={sycl::get_kernel_id<QsaPrefillWorkgroup256Kernel<false>>(),
      sycl::get_kernel_id<QsaPrefillWorkgroup256Kernel<true>>()};
  for (int capture=0;capture<2;++capture) {
    const auto bundle=sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        queue.get_context(),{device},{ids[capture]});
    const auto kernel=bundle.get_kernel(ids[capture]);
    const auto maximum=kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(device);
    const auto local=kernel.get_info<sycl::info::kernel_device_specific::local_mem_size>(device);
    const auto private_bytes=kernel.get_info<sycl::info::kernel_device_specific::private_mem_size>(device);
    if (maximum<256) throw std::runtime_error("WG256exact kernel maximum below required workgroup");
    // Reported local/private values alone do not establish occupancy or spills.
    std::cout<<"prefill_kernel_admission capture="<<capture<<" maximum="<<maximum
        <<" required=256 reported_local="<<local<<" reported_private="<<private_bytes
        <<" requested_dynamic_local=20036 status=PASS\n";
  }
}
