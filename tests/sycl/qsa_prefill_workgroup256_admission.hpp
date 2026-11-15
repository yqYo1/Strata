// Main-authored diagnostic from this project's qsa_topk_prefill_entry contract.
// Repository LICENSE. Outside timed batches;no compiler/resource override.
#pragma once
#include "strata/kernels/qsa_prefill_workgroup256.hpp"
#include "strata/sycl/runtime.hpp"
#include <iostream>
#include <stdexcept>
inline void admit_prefill_workgroup256(sycl::queue &queue) {
  const auto resources=strata::kernels::prefill::prefill_workgroup256_kernel_resources(&queue);
  for (int capture=0;capture<2;++capture) {
    const auto [maximum,local,private_bytes]=resources[capture];
    if (maximum<256 || local<20036+4096) throw std::runtime_error("WG256kernel maximum or device local capacity below requirement");
    // Reported local/private values alone do not establish occupancy or spills.
    std::cout<<"prefill_kernel_admission capture="<<capture<<" maximum="<<maximum
        <<" required=256 device_local="<<local<<" reported_private="<<private_bytes
        <<" requested_dynamic_local=20036 status=PASS\n";
  }
}
