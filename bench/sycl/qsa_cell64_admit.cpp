// Main-authored property-only component; repository LICENSE applies.
// The linked kernels are inspected without submitting pack or attention work.
#include "qsa_cell64_admission.hpp"
#include "strata/sycl/runtime.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    std::cout.setf(std::ios::unitbuf);
    if (argc != 6 || std::string(argv[1]) != "--admit")
      throw std::runtime_error("--admit original32 private32 private64 loader required");
    std::cout << "admission_stage=before-runtime\n";
    auto runtime = strata::sycl_backend::runtime_for();
    std::cout << "admission_stage=runtime-ready device="
              << runtime->device().get_info<sycl::info::device::name>() << '\n';
    const auto properties = qsa_cell64_admit(runtime->compute(),
        {argv[2], argv[3], argv[4]}, argv[5]);
    for (std::size_t arm = 0; arm < properties.size(); ++arm) {
      const auto &p = properties[arm];
      std::cout << "arm=" << arm << " max_workgroup=" << p.maximum_workgroup
                << " subgroup=" << p.subgroup << " arguments=" << p.arguments
                << " reported_local_bytes=" << p.reported_local_bytes
                << " declared_dynamic_local_bytes=" << p.declared_local_bytes
                << " reserve_local_bytes=4096 conservative_local_budget="
                << p.conservative_local_budget << " per_group_local_max="
                << p.per_group_local_max << " spill_descriptor="
                << p.spill_descriptor << " status=PASS\n";
    }
    runtime->wait();
    runtime.reset();
    std::cout << "admission_status=PASS kernel_submissions=0 numerical=UNRUN "
                 "performance=UNRUN physical262144_model=UNRUN\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "admission_status=FAIL error=" << error.what() << '\n';
    return 1;
  }
}
