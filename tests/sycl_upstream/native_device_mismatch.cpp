#include "strata/sycl_upstream/device_metadata.hpp"
#include <iostream>

int main() try {
    namespace native = strata::sycl_upstream;
    native::DeviceFacts facts;
    if (native::query_device_facts(0, &facts) != cudaSuccess) throw std::runtime_error("native metadata unavailable");
    if (facts.architecture_id == uint64_t(sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc))
        throw std::runtime_error("mismatch fixture requires a GPU other than PVC (measured on B570)");
    const auto problem = native::device_arch_problem(0);
    if (problem.find("differs from this binary's SYCL AOT target (intel_gpu_pvc)") == std::string::npos ||
        problem.find(facts.name) == std::string::npos || problem.find(facts.architecture) == std::string::npos)
        throw std::runtime_error("wrong native architecture was not rejected with actual device metadata");
    std::cout << "PASS native_arch_mismatch=1 reason=" << problem << '\n';
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
