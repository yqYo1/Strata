#include "strata/prefill/layer_major_plan.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace strata::prefill::detail;
    int64_t rows, chunk, requested;
    uint64_t budget;
    while (std::cin >> rows >> chunk >> budget >> requested) {
        try {
            const auto p = LayerMajorResidualPlan::make(
                rows, chunk, 10240, budget, requested, LayerMajorResidualDtype::fp16);
            std::cout << "accepted " << p.gpu_rows << ' ' << p.gpu_bytes << ' '
                      << p.host_bytes << ' ' << p.staging_bytes << '\n';
        } catch (const std::invalid_argument&) {
            std::cout << "rejected\n";
        }
    }
    return std::cin.eof() ? 0 : 2;
}
