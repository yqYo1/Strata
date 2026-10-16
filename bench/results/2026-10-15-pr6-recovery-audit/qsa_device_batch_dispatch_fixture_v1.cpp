// Calls the linked production dispatch's zero-row path. No runtime/GPU owner.
#include "strata/kernels/native_qsa_indexer.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const bool reject = std::string(argv[1]) == "reject";
    try {
        strata::kernels::native_qsa_indexer_append_batch(nullptr, 0, 0, 0,
            nullptr, 1e-6f, {}, strata::kernels::qsa_real_shapes(), 262144, {}, nullptr);
        if (reject) { std::fputs("invalid flag accepted by production dispatch\n", stderr); return 1; }
        std::puts("PASS production dispatch zero-row no-op");
    } catch (const std::invalid_argument& e) {
        if (!reject || std::string(e.what()) != "STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH must be 0 or 1") {
            std::fprintf(stderr, "unexpected production rejection: %s\n", e.what()); return 1;
        }
        std::puts("PASS production dispatch strict flag rejection before runtime");
    }
}
