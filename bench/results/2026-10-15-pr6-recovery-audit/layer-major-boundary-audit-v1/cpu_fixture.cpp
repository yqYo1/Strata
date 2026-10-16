// Proposal companion only; NOT compiled or executed. Linux CPU-only helper fixture.
// After patch review/application, caller provides a new private output prefix.
#include "strata/prefill/layer_major_boundary_audit.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    ::setenv("STRATA_PREFILL_BOUNDARY_ROWS", "0,1", 1);
    ::unsetenv("STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE");
    using Audit = strata::prefill::detail::LayerMajorBoundaryAudit;
    Audit::Copy copy = [](void* d, const void* s, size_t bytes, bool) {
        std::memcpy(d, s, bytes); return true;
    };
    for (bool half : {false, true}) {
        const std::string path = std::string(argv[1]) + (half ? ".half" : ".float");
        ::setenv("STRATA_PREFILL_BOUNDARY_PATH", path.c_str(), 1);
        Audit audit(2, 100, 10240, 48, 1, half);
        std::vector<float> r(10240), stored_float(20480);
        std::vector<uint16_t> stored_half(20480);
        for (int layer = 0; layer < 48; ++layer) for (int row = 0; row < 2; ++row) {
            if (layer) {
                for (int i = 0; i < 10240; ++i)
                    r[size_t(i)] = half ? strata::kernels::f32_from_f16(stored_half[size_t(row)*10240+i])
                                            : stored_float[size_t(row)*10240+i];
                assert(audit.input(layer, row, 1, r.data(), false, copy));
            }
            for (int i = 0; i < 10240; ++i) r[size_t(i)] = float(layer + row) + float(i % 97) / 1024;
            assert(audit.source(layer, row, 1, r.data(), copy));
            if (layer != 47) {
                for (int i = 0; i < 10240; ++i) {
                    stored_float[size_t(row)*10240+i] = r[size_t(i)];
                    stored_half[size_t(row)*10240+i] = strata::kernels::f16_from_f32(r[size_t(i)]);
                }
                const void* ptr = half ? static_cast<const void*>(stored_half.data()+row*10240)
                                       : static_cast<const void*>(stored_float.data()+row*10240);
                assert(audit.stored(layer, row, 1, ptr, false, copy));
            }
        }
        audit.complete(); // Consumer must verify complete/equal and exact record/raw counts.
    }
    // These are independent bit semantics expectations, not a GPU conversion test.
    assert(strata::kernels::f16_from_f32(-0.0f) == 0x8000);
    assert(strata::kernels::f16_from_f32(1.0f) == 0x3c00);
    assert(strata::kernels::f32_from_f16(1) == 0x1p-24f);
}
