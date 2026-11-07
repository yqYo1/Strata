#pragma once
// DF2P v1 metadata admission for the Qwen3.8 Flash-Next Q4 drafter.
// Format origin: Niko1221/Strata146df3c4, tools/dflash2_pack.py (MIT).
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {
struct Dflash2Tensor {
    std::string name, dtype;
    int64_t rows = 0, cols = 0;
    uint64_t off = 0, bytes = 0;
};
struct Dflash2PackInfo {
    std::vector<Dflash2Tensor> tensors;
    uint64_t data_offset = 0, data_bytes = 0;
    int mask_token = 0;
    float theta = 10000.f, eps = 1e-6f;
};
// CPU only. Validate every required tensor, shape, encoding and file span before
// creating a runtime/queue or allocating device memory. No tensor payload read.
bool inspect_dflash2_pack(const std::string& path, Dflash2PackInfo& out, std::string& err);
}
