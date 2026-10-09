#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::kernels::cpu { struct ExpertLayout; }

namespace strata {

// Owns the GGUF mappings used to validate and read native expert role tensors.
// Views remain valid until close/open/destruction. Files must not be truncated
// or modified while mapped. No engine, queue or ggml initialization is required.
class NativeRolePlan {
public:
    struct Extent {
        std::string path, tensor;
        uint64_t offset = 0, bytes_per_expert = 0, tensor_bytes = 0;
    };
    NativeRolePlan();
    ~NativeRolePlan();
    NativeRolePlan(const NativeRolePlan&) = delete;
    NativeRolePlan& operator=(const NativeRolePlan&) = delete;

    // A failed open leaves the plan closed, including after a previous success.
    bool open(const std::string& primary, const kernels::cpu::ExpertLayout& layout, std::string& error);
    void close() noexcept;
    bool is_open() const noexcept;
    size_t file_count() const noexcept;
    const Extent* extent(int64_t layer, int role) const noexcept;
    uint64_t blob_bytes(int64_t layer) const noexcept;

    // Assemble [gate | up | down] in a caller-owned, non-overlapping buffer.
    // Each role has its own source tensor/shard and expert stride.
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* destination,
                   size_t capacity, std::string& error) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata
