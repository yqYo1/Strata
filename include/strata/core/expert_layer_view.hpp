#pragma once

// Authored for Strata; repository LICENSE applies. A view owns no memory.
#include <cstdint>
#include <limits>

namespace strata::core {

struct ResidentExpertLayer {
    const uint8_t* data = nullptr;
    uint64_t bytes = 0, stride = 0;
    int64_t experts = 0;

    static ResidentExpertLayer make(const uint8_t* owner, uint64_t owner_bytes,
            uint64_t offset, int64_t count, uint64_t blob_stride) {
        if (!owner || count <= 0 || !blob_stride ||
            uint64_t(count) > std::numeric_limits<uint64_t>::max() / blob_stride)
            return {};
        const uint64_t size = uint64_t(count) * blob_stride;
        const auto address = reinterpret_cast<uintptr_t>(owner);
        if (offset > owner_bytes || size > owner_bytes - offset ||
            owner_bytes > std::numeric_limits<uintptr_t>::max() - address)
            return {};
        return {owner + offset, size, blob_stride, count};
    }
};

} // namespace strata::core
