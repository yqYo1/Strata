#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::core {
// Match ExpertCache::layer_slot_range: equal counts, remainder on the last
// layer. Every adaptive replacement therefore fits its layer's slot.
inline uint64_t expert_slot_aligned_bytes(int64_t bytes) {
    if (bytes <= 0) throw std::invalid_argument("nonpositive expert slot size");
    return (uint64_t(bytes) + 255) / 256 * 256;
}
inline int64_t per_layer_slots_that_fit(const std::vector<int64_t>& bytes,
                                      uint64_t budget, int64_t maximum) {
    if (bytes.empty() || maximum < 0)
        throw std::invalid_argument("invalid per-layer expert slot plan");
    uint64_t round = 0;
    for (const auto b : bytes) {
        const auto aligned = expert_slot_aligned_bytes(b);
        if (aligned > std::numeric_limits<uint64_t>::max() - round)
            throw std::overflow_error("expert slot plan size overflow");
        round += aligned;
    }
    const int64_t layers = int64_t(bytes.size());
    const uint64_t q = std::min<uint64_t>(budget / round, maximum / layers);
    const int64_t full = int64_t(q) * layers;
    const uint64_t last = expert_slot_aligned_bytes(bytes.back());
    const int64_t tail = int64_t(std::min<uint64_t>(
        (budget - q * round) / last,
        uint64_t(std::min(layers - 1, maximum - full))));
    return full + tail;
}
inline std::vector<int64_t> per_layer_slot_sizes(
    const std::vector<int64_t>& bytes, int64_t slots) {
    if (bytes.empty() || slots < 0)
        throw std::invalid_argument("invalid per-layer expert slot plan");
    for (const auto b : bytes) expert_slot_aligned_bytes(b);
    std::vector<int64_t> result;
    result.reserve(size_t(slots));
    const int64_t q = slots / int64_t(bytes.size());
    for (size_t layer = 0; layer < bytes.size(); ++layer) {
        const int64_t count = layer + 1 == bytes.size()
            ? slots - q * int64_t(layer) : q;
        result.insert(result.end(), size_t(count), bytes[layer]);
    }
    return result;
}
} // namespace strata::core
