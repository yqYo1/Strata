#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::core {
// Host view of the actual cache allocation, including the sized arena's end
// sentinel. Loans use the tail's real bytes rather than slots * largest blob.
class ExpertCacheTail {
public:
    ExpertCacheTail(int64_t slots, uint64_t bytes, const uint64_t* offsets, uint64_t blob)
        : slots_(slots), bytes_(bytes), offsets_(offsets), blob_(blob) {
        if (slots < 0 || slots >= INT32_MAX)
            throw std::invalid_argument("invalid cache tail slot count");
        if (offsets) {
            if (offsets[0] != 0 || offsets[slots] != bytes)
                throw std::invalid_argument("invalid cache tail extent");
            for (int64_t i = 1; i <= slots; ++i)
                if (offsets[i] <= offsets[i - 1] || offsets[i] > bytes)
                    throw std::invalid_argument("invalid cache tail offsets");
        } else if (!blob || uint64_t(slots) > UINT64_MAX / blob || uint64_t(slots) * blob != bytes) {
            throw std::invalid_argument("invalid uniform cache tail extent");
        }
    }
    uint64_t bytes_from(int64_t first) const {
        if (first < 0 || first > slots_) throw std::out_of_range("cache tail first slot");
        return offsets_ ? bytes_ - offsets_[first] : uint64_t(slots_ - first) * blob_;
    }
    int64_t slots_for(uint64_t need) const {
        if (need > bytes_) return slots_ + 1; // Explicit no-fit; never a truncated loan.
        if (!offsets_) return int64_t(need / blob_ + (need % blob_ != 0));
        int64_t k = 0;
        while (bytes_from(slots_ - k) < need) ++k;
        return k;
    }
private:
    int64_t slots_;
    uint64_t bytes_;
    const uint64_t* offsets_;
    uint64_t blob_;
};

inline int64_t prefill_request_chunk(int64_t tokens, int64_t maximum) {
    if (tokens <= 0 || maximum <= 0) return 0;
    const int64_t rounded = tokens > INT64_MAX - 255 ? tokens : ((tokens + 255) / 256) * 256;
    return std::min(maximum, rounded);
}

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
