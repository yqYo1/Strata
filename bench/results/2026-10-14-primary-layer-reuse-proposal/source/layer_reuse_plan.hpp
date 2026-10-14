// Proposal only. No GPU headers, allocations, global state or environment reads.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace strata::prefill::reuse {
inline bool mul(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > UINT64_MAX / b) return false;
    out = a * b; return true;
}
inline bool add(uint64_t a, uint64_t b, uint64_t& out) {
    if (a > UINT64_MAX - b) return false;
    out = a + b; return true;
}
struct Request {
    uint64_t tokens = 0, chunk = 0, residual_width = 0;
    uint64_t experts = 0, max_blob = 0;
    // Free bytes after all persistent workspace/Verifier/MTP/KV/cache allocations.
    // additional_device_peak includes bridge DMA buffers and new metadata, not existing scratch.
    uint64_t free_device = 0, reserve_device = 0, additional_device_peak = 0;
    uint64_t available_host = 0, reserve_host = 0, additional_host_peak = 0;
    bool fresh = false, single_gpu = false, borrowed = false;
    bool checkpoints_disabled = false, callbacks_final_rows_only = false;
    bool state_invalidation_contract = false, layer_runner_ready = false;
};
struct Plan {
    uint64_t residual_bytes = 0, stride = 0, resident_experts = 0, slot_bytes = 0;
    uint64_t chunks = 0, device_peak = 0, host_peak = 0;
    bool full_layer = false;
};
// Do not publish a partially valid output: false always resets out to an empty plan.
inline bool plan(const Request& r, Plan& out, std::string& reason) {
    out = {}; reason.clear(); Plan p;
    auto reject = [&](const char* s) { reason = s; return false; };
    if (!r.tokens || !r.chunk || !r.residual_width || !r.experts || !r.max_blob)
        return reject("invalid geometry");
    if (!r.fresh || !r.single_gpu || r.borrowed || !r.checkpoints_disabled ||
        !r.callbacks_final_rows_only || !r.state_invalidation_contract || !r.layer_runner_ready)
        return reject("execution contract unavailable; use unchanged chunk-major");
    uint64_t row_bytes = 0, rounded = 0;
    if (!mul(r.residual_width, sizeof(float), row_bytes) || !mul(r.tokens, row_bytes, p.residual_bytes) ||
        !add(r.max_blob, 255, rounded)) return reject("byte overflow");
    p.stride = rounded & ~uint64_t{255};
    if (!add(p.residual_bytes, r.additional_host_peak, p.host_peak) ||
        !add(p.host_peak, r.reserve_host, rounded)) return reject("host peak overflow");
    if (rounded > r.available_host || p.host_peak > std::numeric_limits<std::size_t>::max())
        return reject("host admission failed; preserve full requested context with chunk-major");
    uint64_t non_slot = 0;
    if (!add(r.reserve_device, r.additional_device_peak, non_slot) || non_slot >= r.free_device)
        return reject("device reserve admission failed");
    p.resident_experts = std::min(r.experts, (r.free_device - non_slot) / p.stride);
    if (!p.resident_experts || !mul(p.resident_experts, p.stride, p.slot_bytes) ||
        !add(p.slot_bytes, r.additional_device_peak, p.device_peak) ||
        p.slot_bytes > std::numeric_limits<std::size_t>::max()) return reject("no bounded resident slot");
    // Cache a bounded subset through every chunk of this layer; stream the remaining experts
    // through the original ring. Never reorder expert accumulation by a group-at-a-time walk.
    p.full_layer = p.resident_experts == r.experts;
    p.chunks = r.tokens / r.chunk + (r.tokens % r.chunk != 0);
    out = p; return true;
}
struct Slice { uint64_t offset = 0, bytes = 0; };
inline bool residual_slice(const Plan& p, uint64_t tokens, uint64_t width, uint64_t first,
                           uint64_t count, Slice& out) {
    out = {}; Slice s; uint64_t row = 0, end = 0;
    if (first > tokens || count > tokens - first || !mul(width, sizeof(float), row) ||
        !mul(first, row, s.offset) || !mul(count, row, s.bytes) ||
        !add(s.offset, s.bytes, end) || end > p.residual_bytes) return false;
    out = s; return true;
}

// Non-owning protocol for an external allocation owner. An unrecorded completion is
// never a completion. Destructor does not free: backend owner must quarantine on drain error.
// Generation prevents a previous layer's successful drain being used for a new layer.
class Lease {
    uint64_t generation_ = 0;
    bool open_ = false, joined_ = false, drained_ = false;
public:
    bool begin(uint64_t& generation) {
        if (open_ || generation_ == UINT64_MAX) return false;
        generation = ++generation_; open_ = true; joined_ = drained_ = false; return true;
    }
    bool producers_joined(uint64_t generation) {
        if (!open_ || generation != generation_) return false;
        joined_ = true; return true;
    }
    // Invoke only after all host producers stop/join and both queues, host staging DMA,
    // bridge DMA and last consumers synchronize successfully, with async errors checked.
    // A failed backend drain must pass success=false even if one queue completed.
    bool record_drain(uint64_t generation, bool success) {
        if (!open_ || generation != generation_ || !joined_ || !success) return false;
        drained_ = true; return true;
    }
    bool close(uint64_t generation) {
        if (!open_ || generation != generation_ || !joined_ || !drained_) return false;
        open_ = false; return true;
    }
    bool may_release() const { return !open_; }
};

// Stable expert selection from host metadata only. Existing ordinary cache entries are
// omitted, so the new bounded slot adds reuse rather than duplicating the 128-slot cache.
inline std::vector<uint64_t> select_experts(const std::vector<bool>& ordinary_resident, uint64_t cap) {
    std::vector<uint64_t> ids;
    for (std::size_t e = 0; e < ordinary_resident.size() && ids.size() < cap; ++e)
        if (!ordinary_resident[e]) ids.push_back(e);
    return ids;
}
} // namespace strata::prefill::reuse
