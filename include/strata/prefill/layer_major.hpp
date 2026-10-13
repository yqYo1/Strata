// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace strata::prefill::layer_major {
inline bool exact_one(const char* value) { return value && std::strcmp(value, "1") == 0; }
inline bool requested() {
    static const bool value = exact_one(std::getenv("STRATA_PREFILL_LAYER_MAJOR"));
    return value;
}
// No signs, whitespace, alternate bases, trailing bytes, or overflowing values.
inline bool prefix_value(const char* text, uint64_t& value) {
    if (!text || !*text) return false;
    uint64_t result = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const uint64_t digit = uint64_t(*p - '0');
        if (result > (262144u - digit) / 10) return false;
        result = result * 10 + digit;
    }
    value = result;
    return true;
}
struct Prefix { bool valid; uint64_t tokens; };
inline Prefix frozen_prefix() {
    static const Prefix value = [] {
        uint64_t tokens = 0;
        return Prefix{prefix_value(std::getenv("STRATA_PREFILL_LAYER_MAJOR_R_GPU"), tokens), tokens};
    }();
    return value;
}
inline uint64_t product(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("layer-major byte product overflow");
    const uint64_t result = a * b;
    if (result > std::numeric_limits<size_t>::max()) throw std::overflow_error("layer-major size_t overflow");
    return result;
}
inline uint64_t add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::overflow_error("layer-major byte sum overflow");
    return a + b;
}
struct Layout {
    uint64_t tokens = 0, width = 0, chunk = 0, gpu_tokens = 0, chunks = 0;
    uint64_t gpu_bytes = 0, host_bytes = 0, cache_bytes = 0, gpu_need = 0, host_need = 0;
};
inline Layout plan(int64_t n, int64_t pos0, int64_t context, int64_t chunk, int64_t width,
                   uint64_t requested_prefix, int64_t experts, int64_t layers,
                   int64_t max_blob, uint64_t reserve) {
    if (n <= 0 || n > 262144 || pos0 != 0 || context <= 0 || context > 262144 || n > context ||
        chunk <= 0 || width <= 0 || requested_prefix > 262144 || experts <= 0 || experts > 512 ||
        layers <= 0 || layers > 512 || max_blob <= 0 || reserve == 0)
        throw std::invalid_argument("layer-major unsupported span/shape/prefix/reserve");
    Layout p;
    p.tokens = uint64_t(n); p.width = uint64_t(width); p.chunk = uint64_t(chunk);
    const uint64_t bounded = requested_prefix < uint64_t(n) ? requested_prefix : uint64_t(n);
    p.gpu_tokens = bounded == uint64_t(n) ? uint64_t(n) : bounded / p.chunk * p.chunk;
    p.chunks = p.tokens / p.chunk + (p.tokens % p.chunk != 0);
    p.gpu_bytes = product(product(p.gpu_tokens, p.width), sizeof(float));
    p.host_bytes = product(product(p.tokens - p.gpu_tokens, p.width), sizeof(float));
    const uint64_t slot = add(uint64_t(max_blob), 255) / 256 * 256;
    p.cache_bytes = product(uint64_t(experts), slot);
    if (p.cache_bytes > uint64_t(std::numeric_limits<int64_t>::max()))
        throw std::overflow_error("layer-major cache API signed extent overflow");
    p.gpu_need = add(add(p.gpu_bytes, p.cache_bytes), reserve);
    // Existing workspace/model/head/session owners are already charged in the
    // fresh samples. Reserve also covers unbounded lazy kernel/provider peaks.
    p.host_need = add(add(p.host_bytes, uint64_t(max_blob)),
                      add(product(product(uint64_t(layers), uint64_t(experts)), sizeof(int32_t)), reserve));
    return p;
}
inline bool chunk_store(const Layout& p, uint64_t row, uint64_t count, bool& gpu, uint64_t& offset) {
    if (row > p.tokens || count > p.tokens - row) return false;
    gpu = row < p.gpu_tokens;
    if (gpu && count > p.gpu_tokens - row) return false;
    offset = product(gpu ? row : row - p.gpu_tokens, p.width);
    return true;
}
// Full input is diagnostic-only and must provide a completion observer.
inline bool full_context_span(int64_t n, int64_t pos0, int64_t capacity, bool callback) {
    return n == 262144 && pos0 == 0 && capacity == 262144 && callback;
}
inline int device_ring_slots(int normal, bool enabled) { return enabled ? 0 : normal; }
inline bool final_layer(int64_t end, int64_t layers) { return layers > 0 && end > 0 && end == layers; }
inline bool layer_ready(int64_t experts, int64_t slots, uint64_t bytes,
                        const int32_t* residency, const uint64_t* offsets) {
    if (experts <= 0 || slots != experts || !bytes || !residency || !offsets) return false;
    for (int64_t e = 0; e < experts; ++e)
        if (residency[e] != e || offsets[e + 1] < offsets[e] || offsets[e + 1] - offsets[e] < bytes) return false;
    return true;
}
// Never restore stack residency/callback pointers before actual reader drains.
// A production failure callback exits without unwinding; CPU fixtures can record it.
template<class Drain, class Restore, class Fail>
void retire(Drain drain, Restore restore, Fail fail) noexcept {
    if (!drain()) { fail(); return; }
    restore();
}
} // namespace strata::prefill::layer_major
