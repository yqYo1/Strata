// Main-authored immutable decode GU cache; repository LICENSE applies.
#pragma once
#include "strata/kernels/cpu/native_expert.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace strata::kernels::cpu::compact_gu {
struct Entry;
struct Key { size_t layer, expert; };
inline constexpr size_t projection_blocks = 6400;
inline constexpr size_t raw_projection_bytes = projection_blocks * 98;

bool available() noexcept;
bool supported(const NativeFmt& format) noexcept;
// Unset/empty/0 disables the optional cache. Reject signs, whitespace and
// overflow before model/runtime setup; this never changes process environment.
bool parse_mib_budget(const char* text, size_t& bytes) noexcept;

// Owns encoded bytes through a synchronous row call, even after the preparing
// cache is retired. Activation/output are borrowed. A lease contains GU only;
// the job's original raw blob must remain alive for hidden quantization/Down.
class Lease {
    std::shared_ptr<const Entry> entry_;
    explicit Lease(std::shared_ptr<const Entry> entry) noexcept;
    friend class Cache;
public:
    Lease() = default;
    explicit operator bool() const noexcept { return bool(entry_); }
    bool matches(const NativeFmt& format) const noexcept;
    bool guards_intact() const noexcept;
    bool gu_rows(NativeGuPhase phase, int nt, std::span<const uint8_t> activation,
                 std::span<float> output, int first, int last) const;
};

// One owner per immutable model/source role. Serialized prepare, then freeze,
// then read-only lookups; no eviction, insertion or source reassignment in use.
// Known bytes exclude allocator/control-block/peak-RSS overhead.
class Cache {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    Cache(size_t layers, size_t experts, size_t known_byte_limit);
    ~Cache();
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    static size_t entry_known_bytes() noexcept;
    bool prepare(Key key, const NativeFmt& format, std::span<const uint8_t> raw_gu);
    void freeze() noexcept;
    Lease lookup(Key key, const NativeFmt& format, NativeGuPhase phase, int nt) const noexcept;
    size_t known_bytes() const noexcept;
    size_t prepared_entries() const noexcept;
    size_t known_byte_limit() const noexcept;
};

// Borrowed immutable host layer, valid through preparation. Coordinates are
// explicit; a transient blob address is never used as model/expert identity.
struct LayerView {
    size_t layer = 0, experts = 0, stride = 0, bytes = 0;
    const uint8_t* data = nullptr;
    NativeFmt format;
};
// All supplied eligible layers or no cache. On admission/encoding/allocation
// failure the caller retains native GU; raw weights remain needed for Down.
std::shared_ptr<const Cache> prepare(size_t layers, size_t experts, size_t byte_limit,
                                     std::span<const LayerView> views, std::string& error);
} // namespace strata::kernels::cpu::compact_gu
