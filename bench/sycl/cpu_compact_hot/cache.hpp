// Main-authored bounded compact-GU experiment; repository LICENSE applies.
// Component only. No production dispatch, source mutation or prefill changes.
#pragma once
#include "compact_iq3xxs.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "ggml.h"
#include <atomic>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace compact_gu_cache {
using strata::kernels::cpu::NativeFmt;
using strata::kernels::cpu::NativeGuPhase;
using Block = compact_iq3xxs::Block;
inline constexpr size_t projection_blocks = compact_iq3xxs::projection_blocks;
inline constexpr size_t raw_projection_bytes = projection_blocks * sizeof(block_iq3_xxs);
inline constexpr size_t storage_blocks = 2 * projection_blocks + 2;

struct Key { size_t layer, expert; };

inline bool available() noexcept {
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    static const bool value = __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return value;
#else
    return false;
#endif
}

inline bool supported(const NativeFmt& f) noexcept {
    return f.gu_type == GGML_TYPE_IQ3_XXS &&
        (f.d_type == GGML_TYPE_IQ4_NL || f.d_type == 42) &&
        f.gu_act == GGML_TYPE_Q8_K && f.n_embd == 2560 && f.n_ff == 640 &&
        f.gu_row == compact_iq3xxs::blocks_per_row * sizeof(block_iq3_xxs) &&
        f.up_off == raw_projection_bytes && f.down_off == 2 * raw_projection_bytes &&
        f.bytes >= f.down_off &&
        f.act_bytes == compact_iq3xxs::blocks_per_row * sizeof(block_q8_K);
}

inline bool same_gu_contract(const NativeFmt& a, const NativeFmt& b) noexcept {
    return a.gu_type == b.gu_type && a.d_type == b.d_type &&
        a.gu_act == b.gu_act && a.n_embd == b.n_embd && a.n_ff == b.n_ff &&
        a.gu_row == b.gu_row && a.up_off == b.up_off &&
        a.down_off == b.down_off && a.act_bytes == b.act_bytes;
}

struct Entry {
    NativeFmt format;
    std::unique_ptr<Block[]> storage = std::make_unique<Block[]>(storage_blocks);

    bool encode(std::span<const uint8_t> raw) {
        if (raw.size_bytes() != 2 * raw_projection_bytes) return false;
        std::memset(&storage[0], 0xa7, sizeof(Block));
        std::memset(&storage[storage_blocks - 1], 0xa7, sizeof(Block));
        for (size_t i = 0; i < 2 * projection_blocks; ++i) {
            if (!compact_iq3xxs::encode_block(raw.subspan(i * 98, 98), storage[i + 1]))
                return false;
        }
        return true;
    }

    bool guards_intact() const noexcept {
        Block guard;
        std::memset(&guard, 0xa7, sizeof guard);
        return std::memcmp(&storage[0], &guard, sizeof guard) == 0 &&
            std::memcmp(&storage[storage_blocks - 1], &guard, sizeof guard) == 0;
    }
};

// A lease owns its view even after the preparing cache/model owner is retired.
// The input activation and output remain borrowed for the synchronous call.
class Lease {
    std::shared_ptr<const Entry> entry_;
    explicit Lease(std::shared_ptr<const Entry> entry) : entry_(std::move(entry)) {}
    friend class Cache;
public:
    Lease() = default;
    explicit operator bool() const noexcept { return bool(entry_); }
    bool guards_intact() const noexcept { return entry_ && entry_->guards_intact(); }

    bool gu_rows(NativeGuPhase phase, int nt, std::span<const uint8_t> activation,
                 std::span<float> output, int first, int last) const {
        if (!entry_ || phase != NativeGuPhase::Decode || nt != 1) return false;
        return compact_iq3xxs::gu_rows(
            {entry_->storage.get() + 1, projection_blocks},
            {entry_->storage.get() + 1 + projection_blocks, projection_blocks},
            activation, output, first, last);
    }
};

// One cache belongs to one immutable model/source generation. Prepare it on one
// host thread, then freeze it before publishing jobs. There is no insertion,
// eviction, model reassignment or environment mutation in a timed lookup.
// A new model/role requires another owner, even when layer/expert IDs coincide.
// Known bytes include the index, Entry objects and guarded block arrays. Shared
// pointer control blocks and allocator overhead require separate RSS evidence.
class Cache {
    size_t layers_, experts_, limit_, used_, prepared_ = 0;
    std::vector<std::shared_ptr<const Entry>> entries_;
    std::atomic<bool> frozen_{false};

    [[gnu::cold, noreturn]] static void invalid(const char* message) {
        throw std::invalid_argument(message);
    }
    static size_t slots(size_t layers, size_t experts, size_t limit) {
        const size_t max = std::numeric_limits<size_t>::max();
        if (!layers || !experts || layers > max / experts ||
            layers * experts > (max - sizeof(Cache)) / sizeof(std::shared_ptr<const Entry>))
            invalid("compact cache index geometry overflow");
        if (sizeof(Cache) + layers * experts * sizeof(std::shared_ptr<const Entry>) > limit)
            invalid("compact cache index exceeds byte budget before allocation");
        return layers * experts;
    }
    bool in_range(Key key) const noexcept {
        return key.layer < layers_ && key.expert < experts_;
    }
public:
    static constexpr size_t entry_known_bytes() noexcept {
        return sizeof(Entry) + storage_blocks * sizeof(Block);
    }
    Cache(size_t layers, size_t experts, size_t known_byte_limit)
        : layers_(layers), experts_(experts), limit_(known_byte_limit), used_(0),
          entries_(slots(layers, experts, known_byte_limit)) {
        if (entries_.capacity() >
            (std::numeric_limits<size_t>::max() - sizeof(Cache)) /
                sizeof(std::shared_ptr<const Entry>))
            invalid("compact cache index byte overflow");
        used_ = sizeof(Cache) + entries_.capacity() * sizeof(std::shared_ptr<const Entry>);
        if (used_ > limit_) invalid("compact cache index exceeds byte budget");
    }
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;

    bool prepare(Key key, const NativeFmt& format, std::span<const uint8_t> raw_gu) {
        if (frozen_.load(std::memory_order_acquire) || !available() || !in_range(key) ||
            !supported(format) || raw_gu.size_bytes() != 2 * raw_projection_bytes ||
            entries_[key.layer * experts_ + key.expert] ||
            entry_known_bytes() > limit_ - used_) return false;
        try {
            auto entry = std::make_shared<Entry>();
            entry->format = format;
            if (!entry->encode(raw_gu)) return false;
            entries_[key.layer * experts_ + key.expert] = std::move(entry);
        } catch (const std::bad_alloc&) {
            return false;
        }
        used_ += entry_known_bytes();
        ++prepared_;
        return true;
    }
    void freeze() noexcept { frozen_.store(true, std::memory_order_release); }

    Lease lookup(Key key, const NativeFmt& format, NativeGuPhase phase, int nt) const noexcept {
        if (!frozen_.load(std::memory_order_acquire) || phase != NativeGuPhase::Decode ||
            nt != 1 || !in_range(key) || !supported(format)) return {};
        const auto& entry = entries_[key.layer * experts_ + key.expert];
        if (!entry || !same_gu_contract(entry->format, format)) return {};
        return Lease(entry);
    }
    size_t known_bytes() const noexcept { return used_; }
    size_t prepared_entries() const noexcept { return prepared_; }
    size_t known_byte_limit() const noexcept { return limit_; }
};
} // namespace compact_gu_cache
