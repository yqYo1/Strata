// Main-authored production owner for the validated compact GU representation.
// Derived only from this project's owned component; repository LICENSE applies.
#include "strata/kernels/cpu/compact_gu.hpp"
#include "strata/kernels/cpu/detail/compact_iq3xxs.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"
#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::kernels::cpu::compact_gu {
using Block = compact_iq3xxs::Block;
static_assert(projection_blocks == compact_iq3xxs::projection_blocks);
constexpr size_t storage_blocks = 2 * projection_blocks + 2;

bool available() noexcept { return cpu_avx2_ok() && !cpu_avx512_ok(); }
bool parse_mib_budget(const char* text, size_t& bytes) noexcept {
    bytes = 0;
    if (!text || !*text) return true;
    size_t value = 0;
    constexpr size_t maximum = std::numeric_limits<size_t>::max() >> 20;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const size_t digit = size_t(*p - '0');
        if (value > (maximum - digit) / 10) return false;
        value = 10 * value + digit;
    }
    bytes = value << 20; return true;
}
bool supported(const NativeFmt& f) noexcept {
    return f.gu_type == GGML_TYPE_IQ3_XXS &&
        (f.d_type == GGML_TYPE_IQ4_NL || f.d_type == 42) &&
        f.gu_act == GGML_TYPE_Q8_K && f.n_embd == 2560 && f.n_ff == 640 &&
        f.gu_row == compact_iq3xxs::blocks_per_row * sizeof(block_iq3_xxs) &&
        f.up_off == raw_projection_bytes && f.down_off == 2 * raw_projection_bytes &&
        f.bytes >= f.down_off &&
        f.act_bytes == compact_iq3xxs::blocks_per_row * sizeof(block_q8_K);
}
bool same_gu_contract(const NativeFmt& a, const NativeFmt& b) noexcept {
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
        for (size_t i = 0; i < 2 * projection_blocks; ++i)
            if (!compact_iq3xxs::encode_block(raw.subspan(i * 98, 98), storage[i + 1])) return false;
        return true;
    }
    bool guards_intact() const noexcept {
        Block guard;
        std::memset(&guard, 0xa7, sizeof guard);
        return std::memcmp(&storage[0], &guard, sizeof guard) == 0 &&
            std::memcmp(&storage[storage_blocks - 1], &guard, sizeof guard) == 0;
    }
};
Lease::Lease(std::shared_ptr<const Entry> entry) noexcept : entry_(std::move(entry)) {}
bool Lease::matches(const NativeFmt& format) const noexcept {
    return entry_ && same_gu_contract(entry_->format, format);
}
bool Lease::guards_intact() const noexcept { return entry_ && entry_->guards_intact(); }
bool Lease::gu_rows(NativeGuPhase phase, int nt, std::span<const uint8_t> activation,
                    std::span<float> output, int first, int last) const {
    if (!entry_ || phase != NativeGuPhase::Decode || nt != 1) return false;
    return compact_iq3xxs::gu_rows({entry_->storage.get() + 1, projection_blocks},
        {entry_->storage.get() + 1 + projection_blocks, projection_blocks},
        activation, output, first, last);
}
struct Cache::Impl {
    size_t layers, experts, limit, used, prepared = 0;
    std::vector<std::shared_ptr<const Entry>> entries;
    std::atomic<bool> frozen{false};
    [[gnu::cold, noreturn]] static void invalid(const char* why) { throw std::invalid_argument(why); }
    static size_t slots(size_t layers, size_t experts, size_t limit) {
        const auto max = std::numeric_limits<size_t>::max();
        if (!layers || !experts || layers > max / experts ||
            layers * experts > (max - sizeof(Cache) - sizeof(Impl)) / sizeof(std::shared_ptr<const Entry>))
            invalid("compact cache index geometry overflow");
        if (sizeof(Cache) + sizeof(Impl) + layers * experts * sizeof(std::shared_ptr<const Entry>) > limit)
            invalid("compact cache index exceeds byte budget before allocation");
        return layers * experts;
    }
    Impl(size_t nl, size_t ne, size_t cap)
        : layers(nl), experts(ne), limit(cap), used(0), entries(slots(nl, ne, cap)) {
        const auto max = std::numeric_limits<size_t>::max();
        if (entries.capacity() > (max - sizeof(Cache) - sizeof(Impl)) / sizeof(std::shared_ptr<const Entry>))
            invalid("compact cache index byte overflow");
        used = sizeof(Cache) + sizeof(Impl) + entries.capacity() * sizeof(std::shared_ptr<const Entry>);
        if (used > limit) invalid("compact cache index exceeds byte budget");
    }
    bool in_range(Key key) const noexcept { return key.layer < layers && key.expert < experts; }
};
Cache::Cache(size_t layers, size_t experts, size_t cap) : impl_(std::make_unique<Impl>(layers, experts, cap)) {}
Cache::~Cache() = default;
size_t Cache::entry_known_bytes() noexcept { return sizeof(Entry) + storage_blocks * sizeof(Block); }
bool Cache::prepare(Key key, const NativeFmt& format, std::span<const uint8_t> raw_gu) {
    auto& s = *impl_;
    if (s.frozen.load(std::memory_order_acquire) || !available() || !s.in_range(key) || !supported(format) ||
        raw_gu.size_bytes() != 2 * raw_projection_bytes || s.entries[key.layer * s.experts + key.expert] ||
        entry_known_bytes() > s.limit - s.used) return false;
    try {
        auto entry = std::make_shared<Entry>(); entry->format = format;
        if (!entry->encode(raw_gu)) return false;
        s.entries[key.layer * s.experts + key.expert] = std::move(entry);
    } catch (const std::bad_alloc&) { return false; }
    s.used += entry_known_bytes(); ++s.prepared; return true;
}
void Cache::freeze() noexcept { impl_->frozen.store(true, std::memory_order_release); }
Lease Cache::lookup(Key key, const NativeFmt& format, NativeGuPhase phase, int nt) const noexcept {
    const auto& s = *impl_;
    if (!s.frozen.load(std::memory_order_acquire) || phase != NativeGuPhase::Decode || nt != 1 ||
        !s.in_range(key) || !supported(format)) return {};
    const auto& entry = s.entries[key.layer * s.experts + key.expert];
    if (!entry || !same_gu_contract(entry->format, format)) return {};
    return Lease(entry);
}
size_t Cache::known_bytes() const noexcept { return impl_->used; }
size_t Cache::prepared_entries() const noexcept { return impl_->prepared; }
size_t Cache::known_byte_limit() const noexcept { return impl_->limit; }
std::shared_ptr<const Cache> prepare(size_t layers, size_t experts, size_t byte_limit,
                                     std::span<const LayerView> views, std::string& error) {
    if (views.empty()) { error = "no eligible immutable GU layers"; return {}; }
    try {
        auto cache = std::make_shared<Cache>(layers, experts, byte_limit);
        const auto max = std::numeric_limits<size_t>::max();
        if (views.size() > max / experts ||
            views.size() * experts > (byte_limit - cache->known_bytes()) / Cache::entry_known_bytes()) {
            error = "all eligible GU views exceed the known-byte budget"; return {};
        }
        for (size_t i = 0; i < views.size(); ++i) {
            const auto& view = views[i];
            if (view.layer >= layers || view.experts != experts || !view.data || !supported(view.format) ||
                view.stride != view.format.bytes || experts > max / view.stride ||
                view.bytes < experts * view.stride ||
                reinterpret_cast<uintptr_t>(view.data) > std::numeric_limits<uintptr_t>::max() - view.bytes) {
                error = "immutable GU source layer has invalid bounds or format"; return {};
            }
            for (size_t j = 0; j < i; ++j) if (views[j].layer == view.layer) {
                error = "duplicate immutable GU source layer"; return {};
            }
        }
        for (const auto& view : views) for (size_t expert = 0; expert < experts; ++expert) {
            const auto raw = std::span<const uint8_t>(view.data + expert * view.stride, view.format.down_off);
            if (!cache->prepare({view.layer, expert}, view.format, raw)) {
                error = "GU view encoding/allocation failed; native fallback retained"; return {};
            }
        }
        cache->freeze(); error.clear(); return cache;
    } catch (const std::exception& problem) { error = problem.what(); return {}; }
}
} // namespace strata::kernels::cpu::compact_gu
