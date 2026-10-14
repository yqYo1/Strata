#!/usr/bin/env python3
"""Regenerate a NON-APPLIED proposal. No Git, subprocess, build, test or GPU calls.

Reads only explicitly pinned repository files, writes source.diff in this folder.
Run from anywhere: python3 /absolute/path/to/this/regenerate.py
"""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PINS = json.loads((HERE / "sourcepins.json").read_text())
old = {}
new = {}
for relative, pin in PINS["sources"].items():
    data = (ROOT / relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise SystemExit("source pin changed; Main must review/rebase: " + relative)
    old[relative] = data.decode()
    new[relative] = old[relative]


def replace(path, before, after, count=1):
    if new[path].count(before) != count:
        raise SystemExit(f"expected {count} anchors in {path}: {before[:100]!r}")
    new[path] = new[path].replace(before, after)


SIZES = "include/strata/core/expert_slot_sizes.hpp"
new[SIZES] = r'''#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::core {
enum class ExpertSlotPartition { last_layer_remainder, balanced };

// Single partition definition for admission, capacity planning and owner lookup.
// Multiplications stay <= slots; do not use layer*slots/layers (can overflow).
inline void expert_layer_slot_range(int64_t slots, int64_t layers, int64_t layer,
                                   int64_t& lo, int64_t& hi,
                                   ExpertSlotPartition partition = ExpertSlotPartition::last_layer_remainder) {
    lo = hi = 0;
    if (slots <= 0 || layers <= 0 || layer < 0 || layer >= layers) return;
    const int64_t q = slots / layers, r = slots % layers;
    if (partition == ExpertSlotPartition::balanced) {
        lo = layer * q + std::min(layer, r);
        hi = lo + q + (layer < r ? 1 : 0);
    } else {
        lo = layer * q;
        hi = layer == layers - 1 ? slots : (layer + 1) * q;
    }
}

inline int64_t expert_slot_layer(int64_t slots, int64_t layers, int64_t slot,
                                ExpertSlotPartition partition = ExpertSlotPartition::last_layer_remainder) {
    if (slot < 0 || slot >= slots || layers <= 0) return -1;
    int64_t a = 0, b = layers;
    // First range with hi > slot, including empty ranges when slots < layers.
    while (a < b) {
        const int64_t m = a + (b - a) / 2;
        int64_t lo = 0, hi = 0;
        expert_layer_slot_range(slots, layers, m, lo, hi, partition);
        if (hi <= slot) a = m + 1; else b = m;
    }
    return a < layers ? a : -1;
}

inline uint64_t expert_slot_aligned_bytes(int64_t bytes) {
    if (bytes <= 0) throw std::invalid_argument("nonpositive expert slot size");
    return (uint64_t(bytes) + 255) / 256 * 256;
}

inline bool expert_balanced_slot_count_valid(int64_t slots, int64_t layers, int64_t experts) {
    if (slots < 0 || layers <= 0 || experts <= 0) return false;
    // ceil(slots/layers) <= experts, without multiplying layers*experts.
    const int64_t q = slots / layers, r = slots % layers;
    return q <= experts && (r == 0 || q < experts);
}

inline std::vector<uint64_t> expert_slot_offsets(const std::vector<int64_t>& bytes) {
    const uint64_t limit = std::min<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max(),
                                             std::numeric_limits<int64_t>::max());
    std::vector<uint64_t> result(bytes.size() + 1, 0);
    for (size_t i = 0; i < bytes.size(); ++i) {
        const uint64_t aligned = expert_slot_aligned_bytes(bytes[i]);
        if (aligned > limit - result[i])
            throw std::overflow_error("expert slot arena exceeds addressable storage");
        result[i + 1] = result[i] + aligned;
    }
    return result;
}

// Raw capacity excludes alignment padding. A zero request means THIS slot's
// blob, never the largest blob of a heterogeneous cache.
inline bool expert_slot_transfer_size(int64_t capacity, int64_t request, uint64_t& bytes) {
    bytes = 0;
    if (capacity <= 0 || request < 0 || request > capacity) return false;
    bytes = uint64_t(request == 0 ? capacity : request);
    return true;
}

inline int64_t per_layer_slots_that_fit(const std::vector<int64_t>& bytes,
                                      uint64_t budget, int64_t maximum,
                                      ExpertSlotPartition partition = ExpertSlotPartition::last_layer_remainder) {
    if (bytes.empty() || maximum < 0 || bytes.size() > uint64_t(std::numeric_limits<int64_t>::max()))
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
    uint64_t left = budget - q * round;
    const int64_t tail_max = std::min(layers - 1, maximum - full);
    if (partition == ExpertSlotPartition::balanced) {
        int64_t tail = 0;
        while (tail < tail_max) {
            const uint64_t next = expert_slot_aligned_bytes(bytes[size_t(tail)]);
            if (next > left) break;
            left -= next;
            ++tail;
        }
        return full + tail;
    }
    const uint64_t last = expert_slot_aligned_bytes(bytes.back());
    return full + int64_t(std::min<uint64_t>(left / last, uint64_t(tail_max)));
}

inline std::vector<int64_t> per_layer_slot_sizes(
    const std::vector<int64_t>& bytes, int64_t slots,
    ExpertSlotPartition partition = ExpertSlotPartition::last_layer_remainder) {
    if (bytes.empty() || slots < 0 || bytes.size() > uint64_t(std::numeric_limits<int64_t>::max()))
        throw std::invalid_argument("invalid per-layer expert slot plan");
    for (const auto b : bytes) expert_slot_aligned_bytes(b);
    std::vector<int64_t> result;
    if (uint64_t(slots) > result.max_size()) throw std::length_error("expert slot count too large");
    result.reserve(size_t(slots));
    for (size_t layer = 0; layer < bytes.size(); ++layer) {
        int64_t lo = 0, hi = 0;
        expert_layer_slot_range(slots, int64_t(bytes.size()), int64_t(layer), lo, hi, partition);
        result.insert(result.end(), size_t(hi - lo), bytes[layer]);
    }
    return result;
}
} // namespace strata::core
'''

CACHE_H = "include/strata/core/expert_cache.hpp"
replace(CACHE_H, '#include <cstddef>', '#include "strata/core/expert_slot_sizes.hpp"\n\n#include <cstddef>')
replace(CACHE_H, '    void set_per_layer_admission(bool on) { per_layer_ = on; }', '''    // Reject policy changes after admission: two admission cursors may alias slots.
    bool set_per_layer_admission(bool on) {
        if (on != per_layer_ && (next_free_ != 0 || admitted_ != 0)) return false;
        per_layer_ = on;
        return true;
    }
    // Choose before open/open_sized. close() keeps this policy for auto retries.
    bool set_slot_partition(ExpertSlotPartition partition) {
        if (valid()) return false;
        partition_ = partition;
        return true;
    }
    ExpertSlotPartition slot_partition() const { return partition_; }
    // Partition owner; meaningful as an ownership contract with per-layer admission.
    int64_t slot_layer(int32_t slot) const {
        return expert_slot_layer(slots_, n_layers_, slot, partition_);
    }
    int64_t slot_capacity(int32_t slot) const {
        if (slot < 0 || slot >= slots_) return 0;
        return slot_bytes_.empty() ? blob_ : slot_bytes_[size_t(slot)];
    }''')
replace(CACHE_H, '                   std::string& err, bool initialize);', '                   std::string& err, bool initialize, uint64_t arena_bytes = 0);\n    bool transfer_size(int32_t slot, int64_t request, uint64_t& bytes, std::string& err) const;')
replace(CACHE_H, '    bool per_layer_ = false;', '    bool per_layer_ = false;\n    ExpertSlotPartition partition_ = ExpertSlotPartition::last_layer_remainder;\n    std::vector<int64_t> slot_bytes_; // raw capacities, without alignment padding')
replace(CACHE_H, '    /// layer taking the remainder.', '    /// layer taking the remainder unless balanced partition was explicitly selected.')

CACHE = "src/core/expert_cache.cpp"
replace(CACHE, '                            std::string& err, bool initialize) {', '                            std::string& err, bool initialize, uint64_t arena_bytes) {')
replace(CACHE, '    const uint64_t want = (uint64_t) n_slots * (uint64_t) blob_bytes;', '''    const uint64_t limit = std::min<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max(),
                                             std::numeric_limits<int64_t>::max());
    if (n_slots > std::numeric_limits<int32_t>::max() ||
        uint64_t(n_layers) > limit / sizeof(int32_t) / uint64_t(n_expert) ||
        (arena_bytes == 0 && uint64_t(n_slots) > limit / uint64_t(blob_bytes)) ||
        arena_bytes > limit) {
        err = "ExpertCache: geometry exceeds addressable storage";
        return false;
    }
    if (partition_ == ExpertSlotPartition::balanced &&
        !expert_balanced_slot_count_valid(n_slots, n_layers, n_expert)) {
        err = "ExpertCache: balanced partition exceeds experts per layer";
        return false;
    }
    const uint64_t want = arena_bytes ? arena_bytes : uint64_t(n_slots) * uint64_t(blob_bytes);''')
replace(CACHE, '"ExpertCache: %lld slots x %lld B = %.2f GiB, but only %.2f GiB of VRAM is free "', '"ExpertCache: %lld slots (max blob %lld B), %.2f GiB total, but only %.2f GiB of VRAM is free "')
start = new[CACHE].index('    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }')
end = new[CACHE].index('\nvoid ExpertCache::close()', start)
new[CACHE] = new[CACHE][:start] + '''    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    if (slot_bytes.size() > uint64_t(std::numeric_limits<int32_t>::max())) {
        err = "ExpertCache: too many sized slots";
        return false;
    }
    std::vector<uint64_t> off;
    try { off = expert_slot_offsets(slot_bytes); }
    catch (const std::exception& e) { err = e.what(); return false; }
    const int64_t mx = *std::max_element(slot_bytes.begin(), slot_bytes.end());
    // Use the actual slot count throughout open_impl. The old byte-sized pseudo
    // slots would overflow int32 cursors and cannot carry a balanced partition.
    if (!open_impl(int64_t(slot_bytes.size()), n_layers, n_expert, mx, err, true, off.back())) return false;
    off_ = std::move(off);
    slot_bytes_ = slot_bytes;
    return true;
}
''' + new[CACHE][end:]
replace(CACHE, '    off_.clear();', '    off_.clear();\n    slot_bytes_.clear();')
start = new[CACHE].index('/// R4.2g.  Layer `l` owns')
end = new[CACHE].index('\nint32_t ExpertCache::slot_of', start)
new[CACHE] = new[CACHE][:start] + '''// Both allocation planning and admission consume the same shared partition.
void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    expert_layer_slot_range(slots_, n_layers_, layer, lo, hi, partition_);
}

bool ExpertCache::transfer_size(int32_t slot, int64_t request, uint64_t& bytes, std::string& err) const {
    if (!expert_slot_transfer_size(slot_capacity(slot), request, bytes)) {
        err = "ExpertCache: transfer exceeds slot raw capacity or has invalid slot/size";
        return false;
    }
    return true;
}
''' + new[CACHE][end:]
replace(CACHE, '    const size_t n = (size_t) (bytes > 0 && bytes <= blob_ ? bytes : blob_);', '    uint64_t n = 0;\n    if (!transfer_size(slot, bytes, n, err)) return false;', count=3)
replace(CACHE, '    const int64_t nb = bytes > 0 && bytes <= blob_ ? bytes : blob_;', '    uint64_t nb = 0;\n    if (!transfer_size(slot, bytes, nb, err)) return false;')
replace(CACHE, '    if (src == nullptr) {\n        err = "ExpertCache::verify_slot: slot outside the arena";', '    if (src == nullptr || host_blob == nullptr) {\n        err = "ExpertCache::verify_slot: invalid slot or null host blob";')

GEN = "src/program/generate.cpp"
replace(GEN, '    bool expert_cache_per_layer = false;', '    bool expert_cache_per_layer = false;\n    bool expert_cache_balanced = false; // experimental SYCL opt-in; legacy default remains')
replace(GEN, '        else if (a == "--expert-cache-per-layer") o.expert_cache_per_layer = true;', '        else if (a == "--expert-cache-per-layer") o.expert_cache_per_layer = true;\n        else if (a == "--expert-cache-balanced") o.expert_cache_balanced = true;')
replace(GEN, '                 "  --expert-cache-per-layer  R4.2g:', '                 "  --expert-cache-balanced  SYCL native experiment: balance total slots over layers; requires\\n"\n                 "                       --expert-cache-per-layer, one GPU, no peer/remote cache. Default off.\\n"\n                 "  --expert-cache-per-layer  R4.2g:')
replace(GEN, '    strata::core::set_coupled_draft(o.coupled_draft);', '''    if (o.expert_cache_balanced) {
#ifndef STRATA_ENABLE_SYCL
        std::fprintf(stderr, "strata generate: --expert-cache-balanced requires SYCL\\n");
        return 2;
#else
        if (!o.expert_cache_per_layer || o.expert_cache == 0 || !o.layer_split.empty() ||
            o.peer_device >= 0 || o.expert_cache_remote[0] > 0 ||
            o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0) {
            std::fprintf(stderr, "strata generate: balanced cache needs --expert-cache-per-layer, "
                                 "a positive/auto cache, one GPU and no peer/remote cache\\n");
            return 2;
        }
#endif
    }
    strata::core::set_coupled_draft(o.coupled_draft);''')
replace(GEN, '    std::vector<int64_t> sized_slots;', '''    const auto cache_partition = o.expert_cache_balanced
        ? strata::core::ExpertSlotPartition::balanced
        : strata::core::ExpertSlotPartition::last_layer_remainder;
    if (!xcache.set_slot_partition(cache_partition)) {
        std::fprintf(stderr, "strata generate: cache partition must be set before open\\n");
        return 2;
    }
    if (o.expert_cache_balanced && !native_pack) {
        std::fprintf(stderr, "strata generate: balanced cache experiment requires a native expert pack\\n");
        return 2;
    }
    if (o.expert_cache_balanced && !auto_cache && o.expert_cache > 0 &&
        !strata::core::expert_balanced_slot_count_valid(o.expert_cache, g.n_layers, g.n_expert)) {
        std::fprintf(stderr, "strata generate: balanced cache exceeds total model expert count\\n");
        return 2;
    }
    std::vector<int64_t> sized_slots;''')
replace(GEN, '''    if (native_pack && o.expert_cache > 0 && o.expert_cache_per_layer) {
        const auto& lay = strata::kernels::cpu::expert_layout();''', '''    if (native_pack && o.expert_cache > 0 && o.expert_cache_per_layer) {
        const auto& lay = strata::kernels::cpu::expert_layout();
        if (o.expert_cache_balanced && (lay.max_blob > uint64_t(INT64_MAX) ||
            lay.max_blob > UINT64_MAX / uint64_t(o.expert_cache))) {
            std::fprintf(stderr, "strata generate: balanced cache byte budget overflows\\n");
            return 2;
        }''')
replace(GEN, 'per_layer_bytes, budget, g.n_layers * g.n_expert);', 'per_layer_bytes, budget, g.n_layers * g.n_expert, cache_partition);')
replace(GEN, 'strata::core::per_layer_slot_sizes(per_layer_bytes, o.expert_cache);', 'strata::core::per_layer_slot_sizes(per_layer_bytes, o.expert_cache, cache_partition);', count=2)
replace(GEN, '                    o.expert_cache);\n                sized_slots', '                    o.expert_cache, cache_partition);\n                sized_slots')
replace(GEN, '        xcache.set_per_layer_admission(o.expert_cache_per_layer);', '''        if (!xcache.set_per_layer_admission(o.expert_cache_per_layer)) {
            std::fprintf(stderr, "strata generate: cannot change cache admission after fills\\n");
            return 2;
        }''')
replace(GEN, '''            int64_t lo = 0, hi = 0;
            xcache.layer_slot_range(0, lo, hi);
            std::fprintf(stderr, "                 R4.2g PER-LAYER: each layer owns %lld slots (%lld..%lld).\\n",
                         (long long) (hi - lo), (long long) lo, (long long) (hi - 1));''', '''            int64_t least = xcache.slots(), most = 0;
            for (int64_t layer = 0; layer < g.n_layers; ++layer) {
                int64_t lo = 0, hi = 0;
                xcache.layer_slot_range(layer, lo, hi);
                least = std::min(least, hi - lo);
                most = std::max(most, hi - lo);
            }
            std::fprintf(stderr, "                 PER-LAYER %s: %lld..%lld slots/layer, %lld allocation bytes.\\n",
                         o.expert_cache_balanced ? "balanced" : "legacy-last-remainder",
                         (long long) least, (long long) most, (long long) xcache.bytes());''')
replace(GEN, '    int64_t prefilled = 0;', '    int64_t prefilled = 0;\n    int32_t first_filled_slot = strata::core::kNotResident;\n    std::pair<int32_t, int32_t> first_filled_pair{};')
replace(GEN, '            ++prefilled;', '''            if (first_filled_slot == strata::core::kNotResident) {
                first_filled_slot = slot;
                first_filled_pair = profile[size_t(i)];
            }
            ++prefilled;''')
replace(GEN, '''        if (prefilled > 0 && !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                srcp->blob(profile[0].first, profile[0].second), err,
                                (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[0].first)))''', '''        if (prefilled > 0 && !xcache.verify_slot(first_filled_slot,
                                srcp->blob(first_filled_pair.first, first_filled_pair.second), err,
                                (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(first_filled_pair.first)))''')
replace(GEN, '''        std::fprintf(stderr, "strata generate: pre-filled %lld of %lld slots from the profile; slot 0 verified\\n",
                     (long long) prefilled, (long long) (per_layer ? xcache.slots() : want));''', '''        std::fprintf(stderr, "strata generate: pre-filled %lld of %lld slots from the profile; verified slot %d\\n",
                     (long long) prefilled, (long long) (per_layer ? xcache.slots() : want), first_filled_slot);''')
# Adaptive copies bypass ExpertCache::fill_slot. Validate the balanced owner and
# raw capacity in BOTH serve and one-shot exchange paths before direct copies.
replace(GEN, '''        if (slot < 0) continue;
        if (cudaMemcpyAsync(src.exchange_buffer(q), cache.device_slot(slot),''', '''        if (slot < 0) continue;
        if (cache.slot_partition() == strata::core::ExpertSlotPartition::balanced &&
            (cache.slot_layer(slot) != s.layer || uint64_t(cache.slot_capacity(slot)) <
                strata::kernels::cpu::expert_layout().blob_bytes(s.layer))) return false;
        if (cudaMemcpyAsync(src.exchange_buffer(q), cache.device_slot(slot),''')
replace(GEN, '''                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot), b,''', '''                const auto& target_cache = gs ? gs->cache : xcache;
                if (o.expert_cache_balanced && (target_cache.slot_layer(slot) != s.layer ||
                    uint64_t(target_cache.slot_capacity(slot)) < strata::kernels::cpu::expert_layout().blob_bytes(s.layer)))
                    return false;
                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot), b,''')
replace(GEN, '''                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(xcache.device_slot(slot), b,''', '''                if (o.expert_cache_balanced && (xcache.slot_layer(slot) != s.layer ||
                    uint64_t(xcache.slot_capacity(slot)) < strata::kernels::cpu::expert_layout().blob_bytes(s.layer)))
                    return false;
                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(xcache.device_slot(slot), b,''')

# A reusable CPU-only fixture is also called by the EXISTING host_engine test.
# This is substantive boundary/capacity coverage, not an equality-of-formulas test.
FIXTURE = "tests/cpu/expert_slot_partition_fixture.hpp"
new[FIXTURE] = r'''#pragma once
#include "strata/core/expert_slot_sizes.hpp"
#include <stdexcept>

inline void check_expert_slot_partition_fixture() {
    using namespace strata::core;
    using P = ExpertSlotPartition;
    auto check = [](bool ok) { if (!ok) throw std::runtime_error("expert slot partition fixture"); };
    auto throws = [&](auto operation) {
        bool caught = false;
        try { operation(); } catch (const std::exception&) { caught = true; }
        check(caught);
    };
    // Explicit independent expected quotas; 128/48 is NOT 2 or 3 in legacy.
    for (const auto partition : {P::last_layer_remainder, P::balanced}) {
        for (const int64_t slots : {int64_t(0), int64_t(1), int64_t(47), int64_t(48),
                                   int64_t(49), int64_t(128), int64_t(512), int64_t(24576)}) {
            int64_t prior = 0, total = 0;
            for (int64_t layer = 0; layer < 48; ++layer) {
                int64_t lo = -1, hi = -1;
                expert_layer_slot_range(slots, 48, layer, lo, hi, partition);
                check(lo == prior && lo <= hi && hi <= slots);
                if (slots == 128) check(hi - lo == (partition == P::balanced ? (layer < 32 ? 3 : 2)
                                                                                           : (layer == 47 ? 34 : 2)));
                if (slots == 1) check(hi - lo == (partition == P::balanced ? (layer == 0 ? 1 : 0)
                                                                                         : (layer == 47 ? 1 : 0)));
                if (slots == 24576) check(hi - lo == 512);
                if (slots == 512 && partition == P::balanced) check(hi - lo == (layer < 32 ? 11 : 10));
                for (int64_t slot = lo; slot < hi; ++slot) check(expert_slot_layer(slots, 48, slot, partition) == layer);
                prior = hi; total += hi - lo;
            }
            check(total == slots && expert_slot_layer(slots, 48, -1, partition) == -1);
            check(expert_slot_layer(slots, 48, slots, partition) == -1);
        }
        int64_t lo = -1, hi = -1;
        expert_layer_slot_range(128, 48, -1, lo, hi, partition); check(lo == 0 && hi == 0);
        expert_layer_slot_range(128, 48, 48, lo, hi, partition); check(lo == 0 && hi == 0);
        expert_layer_slot_range(INT64_MAX, 48, 47, lo, hi, partition); check(hi == INT64_MAX && lo <= hi);
        check(expert_slot_layer(INT64_MAX, 48, INT64_MAX - 1, partition) == 47);
        expert_layer_slot_range(INT64_MAX, 1, 0, lo, hi, partition); check(lo == 0 && hi == INT64_MAX);
    }
    check(expert_slot_layer(3, 0, 0) == -1);
    check(expert_balanced_slot_count_valid(0, 48, 512));
    check(expert_balanced_slot_count_valid(24576, 48, 512));
    check(!expert_balanced_slot_count_valid(24577, 48, 512));
    check(!expert_balanced_slot_count_valid(49, 48, 1));
    check(!expert_balanced_slot_count_valid(-1, 48, 512));
    check(expert_balanced_slot_count_valid(INT64_MAX, INT64_MAX, INT64_MAX));
    check(!expert_balanced_slot_count_valid(INT64_MAX, 1, 512));

    // Heterogeneous raw blobs straddle 256-byte boundaries. Check exact offsets
    // and transfers, including the first byte beyond raw size and beyond padding.
    const std::vector<int64_t> raw{1, 255, 256, 257, 5001, 2057, 9000};
    check(expert_slot_offsets(raw) == std::vector<uint64_t>({0,256,512,768,1280,6400,8704,17920}));
    for (int64_t b : raw) {
        uint64_t n = 123;
        check(expert_slot_transfer_size(b, 0, n) && n == uint64_t(b));
        check(expert_slot_transfer_size(b, b, n) && n == uint64_t(b));
        check(!expert_slot_transfer_size(b, b + 1, n) && n == 0);
        check(!expert_slot_transfer_size(b, -1, n));
    }
    uint64_t n = 1;
    check(!expert_slot_transfer_size(0, 0, n));
    throws([] { expert_slot_offsets({1, 0}); });
    throws([] { expert_slot_offsets({1, -1}); });
    throws([] { expert_slot_offsets({INT64_MAX}); });
    throws([] { per_layer_slot_sizes({}, 1); });
    throws([] { per_layer_slot_sizes({256}, -1); });
    throws([] { per_layer_slots_that_fit({INT64_MAX, INT64_MAX}, UINT64_MAX, 1); });

    // Independent known costs show that same count is not same bytes.
    const std::vector<int64_t> blobs{5001,2057,9000}; // aligned: 5120,2304,9216
    check(per_layer_slot_sizes(blobs, 8) == std::vector<int64_t>({5001,5001,2057,2057,9000,9000,9000,9000}));
    check(per_layer_slot_sizes(blobs, 8, P::balanced) == std::vector<int64_t>({5001,5001,5001,2057,2057,2057,9000,9000}));
    check(expert_slot_offsets(per_layer_slot_sizes(blobs, 8)).back() == 51712);
    check(expert_slot_offsets(per_layer_slot_sizes(blobs, 8, P::balanced)).back() == 40704);
    for (const auto partition : {P::last_layer_remainder, P::balanced}) {
        // Exhaust legal count/budget pairs; cost can drop at legacy round
        // transitions. An ordinary monotonic binary search is invalid there.
        std::vector<uint64_t> costs;
        for (int64_t slots = 0; slots <= 20; ++slots)
            costs.push_back(expert_slot_offsets(per_layer_slot_sizes(blobs, slots, partition)).back());
        for (uint64_t cap = 0; cap <= 120000; cap += 127) {
            int64_t expected = 0;
            for (int64_t slots = 0; slots <= 20; ++slots)
                if (costs[size_t(slots)] <= cap) expected = slots;
            check(per_layer_slots_that_fit(blobs, cap, 20, partition) == expected);
        }
        for (int64_t slots = 1; slots <= 20; ++slots) {
            const uint64_t exact = costs[size_t(slots)];
            int64_t expected_before = 0, expected_exact = 0;
            for (int64_t k = 0; k <= slots; ++k) {
                if (costs[size_t(k)] < exact) expected_before = k;
                if (costs[size_t(k)] <= exact) expected_exact = k;
            }
            check(per_layer_slots_that_fit(blobs, exact - 1, slots, partition) == expected_before);
            check(per_layer_slots_that_fit(blobs, exact, slots, partition) == expected_exact);
        }
        // Each slot in a real-shaped plan must have the owner's raw capacity,
        // across all quota boundaries, including zero-slot layers and >512/layer.
        std::vector<int64_t> sizes48;
        for (int i = 0; i < 48; ++i) sizes48.push_back(raw[size_t(i) % raw.size()]);
        for (const int64_t slots : {int64_t(1), int64_t(47), int64_t(49), int64_t(128),
                                   int64_t(512), int64_t(24576), int64_t(24577)}) {
            const auto plan = per_layer_slot_sizes(sizes48, slots, partition);
            const auto offsets = expert_slot_offsets(plan);
            for (int64_t slot = 0; slot < slots; ++slot) {
                const int64_t layer = expert_slot_layer(slots, 48, slot, partition);
                check(layer >= 0 && layer < 48 && plan[size_t(slot)] == sizes48[size_t(layer)]);
                check(offsets[size_t(slot + 1)] - offsets[size_t(slot)] >= uint64_t(plan[size_t(slot)]));
            }
        }
    }
}
'''
new['tests/cpu/expert_slot_partition.cpp'] = '''#include "expert_slot_partition_fixture.hpp"
#include <cstdio>
int main() {
    check_expert_slot_partition_fixture();
    std::puts("expert slot partition CPU fixture passed");
}
'''
actual_blobs = PINS['external_model_layout']['blob_bytes_by_layer']
model_fixture = '''    // Pinned IQ3_S native_experts.txt metadata, 48 layers / 512 experts.
    // All seven real raw blob sizes are already 256-byte aligned.
    const std::vector<int64_t> actual_blobs{''' + ','.join(map(str, actual_blobs)) + '''};
    for (const int64_t b : actual_blobs) {
        check(expert_slot_aligned_bytes(b - 1) == uint64_t(b));
        check(expert_slot_aligned_bytes(b) == uint64_t(b));
        check(expert_slot_aligned_bytes(b + 1) == uint64_t(b + 256));
        uint64_t count = 0;
        check(expert_slot_transfer_size(b, b - 1, count) && count == uint64_t(b - 1));
        check(!expert_slot_transfer_size(b, b + 1, count));
    }
    check(expert_slot_offsets(per_layer_slot_sizes(actual_blobs, 128)).back() == 281651200);
    check(expert_slot_offsets(per_layer_slot_sizes(actual_blobs, 128, P::balanced)).back() == 260044800);
    check(per_layer_slots_that_fit(actual_blobs, 281651200, 24576, P::balanced) == 138);
    check(expert_slot_offsets(per_layer_slot_sizes(actual_blobs, 138, P::balanced)).back() == 280832000);
    check(expert_slot_offsets(per_layer_slot_sizes(actual_blobs, 139, P::balanced)).back() == 283008000);

'''
replace(FIXTURE, '    // Heterogeneous raw blobs straddle 256-byte boundaries.', model_fixture + '    // Heterogeneous raw blobs straddle 256-byte boundaries.')
CPU = 'tests/cpu/decode_joined_mtp4.cpp'
replace(CPU, '#include "strata/core/decode_service_budget.hpp"', '#include "strata/core/decode_service_budget.hpp"\n#include "expert_slot_partition_fixture.hpp"')
replace(CPU, 'int main() {', 'int main() {\n    check_expert_slot_partition_fixture();')
HOST = "tests/sycl/host_engine.cpp"
replace(HOST, '#include "strata/core/expert_cache.hpp"', '#include "strata/core/expert_cache.hpp"\n#include "../cpu/expert_slot_partition_fixture.hpp"')
replace(HOST, 'int main() {\n  try {', 'int main() {\n  try {\n    check_expert_slot_partition_fixture();')
# Run the existing real allocation/admission/fill fixture with both policies.
replace(HOST, '      constexpr int slots = 8;\n      auto plan', '      constexpr int slots = 8;\n      for (const auto partition : {ExpertSlotPartition::last_layer_remainder, ExpertSlotPartition::balanced}) {\n      require(cache.set_slot_partition(partition), "set partition before allocation");\n      auto plan')
replace(HOST, 'per_layer_slot_sizes(sizes, slots)', 'per_layer_slot_sizes(sizes, slots, partition)')
replace(HOST, 'per_layer_slots_that_fit(sizes, budget, slots)', 'per_layer_slots_that_fit(sizes, budget, slots, partition)')
replace(HOST, 'per_layer_slot_sizes(sizes, n)', 'per_layer_slot_sizes(sizes, n, partition)')
replace(HOST, 'per_layer_slots_that_fit(sizes, cap, slots)', 'per_layer_slots_that_fit(sizes, cap, slots, partition)')
replace(HOST, '        require(hi - lo == (layer == 2 ? 4 : 2), "per-layer remainder range");', '''        require(hi - lo == (partition == ExpertSlotPartition::balanced ? (layer < 2 ? 3 : 2)
                                                                         : (layer == 2 ? 4 : 2)),
                "per-layer partition range");''')
replace(HOST, '          require(cache.admit(layer, s - lo) == s,', '''          require(cache.slot_layer(int32_t(s)) == layer && cache.slot_capacity(int32_t(s)) == sizes[size_t(layer)],
                  "owner and raw capacity");
          require(cache.admit(layer, s - lo) == s,''')
replace(HOST, '      const int64_t shrunk = per_layer_slots_that_fit(sizes, budget * 3 / 4, slots);', '      const int64_t shrunk = per_layer_slots_that_fit(sizes, budget * 3 / 4, slots, partition);')
replace(HOST, 'per_layer_slot_sizes(sizes, shrunk)', 'per_layer_slot_sizes(sizes, shrunk, partition)')
replace(HOST, '''      cache.close();
    }
    for (int reopen''', '''      cache.close();
      }
      require(cache.set_slot_partition(ExpertSlotPartition::balanced), "set balanced");
      require(!cache.open(24577, 48, 512, 1, err), "reject more slots than model experts before allocation");
      require(!cache.open(INT64_MAX, 48, 512, 1, err), "reject cursor overflow before allocation");
      require(cache.set_slot_partition(ExpertSlotPartition::last_layer_remainder), "restore legacy");
    }
    for (int reopen''')

CMAKE = 'CMakeLists.txt'
replace(CMAKE, 'if(STRATA_BUILD_TESTS)\n  enable_testing()', '''if(STRATA_BUILD_TESTS)
  enable_testing()
  add_executable(expert_slot_partition_host_test tests/cpu/expert_slot_partition.cpp)
  target_include_directories(expert_slot_partition_host_test PRIVATE include)
  target_link_libraries(expert_slot_partition_host_test PRIVATE strata_warnings)
  add_test(NAME expert_slot_partition_host COMMAND expert_slot_partition_host_test)
  set_tests_properties(expert_slot_partition_host PROPERTIES TIMEOUT 30)''')

patch = ''.join(''.join(difflib.unified_diff(old.get(path, '').splitlines(keepends=True),
                                           content.splitlines(keepends=True),
                                           fromfile='a/' + path if path in old else '/dev/null',
                                           tofile='b/' + path))
                for path, content in new.items() if old.get(path, '') != content)
(HERE / 'source.diff').write_text(patch)
print(json.dumps({'source_diff_sha256': hashlib.sha256(patch.encode()).hexdigest(),
                  'changed_paths': [p for p in new if old.get(p, '') != new[p]],
                  'status': 'proposal only; unapplied; no tests/builds/GPU executed'}, indent=2))
