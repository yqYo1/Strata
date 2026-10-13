// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/phase_cache_owner.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <utility>

namespace strata::core {
namespace {
bool add(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}
bool round(uint64_t value, uint64_t gran, uint64_t& out) {
    if (!gran) return false;
    return value % gran ? add(value, gran - value % gran, out) : (out = value, true);
}
bool lcm(uint64_t native, uint64_t unit, uint64_t& out) {
    if (!native || !unit) return false;
    const uint64_t factor = unit / std::gcd(native, unit);
    if (native > std::numeric_limits<uint64_t>::max() / factor) return false;
    out = native * factor;
    return true;
}
bool alignment(const PhaseCacheBackend::Caps& caps, uint64_t& out) {
    if (!caps.device_minimum || !caps.context_minimum || caps.granularity % caps.device_minimum) return false;
    // 256-byte payload strides preserve the original native layout. The 2 MiB
    // extent padding is inherited experimental geometry, not a vendor dispatch.
    return lcm(caps.granularity, uint64_t{2} << 20, out) &&
           lcm(out, caps.device_minimum, out) && lcm(out, caps.context_minimum, out) && lcm(out, 256, out);
}
}
PhaseCacheOwner::PhaseCacheOwner(std::shared_ptr<PhaseCacheBackend> backend) : backend_(std::move(backend)) {}
PhaseCacheOwner::~PhaseCacheOwner() {
    if (base_) std::terminate(); // close must succeed before destruction; retain unsafe native owners.
}
bool PhaseCacheOwner::fail(const char* message, std::string& error) {
    poisoned_ = true;
    error = message;
    return false;
}
bool PhaseCacheOwner::reserve(const std::vector<int64_t>& blobs, uint32_t experts,
                              uint64_t virtual_cap, std::string& error, uint32_t extras_per_layer) {
    if (!backend_ || base_ || poisoned_ || blobs.empty() || blobs.size() > 48 || extras_per_layer == 0 || extras_per_layer > 32 || experts < 3 + extras_per_layer || experts > 512)
        return fail("phase cache: invalid owner/layer/expert admission", error);
    PhaseCacheBackend::Caps caps;
    uint64_t gran = 0;
    if (!backend_->caps(caps, error) || !caps.vmm || !caps.free_known || !caps.maximum ||
        !alignment(caps, gran) || !virtual_cap)
        return fail("phase cache: actual VMM/granularity/allocation/free capabilities unavailable", error);
    layers_ = static_cast<uint32_t>(blobs.size());
    experts_ = experts;
    extras_per_layer_ = extras_per_layer;
    granularity_ = gran;
    maximum_ = caps.maximum;
    recommended_ = caps.granularity;
    device_minimum_ = caps.device_minimum;
    context_minimum_ = caps.context_minimum;
    offsets_.assign(static_cast<size_t>(layers_) * (3 + extras_per_layer_), 0);
    strides_.assign(offsets_.size(), 0);
    blobs_.assign(layers_, 0);
    filled_.assign(offsets_.size(), -1);
    staged_.assign(static_cast<size_t>(layers_) * experts_, -1);
    published_.assign(staged_.size(), -1);
    uint64_t largest_blob = 0;
    for (const auto bytes : blobs) {
        if (bytes <= 0) return fail("phase cache: nonpositive source blob", error);
        largest_blob = std::max(largest_blob, static_cast<uint64_t>(bytes));
    }
    if (largest_blob > virtual_cap || largest_blob > maximum_ || largest_blob > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        return fail("phase cache: verification buffer exceeds bounded capacity", error);
    verification_.resize(static_cast<size_t>(largest_blob));
    extents_.assign(layers_ + 1, {}); // All bookkeeping allocated BEFORE native ownership.
    uint64_t at = 0;
    for (uint32_t layer = 0; layer < layers_; ++layer) {
        if (blobs[layer] <= 0) return fail("phase cache: nonpositive native blob size", error);
        uint64_t stride = 0;
        blobs_[layer] = static_cast<uint64_t>(blobs[layer]);
        if (!round(blobs_[layer], 256, stride)) return fail("phase cache: blob stride overflow", error);
        for (uint32_t row = 0; row < 3; ++row) {
            const uint32_t slot = layer * 3 + row;
            offsets_[slot] = at;
            strides_[slot] = stride;
            if (!add(at, stride, at)) return fail("phase cache: base geometry overflow", error);
        }
    }
    if (!round(at, gran, at) || at > maximum_) return fail("phase cache: base physical maximum exceeded", error);
    extents_[0] = {0, at, nullptr, false};
    for (uint32_t layer = 0; layer < layers_; ++layer) {
        const uint64_t stride = strides_[static_cast<size_t>(layer) * 3];
        uint64_t mapped = 0;
        if (stride > std::numeric_limits<uint64_t>::max() / extras_per_layer_ ||
            !round(stride * extras_per_layer_, gran, mapped) || mapped > maximum_)
            return fail("phase cache: extra physical maximum/overflow", error);
        extents_[layer + 1] = {at, mapped, nullptr, false};
        for (uint32_t row = 0; row < extras_per_layer_; ++row) {
            const uint32_t slot = layers_ * 3 + layer * extras_per_layer_ + row;
            if (!add(at, stride * row, offsets_[slot])) return fail("phase cache: extra offset overflow", error);
            strides_[slot] = stride;
        }
        if (!add(at, mapped, at)) return fail("phase cache: extra reservation overflow", error);
    }
    if (at > virtual_cap || at > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        at > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return fail("phase cache: bounded reservation capacity exceeded", error);
    reserved_ = at;
    for (const auto& extent : extents_)
        if (extent.offset % device_minimum_ || extent.bytes % device_minimum_ ||
            extent.offset % context_minimum_ || extent.bytes % context_minimum_ ||
            extent.offset > reserved_ || extent.bytes > reserved_ - extent.offset)
            return fail("phase cache: extent device/context map geometry invalid", error);
    for (size_t slot = 0; slot < offsets_.size(); ++slot) {
        const auto& extent = extents_[extra(static_cast<uint32_t>(slot)) ? slot_layer(static_cast<uint32_t>(slot)) + 1 : 0];
        // Slots are byte subranges of one mapped extent, not individual maps.
        if (offsets_[slot] % 256 || strides_[slot] % 256 || offsets_[slot] < extent.offset ||
            offsets_[slot] - extent.offset > extent.bytes ||
            strides_[slot] > extent.bytes - (offsets_[slot] - extent.offset))
            return fail("phase cache: virtual slot stride/offset/range invalid", error);
    }
    if (reserved_ % context_minimum_) return fail("phase cache: context reservation size invalid", error);
    { PhaseTimedCall timing(timings_.reserve, at);
      base_ = backend_->reserve(at, error);
      timing.result(base_ != nullptr); }
    if (!base_) return fail("phase cache: virtual reservation failed", error);
    if (reinterpret_cast<uintptr_t>(base_) % context_minimum_ ||
        reinterpret_cast<uintptr_t>(base_) % device_minimum_ ||
        at > std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(base_))
        return fail("phase cache: virtual pointer range overflow", error); // retained for checked close
    error.clear();
    return true;
}
bool PhaseCacheOwner::drain(std::string& error) {
    PhaseTimedCall timing(timings_.drain);
    return timing.result(backend_->drain(error));
}
bool PhaseCacheOwner::quiesce(const Quiesce& proof, std::string& error) {
    PhaseTimedCall timing(timings_.quiesce);
    if (!proof) return fail("phase cache: explicit caller quiescence callback missing", error);
    try {
        if (!proof(error) || !drain(error)) {
            poisoned_ = true;
            return false;
        }
    } catch (const std::exception& exception) { return fail(exception.what(), error); }
    return timing.result(true);
}
bool PhaseCacheOwner::budget(uint64_t needed, std::string& error) {
    PhaseCacheBackend::Caps caps;
    uint64_t gran = 0;
    if (!backend_->caps(caps, error) || !caps.vmm || !caps.free_known ||
        !alignment(caps, gran) || gran != granularity_ || caps.granularity != recommended_ ||
        caps.device_minimum != device_minimum_ || caps.context_minimum != context_minimum_ || caps.maximum < maximum_ ||
        needed > caps.free || caps.free - needed < guard_bytes)
        return fail("phase cache: fresh actual free/max/granularity admission failed (512MiB guard)", error);
    return true;
}
bool PhaseCacheOwner::map_extent(size_t index, std::string& error) {
    auto& extent = extents_[index];
    if (extent.physical) return fail("phase cache: mapping already owns a physical handle", error);
    const auto address = reinterpret_cast<uintptr_t>(base_) + extent.offset;
    // physical offset is always zero; each handle backs exactly this extent.
    if (address % device_minimum_ || extent.bytes % device_minimum_ || extent.bytes > maximum_ ||
        extent.offset > reserved_ || extent.bytes > reserved_ - extent.offset)
        return fail("phase cache: physical size/map address/offset invalid", error);
    { PhaseTimedCall timing(timings_.physical, extent.bytes);
      extent.physical = backend_->physical(extent.bytes, error);
      timing.result(extent.physical != nullptr); }
    if (!extent.physical) { poisoned_ = true; return false; }
    // Mark uncertain BEFORE attach: a failed/throwing backend must not authorize deletion.
    extent.maybe_attached = true;
    bool attached = false;
    { PhaseTimedCall timing(timings_.map, extent.bytes);
      attached = timing.result(backend_->attach(extent.physical, static_cast<uint8_t*>(base_) + extent.offset, extent.bytes, error)); }
    if (!attached) {
        poisoned_ = true;
        return false;
    }
    extent.attached = true;
    return true;
}
bool PhaseCacheOwner::detach_extent(size_t index, std::string& error) {
    auto& extent = extents_[index];
    if (!extent.physical) return true;
    // A boolean map/unmap failure provides no proof of the range's state.
    // SYCL unmap requires a known mapped range; never retry unknown ownership.
    if (extent.maybe_attached && !extent.attached)
        return fail("phase cache: uncertain mapping retained; process teardown required", error);
    bool detached = true;
    if (extent.maybe_attached) {
        PhaseTimedCall timing(timings_.unmap, extent.bytes);
        detached = timing.result(backend_->detach(static_cast<uint8_t*>(base_) + extent.offset, extent.bytes, error));
    }
    if (!detached) {
        poisoned_ = true;
        extent.attached = false; // failed detach is uncertain, not known mapped/unmapped
        return false; // DO NOT delete the physical handle or claim unmapped capacity.
    }
    extent.attached = false;
    extent.maybe_attached = false;
    if (!drain(error)) { poisoned_ = true; return false; }
    { PhaseTimedCall timing(timings_.destroy, extent.bytes);
      backend_->destroy_physical(extent.physical);
      timing.result(true); }
    extent.physical = nullptr;
    return true;
}
bool PhaseCacheOwner::map_base(const Quiesce& proof, std::string& error) {
    readers_ready_ = false;
    if (!base_ || poisoned_ || extents_[0].physical) return fail("phase cache: base map state invalid", error);
    if (!quiesce(proof, error) || !budget(extents_[0].bytes, error)) return false;
    return map_extent(0, error);
}
bool PhaseCacheOwner::map_extras(const Quiesce& proof, std::string& error) {
    readers_ready_ = false;
    if (!base_ || poisoned_ || !extents_[0].physical) return fail("phase cache: extra map state invalid", error);
    uint64_t needed = 0;
    for (size_t i = 1; i < extents_.size(); ++i) {
        if (extents_[i].physical || !add(needed, extents_[i].bytes, needed))
            return fail("phase cache: extra map state/size invalid", error);
    }
    if (!quiesce(proof, error) || !budget(needed, error)) return false;
    for (size_t i = 1; i < extents_.size(); ++i) {
        if (map_extent(i, error)) continue;
        const std::string cause = error;
        // Stop before any rollback when attach left an unknown range. Retain
        // all native owners, including known earlier maps and the backend.
        if (extents_[i].maybe_attached && !extents_[i].attached) return false;
        // No inference work is submitted during mapping. Checked drain precedes rollback.
        // Failure remains poisoned even after successful rollback; no automatic map retry.
        if (!drain(error)) return false;
        for (size_t j = i; j > 0; --j)
            if (!detach_extent(j, error)) return false;
        error = cause;
        return false;
    }
    return true;
}
uint32_t PhaseCacheOwner::slot_layer(uint32_t slot) const {
    return extra(slot) ? (slot - layers_ * 3) / extras_per_layer_ : slot / 3;
}
bool PhaseCacheOwner::mapped(uint32_t slot) const {
    if (!base_ || slot >= capacity()) return false;
    const auto& extent = extents_[extra(slot) ? slot_layer(slot) + 1 : 0];
    return extent.physical && extent.attached;
}
void* PhaseCacheOwner::device_slot(uint32_t slot) const {
    return !poisoned_ && mapped(slot) ? static_cast<uint8_t*>(base_) + offsets_[slot] : nullptr;
}
uint64_t PhaseCacheOwner::slot_bytes(uint32_t slot) const {
    return slot < capacity() ? blobs_[slot_layer(slot)] : 0;
}
uint64_t PhaseCacheOwner::slot_stride(uint32_t slot) const {
    return slot < capacity() ? strides_[slot] : 0;
}
std::vector<PhaseCacheOwner::Mapping> PhaseCacheOwner::mapping_snapshot() const {
    std::vector<Mapping> result;
    result.reserve(extents_.size());
    for (const auto& extent : extents_)
        result.push_back({extent.offset, extent.bytes, extent.physical != nullptr, extent.attached,
                          extent.physical && extent.maybe_attached && !extent.attached});
    return result;
}
std::vector<uint32_t> PhaseCacheOwner::layer_slots(uint32_t layer) const {
    if (!base_ || poisoned_ || layer >= layers_) return {};
    std::vector<uint32_t> slots{layer * 3, layer * 3 + 1, layer * 3 + 2};
    for (uint32_t row = 0; row < extras_per_layer_; ++row)
        slots.push_back(layers_ * 3 + layer * extras_per_layer_ + row);
    return slots;
}
int32_t PhaseCacheOwner::vacant_slot(uint32_t layer) const {
    if (poisoned_) return -1;
    for (const uint32_t slot : layer_slots(layer)) if (mapped(slot) && filled_[slot] < 0) return static_cast<int32_t>(slot);
    return -1;
}
int32_t PhaseCacheOwner::resident_slot(uint32_t layer, uint32_t expert) const {
    return readers_ready() && layer < layers_ && expert < experts_ ? published_[static_cast<size_t>(layer) * experts_ + expert] : -1;
}
bool PhaseCacheOwner::fill_checked(uint32_t layer, uint32_t expert, uint32_t slot,
                                  const void* source, uint64_t bytes, std::string& error) {
    readers_ready_ = false;
    if (poisoned_ || layer >= layers_ || expert >= experts_ || slot >= capacity() ||
        slot_layer(slot) != layer || !mapped(slot) || filled_[slot] >= 0 || !source || bytes != blobs_[layer])
        return fail("phase cache: fill source/geometry/mapped admission invalid", error);
    const size_t entry = static_cast<size_t>(layer) * experts_ + expert;
    if (staged_[entry] >= 0) return fail("phase cache: duplicate profile expert", error);
    void* destination = device_slot(slot);
    bool written = false;
    { PhaseTimedCall timing(timings_.h2d, bytes);
      written = timing.result(backend_->write(destination, source, bytes, error)); }
    bool read = false;
    if (written) { PhaseTimedCall timing(timings_.d2h, bytes);
      read = timing.result(backend_->read(verification_.data(), destination, bytes, error)); }
    if (!written || !read) {
        poisoned_ = true;
        return false;
    }
    bool equal = false;
    { PhaseTimedCall timing(timings_.compare, bytes);
      equal = timing.result(std::memcmp(source, verification_.data(), static_cast<size_t>(bytes)) == 0); }
    if (!equal) return fail("phase cache: immutable source/slot byte mismatch", error);
    filled_[slot] = static_cast<int32_t>(expert);
    staged_[entry] = static_cast<int32_t>(slot); // NOT visible through host_residency yet
    return true;
}
bool PhaseCacheOwner::upload(const Publish& callback, std::string& error) {
    PhaseTimedCall timing(timings_.publish);
    if (!callback) return fail("phase cache: checked residency upload callback missing", error);
    for (size_t entry = 0; entry < staged_.size(); ++entry) {
        const int32_t slot = staged_[entry];
        if (slot >= 0 && (static_cast<uint32_t>(slot) >= capacity() || !mapped(static_cast<uint32_t>(slot)) ||
                         slot_layer(static_cast<uint32_t>(slot)) != entry / experts_ ||
                         filled_[static_cast<size_t>(slot)] != static_cast<int32_t>(entry % experts_)))
            return fail("phase cache: residency names an unmapped/unverified/wrong-layer slot", error);
    }
    try {
        if (!callback(staged_.data(), staged_.size(), error) || !drain(error)) {
            poisoned_ = true;
            return false;
        }
    } catch (const std::exception& exception) { return fail(exception.what(), error); }
    std::copy(staged_.begin(), staged_.end(), published_.begin()); // stable host pointer
    readers_ready_ = true;
    return timing.result(true);
}
bool PhaseCacheOwner::publish(const Publish& callback, std::string& error) {
    readers_ready_ = false;
    if (!base_ || poisoned_) return fail("phase cache: closed/poisoned owner cannot publish", error);
    return upload(callback, error);
}
bool PhaseCacheOwner::invalidate_extras(const Publish& callback, std::string& error) {
    PhaseTimedCall timing(timings_.invalidate);
    readers_ready_ = false;
    if (!base_ || poisoned_) return fail("phase cache: invalidation state invalid", error);
    for (auto& slot : staged_) if (slot >= static_cast<int32_t>(layers_ * 3)) slot = -1;
    return timing.result(upload(callback, error));
}
bool PhaseCacheOwner::unmap_extras(const Quiesce& proof, std::string& error) {
    readers_ready_ = false;
    if (!base_ || poisoned_) return fail("phase cache: unmap state invalid", error);
    for (const auto slot : staged_)
        if (slot >= static_cast<int32_t>(layers_ * 3)) return fail("phase cache: extra resident still published", error);
    if (!quiesce(proof, error)) return false;
    for (size_t i = extents_.size(); i > 1; --i) {
        if (!detach_extent(i - 1, error)) return false;
        const uint32_t layer = static_cast<uint32_t>(i - 2);
        for (uint32_t row = 0; row < extras_per_layer_; ++row)
            filled_[layers_ * 3 + layer * extras_per_layer_ + row] = -1;
    }
    readers_ready_ = true; // extra residency already invalidated and checked
    return true;
}
bool PhaseCacheOwner::close(const Quiesce& proof, std::string& error) {
    PhaseTimedCall timing(timings_.close);
    readers_ready_ = false;
    if (!base_) return timing.result(true);
    // Check globally before quiescence or cleanup: close cannot reclaim any
    // ownership while a range's mapping state remains unknown.
    for (const auto& extent : extents_)
        if (extent.physical && extent.maybe_attached && !extent.attached)
            return fail("phase cache: uncertain mapping retained; process teardown required", error);
    if (!quiesce(proof, error)) return false;
    for (size_t i = extents_.size(); i > 0; --i) if (!detach_extent(i - 1, error)) return false;
    if (!drain(error)) { poisoned_ = true; return false; }
    bool released = false;
    { PhaseTimedCall release_timing(timings_.release, reserved_);
      released = release_timing.result(backend_->release(base_, reserved_, error)); }
    if (!released) { poisoned_ = true; return false; }
    base_ = nullptr;
    std::fill(filled_.begin(), filled_.end(), -1);
    std::fill(staged_.begin(), staged_.end(), -1);
    std::fill(published_.begin(), published_.end(), -1);
    return timing.result(true);
}
uint32_t PhaseCacheOwner::mapped_slots() const {
    uint32_t count = 0;
    for (uint32_t slot = 0; slot < capacity(); ++slot) if (mapped(slot)) ++count;
    return count;
}
uint64_t PhaseCacheOwner::mapped_bytes() const {
    uint64_t bytes = 0;
    for (const auto& extent : extents_) if (extent.physical && extent.attached) bytes += extent.bytes;
    return bytes;
}
uint64_t PhaseCacheOwner::retained_physical_bytes() const {
    uint64_t bytes = 0;
    for (const auto& extent : extents_) if (extent.physical) bytes += extent.bytes;
    return bytes;
}
}  // namespace strata::core
